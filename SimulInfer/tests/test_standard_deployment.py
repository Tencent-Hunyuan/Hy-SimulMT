"""Public vLLM chat protocol and opt-in policy behavior."""
import base64
import io
import json
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
import soundfile as sf
from jinja2 import Environment

from simulinfer.audio import part
from simulinfer.budget import render
from simulinfer.config import Model, Settings, load
from simulinfer.translation import ChatAPI, VLLMSession, SRC, TGT, EOT, chat_request


def response(text='translated',finish='stop',prompt=8000):
    return httpx.Response(200,json={'choices':[{'message':{'content':text},
        'finish_reason':finish,'stop_reason':120008,'token_ids':[1]}],
        'usage':{'prompt_tokens':prompt,'completion_tokens':1}})


def test_shipped_default_config_matches_defaults():
    actual=load(Path(__file__).parents[1]/'simulinfer/default_config.toml')
    assert actual==Settings(models={},model_endpoints=(Model(model=""),))
    assert not actual.token_budget_enabled and actual.vad_mode=='firered'
    assert actual.st_vad_mode=='firered_nonstream' and not actual.st_short_speech_guard


@pytest.mark.parametrize('mode,prefix',[('target_only',None),('source_target',None),
    ('target_only','already said '),('source_target',SRC+'source'+TGT+'target ')])
def test_standard_assistant_continuation_matches_expected_prompt(mode,prefix):
    messages=[{'role':'user','content':[{'type':'text','text':'question'}]},
              {'role':'assistant','content':'previous '},
              {'role':'user','content':[{'type':'text','text':'next'}]}]
    before=json.dumps(messages)
    wire=chat_request(messages,mode,prefix)
    assert json.dumps(messages)==before
    assert 'chat_template_kwargs' not in wire
    continuation=prefix if prefix is not None else SRC if mode=='source_target' else None
    assert wire['continue_final_message']==(continuation is not None)
    assert wire['add_generation_prompt']==(continuation is None)
    if continuation is not None:
        assert wire['messages'][-1]['content'][0]['text']==continuation
    # Independently render the released simulst audio-branch semantics.
    rendered='<｜hy_begin▁of▁sentence｜><｜reasoning_mode｜>reasoning_effort:no_think'
    for i,msg in enumerate(wire['messages']):
        rendered += '<｜hy_User｜>' if msg['role']=='user' else '<｜hy_Assistant｜><think></think>'
        rendered += ''.join(p['text'] for p in msg['content'])
        if msg['role']=='assistant' and not(i==len(wire['messages'])-1 and continuation is not None):
            rendered += EOT
    if wire['add_generation_prompt']:rendered+='<｜hy_Assistant｜><think></think>'
    assert rendered==render(messages,mode,prefix)[0]


@pytest.mark.asyncio
async def test_baseline_never_counts_or_rolls_on_token_usage_and_keeps_audio():
    calls=[]
    def answer(request):
        assert request.url.path.endswith('/chat/completions')
        body=json.loads(request.content);calls.append(body)
        assert body['vllm_xargs']['hy_simulst_no_repeat_ngram_size']==3 and 'chat_template_kwargs' not in body
        assert body['repetition_penalty']==1
        return response(prompt=8180)
    settings=replace(Settings(),vad_mode='energy',st_vad_mode='off',window_seconds=0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        s=VLLMSession(Model(),settings,'target_only','zh-en',http=client)
        waveform=np.concatenate([np.full(25600,.1,np.float32),np.full(25600,.001,np.float32)])
        events=[e async for e in s.push(waveform)]
    assert len(calls)==2 and events[1]['is_silence']
    assert not any(e.get('window_reset') or e.get('silence_zero_filled') for e in events)
    raw=calls[1]['messages'][-1]['content'][-1]['input_audio']['data']
    samples,_=sf.read(io.BytesIO(base64.b64decode(raw)),dtype='float32')
    assert len(samples)==25600 and np.all(samples!=0)


@pytest.mark.asyncio
async def test_baseline_silence_skips_model_with_detection_still_enabled():
    def fail(_):raise AssertionError('Silence must not reach the model')
    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        s=VLLMSession(Model(),replace(Settings(),vad_mode='energy',st_vad_mode='off'),'target_only','zh-en',http=client)
        events=[e async for e in s.push(np.zeros(51200,np.float32))]
        assert all(e['is_silence'] and e['silence_skipped'] and not e['model_invoked'] for e in events)


@pytest.mark.asyncio
async def test_baseline_st_retains_full_audio_without_short_token_guard_or_reset():
    calls=[]
    def answer(request):
        assert request.url.path.endswith('/chat/completions')
        d=json.loads(request.content);calls.append(d)
        return response('long recovered sentence' if d['max_tokens']==256 else '')
    cfg=replace(Settings(),vad_mode='energy',st_vad_mode='off',trailing_silence_seconds=1.6)
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        s=VLLMSession(Model(),cfg,'target_only','zh-en',http=client)
        es=[e async for e in s.push(np.r_[np.full(25600,.1,np.float32),np.zeros(25600,np.float32)])]
    assert es[-1]['st_audio_ms']==3200 and es[-1]['st_trimmed_silence_ms']==0
    assert es[-1]['text']=='long recovered sentence' and not es[-1].get('window_reset')
    assert 'st_short_speech_token_limit' not in es[-1]
    assert s.run==0


@pytest.mark.asyncio
async def test_explicit_time_window_resets_and_reports_unresolved_audio():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:response(''))) as client:
        s=VLLMSession(Model(),replace(Settings(),vad_mode='energy',st_vad_mode='off',st_wait_chunks=0,window_seconds=30),
                      'target_only','zh-en',http=client)
        es=[e async for e in s.push(np.full(32*16000,.1,np.float32))]
    reset=next(e for e in es if e.get('window_reset'))
    assert reset['window_cut_trigger']=='time_window'
    assert reset['skipped_untranslated_audio_ms']==30400
    assert es[-1]['run']==1 and len(s.pending)==1


@pytest.mark.asyncio
async def test_length_finish_raises_unless_partial_recovery_is_enabled():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _:response('partial','length'))) as client:
        cfg=Settings()
        api=ChatAPI(Model(),cfg,'target_only','zh-en',client)
        with pytest.raises(RuntimeError,match='max_tokens'):await api.generate([{'role':'user','content':[part(np.ones(25600,np.float32))]}])
        api.settings=replace(cfg,reset_on_max_tokens=True)
        result=await api.generate([{'role':'user','content':[part(np.ones(25600,np.float32))]}])
        assert result.text=='partial' and result.finish_reason=='length'


@pytest.mark.parametrize('body',[
    '[infernce]\nchunk_ms=800',
    '[server]\nchunk_ms=800',
    '[inference]\nst_short_speech_guard="false"',
    '[inference]\nno_repeat_ngram_size=3\nmodel_logits_processors=false',
    '[inference]\nst_short_speech_guard=true\nst_crop_speech=false\nst_vad_mode="off"',
])
def test_misspelled_or_ineffective_options_fail(tmp_path,body):
    p=tmp_path/'test.toml';p.write_text('[models.target_only]\nbackend="vllm"\n'+body)
    with pytest.raises(ValueError):load(p)


@pytest.mark.asyncio
async def test_standard_source_target_is_one_generation_and_keeps_separator():
    calls=[]
    def answer(request):
        d=json.loads(request.content);calls.append(d)
        assert d['continue_final_message'] and not d['add_generation_prompt']
        assert d['messages'][-1]['content'][0]['text']==SRC
        assert TGT not in d['stop'] and d['skip_special_tokens'] is False
        return response('源文'+TGT+'translation')
    async with httpx.AsyncClient(transport=httpx.MockTransport(answer)) as client:
        api=ChatAPI(Model(),Settings(models={'source_target':Model()}),'source_target','zh-en',client)
        reply=await api.generate([{'role':'user','content':[part(np.ones(25600,np.float32))]}])
    assert len(calls)==1 and reply.source=='源文' and reply.text=='translation'
