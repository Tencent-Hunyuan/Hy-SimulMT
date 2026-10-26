import asyncio
import base64
from dataclasses import replace
import io
import json

import httpx
import numpy as np
import pytest
import soundfile as sf

from simulinfer.audio import part
from simulinfer.budget import ContextBudgetError, speech_tokens
from simulinfer.config import Settings as Defaults
from tests.helpers import Model
from simulinfer.translation import ChatAPI, VLLMSession, SRC, TGT
from tests.helpers import Settings, model_transport


def response(text='', *, finish='stop'):
    return httpx.Response(200, json={'choices':[{'message':{'content':text},
        'finish_reason':finish,'stop_reason':120008,'token_ids':[11,12]}]})


def voice(seconds=1.6, value=.1):
    return np.full(round(seconds*16000),value,np.float32)


def test_defaults_use_standard_vllm_and_silence_detection():
    s=Defaults()
    assert s.window_seconds==0 and not s.token_budget_enabled
    assert s.chunk_ms==s.first_chunk_ms==1600 and s.st_wait_chunks==4
    assert not s.reset_after_audible_st and not s.reset_after_silence_st and not s.reset_after_empty_st
    assert not s.reset_on_max_tokens and s.st_fallback_max_seconds==0
    assert not s.st_context and not s.st_trim_trailing_silence and s.skip_initial_silence
    assert s.post_speech_silence_mode=='passthrough' and not s.vad_preroll_enabled
    assert s.vad_mode=='firered' and s.vad_pad_ms==240
    assert s.st_vad_mode=='firered_nonstream' and not s.st_short_speech_guard
    assert s.repetition_penalty==1.0 and s.st_repetition_penalty==1.05
    assert s.no_repeat_ngram_size==s.st_no_repeat_ngram_size==3
    assert not s.no_chinese_target and s.model_logits_processors
    assert s.models['target_only'].chat_protocol=='standard'


@pytest.mark.parametrize('kwargs',[{'max_model_len':400},{'context_reserve_tokens':20},
    {'context_reserve_tokens':4096},{'window_seconds':-1},{'no_repeat_ngram_size':-1},
    {'st_no_repeat_ngram_size':True},{'st_fallback_max_seconds':31},{'repetition_penalty':float('nan')}])
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):Defaults(**kwargs)


def test_audio_expansion_grid_and_local_metadata():
    assert speech_tokens(12800)==5 and speech_tokens(25600)==11
    assert speech_tokens(480000)==195 and speech_tokens(1601)==1
    a=part(voice())
    assert a.num_samples==25600 and 'num_samples' not in json.loads(json.dumps(a))


@pytest.mark.asyncio
async def test_fourth_voiced_wait_resets_and_new_chunk_has_no_old_audio_or_text():
    calls=[]
    def answer(req):
        d=json.loads(req.content);calls.append(d)
        if len(calls)==1:return response('old')
        if d['max_tokens']==256:return response('recovered')
        return response('fresh' if len(calls)==7 else '')
    async with httpx.AsyncClient(transport=model_transport(answer)) as client:
        session=VLLMSession(Model(),Settings(),'target_only','zh-en',http=client)
        es=[e async for e in session.push(voice(9.6))]
        reset=es[4]
        assert reset['restart_reason']=='st_audible_reset' and reset['text']=='recovered'
        assert reset['st_audio_ms']==6400 and not reset['st_context_used']
        assert not reset['recovery_context_retained']
        assert len(calls[-1]['messages'])==1 and 'old' not in json.dumps(calls[-1])
        assert 'recovered' not in json.dumps(calls[-1])
        assert session.full_text=='oldrecoveredfresh'
        assert [e['index'] for e in es]==list(range(6))
        assert es[-1]['run']==1
        assert calls[0]['vllm_xargs']['hy_simulst_no_repeat_ngram_size']==3
        assert calls[-2]['vllm_xargs']['hy_st_no_repeat_ngram_size']==3
        assert all(d['repetition_penalty']==1.05 for d in calls)


@pytest.mark.asyncio
async def test_silence_wait_preserves_counter_and_zero_filled_timeline():
    calls=[]
    def answer(req):
        d=json.loads(req.content);calls.append(d)
        return response('tail' if d['max_tokens']==256 else '')
    async with httpx.AsyncClient(transport=model_transport(answer)) as client:
        session=VLLMSession(Model(),Settings(),'target_only','en-zh',http=client)
        x=np.concatenate([voice(1.6),voice(1.6,0),voice(1.6),voice(3.2,0)])
        es=[e async for e in session.push(x)]
        assert [e['audible_wait_chunks'] for e in es]==[1,1,2,2,2]
        assert es[-1]['restart_reason']=='st_silence_reset'
        assert es[-1]['st_audio_ms']==8000 and es[-1]['st_trimmed_silence_ms']==0
        audio=calls[-1]['messages'][0]['content'][1]['input_audio']['data']
        data,sr=sf.read(io.BytesIO(base64.b64decode(audio)),dtype='float32')
        np.testing.assert_array_equal(data,x)
        assert session.messages==[]


@pytest.mark.asyncio
async def test_no_time_cut_even_after_120_seconds():
    async with httpx.AsyncClient(transport=model_transport(lambda _:response('x'))) as client:
        session=VLLMSession(Model(),Settings(),'target_only','zh-en',http=client)
        es=[e async for e in session.push(voice(128))]
        assert len(es)==80 and session.run==0
        assert all(not e.get('window_reset') for e in es)


@pytest.mark.asyncio
async def test_cut_after_completed_chunk_at_budget_and_count_actual_response():
    calls=[]
    def answer(req):
        d=json.loads(req.content);calls.append(d)
        return response('x'*40)
    async with httpx.AsyncClient(transport=model_transport(answer,limit=512)) as client:
        session=VLLMSession(Model(),Settings(max_model_len=8192,context_reserve_tokens=96),
                            'target_only','zh-en',http=client)
        es=[e async for e in session.push(voice(24))]
        cuts=[(i,e) for i,e in enumerate(es) if e.get('window_reset')]
        assert cuts and all(e['restart_reason']=='token_budget' for _,e in cuts)
        for i,e in cuts:
            assert e['text']=='x'*40 and e['action']=='EMIT'
            if i+1<len(calls):assert len(calls[i+1]['messages'])==1
        assert session.full_text=='x'*600
        assert all(e['prompt_tokens']+e['completion_tokens']<=512 for e in es)


@pytest.mark.asyncio
async def test_waiting_audio_flushed_before_token_cut_and_not_lost():
    calls=[]
    def answer(req):
        d=json.loads(req.content);calls.append(d)
        if d['max_tokens']==256:return response('recovered')
        return response('')
    async with httpx.AsyncClient(transport=model_transport(answer,limit=512)) as client:
        session=VLLMSession(Model(),Settings(context_reserve_tokens=128,st_wait_chunks=0),
                            'target_only','en-zh',http=client)
        es=[e async for e in session.push(voice(40))]
        cut=next(e for e in es if e.get('window_cut_trigger')=='token_budget')
        assert cut['text']=='recovered' and cut['st_attempted']
        assert cut['st_audio_ms']>6400
        assert not cut.get('skipped_untranslated_audio_ms')
        assert cut['st_context_used'] is False
        assert all(d['chat_template_kwargs'].get('harness_prefix') is None for d in calls)


@pytest.mark.asyncio
async def test_empty_st_reset_keeps_true_source_span_and_padded_tail_separate():
    async with httpx.AsyncClient(transport=model_transport(lambda _:response(''))) as client:
        session=VLLMSession(Model(),Settings(),'target_only','zh-en',http=client)
        es=[e async for e in session.push(voice(6.7))]+[e async for e in session.finish()]
        assert es[-1]['empty_st_reset_count']==2
        assert es[-1]['skipped_untranslated_audio_ms']==6700
        assert es[-1]['duration_ms']==6700 and es[-1]['num_chunks']==5
        assert es[-2]['final_padding_ms']==1300 and es[-2]['st_audio_ms']==300
        assert es[-2]['skipped_untranslated_audio_ms']==300


@pytest.mark.asyncio
async def test_usage_mismatch_fails_before_inventing_a_safe_token_budget():
    def answer(req):
        return httpx.Response(200,json={'choices':[{'message':{'content':'x'},'finish_reason':'stop'}],
            'usage':{'prompt_tokens':9999,'completion_tokens':1}})
    async with httpx.AsyncClient(transport=model_transport(answer)) as client:
        api=ChatAPI(Model(),Settings(),'target_only','zh-en',client)
        with pytest.raises(RuntimeError,match='token计数'):
            await api.generate([{'role':'user','content':[part(voice())]}])


@pytest.mark.asyncio
async def test_preflight_does_not_submit_oversized_prompt_to_generation():
    count=0
    def answer(req):
        nonlocal count
        count+=1
        return response('x')
    async with httpx.AsyncClient(transport=model_transport(answer,limit=512)) as client:
        api=ChatAPI(Model(),Settings(context_reserve_tokens=96),'target_only','zh-en',client)
        with pytest.raises(ContextBudgetError):
            await api.generate([{'role':'user','content':[{'type':'text','text':'x'*1000},part(voice())]}])
        assert count==0


@pytest.mark.asyncio
async def test_long_reply_is_returned_once_and_resets_without_erasing_it():
    async with httpx.AsyncClient(transport=model_transport(lambda _:response('partial',finish='length'))) as client:
        session=VLLMSession(Model(),Settings(),'target_only','zh-en',http=client)
        es=[e async for e in session.push(voice(3.2))]
        assert len(es)==2 and all(e['text']=='partial' and e['output_truncated'] for e in es)
        assert all(e['restart_reason']=='max_tokens' for e in es)
        assert session.full_text=='partialpartial'


@pytest.mark.asyncio
async def test_source_and_target_survive_recovery_reset_without_duplicate_text():
    def answer(req):
        d=json.loads(req.content)
        return response(SRC+'源文'+TGT+'translation' if d['max_tokens']==256 else '')
    async with httpx.AsyncClient(transport=model_transport(answer)) as client:
        session=VLLMSession(Model(),Settings(models={'source_target':Model()}),
                            'source_target','zh-en',http=client)
        es=[e async for e in session.push(voice(6.4))]+[e async for e in session.finish()]
        assert es[-2]['source_text']=='源文' and es[-2]['text']=='translation'
        assert es[-2]['window_reset'] and es[-1]['source_text']=='源文'
        assert es[-1]['full_text']=='translation'
