"""Serve SimulInfer with an already running audio-capable llama.cpp model.

From SimulInfer/, install dependencies with ``python -m pip install .``, then:
    export MODEL=/path/to/hymt-A3b-tgt_only-1019-Q8_0.gguf
    MODEL_TYPE=target_only python serve_llamacpp.py

MODEL must match the GGUF path loaded by llama.cpp. LLAMA_URL defaults to
http://127.0.0.1:8080; the web/API service listens on 127.0.0.1:8094 (PORT).
The first audio chunk is 1600ms; CHUNK_MS defaults to 1600 for later chunks.
MODEL_TYPE selects source_target or target_only to match the loaded model.
"""
import os
import json
import asyncio
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from simulinfer.app import build_app
from simulinfer.config import Settings, Model
from simulinfer.translation import ChatAPI, VLLMSession, Reply, SRC, TGT, EOT, EOS, split_reply
from simulinfer.budget import render, speech_tokens, PromptBudget

MODEL = os.environ.get('MODEL', '').strip()
if not MODEL:
    raise ValueError('Set MODEL to the GGUF path loaded by llama.cpp, for example: '
                     'export MODEL=/path/to/model.gguf')
MODE = os.environ.get('MODEL_TYPE', 'source_target')
if MODE not in ('source_target', 'target_only'):
    raise ValueError('MODEL_TYPE must be source_target or target_only')
URL = os.environ.get('LLAMA_URL', 'http://127.0.0.1:8080').rstrip('/')
PORT = int(os.environ.get('PORT', '8094'))
CHUNK = int(os.environ.get('CHUNK_MS', '1600'))
if CHUNK not in (800, 1600):
    raise ValueError('CHUNK_MS must be 800 or 1600')

class LlamaAPI(ChatAPI):
    async def post(self, path, body):
        r = await self.http.post(URL + path, json=body)
        r.raise_for_status()
        return r.json()

    async def initialize(self):
        if getattr(self, 'marker', None):
            return
        r = await self.http.get(URL + '/props')
        r.raise_for_status()
        p = r.json()
        if p.get('model_path') != MODEL:
            raise RuntimeError('MODEL must match the GGUF path loaded by llama.cpp: '
                               f"configured {MODEL!r}, server reports {p.get('model_path')!r}")
        if not p.get('modalities', {}).get('audio'):
            raise RuntimeError('The llama.cpp server must support audio input for the loaded model')
        self.max_model_len = min(16384, p['default_generation_settings']['n_ctx'])
        for text, token in ((SRC,120808),(TGT,120809),(EOT,120008),(EOS,120001),('<eos:6124c78e>',120025)):
            result = await self.post('/tokenize', {'content':text,'add_special':False,'parse_special':True})
            if result['tokens'] != [token]:
                raise RuntimeError('Unexpected special token: '+text)
        self.marker = p['media_marker']

    async def tokenize(self, text):
        await self.initialize()
        result = await self.post('/tokenize', {'content':text,'add_special':False,'parse_special':True})
        return {'tokens':result['tokens'],'count':len(result['tokens']),'max_model_len':self.max_model_len}

    async def budget(self, messages, prefix=None):
        # mtmd splits EACH audio item into 1600ms + 800ms segments,
        # rounding FSK tokens separately. Whole-item vLLM counts differ for ST.
        text, _, items = render(messages, self.mode, prefix)
        count = (await self.tokenize(text))['count']
        for message in messages:
            if message['role'] != 'user':
                continue
            for p in message['content']:
                if p['type'] != 'input_audio':
                    continue
                remaining = p.num_samples
                total, step = 0, 25600
                while remaining > 0:
                    n = min(step, remaining)
                    total += speech_tokens(n)
                    remaining -= n
                    step = 12800
                count += total - 1
        result = PromptBudget(count, self.max_model_len, items)
        self.last_budget = result
        return result

    async def generate(self, messages, *, prefix=None, max_tokens=None, target_only=False,
                       repetition_penalty=None, source_phase=False, st=False, budget=None):
        # st selects fallback generation settings; it does not imply a
        # two-stage source-prefix request. Ordinary ST uses the same completion
        # protocol, with max_tokens and repetition_penalty supplied by caller.
        if source_phase:
            raise RuntimeError('Two-stage source-prefix generation is not supported by this bridge')
        await self.initialize()
        n = max_tokens or self.settings.max_tokens
        budget = budget or await self.budget(messages, prefix=prefix)
        if not budget.fits(n):
            raise RuntimeError('Context budget exhausted')
        text, _, count = render(messages, self.mode, prefix)
        audio = [p['input_audio']['data'] for m in messages if m['role']=='user'
                 for p in m['content'] if p['type']=='input_audio']
        wrapped = '<｜hy_place▁holder▁no▁6｜><｜hy_place▁holder▁no▁5｜><｜hy_place▁holder▁no▁8｜>'
        if len(audio)!=count or text.count(wrapped)!=count:
            raise RuntimeError('Audio placeholder count mismatch')
        prompt = text.replace(wrapped, '<｜hy_place▁holder▁no▁6｜>'+self.marker+'<｜hy_place▁holder▁no▁8｜>')
        result = await self.post('/completion', {'prompt':{'prompt_string':prompt,'multimodal_data':audio},
            'n_predict':n,'temperature':0,'repeat_penalty':repetition_penalty or 1.0,
            'return_tokens':True,'cache_prompt':True,'stream':False})
        if result.get('truncated') or result.get('tokens_evaluated')!=budget.prompt_tokens:
            raise RuntimeError('Truncated context or audio token budget mismatch')
        ids = result.get('tokens', [])
        # This GGUF also declares the speech EOS 120025. Normalize it to
        # the session EOS after removing it from the generated body.
        ends = {120008:EOT,120001:EOS,120025:EOS}
        if not ids or ids[-1] not in ends or any(i in ends for i in ids[:-1]):
            detail = {'stop_type': result.get('stop_type'), 'tail_token_ids': ids[-8:],
                      'tokens_predicted': result.get('tokens_predicted'),
                      'max_tokens': n, 'st': st}
            raise RuntimeError('Incomplete or invalid model turn boundary: ' + json.dumps(detail))
        raw = (await self.post('/detokenize', {'tokens':ids[:-1]}))['content']
        if self.mode == 'source_target' and not target_only and ids[:-1].count(120809)!=(1 if raw else 0):
            raise RuntimeError('Missing or repeated target marker')
        source, target = ('',raw) if target_only else split_reply(raw,self.mode)
        return Reply(source,target,raw,ids,ends[ids[-1]],result['tokens_evaluated'],result['tokens_predicted'],'stop')

model = Model(base_url=URL+'/v1',model=MODEL)
settings = Settings(models={MODE:model},default_model_type=MODE,
    max_sessions=1,first_chunk_ms=1600,chunk_ms=CHUNK,max_model_len=16384,token_budget_enabled=True,max_tokens=128,
    st_context=False,model_logits_processors=False,no_repeat_ngram_size=0,st_no_repeat_ngram_size=0,vad_mode='off',st_vad_mode='off',default_vad=False)
public_settings = Settings(models={MODE:Model(backend='service',base_url=f'http://127.0.0.1:{PORT}/bridge',model=MODEL)},
    default_model_type=MODE,host='127.0.0.1',port=PORT,max_sessions=1,first_chunk_ms=1600,chunk_ms=CHUNK,
    vad_mode='off',st_vad_mode='off',default_vad=False)
app = FastAPI(title='SimulInfer llama.cpp bridge')
lock = asyncio.Lock()

@app.websocket('/bridge/translate/ws')
async def bridge(ws:WebSocket):
    await ws.accept()
    session = None
    try:
        cfg = await asyncio.wait_for(ws.receive_json(),30)
        direction = cfg.get('direction','zh-en')
        if direction not in ('zh-en','en-zh'):
            raise ValueError('Unsupported direction')
        async with lock:
            session = VLLMSession(model,settings,MODE,direction)
            await session.api.close()
            session.api = LlamaAPI(model,settings,MODE,direction)
            await session.prepare()
            ready = session.ready()
            ready.update(backend='llamacpp',streaming='chunked_multiturn_http',persistent_encoder_state=False,
                         input_format='f32le',sample_rate=16000)
            await ws.send_json(ready)
            while True:
                frame = await asyncio.wait_for(ws.receive(),300)
                if frame['type']=='websocket.disconnect':
                    break
                if frame.get('bytes') is not None:
                    raw = frame['bytes']
                    if len(raw)%4 or len(raw)>8*1024*1024:
                        raise ValueError('Invalid float32 PCM frame')
                    samples = np.frombuffer(raw,dtype='<f4').copy()
                    if not np.isfinite(samples).all():
                        raise ValueError('Nonfinite audio samples')
                    async for event in session.push(samples):
                        await ws.send_json(event)
                elif frame.get('text'):
                    if json.loads(frame['text']).get('eof'):
                        async for event in session.finish():
                            await ws.send_json(event)
                        break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({'event':'error','detail':str(e)})
        except Exception:
            pass
    finally:
        if session:
            await session.close()
        try:
            await ws.close()
        except Exception:
            pass

app.mount('/', build_app(public_settings))
if __name__=='__main__':
    uvicorn.run(app,host='127.0.0.1',port=PORT,ws_ping_interval=None)
