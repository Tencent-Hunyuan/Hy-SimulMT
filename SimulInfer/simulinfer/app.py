from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
from dataclasses import replace
import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import uvicorn

from .audio import PCM, decode_file, decode_file_with_info
from .config import Settings, load, response_chunk_ms
from .translation import VLLMSession, ServiceSession, MockSession, validate_mode
from .tts import Speech
from .events import public_event
from . import __version__
from .file_jobs import FileJobs

SSE_KEEPALIVE_SECONDS = 15


def request_options(settings, *, model_type=None, mode=None, vad=None, tts=None):
    if model_type and mode and model_type!=mode:
        raise ValueError('model_type与mode不能指定不同模型')
    selected=model_type or mode or settings.default_model_type or next(iter(settings.models))
    use_vad=settings.default_vad if vad is None else vad
    use_tts=settings.default_tts if tts is None else tts
    if type(use_vad) is not bool or type(use_tts) is not bool:
        raise ValueError('vad和tts必须为true/false')
    if selected not in settings.models:raise ValueError('model_type必须选择已配置模型')
    if use_vad and settings.vad_mode=='off':raise ValueError('VAD未配置')
    if use_tts and settings.tts_backend=='none':raise ValueError('TTS未配置')
    return selected,use_vad,use_tts


def session_settings(settings,use_vad,response_mode=None,model_type=None,*,transport=None):
    if response_mode is not None:
        chunk_ms=response_chunk_ms(response_mode)
        if model_type and settings.models[model_type].backend=='service':
            raise ValueError('上游service控制分块，不能通过本服务设置response_mode')
        settings=replace(settings,chunk_ms=chunk_ms)
    if use_vad:
        model=settings.models.get(model_type) if model_type else None
        local_model=model is None or model.backend in {'vllm','mock'}
        if transport=='ws' and local_model and settings.vad_mode in {'silero','firered'}:
            options={'audio_mode':settings.ws_audio_mode,'default_vad':True}
            if settings.ws_audio_mode=='noise_resistant':
                options.update(st_crop_speech=True,post_speech_silence_mode='passthrough',vad_preroll_enabled=False)
            # Validate only the selected backend: other configured endpoints do
            # not determine this request's audio policy.
            if model is not None:
                options.update(models={model_type:model},model_endpoints=(),default_model_type=model_type)
            settings=replace(settings,**options)
        return settings
    return replace(settings,default_vad=False,audio_mode='standard',vad_mode='off',st_vad_mode='off',st_crop_speech=False,
                   st_short_speech_guard=False,post_speech_silence_mode='passthrough')


async def translate(settings, mode, direction, source, emit, tts=False, voice='', binary_audio=False):
    model = settings.models[mode]
    backend = {'vllm':VLLMSession,'service':ServiceSession,'mock':MockSession}[model.backend]
    session = backend(model,settings,mode,direction)
    speech = None
    tasks = []
    translated = source_text = ''
    terminal = None
    try:
        if model.backend == 'vllm':
            await session.prepare()
        ready = await session.connect() if model.backend == 'service' else session.ready()
        ready.update(mode=mode,direction=direction,tts_enabled=tts,
                     model_type=mode,vad_enabled=settings.vad_mode!='off',
                     source_text_enabled=mode=='source_target',audio_transport='binary' if binary_audio else 'base64')
        if model.backend!='service':ready['response_mode']=settings.response_mode
        if tts:
            speech = Speech(settings,emit,direction,voice)
            await speech.start()
            ready['tts'] = speech.info()
        await emit(ready)

        async def consume(event):
            nonlocal terminal, translated, source_text
            kind = event.get('event')
            if kind == 'error':
                raise RuntimeError(event.get('detail',event.get('error','模型返回错误')))
            if kind == 'done':
                terminal = event
                return
            if kind in ('chunk','flush'):
                text = event.get('text') or ''
                translated += text
                if 'source_text_full' in event:source_text = event['source_text_full']
                else:source_text += event.get('source_text') or ''
                await emit(event)
                if speech:await speech.text(text)
            else:
                await emit(event)

        if model.backend == 'service':
            async def feed():
                async for samples in source:
                    if speech:
                        speech.feed_source(samples)
                        await speech.flush_pending()
                    await session.send(samples)
                if speech:speech.finish_source()
                await session.finish_input()
            async def receive():
                async for event in session.events():await consume(event)
            tasks=[asyncio.create_task(feed()),asyncio.create_task(receive())]
            await asyncio.gather(*tasks)
        else:
            async for samples in source:
                if speech:
                    speech.feed_source(samples)
                    await speech.flush_pending()
                async for event in session.push(samples):await consume(event)
            if speech:speech.finish_source()
            async for event in session.finish():await consume(event)
        if terminal is None:
            raise RuntimeError('模型未返回done，结果不完整')
        if speech:await speech.finish()
        done = {**terminal,'event':'done','mode':mode,'direction':direction,'tts_enabled':tts,
                'model_type':mode,'vad_enabled':settings.vad_mode!='off',
                'full_text':terminal.get('full_text',translated)}
        if model.backend!='service':done['response_mode']=settings.response_mode
        if mode == 'source_target':done['source_text'] = terminal.get('source_text',source_text)
        else:done.pop('source_text',None)
        if speech:done['tts_audio']={'bytes':speech.bytes,'sample_rate':speech.rate,'format':'s16le','channels':1,
                                    'complete':speech.failed_segments == 0,'failed_segments':speech.failed_segments}
        await emit(done)
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        if tasks:await asyncio.gather(*tasks,return_exceptions=True)
        if speech:await speech.close()
        await session.close()


def build_app(settings=None, *, preflight_result=None):
    settings = settings or load()
    file_jobs=FileJobs()
    @contextlib.asynccontextmanager
    async def lifespan(app):
        nonlocal settings
        if settings.model_endpoints:
            from .model_discovery import discover
            settings=await discover(settings)
        if preflight_result is None:
            from .preflight import check
            await check(settings)
        try:yield
        finally:await file_jobs.close()
    app = FastAPI(title='SimulInfer',version=__version__,lifespan=lifespan)
    gate = asyncio.Semaphore(settings.max_sessions)
    state = {'active_sessions':0,'waiting_sessions':0}

    @contextlib.asynccontextmanager
    async def slot():
        state['waiting_sessions'] += 1
        acquired = False
        try:
            await gate.acquire();acquired=True
        finally:state['waiting_sessions']-=1
        try:
            state['active_sessions']+=1
            yield
        finally:
            state['active_sessions']-=1
            if acquired:gate.release()

    @app.get('/health')
    async def health():
        return {'status':'ready',**state,'max_sessions':settings.max_sessions,
                'model_types':list(settings.models),'vad_available':settings.vad_mode!='off',
                'tts_available':settings.tts_backend!='none',
                'models':{k:{'backend':m.backend,'model':m.model} for k,m in settings.models.items()},
                'tts_backend':settings.tts_backend}

    @app.get('/config')
    async def config(response:Response):
        response.headers["Cache-Control"]="no-store"
        default_mode=settings.default_model_type or next(iter(settings.models))
        return {'modes':list(settings.models),'default_mode':default_mode,
                'audio_mode':settings.audio_mode,
                'model_types':list(settings.models),'vad_available':settings.vad_mode!='off',
                'model_labels':{k:m.model.rstrip('/').rsplit('/',1)[-1] for k,m in settings.models.items()},
                'defaults':{'model_type':default_mode,'response_mode':settings.response_mode,
                            'vad':settings.default_vad,'tts':settings.default_tts},
                'response_modes':{'standard':{'label':'标准模式（推荐，1600ms）','chunk_ms':1600},
                                  'short_chunk':{'label':'短分块模式（首块必须1600ms，后续800ms）','chunk_ms':800}},
                'response_mode_available':{k:m.backend!='service' for k,m in settings.models.items()},
                'chunk_ms':settings.chunk_ms,'first_chunk_ms':settings.first_chunk_ms,
                'file_upload_transport':'polling',
                'tts_available':settings.tts_backend!='none',
                'tts_clone_source':settings.clone_source_enabled,
                'mock':all(m.backend=='mock' for m in settings.models.values())}

    @app.get('/')
    async def index():return FileResponse(Path(__file__).parent/'static/index.html',headers={'Cache-Control':'no-store'})

    app.mount('/static',StaticFiles(directory=Path(__file__).parent/'static'),name='static')

    @app.post('/translate/submit')
    async def submit_file(file:UploadFile=File(...),direction:str=Form('zh-en'),
                          model_type:str|None=Form(None),response_mode:str|None=Form(None),
                          vad:bool|None=Form(None),tts:bool|None=Form(None),voice:str=Form('')):
        try:
            mode,use_vad,use_tts=request_options(settings,model_type=model_type,vad=vad,tts=tts)
            request_settings=session_settings(settings,use_vad,response_mode,mode,transport='stream')
            validate_mode(mode,direction,settings)
            data=await file.read(settings.max_upload_mb*1024*1024+1)
            if not data or len(data)>settings.max_upload_mb*1024*1024:
                raise ValueError('文件为空或超过上传大小限制')
        except (ValueError,RuntimeError) as exc:
            raise HTTPException(400,str(exc)) from exc
        finally:await file.close()

        async def run(emit):
            async with slot():
                samples,info=await asyncio.to_thread(decode_file_with_info,data,
                    settings.max_upload_mb*1024*1024//2)
                async def source():
                    for pos in range(0,len(samples),1600):yield samples[pos:pos+1600]
                async def publish(event):
                    if event.get('event') in {'ready','done'}:event={**event,'input_audio':info}
                    event=public_event(event)
                    if event is not None:await emit(event)
                await translate(request_settings,mode,direction,source(),publish,use_tts,voice)
        try:token=file_jobs.create(len(data),run,settings.timeout_seconds)
        except ValueError as exc:raise HTTPException(503,str(exc)) from exc
        return Response(json.dumps({'task_id':token}),media_type='application/json',
                        headers={'Cache-Control':'no-store'})

    @app.get('/translate/task/{task_id}')
    async def poll_file(task_id:str,cursor:int=0):
        try:result=file_jobs.get(task_id).poll(cursor)
        except KeyError as exc:raise HTTPException(404,str(exc)) from exc
        except ValueError as exc:raise HTTPException(400,str(exc)) from exc
        return Response(json.dumps(result,ensure_ascii=False),media_type='application/json',
                        headers={'Cache-Control':'no-store'})

    @app.delete('/translate/task/{task_id}')
    async def cancel_file(task_id:str):
        try:await file_jobs.cancel(task_id)
        except KeyError as exc:raise HTTPException(404,str(exc)) from exc
        return Response(status_code=204,headers={'Cache-Control':'no-store'})

    @app.post('/translate/stream')
    async def stream(file:UploadFile=File(...), direction:str=Form('zh-en'),
                     mode:str|None=Form(None), model_type:str|None=Form(None),
                     response_mode:str|None=Form(None,description='standard: 标准模式（推荐，1600ms）；short_chunk: 短分块模式（首块必须1600ms，后续800ms）。'),
                     vad:bool|None=Form(None),tts:bool|None=Form(None),voice:str=Form('')):
        try:
            mode,use_vad,tts=request_options(settings,model_type=model_type,mode=mode,vad=vad,tts=tts)
            request_settings=session_settings(settings,use_vad,response_mode,mode,transport="stream")
            validate_mode(mode,direction,settings)
            if tts and settings.tts_backend=='none':raise ValueError('TTS未配置')
            data = await file.read(settings.max_upload_mb*1024*1024+1)
            if len(data)>settings.max_upload_mb*1024*1024:raise ValueError('上传文件超过大小限制')
            samples, input_audio = await asyncio.to_thread(decode_file_with_info,data,settings.max_upload_mb*1024*1024//2)
        except (ValueError,RuntimeError) as e:
            raise HTTPException(400,str(e)) from e
        finally:await file.close()

        async def iterator():
            events=asyncio.Queue(maxsize=32)
            async def source():
                for pos in range(0,len(samples),1600):yield samples[pos:pos+1600]
            async def run():
                try:
                    async with slot():
                        async def emit(event):
                            if event.get('event') in {'ready', 'done'}:
                                event = {**event, 'input_audio': input_audio}
                            outgoing=public_event(event)
                            if outgoing is not None:await events.put(outgoing)
                        await translate(request_settings,mode,direction,source(),emit,tts,voice)
                except Exception as e:
                    await events.put({'event':'error','detail':str(e)})
                finally:
                    if not asyncio.current_task().cancelling():await events.put(None)
            task=asyncio.create_task(run())
            try:
                # Flush a response even while a session slot/model is pending.
                # SSE comments keep proxies alive without changing the JSON protocol.
                yield ': connected\n\n'
                while True:
                    try:
                        event=await asyncio.wait_for(events.get(),SSE_KEEPALIVE_SECONDS)
                    except asyncio.TimeoutError:
                        yield ': keep-alive\n\n'
                        continue
                    if event is None:break
                    yield 'data: '+json.dumps(event,ensure_ascii=False)+'\n\n'
                await task
            finally:
                task.cancel();await asyncio.gather(task,return_exceptions=True)
        return StreamingResponse(iterator(),media_type='text/event-stream',
            # code-server applies compression unless no-transform is present;
            # compression can hold the first bytes until the request times out.
            headers={'Cache-Control':'no-cache, no-transform','X-Accel-Buffering':'no'})

    @app.websocket('/translate/ws')
    async def realtime(ws:WebSocket):
        await ws.accept()
        tasks=[];send_lock=asyncio.Lock()
        try:
            cfg=json.loads(await asyncio.wait_for(ws.receive_text(),settings.timeout_seconds))
            if not isinstance(cfg,dict):raise ValueError('首条消息必须为JSON对象')
            mode,use_vad,tts=request_options(settings,model_type=cfg.get('model_type'),
                mode=cfg.get('mode'),vad=cfg.get('vad'),tts=cfg.get('tts'))
            request_settings=session_settings(settings,use_vad,cfg.get('response_mode'),mode,transport='ws')
            direction=cfg.get('direction','zh-en')
            validate_mode(mode,direction,settings)
            if type(tts) is not bool:raise ValueError('tts必须为布尔值')
            if tts and settings.tts_backend=='none':raise ValueError('TTS未配置')
            voice=cfg.get('voice','')
            if not isinstance(voice,str):raise ValueError('voice必须为字符串')
            fmt=cfg.get('input_format','s16le')
            rate=cfg.get('input_sample_rate',cfg.get('sample_rate',16000))
            channels=cfg.get('input_channels',cfg.get('channels',1))
            converter=PCM(fmt,rate,channels)
            encoding=cfg.get('audio_transport','binary')
            if encoding not in {'binary','base64'}:raise ValueError('audio_transport需为binary或base64')
            incoming=asyncio.Queue(maxsize=settings.input_queue_frames)
            ended=False
            report_input=cfg.get("input_progress") is True
            reported_samples=0
            async def send(event):
                async with send_lock:
                    if event.get('event') in {'ready', 'done'}:
                        event = {**event, 'input_audio': converter.info()}
                    event=public_event(event)
                    if event is None:return
                    if event.get('event')=='audio' and encoding=='binary':
                        e=dict(event);raw=base64.b64decode(e.pop('data'))
                        e.update(bytes=len(raw),encoding='binary-next')
                        await ws.send_json(e);await ws.send_bytes(raw)
                    else:await ws.send_json(event)
            async def receive():
                nonlocal ended,reported_samples
                while True:
                    message=await asyncio.wait_for(ws.receive(),settings.timeout_seconds)
                    if message['type']=='websocket.disconnect':raise WebSocketDisconnect()
                    if message.get('bytes') is not None:
                        if ended:raise ValueError('EOF后不得继续发送音频')
                        raw=message['bytes']
                        if len(raw)>settings.max_ws_message_mb*1024*1024:raise ValueError('单条音频消息过大')
                        samples=converter.feed(raw)
                        if report_input and converter.output_samples-reported_samples>=16000:
                            reported_samples=converter.output_samples
                            await send({'event':'input_progress','received_ms':reported_samples/16})
                        # Bound each queued item independent of network frame size.
                        for pos in range(0,len(samples),1600):await incoming.put(samples[pos:pos+1600])
                    elif message.get('text'):
                        value=json.loads(message['text'])
                        if not isinstance(value,dict):raise ValueError('控制消息必须是JSON对象')
                        if value.get('eof'):
                            if ended:raise ValueError('重复EOF')
                            tail=converter.finish()
                            for pos in range(0,len(tail),1600):await incoming.put(tail[pos:pos+1600])
                            ended=True;await incoming.put(None)
                        elif value.get('cancel'):raise WebSocketDisconnect()
                        else:raise ValueError('后续文本消息仅支持eof或cancel')
            async def source():
                while True:
                    samples=await incoming.get()
                    if samples is None:return
                    yield samples
            async def run():
                async with slot():
                    await translate(request_settings,mode,direction,source(),send,tts,voice,encoding=='binary')
            reader=asyncio.create_task(receive());worker=asyncio.create_task(run());tasks=[reader,worker]
            done,_=await asyncio.wait(tasks,return_when=asyncio.FIRST_COMPLETED)
            for task in done:task.result()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            with contextlib.suppress(Exception):await ws.send_json({'event':'error','detail':str(e)})
        finally:
            for task in tasks:task.cancel()
            if tasks:await asyncio.gather(*tasks,return_exceptions=True)
            with contextlib.suppress(Exception):await ws.close()
    return app


def main():
    parser=argparse.ArgumentParser(description='SimulInfer 流式同传服务')
    parser.add_argument('--config');parser.add_argument('--mock',action='store_true')
    parser.add_argument('--init-config',metavar='PATH',help='生成配置，不覆盖已有文件')
    parser.add_argument('--profile',choices=['standard','noise_resistant'],default='standard',
                        help='选择生成配置的音频模式，与--init-config一起使用')
    parser.add_argument('--check',action='store_true',help='检查配置/模型协议/VAD后退出，不做生成')
    parser.add_argument('--host');parser.add_argument('--port',type=int)
    args=parser.parse_args()
    if args.init_config:
        text=(Path(__file__).parent/'default_config.toml').read_text()
        if args.profile=='noise_resistant':
            text=text.replace('audio_mode = "standard"','audio_mode = "noise_resistant"')
            text=text.replace('# st_short_speech_guard = false','st_short_speech_guard = true')
        with Path(args.init_config).open('x') as f:f.write(text)
        print(f'配置已写入 {args.init_config}',flush=True)
        return
    if args.profile!='standard':parser.error('--profile用于生成配置；运行时请编辑配置中的audio_mode')
    settings=load(args.config,mock=args.mock)
    if settings.model_endpoints:
        from .model_discovery import discover
        settings=asyncio.run(discover(settings))
    from .preflight import check
    result=asyncio.run(check(settings))
    print('启动检查: '+json.dumps(result,ensure_ascii=False),flush=True)
    if args.check:return
    uvicorn.run(build_app(settings,preflight_result=result),host=args.host or settings.host,port=args.port or settings.port,
                ws_max_size=settings.max_ws_message_mb*1024*1024,ws_ping_interval=None)


if __name__=='__main__':main()
