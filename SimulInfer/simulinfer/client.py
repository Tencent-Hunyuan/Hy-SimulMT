"""CLI for file/SSE and true real-time WS, with optional returned TTS WAV."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import wave

import httpx
import numpy as np
import websockets

from .audio import decode_file


async def run(args):
    output = bytearray();sample_rate = None;events=[];audio_header=None
    options={}
    if args.mode is not None:options['model_type']=args.mode
    if args.response_mode is not None:options['response_mode']=args.response_mode
    if args.vad is not None:options['vad']=args.vad
    if args.tts is not None:options['tts']=args.tts
    def receive(event):
        nonlocal sample_rate,audio_header
        events.append(event)
        if event.get('event')=='error':raise RuntimeError(event.get('detail','请求失败'))
        if event.get('event')=='audio':
            rate=event['sample_rate']
            if sample_rate is not None and sample_rate!=rate:raise ValueError('TTS采样率变化')
            sample_rate=rate
            if event.get('encoding')=='binary-next':audio_header=event
            else:output.extend(base64.b64decode(event['data']))
        else:print(json.dumps(event,ensure_ascii=False),flush=True)
    if args.transport=='stream':
        async with httpx.AsyncClient(timeout=args.timeout,trust_env=False) as client:
            with Path(args.audio).open('rb') as f:
                async with client.stream('POST',args.url.rstrip('/')+'/translate/stream',
                        data={'direction':args.direction,'voice':args.voice,
                              **{k:str(v).lower() if isinstance(v,bool) else v for k,v in options.items()}},
                        files={'file':(Path(args.audio).name,f,'audio/wav')}) as r:
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if line.startswith('data:'):receive(json.loads(line[5:]))
    else:
        raw=Path(args.audio).read_bytes();samples=decode_file(raw,100*1024*1024)
        url=args.url.rstrip('/').replace('https://','wss://').replace('http://','ws://')+'/translate/ws'
        async with websockets.connect(url,max_size=16*1024*1024,ping_interval=None) as ws:
            await ws.send(json.dumps({**options,'direction':args.direction,'input_format':'f32le',
                                      'input_sample_rate':16000,'input_channels':1,
                                      'voice':args.voice,'audio_transport':'binary'}))
            ready=json.loads(await asyncio.wait_for(ws.recv(),args.timeout));receive(ready)
            if ready.get('event')!='ready':raise RuntimeError('没有收到ready')
            async def upload():
                for p in range(0,len(samples),1600):
                    block=samples[p:p+1600]
                    await ws.send(np.asarray(block,dtype='<f4').tobytes())
                    if args.realtime:await asyncio.sleep(len(block)/16000)
                await ws.send(json.dumps({'eof':True}))
            sender=asyncio.create_task(upload())
            try:
                while True:
                    data=await asyncio.wait_for(ws.recv(),args.timeout)
                    if isinstance(data,bytes):
                        if audio_header is None or audio_header['bytes']!=len(data):raise ValueError('音频帧缺少对应元信息')
                        output.extend(data);audio_header=None
                    else:
                        event=json.loads(data);receive(event)
                        if event.get('event')=='done':break
                await sender
            finally:
                sender.cancel();await asyncio.gather(sender,return_exceptions=True)
    if not events or events[-1].get('event')!='done':raise RuntimeError('连接结束但缺少done')
    if args.events:
        p=Path(args.events);p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(events,ensure_ascii=False,indent=2))
    if args.tts_output and output:
        p=Path(args.tts_output);p.parent.mkdir(parents=True,exist_ok=True)
        with wave.open(str(p),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(sample_rate);w.writeframes(output)
        print(f'TTS音频已保存: {p}',flush=True)


def main():
    parser=argparse.ArgumentParser(description='同传客户端：WS实时输入或文件上传SSE')
    parser.add_argument('audio');parser.add_argument('--url',default='http://127.0.0.1:8080')
    parser.add_argument('--transport',choices=['stream','ws'],default='ws')
    parser.add_argument('--model-type','--mode',dest='mode',choices=['target_only','source_target'])
    parser.add_argument('--response-mode',choices=['standard','short_chunk'],
                        help='standard：标准模式（推荐，1600ms）；short_chunk：短分块模式（首块必须1600ms，后续800ms）')
    parser.add_argument('--direction',choices=['zh-en','en-zh'],default='zh-en')
    parser.add_argument('--realtime',action='store_true',help='WS按100ms实时节奏发送')
    parser.add_argument('--vad',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--tts',action=argparse.BooleanOptionalAction,default=None);parser.add_argument('--voice',default='')
    parser.add_argument('--tts-output',default='outputs/translated.wav')
    parser.add_argument('--events',default='outputs/events.json');parser.add_argument('--timeout',type=float,default=900)
    asyncio.run(run(parser.parse_args()))


if __name__=='__main__':main()
