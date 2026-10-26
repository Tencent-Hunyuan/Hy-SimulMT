"""User audio formats normalize to mono16k without resetting at packet boundaries."""
import base64
import io
import json
from pathlib import Path
import subprocess

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from simulinfer.app import build_app
from simulinfer.audio import PCM, decode_file, decode_file_with_info
from simulinfer.config import load


def signal(rate, channels, seconds=1.237):
    n=round(rate*seconds)
    a=(.2*np.sin(2*np.pi*440*np.arange(n)/rate)).astype(np.float32)
    return a if channels==1 else np.stack([a*(1-.05*c) for c in range(channels)],axis=1)


def wav(a, rate):
    out=io.BytesIO();sf.write(out,a,rate,format='WAV',subtype='FLOAT');return out.getvalue()


@pytest.mark.parametrize('rate',[8000,16000,22050,24000,44100,48000,96000,192000])
@pytest.mark.parametrize('channels',[1,2,8])
def test_file_and_irregular_stream_agree_and_preserve_duration(rate,channels):
    a=signal(rate,channels);raw=np.asarray(a,dtype='<f4').tobytes()
    expected,meta=decode_file_with_info(wav(a,rate),100000)
    converter=PCM('f32le',rate,channels);pieces=[]
    for i in range(0,len(raw),1117):pieces.append(converter.feed(raw[i:i+1117]))
    pieces.append(converter.finish());actual=np.concatenate(pieces)
    np.testing.assert_array_equal(actual,expected)
    assert abs(len(actual)-len(a)*16000/rate)<=.5
    assert meta['sample_rate']==rate and meta['channels']==channels
    assert meta['output_sample_rate']==16000 and meta['output_channels']==1
    assert converter.info()['input_frames']==len(a)
    assert len(converter.finish())==0
    with pytest.raises(ValueError,match='EOF'):converter.feed(b'')


@pytest.mark.parametrize('fmt,dtype,scale',[('f32le','<f4',1),('s16le','<i2',32768)])
def test_existing_mono16k_path_is_bit_identical(fmt,dtype,scale):
    a=signal(16000,1)
    raw=(a*scale).astype(dtype).tobytes()
    c=PCM(fmt);parts=[c.feed(raw[i:i+313]) for i in range(0,len(raw),313)]+[c.finish()]
    expected=np.frombuffer(raw,dtype=dtype).astype(np.float32)/scale
    np.testing.assert_array_equal(np.concatenate(parts),expected)


def test_downmix_uses_both_channels_without_gain_normalization():
    a=np.stack([np.full(8000,.1),np.full(8000,.3)],axis=1).astype('<f4')
    c=PCM('f32le',16000,2);out=c.feed(a.tobytes())
    np.testing.assert_allclose(out,.2,rtol=0,atol=2e-8)
    assert len(c.finish())==0


def test_resampling_filters_out_of_band_aliasing():
    rate=48000;t=np.arange(rate)/rate
    for hz,min_rms,max_rms in [(1000,.13,.15),(12000,0,.0001)]:
        a=(.2*np.sin(2*np.pi*hz*t)).astype('<f4');c=PCM('f32le',rate)
        out=np.concatenate([c.feed(a.tobytes()),c.finish()])[1600:-1600]
        rms=float(np.sqrt(np.mean(out*out)))
        assert min_rms<=rms<=max_rms,(hz,rms)


@pytest.mark.parametrize('fmt,width',[('f32le',4),('s16le',2)])
def test_multichannel_eof_rejects_incomplete_interleaved_frame(fmt,width):
    c=PCM(fmt,48000,2);c.feed(bytes(width))
    with pytest.raises(ValueError,match='不完整'):c.finish()


@pytest.mark.parametrize('rate,channels',[(0,1),(True,1),(16000.0,1),(384000,1),(16000,0),(16000,9),(16000,True)])
def test_invalid_input_metadata_rejected(rate,channels):
    with pytest.raises(ValueError):PCM('f32le',rate,channels)


def test_decode_limit_is_measured_after_conversion():
    for rate in [8000,48000]:
        raw=wav(signal(rate,2,1),rate)
        assert len(decode_file(raw,16000))==16000
        with pytest.raises(ValueError,match='上限'):decode_file(raw,15999)
    with pytest.raises(ValueError):decode_file(b'',100)
    with pytest.raises(ValueError):decode_file(wav(np.full(20,np.nan,np.float32),48000),100)


@pytest.mark.parametrize('rate,channels',[(8000,1),(44100,2),(48000,2)])
def test_http_file_conversion_metadata_and_timeline(rate,channels):
    with TestClient(build_app(load(mock=True))) as client:
        r=client.post('/translate/stream',files={'file':('source.wav',wav(signal(rate,channels,3.7),rate),'audio/wav')})
        assert r.status_code==200,r.text
        es=[json.loads(l[5:]) for l in r.text.splitlines() if l.startswith('data:')]
        assert es[0]['input_audio']['sample_rate']==rate
        assert es[-1]['event']=='done' and es[-1]['duration_ms']==3700
        assert es[-1]['input_audio']['output_samples']==59200


@pytest.mark.parametrize('fmt',["f32le","s16le"])
@pytest.mark.parametrize('rate,channels',[(8000,1),(44100,2),(48000,2)])
def test_websocket_eof_flush_preserves_tail(fmt,rate,channels):
    a=signal(rate,channels,3.7)
    raw=a.astype('<f4').tobytes() if fmt=='f32le' else np.rint(a*32768).astype('<i2').tobytes()
    with TestClient(build_app(load(mock=True))) as client:
        with client.websocket_connect('/translate/ws') as ws:
            ws.send_json({'input_format':fmt,'input_sample_rate':rate,'input_channels':channels})
            ready=ws.receive_json();assert ready['input_audio']['sample_rate']==rate
            for i in range(0,len(raw),10007):ws.send_bytes(raw[i:i+10007])
            ws.send_json({'eof':True});events=[]
            while True:
                e=ws.receive_json();events.append(e)
                assert e['event']!='error',e
                if e['event']=='done':break
        assert events[-1]['duration_ms']==3700
        assert events[-1]['input_audio']['output_samples']==59200
        chunks=[e for e in events if e['event']=='chunk']
        assert [(e['start_ms'],e['end_ms']) for e in chunks]==[(0,1600),(1600,3200),(3200,3700)]


def test_websocket_alias_metadata_and_invalid_rate():
    with TestClient(build_app(load(mock=True))) as client:
        with client.websocket_connect('/translate/ws') as ws:
            ws.send_json({'sample_rate':48000,'channels':2})
            assert ws.receive_json()['input_audio']['channels']==2
            ws.send_bytes(np.zeros(4800*2,dtype='<i2').tobytes());ws.send_json({'eof':True})
            while ws.receive_json()['event']!='done':pass
        with client.websocket_connect('/translate/ws') as ws:
            ws.send_json({'input_sample_rate':False})
            assert ws.receive_json()['event']=='error'


def test_resampling_sessions_do_not_share_filter_state():
    a=signal(44100,1,.7);b=signal(48000,1,.7)*.5
    converters=[PCM('f32le',44100),PCM('f32le',48000)];out=[[],[]]
    for start in range(0,max(len(a),len(b)),731):
        for i,x in enumerate([a,b]):out[i].append(converters[i].feed(x[start:start+731].tobytes()))
    for i,(x,rate) in enumerate([(a,44100),(b,48000)]):
        out[i].append(converters[i].finish());expected=decode_file(wav(x,rate),50000)
        np.testing.assert_array_equal(np.concatenate(out[i]),expected)


def test_compressed_fallback_is_bounded_and_keeps_duration(tmp_path):
    import imageio_ffmpeg
    source=tmp_path/'source.wav';source.write_bytes(wav(signal(48000,2,1.25),48000))
    compressed=tmp_path/'source.m4a'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(source),'-c:a','aac',str(compressed)],check=True)
    x,meta=decode_file_with_info(compressed.read_bytes(),24000)
    assert meta['decoder']=='ffmpeg'
    assert abs(len(x)/16000-1.25)<.05
    with pytest.raises(ValueError,match='上限'):decode_file(compressed.read_bytes(),16000)
    with pytest.raises(ValueError,match='无法解码'):decode_file(b'invalid audio container',16000)
