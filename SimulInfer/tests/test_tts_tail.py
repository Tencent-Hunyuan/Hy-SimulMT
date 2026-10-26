import numpy as np
import pytest
from simulinfer.tts_tail import SpeechTail
from simulinfer.tts_audio import TTSStreamError


class Detector:
    def __init__(self, spans):self.spans=spans
    def analyze(self, samples, offset):
        mask=np.zeros(len(samples),bool)
        for left,right in self.spans:
            a=max(0,round(left*16000)-offset);b=min(len(samples),round(right*16000)-offset)
            if b>a:mask[a:b]=True
        return {'_vad_speech_mask':mask}


async def filtered(raw,spans):
    output=[]
    async def emit(b):output.append(b)
    gate=SpeechTail(16000,emit,detector=Detector(spans));await gate.start()
    for off in range(0,len(raw),317):await gate.feed(raw[off:off+317])
    await gate.finish()
    return b''.join(output)


@pytest.mark.asyncio
async def test_filter_keeps_speech_padding_and_interior_pause():
    raw=np.arange(16000*4,dtype=np.int16).tobytes()
    actual=await filtered(raw,[(.4,1),(1.5,2.6)])
    expected=raw[round(.16*16000)*2:round(2.84*16000)*2]
    assert actual==expected


@pytest.mark.asyncio
async def test_long_nonspeech_output_is_aborted_not_committed_as_success():
    output=[]
    async def emit(b):output.append(b)
    gate=SpeechTail(16000,emit,detector=Detector([(0,.6)]));await gate.start()
    with pytest.raises(TTSStreamError,match='unverified'):
        await gate.feed(np.ones(16000*15,dtype='<i2').tobytes())
    assert len(b''.join(output))==round(.84*16000)*2
    assert gate.position<=round(3.2*16000)


@pytest.mark.asyncio
async def test_entire_nonspeech_output_is_not_published():
    output=[]
    async def emit(b):output.append(b)
    gate=SpeechTail(16000,emit,detector=Detector([]));await gate.start()
    await gate.feed(np.ones(16000,dtype='<i2').tobytes())
    with pytest.raises(TTSStreamError,match='no detected speech'):await gate.finish()
    assert not output
