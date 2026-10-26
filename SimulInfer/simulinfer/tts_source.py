"""Session-owned source voice snapshots, measured on the source audio timeline."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from .tts_history import wav_data_url


@dataclass(frozen=True)
class VoiceSnapshot:
    revision: int
    end_sample: int
    seconds: float
    audio: str


class SourceVoice:
    def __init__(self, first_seconds=2.4, interval_seconds=10):
        self.first=round(first_seconds*16000)
        self.window=round(interval_seconds*16000)
        self.next_at=self.first
        self.total=0
        self.parts=deque()
        self.samples=0
        self.snapshots=deque(maxlen=16)
        self.revision=0

    def feed(self, samples):
        data=np.asarray(samples,dtype=np.float32).reshape(-1)
        if not np.isfinite(data).all():raise ValueError('Source audio contains NaN or Inf')
        offset=0
        while offset<len(data):
            count=min(len(data)-offset,self.next_at-self.total)
            piece=data[offset:offset+count].copy();offset+=count
            self.parts.append(piece);self.samples+=count;self.total+=count
            while self.samples>self.window:
                excess=self.samples-self.window
                head=self.parts.popleft()
                if len(head)>excess:
                    self.parts.appendleft(head[excess:]);self.samples-=excess
                else:self.samples-=len(head)
            if self.total==self.next_at:
                self._snapshot()
                self.next_at=(self.total//self.window+1)*self.window

    def _snapshot(self):
        if not self.parts:return
        audio=np.concatenate(self.parts)
        peak=float(np.max(np.abs(audio)))
        if peak<1e-4:return  # Keep the existing voice for silent update windows.
        if peak<.5 or peak>.95:audio=audio*(.7/peak)
        pcm=np.clip(np.rint(np.clip(audio,-1,1)*32768),-32768,32767).astype('<i2').tobytes()
        self.revision+=1
        snapshot=VoiceSnapshot(self.revision,self.total,len(audio)/16000,wav_data_url(pcm,16000))
        self.snapshots.append(snapshot)

    def latest(self, end_ms=None):
        if not self.snapshots:return None
        if end_ms is None:return self.snapshots[-1]
        available=[s for s in self.snapshots if s.end_sample<=round(end_ms*16)]
        return available[-1] if available else None

    def finish(self):
        # A short final utterance can still supply a voice reference. The model
        # needs enough samples for its speaker encoder; do not invent padding.
        if not self.snapshots and self.samples>=6400:self._snapshot()

    def clear(self):
        self.parts.clear();self.snapshots.clear();self.samples=0
