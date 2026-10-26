class PCMRecorder extends AudioWorkletProcessor {
  constructor() {
    super();this.samples=[];this.blockSize=Math.max(1,Math.round(sampleRate*.1));
    this.stopped=false;
    this.port.onmessage=e=>{if(e.data==='flush'){this.stopped=true;this.flush();this.port.postMessage({event:'flushed'})}};
  }
  flush(){if(this.samples.length){const a=Int16Array.from(this.samples);this.port.postMessage(a.buffer,[a.buffer]);this.samples=[]}}
  process(inputs){
    const a=inputs[0]?.[0];if(!a||this.stopped)return true;
    for(const value of a){
      // Preserve the AudioContext rate. Stateful antialiasing resampling runs
      // in the SimulInfer, not independently on each microphone callback.
      this.samples.push(Math.max(-32768,Math.min(32767,Math.round(value*32768))));
      if(this.samples.length>=this.blockSize)this.flush();
    }return true;
  }
}
registerProcessor('pcm-recorder',PCMRecorder);
