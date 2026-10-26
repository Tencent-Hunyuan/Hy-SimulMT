import {serviceURL, websocketURL} from './urls.js';
import {t, setMessage, setLiteral, uiError} from './i18n.js?v=0.5.14';
const $=id=>document.getElementById(id);
let ws,context,media,node,playingAt=0,busy=false,pendingAudio=null,ttsSources=[],captureFlushed=null;
let ttsFinished=false;
let settings=null,inputNode=null,mutedNode=null,captureStats=null,captureTimer=null;
function updateCaptureStatus(force=false){
  if(!captureStats||(!busy&&!force))return;
  const now=performance.now(),s=captureStats;
  if(context?.state==='suspended'&&!s.stopping)context.resume().catch(()=>{});
  const seconds=s.samples/s.rate;
  $('capture-status').hidden=false;
  let message=t('captured',{captured:seconds.toFixed(1),received:(s.receivedMs/1000).toFixed(1),processed:(s.processedMs/1000).toFixed(1)});
  if(s.stopping)message+=' · '+t('captureFinishing');
  else if(now-s.started>3000&&now-s.lastFrame>3000)message+=' · '+t('captureNoFrames');
  else if(now-s.lastVoice>3000)message+=' · '+t('captureQuiet');
  else if(seconds*1000-s.receivedMs>5000)message+=' · '+t('captureSending');
  else if(s.receivedMs-s.processedMs>5000)message+=' · '+t('captureProcessing');
  else if(s.lastAction==='WAIT')message+=' · '+t('captureWaiting');
  $('capture-status').textContent=message;
}
function captureFailed(error){
  fail(error);stopCapture();ws?.close();ws=null;setBusy(false);
}
function sendCaptured(raw){
  if(!captureStats||ws?.readyState!==WebSocket.OPEN)return;
  if(ws.bufferedAmount>1024*1024){captureFailed(uiError('networkSlow'));return}
  const samples=new Int16Array(raw);let peak=0;
  for(const x of samples)peak=Math.max(peak,Math.abs(x));
  const now=performance.now();captureStats.samples+=samples.length;captureStats.lastFrame=now;
  if(peak>98)captureStats.lastVoice=now;
  ws.send(raw);
}

function updateDirection(){
  const toEnglish=$('direction').value==='zh-en';
  setMessage($('source-language'),toEnglish?'chinese':'english');
  setMessage($('target-language'),toEnglish?'english':'chinese');
  const label=t(toEnglish?'swapToChinese':'swapToEnglish');
  $('direction-swap').setAttribute('aria-label',label);
  $('direction-swap').title=label;
}
$('direction-swap').onclick=()=>{
  if(busy||!settings)return;
  $('direction').value=$('direction').value==='zh-en'?'en-zh':'zh-en';
  updateDirection();
};

document.addEventListener('uilanguagechange',()=>{updateDirection();updateCaptureStatus(true)});
updateDirection();

function defaultModel(){return settings?.defaults.model_type}
function cadenceAvailable(){return settings?.response_mode_available?.[defaultModel()]!==false}
function showModel(mode){
  $('source-panel').hidden=mode!=='source_target';
  const name=settings?.model_labels?.[mode]||'';
  setMessage($('model-info'),mode==='source_target'?'modelSourceTarget':'modelTarget',{name});
}
function updateCadence(){
  const available=cadenceAvailable();
  $('response-mode').disabled=busy||!settings||!available;
  $('cadence-help').hidden=true;
}
async function initialize(){
  try{
    const response=await fetch(serviceURL('config'),{cache:'no-store'});
    if(!response.ok)throw uiError('configHttp',{status:response.status});
    const data=await response.json();
    if(!data.defaults||!data.modes?.includes(data.defaults.model_type))throw uiError('configInvalid');
    if(!['standard','short_chunk'].includes(data.defaults.response_mode))throw uiError('versionMismatch');
    settings=data;
    $('response-mode').value='standard';
    $('tts').checked=settings.defaults.tts;
    $('mock').hidden=!settings.mock;
    $('voice-field').hidden=!!settings.tts_clone_source;
    setMessage($('tts-label'),settings.tts_clone_source?'cloneTranslation':'readTranslation');
    showModel(defaultModel());updateCadence();setBusy(false);setMessage($('status'),'ready');
  }catch(e){fail(e);setBusy(false)}
}
function fail(e){$('error').hidden=false;if(e.i18nKey)setMessage($('error'),e.i18nKey,e.i18nParams);else setLiteral($('error'),e.message||String(e));setMessage($('status'),'failed');}
function setBusy(value){busy=value;for(const id of ['upload','mic','response-mode','direction','direction-swap','tts','voice','file'])$(id).disabled=!settings||value||(id==='response-mode'&&!cadenceAvailable())||(id==='tts'&&!settings.tts_available);$('stop').disabled=!value||!ws;}
async function prepare(){
  $('error').hidden=true;$('source').textContent='';$('translation').textContent='';$('progress').textContent='';showModel(defaultModel());pendingAudio=null;
  $('tts-history').hidden=true;
  $('tts-warning').hidden=true;$('tts-status').hidden=true;ttsFinished=false;
  $('input-audio').hidden=true;
  $('capture-status').hidden=true;
  captureStats=null;
  ttsSources.forEach(s=>{try{s.stop()}catch{}});ttsSources=[];
  if(context){await context.close();context=null}
  if($('tts').checked){context=new AudioContext();await context.resume();playingAt=context.currentTime}
}
function decode64(s){const raw=atob(s);return Uint8Array.from(raw,c=>c.charCodeAt(0)).buffer}
function play(raw,sr,seq){
  if(!context)return;
  const view=new DataView(raw),a=new Float32Array(raw.byteLength/2);
  for(let i=0;i<a.length;i++)a[i]=view.getInt16(i*2,true)/32768;
  const buffer=context.createBuffer(1,a.length,sr);buffer.copyToChannel(a,0);
  const source=context.createBufferSource(),gain=context.createGain();source.buffer=buffer;source.connect(gain).connect(context.destination);
  source.ttsSeq=seq;source.ttsGain=gain;
  if(playingAt<=context.currentTime)playingAt=context.currentTime+.20;
  source.ttsStart=playingAt;source.start(playingAt);playingAt+=buffer.duration;source.ttsEnd=playingAt;
  $('tts-status').hidden=false;setMessage($('tts-status'),'ttsPlaying');
  ttsSources.push(source);source.onended=()=>{ttsSources=ttsSources.filter(x=>x!==source);source.disconnect();gain.disconnect();if(!ttsSources.length&&ttsFinished)setMessage($('tts-status'),'ttsFinished')};
}
function handle(e){
  if(e.event==='tts_error'){
    if(context){
      const now=context.currentTime;
      for(const source of ttsSources.filter(s=>s.ttsSeq===e.seq)){
        try{
          if(source.ttsStart>now)source.stop();
          else{source.ttsGain.gain.setValueAtTime(1,now);source.ttsGain.gain.linearRampToValueAtTime(0,now+.015);source.stop(now+.015)}
        }catch{}
      }
      ttsSources=ttsSources.filter(s=>s.ttsSeq!==e.seq);
      playingAt=Math.max(now,...ttsSources.map(s=>s.ttsEnd));
    }
    $('tts-warning').hidden=false;setMessage($('tts-warning'),'ttsPartial');return;
  }
  if(e.event==='tts_start'){
    $('tts-status').hidden=false;setMessage($('tts-status'),ttsSources.length?'ttsPlaying':'ttsGenerating');return;
  }
  if(e.event==='tts_done'){
    ttsFinished=true;
    if(!ttsSources.length){$('tts-status').hidden=false;setMessage($('tts-status'),'ttsFinished')}
    return;
  }
  if(e.event==='input_progress'){
    if(captureStats){captureStats.receivedMs=e.received_ms;updateCaptureStatus()}
    return;
  }
  if(e.event==='voice_clone'){
    $('tts-history').hidden=false;setMessage($('tts-history'),'cloned',{seconds:e.reference_seconds.toFixed(1)});
    return;
  }
  if(e.event==='tts_unavailable'){
    $('tts-history').hidden=false;setLiteral($('tts-history'),e.detail);return;
  }
  if(e.event==='error')throw (e.detail?Error(e.detail):uiError('serviceError'));
  if(e.event==='ready'){
    showModel(e.model_type||e.mode||defaultModel());
    if(typeof e.source_text_enabled==='boolean')$('source-panel').hidden=!e.source_text_enabled;
    setMessage($('status'),'connected');
    if(e.input_audio){
      const a=e.input_audio;$('input-audio').hidden=false;
      if(a.sample_rate)setMessage($('input-audio'),'audioFormat',{rate:a.sample_rate,channels:a.channels,outputRate:a.output_sample_rate,outputChannels:a.output_channels});
      else setMessage($('input-audio'),'audioDecoded',{rate:a.output_sample_rate,channels:a.output_channels});
    }
    if(e.tts_enabled){
      const history=e.tts?.continuity;
      $('tts-history').hidden=false;
      const key=e.tts?.clone_source?'ttsCloning':history==='paired_reference_audio'||history==='backend_paired_history'
        ? 'ttsHistory':history==='backend_session_unverified'?'ttsSession':'ttsIndependent';
      setMessage($('tts-history'),key,{seconds:e.tts?.history_window_s});
    }
    return;
  }
  if(e.event==='audio'){
    if(e.encoding==='binary-next')pendingAudio=e;
    else play(decode64(e.data),e.sample_rate,e.seq);
  }
  if(e.event==='chunk'||e.event==='flush'){
    if(captureStats){captureStats.processedMs=e.end_ms||captureStats.processedMs;captureStats.lastAction=e.action;updateCaptureStatus()}
    $('translation').textContent+=e.text||'';
    if(e.source_text_full!==undefined)$('source').textContent=e.source_text_full;
    else $('source').textContent+=e.source_text||'';
    $('progress').textContent=`${((e.end_ms||0)/1000).toFixed(1)}s · ${e.action||'WAIT'}`;
    setMessage($('status'),e.text?'translating':'waitingMore');
  }
  if(e.event==='done'){
    if(e.full_text!==undefined)$('translation').textContent=e.full_text;
    if(e.source_text!==undefined)$('source').textContent=e.source_text||'';
    if(captureStats){captureStats.receivedMs=e.duration_ms;captureStats.processedMs=e.duration_ms;updateCaptureStatus()}
    const warning=e.complete===false||e.unresolved_audio_ms||e.skipped_untranslated_audio_ms;
    setMessage($('status'),warning?'incomplete':'done');
  }
}
$('file').onchange=()=>{const f=$('file').files[0];if(f)setLiteral($('status'),f.name)};
$('upload').onclick=async()=>{
  if(busy||!settings)return;const file=$('file').files[0];if(!file)return fail(uiError('chooseFile'));
  setBusy(true);
  try{
    await prepare();setMessage($('status'),'uploading');
    const data=new FormData();data.append('file',file);data.append('direction',$('direction').value);data.append('tts',$('tts').checked);data.append('voice',$('voice').value);
    const r=await fetch(serviceURL('translate/stream'),{method:'POST',body:data});if(!r.ok)throw Error(await r.text());
    const reader=r.body.getReader(),decoder=new TextDecoder();let buf='',doneSeen=false;
    while(true){const {value,done}=await reader.read();if(done)break;buf+=decoder.decode(value,{stream:true});
      let pos;while((pos=buf.indexOf('\n\n'))>=0){const item=buf.slice(0,pos);buf=buf.slice(pos+2);for(const line of item.split('\n'))if(line.startsWith('data:')){const e=JSON.parse(line.slice(5));handle(e);if(e.event==='done')doneSeen=true}}}
    if(!doneSeen)throw uiError('missingDone');
  }catch(e){fail(e)}finally{setBusy(false)}
};
async function stopCapture(){
  if(captureTimer){clearInterval(captureTimer);captureTimer=null}
  if(node){node.disconnect();node=null}
  if(inputNode){inputNode.disconnect();inputNode=null}
  if(mutedNode){mutedNode.disconnect();mutedNode=null}
  if(media){media.getTracks().forEach(t=>t.stop());media=null}
}
$('mic').onclick=async()=>{
  if(busy||!settings)return;setBusy(true);
  try{
    await prepare();
    if(!navigator.mediaDevices?.getUserMedia)throw uiError('micSecure');
    context=context||new AudioContext();await context.resume();
    media=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:true,noiseSuppression:true}});
    ws=new WebSocket(websocketURL());ws.binaryType='arraybuffer';
    let completed=false;
    const socket=ws;
    const ready=new Promise((resolve,reject)=>{
      ws.onopen=()=>ws.send(JSON.stringify({direction:$('direction').value,input_format:'s16le',input_sample_rate:context.sampleRate,input_channels:1,tts:$('tts').checked,voice:$('voice').value,audio_transport:'binary',input_progress:true}));
      ws.onerror=()=>{const error=uiError('socketFailed');reject(error);fail(error)};
      ws.onmessage=e=>{try{if(e.data instanceof ArrayBuffer){if(!pendingAudio)throw uiError('audioMetadata');play(e.data,pendingAudio.sample_rate,pendingAudio.seq);pendingAudio=null;return}const d=JSON.parse(e.data);handle(d);if(d.event==='ready')resolve();if(d.event==='done'){completed=true;socket.close();if(ws===socket)ws=null;stopCapture();setBusy(false)}}catch(err){reject(err);fail(err);stopCapture();ws?.close();ws=null;setBusy(false)}};
      ws.onclose=()=>{if(!completed&&ws===socket){const error=uiError('socketClosed');reject(error);fail(error);stopCapture();ws=null;setBusy(false)}};
    });
    await ready;
    await context.resume();
    if(context.state!=='running')throw uiError('captureSuspended');
    const now=performance.now();captureStats={samples:0,rate:context.sampleRate,started:now,lastFrame:now,lastVoice:now,receivedMs:0,processedMs:0,lastAction:'',stopping:false};
    if(context.audioWorklet&&typeof AudioWorkletNode!=='undefined'){
      await context.audioWorklet.addModule(serviceURL('static/recorder.js?v=0.5.7'));
      node=new AudioWorkletNode(context,'pcm-recorder');
      node.onprocessorerror=()=>captureFailed(uiError('captureInterrupted'));
      node.port.onmessage=e=>{
        if(e.data?.event==='flushed'){captureFlushed?.();captureFlushed=null;return}
        sendCaptured(e.data);
      };
    }else{
      node=context.createScriptProcessor(4096,1,1);
      node.onaudioprocess=e=>{
        const source=e.inputBuffer.getChannelData(0),pcm=new Int16Array(source.length);
        for(let i=0;i<source.length;i++)pcm[i]=Math.max(-32768,Math.min(32767,Math.round(source[i]*32768)));
        sendCaptured(pcm.buffer);
      };
    }
    inputNode=context.createMediaStreamSource(media);
    mutedNode=context.createGain();mutedNode.gain.value=0;
    inputNode.connect(node);node.connect(mutedNode).connect(context.destination);
    for(const track of media.getAudioTracks())track.onended=()=>{if(captureStats&&!captureStats.stopping&&busy)captureFailed(uiError('micDisconnected'))};
    captureTimer=setInterval(updateCaptureStatus,1000);updateCaptureStatus();
    $('stop').disabled=false;setMessage($('status'),'recording');
  }catch(e){fail(e);await stopCapture();ws?.close();ws=null;setBusy(false)}
};
$('stop').onclick=async()=>{
  if(captureStats)captureStats.stopping=true;
  $('stop').disabled=true;setMessage($('status'),'finishing');
  try{
    if(node?.port){await new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>{captureFlushed=null;reject(uiError('captureFlush'))},2000);
      captureFlushed=()=>{clearTimeout(timer);resolve()};node.port.postMessage('flush');
    })}
    await stopCapture();if(ws?.readyState===WebSocket.OPEN)ws.send(JSON.stringify({eof:true}));
  }catch(e){fail(e);await stopCapture();ws?.close();ws=null;setBusy(false)}
};

setBusy(false);
await initialize();
