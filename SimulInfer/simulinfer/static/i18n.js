const messages={
  en:{
    pageTitle:'Hy-SimulMT · Live translation',
    interfaceLanguage:'Interface language',headline:'Listen. Translate.',
    intro:'Upload audio or start a live conversation, with optional spoken translation.',
    demoNotice:'Demo mode: text is simulated and audio is a test tone, not a real translation.',
    chunkMode:'Chunking mode',standardMode:'Standard',translationDirection:'Translation direction',
    chinese:'Chinese',english:'English',swapToEnglish:'Switch to Chinese → English',swapToChinese:'Switch to English → Chinese',
    readTranslation:'Read translation aloud',cloneTranslation:'Clone voice & read aloud',voice:'Voice',defaultVoice:'Use the default TTS voice',
    loadingModel:'Loading the deployed model…',modelSourceTarget:'Model: {name} · Source and translation',modelTarget:'Model: {name} · Translation only',
    chooseAudio:'Choose audio',upload:'Translate file',or:'or',startLive:'Start live',finish:'Finish',
    inputHint:'Mono and multichannel audio at common sample rates are converted automatically. Microphone access requires localhost or HTTPS.',
    sourceText:'Source',translation:'Translation',waitingSpeech:'Waiting for speech…',
    connecting:'Connecting to the service…',ready:'Ready',failed:'Request failed',connected:'Connected, waiting for audio…',
    noSpeech:'No speech detected',translating:'Translating…',waitingMore:'Waiting for more speech…',done:'Done',incomplete:'Finished; some audio could not be translated',
    uploading:'Uploading…',recording:'Listening…',finishing:'Finishing the translation and remaining audio…',
    captured:'Captured {captured}s · Received {received}s · Processed {processed}s',
    captureFinishing:'Processing remaining audio',captureNoFrames:'No microphone audio; check permissions and keep this page in the foreground',
    captureQuiet:'Very quiet audio; check your microphone',captureSending:'Sending audio; check the network or wait for the service',
    captureProcessing:'The model is processing audio; please wait',captureWaiting:'Waiting for more speech',
    cloned:'Using the input voice · {seconds}s reference audio',
    audioFormat:'Input: {rate} Hz / {channels} channels → Model: {outputRate} Hz / {outputChannels} channels',
    audioDecoded:'Audio decoded to {rate} Hz / {channels} channels for the model',
    ttsGenerating:'Generating speech…',ttsPlaying:'Playing speech…',ttsFinished:'Speech playback finished',
    ttsPartial:'A speech segment was interrupted. Translation continues; some words may not have been read aloud.',
    ttsCloning:'Automatically cloning the input voice for spoken translation',
    ttsHistory:'Spoken translation uses approximately {seconds}s of preceding audio with its matching text.',
    ttsSession:'TTS connected; audio history support has not been confirmed by the backend.',
    ttsIndependent:'TTS synthesizes each segment independently, without preceding audio.',
    networkSlow:'The network or model is too slow. End the session and reconnect.',
    configHttp:'Unable to load service configuration (HTTP {status})',
    configInvalid:'Invalid service configuration. Check the deployment address.',
    versionMismatch:'The page and backend versions do not match. Restart the service and refresh the page.',
    serviceError:'Service error',chooseFile:'Choose an audio file first.',missingDone:'The connection ended without a completion event.',
    gatewayTimeout:'The gateway timed out (HTTP 504). Translation did not complete. Check the port forwarding connection before retrying.',
    serviceUnavailable:'The translation service is unavailable (HTTP {status}). Please try again later.',
    uploadTooLarge:'The audio file exceeds the upload size limit (HTTP 413).',uploadHttp:'Audio upload failed (HTTP {status}).',
    uploadProtocol:'The file task returned an invalid response. Please reconnect.',uploadCancelled:'File translation cancelled.',
    micSecure:'Microphone access requires localhost or HTTPS.',socketFailed:'WebSocket connection failed. Check the service address and proxy.',
    audioMetadata:'Audio metadata is missing.',socketClosed:'The WebSocket closed before completion. Please reconnect.',
    captureSuspended:'Audio capture was paused by the browser. Start again.',captureInterrupted:'Microphone capture was interrupted. Please reconnect.',
    micDisconnected:'The microphone disconnected. Check the device and reconnect.',captureFlush:'The final microphone audio was not acknowledged. Please reconnect.'
  },
  zh:{
    pageTitle:'Hy-SimulMT · 流式同传',
    interfaceLanguage:'界面语言',headline:'边听，边译。',
    intro:'上传音频，或开启实时对话，可选择朗读译文。',
    demoNotice:'当前是协议演示模式：文本为模拟结果，声音为提示音，不是真实翻译。',
    chunkMode:'分块模式',standardMode:'标准模式',translationDirection:'翻译方向',
    chinese:'中文',english:'英文',swapToEnglish:'切换为中文 → 英文',swapToChinese:'切换为英文 → 中文',
    readTranslation:'朗读译文',cloneTranslation:'克隆音色并朗读译文',voice:'音色',defaultVoice:'使用 TTS 默认音色',
    loadingModel:'正在读取已部署模型…',modelSourceTarget:'当前模型：{name} · 自动显示源文和译文',modelTarget:'当前模型：{name} · 仅显示译文',
    chooseAudio:'选择音频',upload:'上传并翻译',or:'或',startLive:'开始实时对话',finish:'结束并收尾',
    inputHint:'支持常见采样率的单声道或多声道音频，上传后自动转换供模型使用。麦克风需要 localhost 或 HTTPS。',
    sourceText:'源文',translation:'译文',waitingSpeech:'等待语音…',
    connecting:'正在连接服务…',ready:'已就绪',failed:'请求失败',connected:'已连接，等待音频…',
    noSpeech:'未检测到人声',translating:'正在翻译',waitingMore:'等待更多语音…',done:'已完成',incomplete:'处理结束，部分音频未能翻译',
    uploading:'正在上传…',recording:'正在收音…',finishing:'正在收尾并生成剩余语音…',
    captured:'已采集 {captured} 秒 · 服务收到 {received} 秒 · 已处理 {processed} 秒',
    captureFinishing:'正在处理剩余音频',captureNoFrames:'麦克风未送出音频，请检查权限并保持页面在前台',
    captureQuiet:'声音很小，请检查麦克风',captureSending:'音频发送中，请检查网络或等待服务处理',
    captureProcessing:'模型正在处理，请稍候',captureWaiting:'等待更多语音',
    cloned:'已使用输入音色朗读 · {seconds} 秒参考语音',
    audioFormat:'输入 {rate} Hz / {channels} 声道 → 模型 {outputRate} Hz / {outputChannels} 声道',
    audioDecoded:'文件已解码为 {rate} Hz / {channels} 声道供模型使用',
    ttsGenerating:'正在生成译音…',ttsPlaying:'正在朗读…',ttsFinished:'译音播放完毕',
    ttsPartial:'有一段译音未完成，部分文字可能未朗读；同传继续。',
    ttsCloning:'自动克隆输入音色并朗读译文',
    ttsHistory:'译音保留前文语音，历史窗口约 {seconds} 秒；完整句段与文字成对保留。',
    ttsSession:'已连接会话式 TTS，后端未声明是否保留前文语音。',
    ttsIndependent:'译音独立合成，当前 TTS 接口不使用前文语音。',
    networkSlow:'网络或模型处理过慢，请结束后重新连接',
    configHttp:'读取服务配置失败（HTTP {status}）',configInvalid:'服务配置无效，请检查部署地址',
    versionMismatch:'页面与后端版本不一致，请重启服务后刷新页面',
    serviceError:'服务错误',chooseFile:'请先选择音频文件',missingDone:'连接结束但没有收到完成事件',
    gatewayTimeout:'网关等待超时（HTTP 504），本次翻译未完成。请检查端口转发连接后重试。',
    serviceUnavailable:'翻译服务暂时不可用（HTTP {status}），请稍后重试。',
    uploadTooLarge:'音频文件超过上传大小限制（HTTP 413）。',uploadHttp:'音频上传失败（HTTP {status}）。',
    uploadProtocol:'文件任务响应异常，请重新连接。',uploadCancelled:'已取消文件翻译。',
    micSecure:'麦克风需要 localhost 或 HTTPS 安全环境',socketFailed:'WebSocket 连接失败，请检查服务地址和代理连接',
    audioMetadata:'音频缺少元信息',socketClosed:'WebSocket 在结果完成前断开，请重新连接',
    captureSuspended:'浏览器暂停了音频采集，请重新点击开始',captureInterrupted:'麦克风采集器中断，请重新连接',
    micDisconnected:'麦克风已断开，请检查设备并重新连接',captureFlush:'麦克风尾帧未确认，请重新连接'
  }
};
const storageKey='hy-simulmt-ui-language';
const parameters=new WeakMap();
let locale='en';
try{if(localStorage.getItem(storageKey)==='zh')locale='zh'}catch{}
export function t(key,params={}){
  return (messages[locale][key]??messages.en[key]??key).replace(/\{(\w+)\}/g,(_,name)=>String(params[name]??''));
}
export function setMessage(element,key,params={}){
  element.dataset.i18n=key;parameters.set(element,params);element.textContent=t(key,params);
}
export function setLiteral(element,value){
  delete element.dataset.i18n;parameters.delete(element);element.textContent=value;
}
export function uiError(key,params={}){
  const error=new Error(t(key,params));error.i18nKey=key;error.i18nParams=params;return error;
}
function applyLocale(){
  document.documentElement.lang=locale==='zh'?'zh-CN':'en';
  document.querySelectorAll('[data-i18n]').forEach(el=>{el.textContent=t(el.dataset.i18n,parameters.get(el))});
  for(const [data,attr] of [['i18nPlaceholder','placeholder'],['i18nAria','aria-label'],['i18nEmpty','data-empty-label']]){
    document.querySelectorAll(`[data-${data.replace(/[A-Z]/g,c=>'-'+c.toLowerCase())}]`).forEach(el=>el.setAttribute(attr,t(el.dataset[data])));
  }
  for(const lang of ['en','zh'])document.getElementById('ui-'+lang).setAttribute('aria-pressed',String(locale===lang));
  document.dispatchEvent(new Event('uilanguagechange'));
}
for(const lang of ['en','zh'])document.getElementById('ui-'+lang).onclick=()=>{
  locale=lang;try{localStorage.setItem(storageKey,lang)}catch{}applyLocale();
};
applyLocale();
