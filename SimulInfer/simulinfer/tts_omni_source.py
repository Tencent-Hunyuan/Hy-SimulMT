"""Optional source-voice conditioning for a Qwen vLLM-Omni Speech API process."""
import asyncio
import base64
from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import threading

KEY='simulinfer_source_audio'
VERSION='source-voice-history-v1'


class SpeakerEncoder:
    def __init__(self, model_path):
        import torch
        from safetensors import safe_open
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSSpeakerEncoderConfig
        from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSSpeakerEncoder
        from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import mel_spectrogram
        root=Path(model_path)
        if not root.is_dir():
            from huggingface_hub import snapshot_download
            root=Path(snapshot_download(model_path,allow_patterns=['config.json','*.safetensors','*.safetensors.index.json']))
        config=json.loads((root/'config.json').read_text())
        self.encoder=Qwen3TTSSpeakerEncoder(Qwen3TTSSpeakerEncoderConfig(**config['speaker_encoder_config'])).eval()
        weights={}
        for path in root.glob('*.safetensors'):
            with safe_open(path,framework='pt',device='cpu') as f:
                for key in f.keys():
                    if key.startswith('speaker_encoder.'):
                        weights[key.removeprefix('speaker_encoder.')]=f.get_tensor(key).float()
        self.encoder.load_state_dict(weights,strict=True)
        self.mel_spectrogram=mel_spectrogram
        self.cache=OrderedDict();self.lock=threading.Lock()

    def encode(self, audio):
        import torch
        import soundfile as sf
        import numpy as np
        import soxr
        key=hashlib.sha256(audio.encode()).hexdigest()
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key);return self.cache[key],key
            if not audio.startswith('data:audio/wav;base64,') or len(audio)>2_000_000:
                raise ValueError('Source voice reference must be an inline WAV under 2MB')
            raw=base64.b64decode(audio.split(',',1)[1],validate=True)
            with sf.SoundFile(io.BytesIO(raw)) as f:
                if not 8000<=f.samplerate<=96000 or f.channels!=1 or not .4<=len(f)/f.samplerate<=20:
                    raise ValueError('Source voice reference must be 0.4–20s mono audio')
                rate=f.samplerate;samples=f.read(dtype='float32')
            if not np.isfinite(samples).all() or np.max(np.abs(samples))<1e-4:
                raise ValueError('Source voice reference is silent or invalid')
            if rate!=24000:samples=soxr.resample(samples,rate,24000).astype(np.float32)
            with torch.inference_mode():
                mel=self.mel_spectrogram(torch.from_numpy(samples).unsqueeze(0),n_fft=1024,num_mels=128,
                    sampling_rate=24000,hop_size=256,win_size=1024,fmin=0,fmax=12000).transpose(1,2)
                embedding=self.encoder(mel)[0].tolist()
            if not np.isfinite(embedding).all():raise ValueError('Invalid speaker embedding')
            self.cache[key]=embedding
            while len(self.cache)>32:self.cache.popitem(last=False)
            return embedding,key


def install(model_path):
    from vllm_omni.entrypoints.openai import api_server
    from vllm_omni.entrypoints.openai.tts_adapters import TTS_ADAPTER_REGISTRY
    from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
    if getattr(api_server,'_simulinfer_source_voice',False):return
    if not callable(getattr(Qwen3TTSAdapter,'build',None)):
        raise RuntimeError('This vLLM-Omni does not support the Qwen TTS adapter API')
    encoder=SpeakerEncoder(model_path)

    class SourceVoiceAdapter(Qwen3TTSAdapter):
        def validate(self, request):
            source=(request.extra_params or {}).get(KEY)
            if source is None:source=getattr(request,'_simulinfer_source_audio',None)
            if source is not None:
                if request.task_type!='Base':return 'Source voice cloning requires Qwen Base'
                if not isinstance(source,str) or not source.startswith('data:audio/wav;base64,') or len(source)>2_000_000:
                    return 'Source voice cloning requires an inline WAV under 2MB'
            return super().validate(request)

        async def build(self, request, sampling_params_list, has_inline_ref_audio):
            source=(request.extra_params or {}).get(KEY)
            if source is None:source=getattr(request,'_simulinfer_source_audio',None)
            clean=request
            if source is not None:
                # Do not propagate source WAVs into sampling logs or parameters.
                extra={k:v for k,v in (request.extra_params or {}).items() if k!=KEY}
                object.__setattr__(request,'_simulinfer_source_audio',source)
                clean=request.model_copy(update={'extra_params':extra or None})
                request.extra_params=extra or None
            prepared=await super().build(clean,sampling_params_list,has_inline_ref_audio)
            if source is not None:
                embedding,digest=await asyncio.to_thread(encoder.encode,source)
                # Keep the source-conditioned artifacts separate from the
                # generic cache for the same generated-reference waveform.
                waveform,sr,_=await self.ctx.server._resolve_ref_audio(clean.ref_audio)
                prepared.tts_params['ref_audio']=[[waveform,sr]]
                prepared.warmup_artifact_key=None
                prepared.tts_params['voice_clone_prompt']=[{'ref_spk_embedding':embedding}]
                prepared.tts_params['_simulinfer_source_voice']=True
                prepared.prompt['additional_information']=prepared.tts_params
                salt=prepared.prompt.get('cache_salt','')
                prepared.prompt['cache_salt']=hashlib.sha256((salt+':'+digest).encode()).hexdigest()
            return prepared

    TTS_ADAPTER_REGISTRY['qwen3_tts']=SourceVoiceAdapter
    @api_server.router.get('/v1/simulinfer/tts_capabilities')
    async def capabilities():
        return {'source_voice_history':VERSION,'source_audio_key':KEY}
    api_server._simulinfer_source_voice=True


def main():
    import os,sys
    model=os.environ.get('SIMULINFER_TTS_MODEL','')
    if not model:raise SystemExit('SIMULINFER_TTS_MODEL must identify the Qwen Base model')
    from .tts_omni_plugin import register
    register()
    install(model)
    sys.argv[0]='vllm-omni'
    from vllm_omni.entrypoints.cli.main import main as omni_main
    omni_main()


if __name__=='__main__':main()
