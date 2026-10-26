"""Validate the configured protocol and optional VAD before accepting traffic."""
import asyncio

from .translation import ChatAPI
from .vad import create_activity, FireRedNonstreamActivity


async def check(settings):
    results=[]
    for mode,model in settings.models.items():
        if model.backend!='vllm':
            results.append({'mode':mode,'backend':model.backend,'checked':False})
            continue
        api=ChatAPI(model,settings,mode,'zh-en')
        try:
            result=await api.check()
            result['no_repeat_processors']=await api.check_repetition_processors()
            results.append(result)
        finally:await api.close()
    if any(m.backend=='vllm' for m in settings.models.values()):
        await asyncio.to_thread(create_activity,settings)
        if settings.st_vad_mode=='firered_nonstream':
            await asyncio.to_thread(FireRedNonstreamActivity,settings.st_vad_model_path,
                                    settings.st_vad_threshold,cpu_threads=settings.vad_cpu_threads)
    if settings.tts_tail_filter:
        from .tts_tail import SpeechTail
        tail = SpeechTail(settings.tts_sample_rate,None)
        await tail.start()
    if settings.clone_source_enabled:
        from .tts import check_source_voice_service
        await check_source_voice_service(settings)
    return {'models':results,'vad_mode':settings.vad_mode,'st_vad_mode':settings.st_vad_mode,
            'model_logits_processors_declared':settings.model_logits_processors}
