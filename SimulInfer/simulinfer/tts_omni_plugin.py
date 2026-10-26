"""Model-process hook used only by the source-cloning TTS launcher."""
import os
from contextvars import ContextVar
from functools import wraps

_source_conditioning=ContextVar('simulinfer_source_conditioning',default=False)


def register():
    if not os.environ.get('SIMULINFER_TTS_MODEL'):return
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
    cls=Qwen3TTSPromptEmbedsBuilder
    if getattr(cls,'_simulinfer_source_cache',False):return
    build=cls.build_prompt_embeds
    put=cls.put_ref_audio_artifacts

    @wraps(build)
    def build_with_source(self,*,task_type,info_dict):
        token=_source_conditioning.set(bool(info_dict.get('_simulinfer_source_voice')))
        try:return build(self,task_type=task_type,info_dict=info_dict)
        finally:_source_conditioning.reset(token)

    @wraps(put)
    def cache_reference(self,cache_key,*,ref_code=None,ref_spk_embedding=None):
        # The reference WAV still identifies its codec content. A separately
        # supplied source speaker must never replace that WAV's own speaker.
        if _source_conditioning.get():ref_spk_embedding=None
        return put(self,cache_key,ref_code=ref_code,ref_spk_embedding=ref_spk_embedding)

    cls.build_prompt_embeds=build_with_source
    cls.put_ref_audio_artifacts=cache_reference
    cls._simulinfer_source_cache=True
