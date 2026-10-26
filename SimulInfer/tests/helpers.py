"""A deterministic HTTP model stub with real audio lengths and token usage."""
import base64
import io
import json
import re

import httpx
import soundfile as sf

from simulinfer.budget import render
from simulinfer.config import Settings as ProductionSettings, Model as ProductionModel


def Model(**kwargs):
    return ProductionModel(**{'chat_protocol':'harness',**kwargs})


def Settings(**kwargs):
    # Protocol tests need no torch/model download; Silero has separate tests.
    return ProductionSettings(**{'vad_mode':'energy', 'default_vad':kwargs.get('vad_mode')!='off', 'no_chinese_target':False,
                                 'model_audio_format':'float32',
                                 'model_logits_processors':True,
                                 'no_repeat_ngram_size':3,'st_no_repeat_ngram_size':3,
                                 'post_speech_silence_mode':'zero_fill',
                                 'token_budget_enabled':True,'window_seconds':0,
                                 'reset_after_audible_st':True,'reset_after_silence_st':True,
                                 'reset_after_empty_st':True,'reset_on_max_tokens':True,
                                 'st_fallback_max_seconds':30,'repetition_penalty':1.05,
                                 'vad_preroll_enabled':True,'vad_pad_ms':240,
                                 'st_crop_speech':False,'st_vad_mode':'off',
                                 'st_short_speech_guard':False, **kwargs})


def tokens(text):
    parts = re.findall(r'<｜.*?｜>|[\s\S]', text)
    return [sum(map(ord, p)) % 100000 for p in parts]


def model_transport(handler, *, limit=8192):
    def respond(request):
        body = json.loads(request.content)
        if request.url.path.endswith('/tokenize'):
            ids = tokens(body['prompt'])
            return httpx.Response(200, json={'count':len(ids), 'tokens':ids, 'max_model_len':limit})
        response = handler(request)
        if response.is_error:
            return response
        data = json.loads(response.content)
        if 'usage' not in data:
            class Part(dict): pass
            for msg in body.get('messages', []):
                if msg['role'] != 'user': continue
                for i,p in enumerate(msg['content']):
                    if p['type'] == 'input_audio':
                        a=Part(p)
                        with sf.SoundFile(io.BytesIO(base64.b64decode(p['input_audio']['data']))) as f:
                            a.num_samples=len(f)
                        msg['content'][i]=a
            kwargs=body.get('chat_template_kwargs', {})
            text, expansion, _ = render(body.get('messages', []),kwargs.get('output_mode','target_only'),kwargs.get('harness_prefix'))
            choice=data['choices'][0]
            data['usage']={'prompt_tokens':len(tokens(text))+expansion,
                           'completion_tokens':len(tokens(choice.get('message',{}).get('content','')))+1}
        return httpx.Response(response.status_code,json=data)
    return httpx.MockTransport(respond)
