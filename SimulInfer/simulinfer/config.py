from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import tomllib

RESPONSE_MODES = {'standard': 1600, 'short_chunk': 800}


def response_chunk_ms(value):
    if not isinstance(value,str) or value not in RESPONSE_MODES:
        raise ValueError('response_mode 必须为standard（标准1600ms）或short_chunk（首块必须1600ms，后续800ms）')
    return RESPONSE_MODES[value]


@dataclass(frozen=True)
class Model:
    backend: str = "vllm"
    base_url: str = "http://127.0.0.1:8800/v1"
    model: str = "simulst"
    api_key_env: str = "VLLM_API_KEY"
    chat_protocol: str = "standard"
    checkpoint: str = ""


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8080
    max_sessions: int = 4
    max_upload_mb: int = 100
    max_ws_message_mb: int = 2
    input_queue_frames: int = 32
    timeout_seconds: float = 900
    first_chunk_ms: int = 1600
    chunk_ms: int = 1600
    model_audio_format: str = "pcm16"
    window_seconds: float = 0
    audio_mode: str = "standard"
    ws_audio_mode: str = "noise_resistant"
    token_budget_enabled: bool = False
    max_model_len: int = 8192
    context_reserve_tokens: int = 256
    max_tokens: int = 64
    st_max_tokens: int = 256
    st_wait_chunks: int = 4
    st_context: bool = False
    reset_after_audible_st: bool = False
    reset_after_silence_st: bool = False
    reset_after_empty_st: bool = False
    reset_on_max_tokens: bool = False
    st_fallback_max_seconds: float = 0
    st_trim_trailing_silence: bool = False
    st_crop_speech: bool = True
    simulst_crop_speech: bool = False
    st_segment_pad_ms: float = 240
    repetition_penalty: float = 1.0
    st_repetition_penalty: float = 1.05
    model_logits_processors: bool = True
    no_repeat_ngram_size: int = 3
    st_no_repeat_ngram_size: int = 3
    vad_mode: str = "firered"
    vad_model_path: str = ""
    vad_threshold: float = 0.4
    vad_release_threshold: float = 0.35
    vad_pad_ms: float = 240
    vad_leading_pad_ms: float = 160
    vad_full_chunk_speech_ms: float = 1000
    vad_cpu_threads: int = 1
    silence_rms: float = 0.003
    trailing_silence_seconds: float = 2.4
    skip_initial_silence: bool = True
    vad_preroll_enabled: bool = False
    post_speech_silence_mode: str = "passthrough"
    st_vad_mode: str = "firered_nonstream"
    st_vad_model_path: str = ""
    st_vad_threshold: float = 0.4
    st_short_speech_guard: bool | None = None
    no_chinese_target: bool = False
    tts_backend: str = "none"
    tts_url: str = "http://127.0.0.1:8003"
    tts_model: str = "tts"
    tts_voice: str = ""
    tts_sample_rate: int = 24000
    tts_api_key_env: str = "TTS_API_KEY"
    tts_min_chars: int = 40
    tts_batch_ms: int = 650
    tts_timeout_seconds: float = 30.0
    tts_max_new_tokens: int = 0
    tts_tail_filter: bool = False
    tts_history_mode: str = "auto"
    tts_history_window_s: float = 10.0
    tts_history_max_chars: int = 240
    tts_max_audio_seconds: float = 120.0
    tts_task_type: str = "CustomVoice"
    tts_reference_audio: str = ""
    tts_reference_text: str = ""
    tts_speed: float = 1.0
    tts_speed_zh_en: float = 1.2
    tts_clone_source: bool | None = None
    tts_clone_first_seconds: float = 2.4
    tts_clone_interval_seconds: float = 10.0
    models: dict[str, Model] = field(default_factory=lambda: {"target_only": Model()})
    model_endpoints: tuple[Model, ...] = ()
    default_model_type: str = ""
    default_vad: bool = True
    default_tts: bool = False
    wait_fallback_seconds: float | None = None

    @property
    def clone_source_enabled(self):
        return (self.tts_backend=='vllm_omni' and self.tts_task_type=='Base' and not self.tts_reference_audio
                if self.tts_clone_source is None else self.tts_clone_source)

    @property
    def response_mode(self):
        return 'short_chunk' if self.chunk_ms==800 else 'standard'

    @property
    def st_guard_enabled(self):
        return (self.audio_mode=='noise_resistant' if self.st_short_speech_guard is None
                else self.st_short_speech_guard)

    @property
    def simulst_crop_enabled(self):
        return self.audio_mode=='noise_resistant' and self.simulst_crop_speech

    @property
    def st_crop_enabled(self):
        return self.st_crop_speech and self.vad_mode in {'silero','firered'}

    def __post_init__(self):
        if self.ws_audio_mode not in {'standard','noise_resistant'}:
            raise ValueError('ws_audio_mode 必须为standard或noise_resistant')
        if self.audio_mode not in {'standard','noise_resistant'}:
            raise ValueError('audio_mode 必须为standard或noise_resistant')
        if self.audio_mode=='noise_resistant':
            if self.vad_mode not in {'silero','firered'} or not self.default_vad:
                raise ValueError('抗噪模式需要启用Silero或FireRed人声检测')
            if self.post_speech_silence_mode!='passthrough' or self.vad_preroll_enabled:
                raise ValueError('抗噪模式不可同时启用zero_fill或preroll')
            if any(m.backend=='service' for m in (*self.models.values(),*self.model_endpoints)):
                raise ValueError('抗噪模式需要vllm后端，本地无法裁剪上游service的模型输入')
            if not self.st_crop_speech:raise ValueError('抗噪模式需要开启ST人声裁剪')
        if self.st_short_speech_guard is not None and type(self.st_short_speech_guard) is not bool:
            raise ValueError('st_short_speech_guard 必须为true/false，省略时跟随音频模式')
        if not math.isfinite(self.st_segment_pad_ms) or not 0<=self.st_segment_pad_ms<=1000:
            raise ValueError('st_segment_pad_ms 必须为0～1000毫秒')
        for key in ('token_budget_enabled','reset_on_max_tokens','vad_preroll_enabled',
                    'model_logits_processors','no_chinese_target','skip_initial_silence',
                    'st_context','st_trim_trailing_silence','st_crop_speech','simulst_crop_speech',
                    'reset_after_audible_st','reset_after_silence_st','reset_after_empty_st',
                    'default_vad','default_tts'):
            if type(getattr(self,key)) is not bool:raise ValueError(f'{key} 必须为true/false')
        if (not self.models and not self.model_endpoints) or set(self.models) - {"target_only", "source_target"}:
            raise ValueError("models 必须配置 target_only 和/或 source_target")
        if self.models and self.model_endpoints:
            raise ValueError('自动模型发现与显式models配置不能混用')
        if self.default_model_type and self.default_model_type not in self.models and not self.model_endpoints:
            raise ValueError('defaults.model_type 必须指向已配置的模型')
        if self.default_vad and self.vad_mode=='off':
            raise ValueError('defaults.vad=true 需要启用VAD；设置vad_mode或defaults.vad=false')
        if self.default_tts and self.tts_backend=='none':
            raise ValueError('defaults.tts=true 需要配置TTS后端')
        if self.wait_fallback_seconds is not None and (
                not math.isfinite(self.wait_fallback_seconds) or self.wait_fallback_seconds<0):
            raise ValueError('wait_fallback_seconds 必须为非负秒数')
        for m in (*self.models.values(),*self.model_endpoints):
            if m.backend not in {"vllm", "service", "mock"}:
                raise ValueError("backend 必须为 vllm、service 或 mock")
            if m.backend != "mock" and not m.base_url.startswith(("http://", "https://")):
                raise ValueError("base_url 必须为 http/https")
            if m.chat_protocol not in {'standard', 'harness'}:
                raise ValueError('chat_protocol 必须为 standard 或 SimulInfer')
        if any(m.backend!='vllm' for m in self.model_endpoints):
            raise ValueError('自动模型发现使用vllm后端')
        if (any(m.backend=='vllm' for m in (*self.models.values(),*self.model_endpoints))
                and not self.model_logits_processors
                and (self.no_repeat_ngram_size or self.st_no_repeat_ngram_size or self.no_chinese_target)):
            raise ValueError('严格n-gram/目标语言过滤需要模型侧处理器；配置 model_logits_processors=true '
                             '并按扩展部署说明加载处理器。原生部署使用0/false；repetition_penalty仍可用。')
        if self.chunk_ms not in (800, 1600) or self.first_chunk_ms != 1600:
            raise ValueError("首块固定1600ms，后续块支持800/1600ms")
        if self.model_audio_format not in {"pcm16", "float32"}:
            raise ValueError("model_audio_format 必须为 pcm16 或 float32")
        if self.window_seconds != 0 and not 1.6 <= self.window_seconds <= 120:
            raise ValueError("window_seconds 为0关闭定时切窗，或在1.6～120之间")
        if type(self.max_model_len) is not int or self.max_model_len < 512:
            raise ValueError("max_model_len 需为不小于512的整数")
        if (type(self.context_reserve_tokens) is not int or
                not self.max_tokens + 32 <= self.context_reserve_tokens < self.max_model_len // 2):
            raise ValueError("context_reserve_tokens 需至少为max_tokens+32，且小于上下文上限的一半")
        if self.max_tokens >= self.max_model_len or self.st_max_tokens >= self.max_model_len:
            raise ValueError("输出token预算必须小于上下文上限")
        if (not math.isfinite(self.st_fallback_max_seconds) or
                (self.st_fallback_max_seconds != 0 and not 1.6 <= self.st_fallback_max_seconds <= 30)):
            raise ValueError("st_fallback_max_seconds 为0不裁剪，或在1.6～30之间")
        if self.st_guard_enabled and self.st_vad_mode=='off' and not self.st_crop_enabled:
            raise ValueError('st_short_speech_guard 需要人声裁剪记录或ST二次VAD以统计人声时长')
        for value in (self.no_repeat_ngram_size, self.st_no_repeat_ngram_size):
            if type(value) is not int or not 0 <= value <= 32:
                raise ValueError("no_repeat_ngram_size 需为0～32整数，0表示关闭")
        for value in (self.repetition_penalty, self.st_repetition_penalty):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("重复惩罚需为有限正数")
        if not 1 <= self.max_sessions <= 128:
            raise ValueError("max_sessions 需在1～128之间")
        if self.max_tokens < 1 or self.st_max_tokens < 1 or self.st_wait_chunks < 0:
            raise ValueError("输出token上限需为正，st_wait_chunks需非负")
        if self.timeout_seconds <= 0 or self.max_upload_mb < 1 or self.input_queue_frames < 1:
            raise ValueError("超时、上传大小和队列大小需为正")
        if self.vad_mode not in {"energy", "silero", "firered", "off"}:
            raise ValueError("vad_mode 必须为 energy、silero、firered 或 off")
        if not 0 < self.vad_release_threshold <= self.vad_threshold < 1:
            raise ValueError("VAD 阈值需满足 0 < vad_release_threshold <= vad_threshold < 1")
        if not math.isfinite(self.vad_pad_ms) or not 0 <= self.vad_pad_ms <= 1000:
            raise ValueError("vad_pad_ms 需在0～1000之间")
        if not math.isfinite(self.vad_leading_pad_ms) or not 0 <= self.vad_leading_pad_ms <= 1000:
            raise ValueError("vad_leading_pad_ms 需在0～1000之间")
        if not math.isfinite(self.vad_full_chunk_speech_ms) or self.vad_full_chunk_speech_ms < 0:
            raise ValueError("vad_full_chunk_speech_ms 需为有限非负数")
        if type(self.vad_cpu_threads) is not int or self.vad_cpu_threads < 1:
            raise ValueError("vad_cpu_threads 需为正整数")
        if not isinstance(self.vad_model_path, str):
            raise ValueError("vad_model_path 必须为字符串")
        if self.post_speech_silence_mode not in {"passthrough", "zero_fill"}:
            raise ValueError("post_speech_silence_mode 必须为 passthrough 或 zero_fill")
        if self.st_vad_mode not in {"off", "firered_nonstream"}:
            raise ValueError("st_vad_mode 必须为 off 或 firered_nonstream")
        if not 0 <= self.st_vad_threshold <= 1:
            raise ValueError("st_vad_threshold 需在0～1之间")
        if not all(math.isfinite(v) and v >= 0 for v in (self.silence_rms, self.trailing_silence_seconds)):
            raise ValueError("静音阈值不得为负")
        if self.tts_backend not in {"none", "session", "openai", "vllm_omni", "mock"}:
            raise ValueError("tts_backend 必须为 none/session/openai/vllm_omni/mock")
        if self.tts_min_chars < 1 or not 8000 <= self.tts_sample_rate <= 96000:
            raise ValueError("TTS 参数无效")
        if type(self.tts_tail_filter) is not bool:
            raise ValueError('tts_tail_filter must be true/false')
        if type(self.tts_batch_ms) is not int or not 0 <= self.tts_batch_ms <= 2000:
            raise ValueError('tts_batch_ms must be between 0 and 2000')
        if not math.isfinite(self.tts_timeout_seconds) or not 1 <= self.tts_timeout_seconds <= 300:
            raise ValueError('tts_timeout_seconds must be between 1 and 300')
        if type(self.tts_max_new_tokens) is not int or not 0 <= self.tts_max_new_tokens <= 4096:
            raise ValueError('tts_max_new_tokens must be between 0 and 4096')
        if self.tts_history_mode not in {'auto', 'none', 'paired'}:
            raise ValueError('tts_history_mode 必须为 auto/none/paired')
        if (not math.isfinite(self.tts_history_window_s) or not 1 <= self.tts_history_window_s <= 20
                or self.tts_history_max_chars < 1):
            raise ValueError('TTS 历史窗口需为1～20秒，字符上限需为正')
        if not math.isfinite(self.tts_max_audio_seconds) or not 1 <= self.tts_max_audio_seconds <= 300:
            raise ValueError('tts_max_audio_seconds 需为1～300秒')
        for key in ('tts_speed','tts_speed_zh_en'):
            value=getattr(self,key)
            if type(value) not in (int,float) or not math.isfinite(value) or not .5 <= value <= 2:
                raise ValueError(f'{key} 需为0.5～2')
        if self.tts_backend not in {'none', 'mock'} and not self.tts_url.startswith(('http://', 'https://')):
            raise ValueError('tts_url 必须为 http/https 地址')
        if self.tts_backend == 'openai' and self.tts_history_mode == 'paired':
            raise ValueError('普通 OpenAI speech API 不承诺历史音频支持；请用 session 或 vllm_omni Base')
        if self.tts_clone_source is not None and type(self.tts_clone_source) is not bool:
            raise ValueError('tts_clone_source must be true/false')
        if (not math.isfinite(self.tts_clone_first_seconds) or not math.isfinite(self.tts_clone_interval_seconds)
                or not .4 <= self.tts_clone_first_seconds <= self.tts_clone_interval_seconds <= 20):
            raise ValueError('TTS clone reference durations must satisfy 0.4 <= first <= interval <= 20')
        if self.clone_source_enabled and not (self.tts_backend=='vllm_omni' and self.tts_task_type=='Base'):
            raise ValueError('Input voice cloning requires vllm_omni Base')
        if self.tts_backend == 'vllm_omni':
            if self.tts_task_type not in {'Base', 'CustomVoice'}:
                raise ValueError('vllm_omni 支持 Base 或 CustomVoice')
            if self.tts_history_mode == 'paired' and self.tts_task_type != 'Base':
                raise ValueError('成对历史音频只支持 Qwen3-TTS Base，CustomVoice 不接受参考音频')
            if self.tts_task_type == 'Base' and not self.clone_source_enabled and (not self.tts_reference_audio or not self.tts_reference_text.strip()):
                raise ValueError('Base 需要 tts_reference_audio 与准确的 tts_reference_text')
            if self.tts_task_type == 'CustomVoice' and (self.tts_reference_audio or self.tts_reference_text):
                raise ValueError('CustomVoice 不支持参考音频，请使用 Base')


def load(path: str | None = None, *, mock=False) -> Settings:
    if mock:
        return Settings(models={k: Model(backend="mock") for k in ("target_only", "source_target")},
                        tts_backend="mock")
    path = path or os.getenv("SIMULINFER_CONFIG")
    if not path:
        return Settings()
    data = tomllib.loads(Path(path).read_text())
    allowed = {'server', 'defaults', 'inference', 'vad', 'tts', 'models', 'model'}
    if set(data)-allowed:
        raise ValueError('未知配置段: '+', '.join(sorted(set(data)-allowed)))
    fields = {
        'server': {'host','port','max_sessions','max_upload_mb','max_ws_message_mb',
                   'input_queue_frames','timeout_seconds'},
        'tts': {k for k in Settings.__dataclass_fields__ if k.startswith('tts_')},
    }
    fields['inference'] = set(Settings.__dataclass_fields__)-fields['server']-fields['tts']-{
        'models','model_endpoints','default_model_type','default_vad','default_tts'}
    fields['defaults']={'model_type','response_mode','vad','tts'}
    fields['vad']={'enabled','mode','stream_model_path','nonstream_model_path','threshold',
        'release_threshold','st_threshold','leading_pad_ms','trailing_pad_ms',
        'full_chunk_speech_ms','zero_fill_silence','short_speech_guard','cpu_threads'}
    tts_aliases={'backend':'tts_backend','base_url':'tts_url','url':'tts_url',
        'model':'tts_model','voice':'tts_voice','sample_rate':'tts_sample_rate',
        'min_chars':'tts_min_chars','api_key_env':'tts_api_key_env'}
    fields['tts']=fields['tts']|set(tts_aliases)|{'enabled'}
    for section, keys in fields.items():
        if not isinstance(data.get(section,{}),dict):
            raise ValueError(f'[{section}] 必须为配置表')
        unknown=set(data.get(section,{}))-keys
        if unknown:raise ValueError(f'[{section}] 未知或放错位置的参数: '+', '.join(sorted(unknown)))
    options = {**data.get("server", {}), **data.get("inference", {})}
    tts=data.get('tts',{})
    for key,value in tts.items():
        if key=='enabled':continue
        target=tts_aliases.get(key,key)
        if target in options:raise ValueError(f'TTS参数重复: {target}')
        options[target]=value
    if 'enabled' in tts:
        if type(tts['enabled']) is not bool:raise ValueError('tts.enabled必须为true/false')
        if not tts['enabled']:options['tts_backend']='none'
        elif 'tts_backend' not in options:options['tts_backend']='openai'
    vad=data.get('vad',{})
    if vad:
        if 'enabled' in vad and type(vad['enabled']) is not bool:raise ValueError('vad.enabled必须为true/false')
        mode=vad.get('mode','firered')
        if vad.get('enabled',True) is False:mode='off'
        def set_vad(key,value):
            if key in options:raise ValueError(f'VAD参数重复: {key}')
            options[key]=value
        set_vad('vad_mode',mode)
        aliases={'stream_model_path':'vad_model_path','nonstream_model_path':'st_vad_model_path',
            'threshold':'vad_threshold','release_threshold':'vad_release_threshold',
            'st_threshold':'st_vad_threshold','leading_pad_ms':'vad_leading_pad_ms',
            'trailing_pad_ms':'vad_pad_ms','full_chunk_speech_ms':'vad_full_chunk_speech_ms',
            'cpu_threads':'vad_cpu_threads','short_speech_guard':'st_short_speech_guard'}
        for key,value in vad.items():
            if key in aliases:set_vad(aliases[key],value)
        if 'zero_fill_silence' in vad:
            if type(vad['zero_fill_silence']) is not bool:raise ValueError('zero_fill_silence必须为true/false')
            set_vad('post_speech_silence_mode','zero_fill' if vad['zero_fill_silence'] else 'passthrough')
        if mode!='off' and (vad.get('nonstream_model_path') or vad.get('short_speech_guard')):
            options.setdefault('st_vad_mode','firered_nonstream')
    defaults=data.get('defaults',{})
    if 'response_mode' in defaults:
        chunk_ms=response_chunk_ms(defaults['response_mode'])
        if 'chunk_ms' in options and options['chunk_ms']!=chunk_ms:
            raise ValueError('defaults.response_mode与inference.chunk_ms不一致，请保留一处设置')
        options['chunk_ms']=chunk_ms
    options.update(default_model_type=defaults.get('model_type',''),
                   default_vad=defaults.get('vad',options.get('vad_mode','firered')!='off'),
                   default_tts=defaults.get('tts',False))
    if options.get("vad_model_path"):
        model_path = Path(options["vad_model_path"]).expanduser()
        if not model_path.is_absolute():
            model_path = Path(path).resolve().parent / model_path
        options["vad_model_path"] = str(model_path.resolve())
    if options.get("st_vad_model_path"):
        model_path = Path(options["st_vad_model_path"]).expanduser()
        if not model_path.is_absolute():
            model_path = Path(path).resolve().parent / model_path
        options["st_vad_model_path"] = str(model_path.resolve())
    if options.get('tts_reference_audio'):
        reference = Path(options['tts_reference_audio']).expanduser()
        if not reference.is_absolute():
            reference = Path(path).resolve().parent / reference
        options['tts_reference_audio'] = str(reference.resolve())
    def parse_model(values, *, discover=False):
        if not isinstance(values,dict):raise ValueError('模型服务必须为配置表')
        values=dict(values)
        if 'name' in values:
            if 'model' in values:raise ValueError('name和model不能同时设置')
            values['model']=values.pop('name')
        if discover:values.setdefault('model','')
        if values.get('checkpoint'):
            cp=Path(values['checkpoint']).expanduser()
            relative=Path(path).resolve().parent/cp
            if cp.is_absolute() or str(cp).startswith('.') or relative.is_dir():
                values['checkpoint']=str(relative.resolve())
        return Model(**values)
    model_data=data.get('models',{})
    models={}
    if 'model' in data:
        if model_data:raise ValueError('[model]不能与models同时配置')
        options['model_endpoints']=(parse_model(data['model'],discover=True),)
    elif isinstance(model_data,list):
        options['model_endpoints']=tuple(parse_model(v,discover=True) for v in model_data)
    elif isinstance(model_data,dict):
        models={key:parse_model(values) for key,values in model_data.items()}
    else:raise ValueError('models必须为配置表或服务列表')
    options['models']=models
    return Settings(**options)
