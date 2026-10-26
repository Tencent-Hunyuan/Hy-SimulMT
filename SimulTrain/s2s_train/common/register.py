"""通用注册逻辑，由各变体传入自己的模型标识与模板参数。

为什么必须注册 composite model（已验证的源码行为）：
- visual.py:177  未注册 -> get_forbidden_modules 返回空集，
                 freeze_vision_tower / freeze_language_model 被静默忽略
- visual.py:212  未注册 -> patch_target_modules 原样返回，不做过滤
- misc.py:49lora_target: all 按叶子名收集，声学编码器与多数 LLM
                 共用 q_proj/k_proj/v_proj/out_proj 等名字

不注册的后果：LoRA 会被注入到本应冻结的声学编码器中，且全参模式会
连同声学编码器一起训练，两者都不会报错。
"""

from transformers import AutoConfig, AutoModelForCausalLM

_REGISTERED: set[str] = set()


def register_variant(
    model_type: str,
    config_class,
    template_name: str,
    audio_token: str,
    num_speech_tokens: int,
    template_kwargs: dict,
    speech_tokens_for=None,
) -> None:
    """注册一个变体的 config/model/composite/plugin/template。

    speech_tokens_for：变长 Adapter 时传 config.speech_tokens_for，
    让 mm_plugin 按每条音频的时长展开占位符。定长（Q-Former）传 None。

    同一进程内重复调用同一 model_type 会被跳过。
    """
    if model_type in _REGISTERED:
        return

    from .modeling_speech_llm import SpeechLLMForCausalLM

    # Transformers 要求 model.config_class 与注册的 config 类完全一致，
    # 所以为每个变体动态生成一个绑定了自身 config 的模型子类。
    model_class = type(
        f"SpeechLLMForCausalLM_{model_type}",
        (SpeechLLMForCausalLM,),
        {"config_class": config_class},
    )

    AutoConfig.register(model_type, config_class)
    AutoModelForCausalLM.register(config_class, model_class)

    from .audio_processing import register_audio_processing

    register_audio_processing(config_class)

    _register_composite(model_type)
    _register_plugin_and_template(
        model_type,
        template_name,
        audio_token,
        num_speech_tokens,
        template_kwargs,
        speech_tokens_for=speech_tokens_for,
    )
    _REGISTERED.add(model_type)


def _register_composite(model_type: str) -> None:
    from llamafactory.model.model_utils.visual import COMPOSITE_MODELS, _register_composite_model

    if model_type in COMPOSITE_MODELS:
        return

    _register_composite_model(
        model_type=model_type,
        projector_keys=["speech_adapter"],
        # LlamaFactory 把声学编码器归到 vision_model_keys（它没有 audio 这一类），
        # 冻结逻辑复用 freeze_vision_tower。这个键名必须与模型的属性名一致，
        # 否则 freeze 找不到模块、日志里也不会打 not trainable。
        vision_model_keys=["speech_encoder"],
        language_model_keys=["language_model"],
        # 兜底：即使前缀匹配失效，也不允许 LoRA 落到语音侧
        lora_conflict_keys=["speech_encoder", "speech_adapter"],
    )


def _register_plugin_and_template(
    model_type: str,
    template_name: str,
    audio_token: str,
    num_speech_tokens: int,
    template_kwargs: dict,
    speech_tokens_for=None,
) -> None:
    from llamafactory.data.mm_plugin import PLUGINS, get_mm_plugin, register_mm_plugin
    from llamafactory.data.template import TEMPLATES, register_template

    from .mm_plugin_speech import SpeechAudioPlugin

    if model_type not in PLUGINS:
        register_mm_plugin(model_type, SpeechAudioPlugin)

    TEMPLATES.pop(template_name, None)
    register_template(
        name=template_name,
        mm_plugin=get_mm_plugin(
            name=model_type,
            audio_token=audio_token,
            num_speech_tokens=num_speech_tokens,
            # 变长模式传 config.speech_tokens_for，plugin 会按每条音频时长
            # 展开占位符。传 None 则退回 num_speech_tokens 的定长行为。
            speech_tokens_for=speech_tokens_for,
        ),
        **template_kwargs,
    )
