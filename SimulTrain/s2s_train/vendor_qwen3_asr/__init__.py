"""Qwen3-ASR 的建模代码（vendored，原样搬运）。

为什么要 vendor 而不是 pip install qwen-asr：
1. transformers 4.57.6 不认识 model_type='qwen3_asr'（CONFIG_MAPPING 里只有
   qwen3_omni_moe），而 Qwen3-ASR-1.7B 的模型目录下也没有 modeling_*.py，
   所以 trust_remote_code 同样救不了 —— 必须自带代码。
2. qwen_asr 包的 __init__.py 会连带 import nagisa / vllm 等推理工具链依赖，
   而我们只需要 audio_tower 这一个编码器，装整个包会污染环境。

来源：PyPI 包 qwen-asr==0.0.6 中的
      qwen_asr/core/transformers_backend/{configuration,modeling,processing}_qwen3_asr.py
      （对应 https://github.com/QwenLM/Qwen3-ASR）
文件内容未作修改（相对 import 本来就是 `from .configuration_qwen3_asr`，
放进本包后依然成立）。升级时直接覆盖这三个文件即可。

本项目只用 Qwen3ASRAudioEncoder（即 audio_tower），不用它的文本 decoder ——
文本侧由 HY-MT2 承担。
"""

from .configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
from .modeling_qwen3_asr import Qwen3ASRAudioEncoder

__all__ = ["Qwen3ASRAudioEncoderConfig", "Qwen3ASRAudioEncoder"]
