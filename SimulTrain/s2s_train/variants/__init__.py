"""变体注册表：新增模型尺寸时在此登记。"""

from importlib import import_module

# 变体名 -> 模块路径
VARIANTS = {
    "hy_v3_speech_merged": "s2s_train.variants.hy_v3_speech_merged",
}


def load_variant(name: str):
    if name not in VARIANTS:
        raise SystemExit(f"未知变体 `{name}`，可选：{sorted(VARIANTS)}")

    return import_module(VARIANTS[name])
