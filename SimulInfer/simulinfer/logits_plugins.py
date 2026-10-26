"""Resolve automatic n-gram plugins for the selected vLLM runner on demand."""
from importlib import import_module


def __getattr__(name):
    if name not in {'SimulstNoRepeatNgramProcessor', 'STNoRepeatNgramProcessor', 'NoChineseTargetProcessor'}:
        raise AttributeError(name)
    from .launch import logits_api
    module = '.logits_v2' if logits_api() == 'v2' else ('.target_language' if name == 'NoChineseTargetProcessor' else '.repetition')
    return getattr(import_module(module, __package__), name)
