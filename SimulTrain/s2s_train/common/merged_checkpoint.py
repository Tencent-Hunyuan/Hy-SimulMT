"""Utilities for the exported ``HYV3SpeechForCausalLM`` checkpoint.

The exported directory contains a normal HYV3 language model plus
``audio_tower.speech_encoder`` and ``audio_tower.speech_adapter`` tensors.
This module never writes to that directory.  It creates a lightweight LLM
view under the training output directory and loads the two audio components
with exact key validation.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open


LLM_PREFIXES = ("model.", "lm_head.")
ENCODER_PREFIX = "audio_tower.speech_encoder."
ADAPTER_PREFIX = "audio_tower.speech_adapter."
EXPECTED_ARCHITECTURE = "HYV3SpeechForCausalLM"


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"missing checkpoint file: {path}") from exc


def inspect_merged_checkpoint(model_dir: str | Path) -> dict:
    """Validate the merged export and return its index/config metadata."""
    root = Path(model_dir).resolve()
    config = _read_json(root / "config.json")
    index = _read_json(root / "model.safetensors.index.json")
    architectures = config.get("architectures") or []
    if EXPECTED_ARCHITECTURE not in architectures:
        raise RuntimeError(
            f"{root} is not a {EXPECTED_ARCHITECTURE} export: architectures={architectures!r}"
        )
    if config.get("model_type") != "hy_v3":
        raise RuntimeError(f"expected model_type='hy_v3', got {config.get('model_type')!r}")
    audio_config = config.get("audio_config")
    if not isinstance(audio_config, dict):
        raise RuntimeError(f"{root}/config.json does not contain audio_config")

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"invalid or empty weight_map: {root}/model.safetensors.index.json")

    groups = {
        "llm": [key for key in weight_map if key.startswith(LLM_PREFIXES)],
        "encoder": [key for key in weight_map if key.startswith(ENCODER_PREFIX)],
        "adapter": [key for key in weight_map if key.startswith(ADAPTER_PREFIX)],
    }
    recognized = set(groups["llm"] + groups["encoder"] + groups["adapter"])
    unknown = sorted(set(weight_map) - recognized)
    if unknown:
        raise RuntimeError(f"unrecognized merged-checkpoint keys: {unknown[:8]}")
    if not groups["llm"] or not groups["encoder"] or not groups["adapter"]:
        raise RuntimeError(
            "merged checkpoint is incomplete: "
            f"llm={len(groups['llm'])}, encoder={len(groups['encoder'])}, adapter={len(groups['adapter'])}"
        )

    files = sorted(set(weight_map.values()))
    missing = [name for name in files if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint shards are missing: {missing}")

    return {
        "root": root,
        "config": config,
        "index": index,
        "weight_map": weight_map,
        "groups": groups,
        "files": files,
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_llm_view(model_dir: str | Path, view_dir: str | Path) -> Path:
    """Create a small pure-HYV3 view without copying any model weights."""
    inspected = inspect_merged_checkpoint(model_dir)
    root: Path = inspected["root"]
    view = Path(view_dir).resolve()
    view.mkdir(parents=True, exist_ok=True)

    llm_weight_map = {
        key: filename
        for key, filename in inspected["weight_map"].items()
        if key.startswith(LLM_PREFIXES)
    }
    llm_files = sorted(set(llm_weight_map.values()))
    for filename in llm_files:
        target = root / filename
        link = view / filename
        if link.is_symlink() and Path(os.path.realpath(link)) == target:
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(target)

    config = dict(inspected["config"])
    config["architectures"] = ["HYV3ForCausalLM"]
    config.pop("audio_config", None)
    config.pop("audio_token_id", None)
    config.pop("turn_end_token_id", None)

    # Normalize the 4.57-era HYV3 config to the 5.8 schema used by the
    # training environment. Unknown legacy fields are harmless, but omitting
    # these two new fields is not: 5.8 would silently use its default expert
    # width (1536 instead of 768), producing an incompatible model skeleton.
    if "expert_hidden_dim" in config:
        config["moe_intermediate_size"] = int(config["expert_hidden_dim"])
    dense_layers = int(config.get("first_k_dense_replace", 0))
    num_layers = int(config["num_hidden_layers"])
    if dense_layers < 0 or dense_layers > num_layers:
        raise RuntimeError(
            f"invalid first_k_dense_replace={dense_layers} for num_hidden_layers={num_layers}"
        )
    config["mlp_layer_types"] = ["dense"] * dense_layers + ["sparse"] * (num_layers - dense_layers)
    filtered_index = {
        "metadata": dict(inspected["index"].get("metadata") or {}),
        "weight_map": llm_weight_map,
    }

    passthrough = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "generation_config.json",
    )
    for filename in passthrough:
        source = root / filename
        link = view / filename
        if not source.is_file():
            continue
        if link.is_symlink() and Path(os.path.realpath(link)) == source:
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(source)

    _atomic_write(
        view / "model.safetensors.index.json",
        (json.dumps(filtered_index, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    # config is written last and acts as the ready marker for other ranks.
    _atomic_write(
        view / "config.json",
        (json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    return view


def _files_for_prefix(model_dir: Path, prefix: str) -> list[Path]:
    inspected = inspect_merged_checkpoint(model_dir)
    names = sorted(
        {
            filename
            for key, filename in inspected["weight_map"].items()
            if key.startswith(prefix)
        }
    )
    if not names:
        raise RuntimeError(f"no {prefix}* tensors in {model_dir}")
    return [model_dir / name for name in names]


def _load_prefixed_state(model_dir: Path, prefix: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for shard_path in _files_for_prefix(model_dir, prefix):
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            for full_key in shard.keys():
                if not full_key.startswith(prefix):
                    continue
                key = full_key[len(prefix) :]
                if key in state:
                    raise RuntimeError(f"duplicate checkpoint tensor: {full_key}")
                tensor = shard.get_tensor(full_key)
                state[key] = tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
    if not state:
        raise RuntimeError(f"no {prefix}* tensors were loaded from {model_dir}")
    return state


def load_merged_audio_encoder(
    model_dir: str | Path,
    dtype: torch.dtype,
    attn_impl: str | None = None,
):
    """Load ``audio_tower.speech_encoder`` with strict architecture checks."""
    from s2s_train.vendor_qwen3_asr import Qwen3ASRAudioEncoder, Qwen3ASRAudioEncoderConfig
    from s2s_train.vendor_qwen3_asr.modeling_qwen3_asr import SinusoidsPositionEmbedding

    root = Path(model_dir).resolve()
    raw = inspect_merged_checkpoint(root)["config"]
    audio_config = raw["audio_config"]
    encoder_config = Qwen3ASRAudioEncoderConfig(
        **{key: value for key, value in audio_config.items() if not key.startswith("_")}
    )
    if attn_impl:
        encoder_config._attn_implementation = attn_impl

    with torch.device("meta"):
        encoder = Qwen3ASRAudioEncoder(encoder_config)
    state = _load_prefixed_state(root, ENCODER_PREFIX, dtype)
    missing, unexpected = encoder.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError(
            "merged audio encoder does not match the vendored Qwen3-ASR architecture: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    for module in encoder.modules():
        if isinstance(module, SinusoidsPositionEmbedding) and module.positional_embedding.is_meta:
            length, channels = module.positional_embedding.shape
            rebuilt = SinusoidsPositionEmbedding(length, channels)
            module.register_buffer(
                "positional_embedding",
                rebuilt.positional_embedding.to(dtype=dtype),
                persistent=False,
            )
    remaining = [name for name, value in encoder.named_parameters() if value.is_meta]
    remaining += [name for name, value in encoder.named_buffers() if value.is_meta]
    if remaining:
        raise RuntimeError(f"audio encoder still has meta tensors: {remaining[:8]}")
    return encoder


def load_merged_adapter(adapter: torch.nn.Module, model_dir: str | Path, dtype: torch.dtype) -> None:
    """Load ``audio_tower.speech_adapter`` into the supplied FSK adapter."""
    root = Path(model_dir).resolve()
    state = _load_prefixed_state(root, ADAPTER_PREFIX, dtype)
    expected = set(adapter.state_dict())
    found = set(state)
    if expected != found:
        raise RuntimeError(
            "merged speech adapter keys do not match FSKAdapter: "
            f"missing={sorted(expected - found)}, unexpected={sorted(found - expected)}"
        )
    adapter.load_state_dict(state, strict=True)


__all__ = [
    "ADAPTER_PREFIX",
    "ENCODER_PREFIX",
    "build_llm_view",
    "inspect_merged_checkpoint",
    "load_merged_adapter",
    "load_merged_audio_encoder",
]
