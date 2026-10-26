#!/usr/bin/env python3
"""Read-only compatibility check for a merged A3B speech checkpoint."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from s2s_train.common.configuration_speech_llm import SpeechLLMConfig
from s2s_train.common.merged_checkpoint import (
    build_llm_view,
    inspect_merged_checkpoint,
    load_merged_adapter,
    load_merged_audio_encoder,
)
from s2s_train.common.modeling_speech_adapter import FSKAdapter
from s2s_train.variants import hy_v3_speech_merged as variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--skip-encoder-load", action="store_true")
    args = parser.parse_args()

    inspected = inspect_merged_checkpoint(args.model)
    config = inspected["config"]
    audio = config["audio_config"]
    adapter_config = SpeechLLMConfig(
        encoder_dim=int(audio["encoder_dim"]),
        llm_dim=int(audio["llm_dim"]),
        adapter_type="fsk",
        fsk_size=int(audio["fsk_size"]),
    )
    adapter = FSKAdapter(adapter_config)
    load_merged_adapter(adapter, args.model, torch.bfloat16)

    # transformers 5.8 stores MoE experts as two 3D parameters while this
    # export stores one gate/up/down tensor per expert.  The generic checkpoint
    # conversion mapping must contain the hy_v3 -> qwen2_moe fusion rules.
    import transformers

    if tuple(int(part) for part in transformers.__version__.split(".")[:2]) >= (5, 0):
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping

        hy_v3_rules = get_checkpoint_conversion_mapping("hy_v3")
        if not hy_v3_rules:
            raise RuntimeError(
                f"transformers {transformers.__version__} has no hy_v3 MoE checkpoint conversion mapping"
            )
        conversion_rule_count = len(hy_v3_rules)
    else:
        # 4.57 uses ModuleList experts directly and needs no fusion conversion.
        conversion_rule_count = 0

    with tempfile.TemporaryDirectory(prefix="a3b-merged-preflight-") as temporary:
        view = build_llm_view(args.model, Path(temporary) / "llm")
        from transformers import AutoConfig, AutoTokenizer

        llm_config = AutoConfig.from_pretrained(view, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(view, local_files_only=True)
        variant.verify_tokens(tokenizer)
        raw_view_config = json.loads((view / "config.json").read_text(encoding="utf-8"))
        expected_expert_dim = int(config["expert_hidden_dim"])
        if raw_view_config.get("moe_intermediate_size") != expected_expert_dim:
            raise RuntimeError(
                "LLM view did not migrate expert_hidden_dim to moe_intermediate_size: "
                f"{raw_view_config.get('moe_intermediate_size')} != {expected_expert_dim}"
            )
        dense_layers = int(config.get("first_k_dense_replace", 0))
        expected_layer_types = ["dense"] * dense_layers + ["sparse"] * (
            int(config["num_hidden_layers"]) - dense_layers
        )
        if raw_view_config.get("mlp_layer_types") != expected_layer_types:
            raise RuntimeError("LLM view has the wrong dense/sparse layer layout")

    encoder_parameter_count = None
    if not args.skip_encoder_load:
        encoder = load_merged_audio_encoder(args.model, torch.bfloat16)
        encoder_parameter_count = sum(parameter.numel() for parameter in encoder.parameters())

    result = {
        "model": str(inspected["root"]),
        "architecture": config["architectures"][0],
        "model_type": config["model_type"],
        "llm_config_class": type(llm_config).__name__,
        "moe_intermediate_size": raw_view_config["moe_intermediate_size"],
        "dense_layers": raw_view_config["mlp_layer_types"].count("dense"),
        "sparse_layers": raw_view_config["mlp_layer_types"].count("sparse"),
        "llm_tensors": len(inspected["groups"]["llm"]),
        "encoder_tensors": len(inspected["groups"]["encoder"]),
        "adapter_tensors": len(inspected["groups"]["adapter"]),
        "adapter_parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        "encoder_parameters": encoder_parameter_count,
        "source_token_id": tokenizer.encode(variant.SOURCE_TOKEN, add_special_tokens=False)[0],
        "target_token_id": tokenizer.encode(variant.TARGET_TOKEN, add_special_tokens=False)[0],
        "transformers_version": transformers.__version__,
        "hy_v3_conversion_rules": conversion_rule_count,
        "source_model_modified": False,
        "passed": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
