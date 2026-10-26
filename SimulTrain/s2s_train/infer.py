"""推理脚本：语音 + 指令 -> 文本。

用法：
    python -m s2s_train.infer \
      --checkpoint /path/to/exp/stage_b \
      --audio /path/to/a.wav \
      --prompt "请将这段语音翻译为英语，只输出翻译结果。"

要点：
1. prompt 必须复现训练时模板产出的格式，否则模型看到的分布与训练不一致。
2. 语音占位符数量必须等于 Adapter 输出长度，否则前向会直接报错。
3. LoRA checkpoint 会自动叠加到 LLM 上；Adapter 从 speech_adapter.safetensors 读取。
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="语音到文本推理")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="训练输出目录（需含 composite/ 与 speech_adapter.safetensors）",
    )
    parser.add_argument("--audio", type=Path, nargs="+", required=True, help="音频文件，可传多个")
    parser.add_argument(
        "--prompt",
        default="请将这段语音翻译为英语，只输出翻译结果。",
        help="指令文本（不含 <audio> 占位符，脚本会自动插入）",
    )
    parser.add_argument("--variant", default=None, help="模型变体，默认从 checkpoint 的 model_type 推断")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 表示贪心解码")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="使用哪张 GPU（物理卡号）。不指定则自动挑空闲显存最多的卡",
    )
    parser.add_argument("--cpu", action="store_true", help="强制用 CPU 推理（默认优先 GPU）")
    parser.add_argument("--output", type=Path, help="将结果写入 JSONL 文件")
    return parser.parse_args()


def resolve_composite_dir(checkpoint: Path) -> Path:
    """定位 composite 目录（含 config.json / tokenizer / 特征提取器配置）。"""
    for candidate in (checkpoint / "composite", checkpoint):
        if (candidate / "config.json").is_file():
            return candidate

    raise SystemExit(f"未找到 composite 配置目录：{checkpoint}")


def _report_missing(missing_keys) -> None:
    """声学编码器与冻结的 LLM 都可能被有意排除在 checkpoint 之外，不算缺失。

    - speech_encoder.*：冻结时不写入 checkpoint
    - language_model.*：save_frozen_llm=false 时不写入，推理从 llm_path 加载
    两者都已在建模时从原始权重加载好，缺失是预期行为。
    """
    skipped = ("speech_encoder.", "language_model.")
    real_missing = [key for key in missing_keys if not key.startswith(skipped)]
    if any(key.startswith("language_model.") for key in missing_keys):
        print("[info] checkpoint 未含 LLM 权重（冻结未保存），已从 llm_path 加载原始权重。")

    if real_missing:
        print(f"[warn] 以下权重未加载（前 3 个）：{real_missing[:3]}")


# 组合模型 BF16 约 5GB，加上激活与 KV cache 留一些余量
MIN_FREE_MIB = 8000


def _pick_device(gpu_id: int | None) -> torch.device:
    """选择推理设备。

    gpu_id 指定时用该卡；否则自动挑空闲显存最多的一张。
    显存不足直接报错，而不是等CUDA OOM 或悄悄退化到 CPU。

    注意：若外部已设置 CUDA_VISIBLE_DEVICES，torch 看到的编号是重映射后的，
    此时 --gpu 应按重映射后的序号传（通常直接不传，用外部变量控制即可）。
    """
    if not torch.cuda.is_available():
        print("[warn] 未检测到可用 GPU，将使用 CPU 推理（会明显更慢）。")
        return torch.device("cpu")

    count = torch.cuda.device_count()
    if gpu_id is None:
        # torch.cuda.mem_get_info 需要先切到目标卡
        free_by_id = []
        for index in range(count):
            with torch.cuda.device(index):
                free_bytes, _ = torch.cuda.mem_get_info()

            free_by_id.append((free_bytes, index))

        free_bytes, gpu_id = max(free_by_id)
        print(f"[info] 自动选择 GPU {gpu_id}（可见卡 {count} 张）")
    else:
        if not 0 <= gpu_id < count:
            raise SystemExit(f"GPU {gpu_id} 不存在，当前可见 {count} 张卡（编号 0~{count - 1}）")

        with torch.cuda.device(gpu_id):
            free_bytes, _ = torch.cuda.mem_get_info()

    free_mib = free_bytes // (1024 * 1024)
    if free_mib < MIN_FREE_MIB:
        raise SystemExit(
            f"GPU {gpu_id} 空闲显存不足：{free_mib} MiB（至少需要 {MIN_FREE_MIB} MiB）。\n"
            f"请用 --gpu <其他卡号> 换卡，或加 --cpu 用 CPU 推理。"
        )

    torch.cuda.set_device(gpu_id)
    print(f"[info] 使用 GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}  空闲 {free_mib} MiB")
    return torch.device(f"cuda:{gpu_id}")


def resolve_variant(composite_dir: Path, override: str | None) -> str:
    if override:
        return override

    model_type = json.loads((composite_dir / "config.json").read_text(encoding="utf-8")).get("model_type", "")
    from s2s_train.variants import VARIANTS, load_variant

    for name in VARIANTS:
        module = load_variant(name)
        if model_type == module.MODEL_TYPE:
            return name
        if model_type.endswith(name):
            return name

    raise SystemExit(f"无法从 model_type `{model_type}` 推断变体，请用 --variant 指定")


def build_prompt(template, tokenizer, variant, prompt: str, num_speech_tokens: int) -> str:
    """复现训练时的 prompt 格式：前缀 + user(指令 + 占位符) + assistant 起始。"""
    audio_tokens = variant.AUDIO_TOKEN * num_speech_tokens
    encoded = template.encode_oneturn(
        tokenizer=tokenizer,
        messages=[
            # 占位符在指令**之后** —— 训练数据一律是「指令<audio>」，
            # mm_plugin 也是就地展开。写成 audio_tokens + prompt 会让
            # 顺序反过来，表现为先转写正确、再抄一遍指令、无限循环
            #（同一个坑 eval_gen.py 里已经修过，见那里的注释）。
            {"role": "user", "content": prompt + audio_tokens},
            {"role": "assistant", "content": ""},
        ],
    )
    # encode_oneturn 返回 (prompt_ids, response_ids)。必须把 response 里的
    # 生成提示 <｜hy_Assistant｜><think></think> 接到提示末尾，否则提示停在
    # 音频占位符上，模型会把它当成「用户话没说完」继续接用户的话，
    # 输出与音频完全无关（详见 eval_gen.py 里同一处修复的注释）。
    prompt_ids, resp_ids = encoded
    if resp_ids and resp_ids[-1] == tokenizer.eos_token_id:
        resp_ids = resp_ids[:-1]  # content="" 时末尾那个 eos 不要
    return prompt_ids + resp_ids


def main() -> None:
    args = parse_args()
    composite_dir = resolve_composite_dir(args.checkpoint)
    variant_name = resolve_variant(composite_dir, args.variant)

    from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    from s2s_train.common.mm_plugin_speech import decode_audio
    from s2s_train.variants import load_variant

    variant = load_variant(variant_name)
    config = variant.CONFIG_CLASS.from_pretrained(composite_dir)
    variant.register(config=config)

    device = torch.device("cpu") if args.cpu else _pick_device(args.gpu)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(composite_dir, local_files_only=True)
    processor = AutoProcessor.from_pretrained(composite_dir, local_files_only=True)
    feature_extractor = processor.feature_extractor

    # Adapter 权重优先取 checkpoint 根目录下的产物
    adapter_weights = args.checkpoint / "speech_adapter.safetensors"
    if adapter_weights.is_file():
        config.speech_adapter_path = str(adapter_weights)
    elif not config.speech_adapter_path:
        print("[warn] 未找到 speech_adapter.safetensors，Adapter 为随机初始化，输出无意义。")

    model = AutoModelForCausalLM.from_pretrained(composite_dir, config=config, dtype=dtype)

    # 全参训练的 LLM 权重、或 LoRA adapter，都在 checkpoint 根目录
    if (args.checkpoint / "adapter_config.json").is_file():
        from peft import PeftModel

        # LoRA 是在整个组合模型上注入的（键形如
        # base_model.model.language_model.model.layers.*），
        # 所以必须挂在model 上，不能只挂 model.language_model。
        model = PeftModel.from_pretrained(model, args.checkpoint)
        model = model.merge_and_unload()
        print(f"[info] 已合并 LoRA: {args.checkpoint}")
    elif (args.checkpoint / "model.safetensors.index.json").is_file():
        from transformers.modeling_utils import load_sharded_checkpoint

        result = load_sharded_checkpoint(model, str(args.checkpoint), strict=False)
        _report_missing(result.missing_keys)
        print(f"[info] 已加载全参权重（分片）: {args.checkpoint}")
    elif (args.checkpoint / "model.safetensors").is_file():
        from safetensors.torch import load_file

        state = load_file(str(args.checkpoint / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        _report_missing(missing)
        if unexpected:
            print(f"[warn] checkpoint 中有多余权重（前 3 个）：{unexpected[:3]}")

        print(f"[info] 已加载全参权重: {args.checkpoint}")

    model = model.to(device=device).eval()

    from llamafactory.data.template import TEMPLATES

    template = TEMPLATES[variant.TEMPLATE_NAME]
    template.fix_special_tokens(tokenizer)

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "num_beams": args.num_beams,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generate_kwargs.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    else:
        generate_kwargs.update(do_sample=False)

    is_fixed_len = getattr(config, "adapter_type", "qformer") == "qformer"
    n_desc = config.num_speech_tokens if is_fixed_len else "随时长变(fsk)"
    print(f"[info] 变体={variant_name} 语音token={n_desc} 设备={device}")
    print(f"[info] 指令: {args.prompt}")
    print()

    results = []
    # 定长时 prompt 对所有音频相同，可提到循环外；变长时每条不同
    prompt_ids = (
        build_prompt(template, tokenizer, variant, args.prompt, config.num_speech_tokens)
        if is_fixed_len
        else None
    )
    for audio_path in args.audio:
        if not audio_path.is_file():
            raise SystemExit(f"音频不存在：{audio_path}")

        waveform = decode_audio(str(audio_path), feature_extractor.sampling_rate)
        features = feature_extractor(
            [waveform],
            sampling_rate=feature_extractor.sampling_rate,
            return_attention_mask=True,
            # 固定长度模式补到 3000 帧；FSK 模式按真实长度处理，避免白算 padding。
            padding="max_length" if is_fixed_len else "longest",
            return_tensors="pt",
        )

        if not is_fixed_len:
            # 占位符数量必须与 Adapter 输出严格相等，两边都走
            # config.speech_tokens_for，算法同源
            # 变长时下面用 padding="longest" -> floor 取整
            n_speech = config.speech_tokens_for_samples(
                len(waveform), pad_to_max=is_fixed_len
            )
            prompt_ids = build_prompt(
                template, tokenizer, variant, args.prompt, n_speech
            )

        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output_ids = model.generate_from_speech(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            input_features=features["input_features"].to(device=device, dtype=dtype),
            feature_attention_mask=features["attention_mask"].to(device),
            **generate_kwargs,
        )
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        duration = len(waveform) / feature_extractor.sampling_rate
        print(f"{audio_path.name}  ({duration:.1f}s)")
        print(f"  -> {text}")
        results.append({"audio": str(audio_path), "duration_sec": round(duration, 2), "output": text})

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"\n[info] 结果已写入 {args.output}")


if __name__ == "__main__":
    main()
