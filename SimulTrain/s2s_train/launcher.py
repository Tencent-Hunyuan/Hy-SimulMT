"""训练入口。

用法：
    python -m s2s_train.launcher <config.yaml> [key=value ...]

配置里通过 `variant` 指定模型变体（默认 hy_v3_speech_merged）。

流程：
1. 读 YAML，取出本框架自定义键（variant/模型路径/adapter 维度等）
2. 生成 composite 模型目录：config.json + LLM tokenizer + Hy-SimulMT 特征提取器配置
3. 注册变体的 config/model/composite/plugin/template
4. 其余参数交给 LlamaFactory 的 run_exp
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml

CUSTOM_KEYS = (
    "variant",
    "merged_model_path",
    "encoder_path",
    "llm_path",
    "speech_adapter_path",
    "composite_dir",
    "encoder_dim",
    "encoder_type",
    # 是否训练声学编码器。属于我们自己的 config 字段，必须列在这里 ——
    # 否则会被当成 LlamaFactory 的参数传下去，报 "not used by HfArgumentParser"。
    "train_encoder",
    # Adapter 结构：qformer（定长）/ fsk（变长，输出随音频时长变）
    "adapter_type",
    "fsk_size",
    "qformer_hidden",
    "qformer_layers",
    "qformer_heads",
    "qformer_ffn_dim",
    "cross_attention_frequency",
    "qformer_dropout",
    "qformer_layer_norm_eps",
    "query_len",
    "use_pooling",
    "pool_stride",
    "mlp_hidden",
    "save_frozen_llm",
    # 早停：自己实现以支持改善阈值，见 callbacks.EarlyStopCallback
    "early_stop_patience",
    "early_stop_threshold",
    "early_stop_metric",
    "early_stop_greater_is_better",
    # 生成式评测：字符准确率，见 callbacks.GenEvalMetricCallback
    "gen_eval_samples",
    "gen_eval_media_dir",
    "gen_eval_batch_size",
    "gen_eval_max_new_tokens",
    "gen_eval_langs",
    # 评测产物的落地目录；留空则写在 output_dir 下（与 checkpoint 同层）
    "gen_eval_dir",
    # 换最优指标口径时清掉 checkpoint 里的旧值，见 callbacks.ResetBestMetricCallback
    "reset_best_metric",
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "tokenizer.model",
    "generation_config.json",
)

PLACEHOLDER = "PLEASE_OVERRIDE"
DEFAULT_VARIANT = "hy_v3_speech_merged"


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """原子落盘：先写同目录临时文件，再 os.replace 改名到位。

    os.replace 在同一文件系统内是原子改名，读者只会看到「旧的完整文件」或
    「新的完整文件」，不存在被截断成 0 字节的中间态。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        # mkstemp 建的文件是 0600，直接改名过去会让 composite 目录变成只有
        # 属主可读。这里对齐普通写文件的 0644，否则换用户跑推理会读不了。
        os.chmod(tmp, 0o644)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _is_global_rank0() -> bool:
    """torchrun 会给每个进程注入 RANK，非分布式启动时该变量不存在。"""
    return int(os.environ.get("RANK", "0")) == 0


def _wait_for_composite(out_dir: Path, expected: bytes, timeout: float = 1800.0) -> None:
    """非 rank0 进程等 rank0 把 composite 目录写完。

    判定用 config.json 的内容，因为它由 build_composite_dir 最后写，
    是「目录已就绪」的提交标记。
    读文件时要容忍 ESTALE/ENOENT：rank0 的 os.replace 会让路径的旧 inode
    失效，正好卡在那一瞬间读就会拿到这两个错，重试即可。
    """
    cfg = out_dir / "config.json"
    deadline = time.monotonic() + timeout
    while True:
        try:
            if cfg.read_bytes() == expected:
                return
        except (FileNotFoundError, OSError):
            pass
        if time.monotonic() > deadline:
            raise SystemExit(
                f"等待 rank0 生成 composite 目录超时（{timeout:.0f}s）：{out_dir}\n"
                "通常意味着 rank0 已经异常退出，去看它的日志。"
            )
        time.sleep(1.0)


def _atomic_copy(src: Path, dst: Path) -> None:
    """内容一致就跳过，否则原子覆盖。"""
    data = src.read_bytes()
    try:
        if dst.is_file() and dst.stat().st_size == len(data) and dst.read_bytes() == data:
            return
    except OSError:
        pass  # ESTALE 之类：当作「需要重写」处理
    _atomic_write_bytes(dst, data)


def build_composite_dir(custom: dict, out_dir: Path, variant) -> Path:
    from transformers import AutoConfig

    merged_model_path = custom.get("merged_model_path")
    if merged_model_path:
        from s2s_train.common.merged_checkpoint import build_llm_view, inspect_merged_checkpoint

        merged_model_path = str(Path(merged_model_path).resolve())
        inspected = inspect_merged_checkpoint(merged_model_path)
        llm_view = out_dir / "llm_view"
        if _is_global_rank0():
            build_llm_view(merged_model_path, llm_view)
        else:
            deadline = time.monotonic() + 1800.0
            while not (llm_view / "config.json").is_file():
                if time.monotonic() > deadline:
                    raise SystemExit(f"waiting for merged LLM view timed out: {llm_view}")
                time.sleep(1.0)
        encoder_path = Path(merged_model_path)
        llm_path = llm_view
        audio_config = inspected["config"]["audio_config"]
        if int(audio_config.get("encoder_dim", 0)) != 2048:
            raise SystemExit(f"merged checkpoint encoder_dim is not 2048: {audio_config!r}")
        custom.setdefault("encoder_type", "hy_simulmt_audio")
        custom.setdefault("adapter_type", "fsk")
        custom.setdefault("fsk_size", int(audio_config.get("fsk_size", 2)))
    else:
        encoder_path = Path(custom["encoder_path"])
        llm_path = Path(custom["llm_path"])
    if not encoder_path.is_dir():
        raise SystemExit(f"声学编码器目录不存在：{encoder_path}")
    if not llm_path.is_dir():
        raise SystemExit(f"LLM 目录不存在：{llm_path}")

    # 未覆盖的占位符视为「不加载已有 Adapter」，即从随机初始化开始
    adapter_path = custom.get("speech_adapter_path")
    if adapter_path == PLACEHOLDER:
        adapter_path = None

    llm_config = AutoConfig.from_pretrained(llm_path, local_files_only=True)

    from s2s_train.common.configuration_speech_llm import ENCODER_SPECS

    encoder_type = custom.get("encoder_type", "hy_simulmt_audio")
    if encoder_type not in ENCODER_SPECS:
        raise SystemExit(
            f"未登记的 encoder_type={encoder_type!r}，可选 {sorted(ENCODER_SPECS)}"
        )

    # encoder_dim 取 ENCODER_SPECS 里登记的值，不再从编码器 config 猜。
    #
    # audio_config.d_model=1024，但 audio tower 末尾还有
    # proj1/proj2/conv_out，真实输出是 output_dim=2048 —— 按 d_model 取会让
    # cross-attention 的 encoder_hidden_size 差一倍。
    encoder_dim = custom.get("encoder_dim", ENCODER_SPECS[encoder_type]["dim"])

    config = variant.CONFIG_CLASS(
        encoder_path=str(encoder_path),
        llm_path=str(llm_path),
        speech_adapter_path=adapter_path,
        merged_model_path=merged_model_path or "",
        encoder_dim=encoder_dim,
        encoder_type=encoder_type,
        train_encoder=bool(custom.get("train_encoder", False)),
        adapter_type=custom.get("adapter_type", "qformer"),
        fsk_size=custom.get("fsk_size", 2),
        qformer_hidden=custom.get("qformer_hidden", 768),
        qformer_layers=custom.get("qformer_layers", 8),
        qformer_heads=custom.get("qformer_heads", 12),
        qformer_ffn_dim=custom.get("qformer_ffn_dim", 3072),
        cross_attention_frequency=custom.get("cross_attention_frequency", 2),
        qformer_dropout=custom.get("qformer_dropout", 0.1),
        qformer_layer_norm_eps=custom.get("qformer_layer_norm_eps", 1e-12),
        query_len=custom.get("query_len", 150),
        use_pooling=custom.get("use_pooling", False),
        pool_stride=custom.get("pool_stride", 5),
        mlp_hidden=custom.get("mlp_hidden", 2560),
        llm_dim=llm_config.hidden_size,
        audio_token_id=variant.AUDIO_TOKEN_ID,
        save_frozen_llm=custom.get("save_frozen_llm", True),
    )
    # 维度写错时训练照样能跑、loss 也降，但语音表征是错的 —— 必须启动即崩
    config.validate_encoder_dim()

    preprocessor = encoder_path / "preprocessor_config.json"
    if not preprocessor.is_file():
        raise SystemExit(f"缺少编码器的 preprocessor_config.json：{preprocessor}")

    # config.json 由 build_composite_dir 最后写，充当「目录已就绪」的提交标记：
    # 读者一旦看到正确的 config.json，前面几个文件必然都已经落盘。
    # 替代 config.save_pretrained()：后者内部是非原子的 open("w")。
    # 本变体没有 register_for_auto_class，save_pretrained 除了 config.json
    # 不会再写别的文件，所以换成 to_json_string 是等价的（已按 md5 逐字节核对）。
    expected = (config.to_json_string(use_diff=True)).encode("utf-8")

    # ---- 只有全局 rank0 写，其余进程等它写完 ----
    # 为什么必须选主，而不是「大家都写、反正内容一样」（这是踩过两次的地方）：
    #
    # 1. torchrun 在每个进程里都完整跑一遍 launcher.main()，所以 8 机 64 个
    #    进程会同时往共享盘上同一个 composite 目录写。第一版用
    #    save_pretrained + shutil.copy2，两者都是「先截断成 0 再写」，
    #    别的进程正好在这个窗口读就会拿到空文件：
    #        json.decoder.JSONDecodeError: Expecting value: line 1 column 1
    #    该进程所在整机退出后，其余 rank 永远卡在 main_process_first 的 barrier
    #    上（world size 组网成功、GPU 只占 1.5 GB、日志再无输出）。
    #
    # 2. 换成原子写（临时文件 + os.replace）解决了截断问题，但 共享文件系统
    #    是 fuseblk 网络文件系统：每次 replace 都让该路径的旧 inode 失效，
    #    64 个进程连续 replace 同一路径时，正在读的进程会拿到
    #        OSError: [Errno 116] Stale file handle
    #    也就是说在 fuse 上「并发 replace 同一路径」本身就有害，光靠原子性不够。
    #
    # 所以这里让 rank0 独写、其余等待，把并发彻底消掉。写入仍走原子写：
    # 上一轮训练遗留的旧文件会被完整替换，读者不会看到半个文件。
    if _is_global_rank0():
        out_dir.mkdir(parents=True, exist_ok=True)

        # tokenizer 走 LLM
        for name in TOKENIZER_FILES:
            src = llm_path / name
            if src.is_file():
                _atomic_copy(src, out_dir / name)

        # 保留源模型的声学参数，但使用项目自己的中性特征提取器类型。
        feature_config = json.loads(preprocessor.read_text(encoding="utf-8"))
        feature_config["feature_extractor_type"] = "HySimulMTAudioFeatureExtractor"
        feature_config["processor_class"] = "HySimulMTProcessor"
        _atomic_write_bytes(
            out_dir / "preprocessor_config.json",
            (json.dumps(feature_config, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        _atomic_write_bytes(
            out_dir / "processor_config.json",
            (json.dumps({"processor_class": "HySimulMTProcessor"}, indent=2) + "\n").encode("utf-8"),
        )
        _atomic_write_bytes(out_dir / "config.json", expected)
    else:
        _wait_for_composite(out_dir, expected)

    return out_dir


def _validate_raw(raw: dict) -> None:
    """YAML 行内注释未留空格时会被解析进值里，提前拦截。"""
    bad = [key for key, value in raw.items() if isinstance(value, str) and "#" in value]
    if bad:
        raise SystemExit(f"配置项的值里出现 '#'，请在行内注释前加空格：{bad}")

    # 模型、数据与输出路径都必须由外部提供，本包只带代码与配置模板
    required = ("dataset", "dataset_dir", "output_dir")
    missing = [key for key in required if not raw.get(key) or raw.get(key) == PLACEHOLDER]

    merged_model_path = raw.get("merged_model_path")
    if not merged_model_path or merged_model_path == PLACEHOLDER:
        if not raw.get("llm_path") or raw.get("llm_path") == PLACEHOLDER:
            missing.append("llm_path")

    def _given(key: str) -> bool:
        return bool(raw.get(key)) and raw.get(key) != PLACEHOLDER

    if not merged_model_path and not _given("encoder_path"):
        missing.append("encoder_path")

    if missing:
        raise SystemExit(
            f"以下参数必须从外部传入：{missing}\n"
            "推荐把路径写进仓库根目录的 .env（可从 .env.example 复制），"
            "或在命令行用 key=value 覆盖。"
        )

    info = Path(raw["dataset_dir"]) / "dataset_info.json"
    if not info.is_file():
        raise SystemExit(f"数据集注册文件不存在：{info}")

    if raw.get("media_dir") == PLACEHOLDER:  # 未指定时 LlamaFactory 回退到 dataset_dir
        raw.pop("media_dir")


def _parse_value(text: str):
    """解析命令行传入的值。

    两个坑都来自 YAML 1.1：
    1. 科学计数法必须带小数点，`2e-4` 会被当成字符串，
       传给 Trainer 后会在比较时抛 TypeError，所以补一层浮点兜底。
    2. `no` / `yes` / `on` / `off` 会被解析成布尔值，
       而 `eval_strategy=no`、`save_strategy=no` 这些必须保持字符串，
       所以只认 true/false 两种字面量为布尔。
    """
    value = yaml.safe_load(text)

    if isinstance(value, bool) and text.strip().lower() not in ("true", "false"):
        return text

    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value

    return value


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].endswith((".yaml", ".yml")):
        raise SystemExit("用法: python -m s2s_train.launcher <config.yaml> [key=value ...]")

    config_path = Path(sys.argv[1]).resolve()
    overrides = sys.argv[2:]
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    for item in overrides:  # 允许命令行覆盖，例如 output_dir=...
        if "=" in item:
            key, value = item.split("=", 1)
            raw[key.strip()] = _parse_value(value)

    _validate_raw(raw)
    custom = {key: raw.pop(key) for key in CUSTOM_KEYS if key in raw}

    from s2s_train.variants import load_variant

    variant_name = custom.get("variant", DEFAULT_VARIANT)
    variant = load_variant(variant_name)

    # composite 目录默认放在 output_dir 下，避免往代码目录里写文件。
    # 它只有几 MB（config + tokenizer + 特征提取器配置），不含权重。
    if custom.get("composite_dir"):
        composite_dir = Path(custom["composite_dir"])
    else:
        composite_dir = Path(raw["output_dir"]) / "composite"

    composite_dir = build_composite_dir(custom, composite_dir.resolve(), variant)

    config = variant.CONFIG_CLASS.from_pretrained(composite_dir)
    # 传 config 而不是 num_speech_tokens：变长 Adapter 没有固定的 token 数，
    # 要靠 config.speech_tokens_for 让 mm_plugin 按每条音频时长展开占位符。
    variant.register(config=config)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(composite_dir, local_files_only=True)
    if hasattr(variant, "verify_tokens"):
        variant.verify_tokens(tokenizer)

    raw["model_name_or_path"] = str(composite_dir)
    raw.setdefault("template", variant.TEMPLATE_NAME)

    from llamafactory.train.tuner import run_exp

    from s2s_train.common.callbacks import (
        BestAdapterMirrorCallback,
        EarlyStopCallback,
        GenEvalMetricCallback,
        PruneOptimizerStateCallback,
        ResetBestMetricCallback,
        SpeechAdapterSaveCallback,
    )

    print(f"[launcher] 变体: {variant_name} (model_type={variant.MODEL_TYPE})")
    print(f"[launcher] composite dir: {composite_dir}")
    if getattr(config, "merged_model_path", ""):
        print(f"[launcher] merged checkpoint (read-only): {config.merged_model_path}")
        print(f"[launcher] pure LLM view: {config.llm_path}")
    if config.adapter_type == "qformer":
        print(
            f"[launcher] 语音 token 数: {config.num_speech_tokens}"
            f" (定长, use_pooling={config.use_pooling})"
        )
    else:
        # 变长：num_speech_tokens 会抛 AttributeError（故意的，防止别处按定长假设用）。
        # 打几个代表性时长，方便一眼确认压缩比是否符合预期。
        preview = "  ".join(
            f"{sec}s->{config.speech_tokens_for(int(sec * 100))}" for sec in (1, 3, 5, 10, 30)
        )
        print(
            f"[launcher] 语音 token 数: 变长 {config.adapter_type} k={config.fsk_size}"
            f"  ({preview})"
        )

    # 顺序有讲究：回调按列表序被调用，注入指标的必须排在读取指标的前面
    callbacks = [
        SpeechAdapterSaveCallback(),
        # 必须排在 SpeechAdapterSaveCallback 之后：要复制它刚写进 checkpoint 的 adapter
        BestAdapterMirrorCallback(),
        PruneOptimizerStateCallback(),
    ]

    if custom.get("reset_best_metric"):
        callbacks.append(
            ResetBestMetricCallback(
                expected_metric=str(raw.get("metric_for_best_model", "")),
                greater_is_better=bool(raw.get("greater_is_better", False)),
            )
        )

    if custom.get("gen_eval_samples"):
        callbacks.append(
            GenEvalMetricCallback(
                samples_path=str(custom["gen_eval_samples"]),
                media_dir=str(
                    custom.get("gen_eval_media_dir") or raw.get("media_dir") or raw["dataset_dir"]
                ),
                variant_name=variant_name,
                composite_dir=str(composite_dir),
                batch_size=int(custom.get("gen_eval_batch_size", 16)),
                max_new_tokens=int(custom.get("gen_eval_max_new_tokens", 110)),
                langs=str(custom.get("gen_eval_langs", "") or ""),
                eval_dir=str(custom.get("gen_eval_dir", "") or ""),
            )
        )
        lang_desc = custom.get("gen_eval_langs") or "全部语种"
        print(
            f"[launcher] 生成式评测: {custom['gen_eval_samples']}"
            f"（字符准确率，语种={lang_desc}，多卡分片）"
        )

    patience = custom.get("early_stop_patience")
    if patience:
        threshold = float(custom.get("early_stop_threshold", 0.0))
        metric = str(custom.get("early_stop_metric", "eval_loss"))
        greater = custom.get("early_stop_greater_is_better")
        if greater is not None:
            greater = bool(greater)
        callbacks.append(
            EarlyStopCallback(
                patience=int(patience),
                threshold=threshold,
                metric=metric,
                greater_is_better=greater,
            )
        )
        print(
            f"[launcher] 早停: 连续 {patience} 次评估 {metric} 改善 < {threshold} 即停止"
            "（保留最后一个 checkpoint）"
        )

    run_exp(args=raw, callbacks=callbacks)


if __name__ == "__main__":
    main()
