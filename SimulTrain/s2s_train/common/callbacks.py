"""Adapter 单独保存回调。

LoRA 模式下 Adapter 既不是 LoRA 目标，也不在 PEFT 的 modules_to_save 中
（adapter.py:251），默认不会被写入 checkpoint。用回调统一保存可以让
全参 / LoRA / 冻结三种模式产物行为一致。
"""

import json
import os
from pathlib import Path

from safetensors.torch import save_file
from transformers import TrainerCallback

ADAPTER_WEIGHTS = "speech_adapter.safetensors"
ADAPTER_CONFIG = "speech_adapter_config.json"


def _unwrap(model):
    """剥掉 PEFT / DDP / torch.compile 包装，取到 Hy-SimulMT 模型。"""
    seen = 0
    while seen < 10:
        if hasattr(model, "speech_adapter"):
            return model

        for attr in ("module", "base_model", "model", "_orig_mod"):
            inner = getattr(model, attr, None)
            if inner is not None and inner is not model:
                model = inner
                break
        else:
            return None

        seen += 1

    return None


def _normalize_keys(state: dict) -> dict:
    """去掉 PEFT modules_to_save 包装引入的前缀。

    LoRA 模式下 additional_target会让 PEFT 用 ModulesToSaveWrapper 包住
    Adapter，state_dict 里同时出现 `modules_to_save.default.*`（训练后的
    权重）与 `original_module.*`（原始副本）。这里只保留训练后的权重，并
    还原成干净的键名，保证三种模式产出的文件可以互相加载。
    """
    marker = "modules_to_save.default."
    trained = {key[len(marker) :]: value for key, value in state.items() if key.startswith(marker)}
    if trained:
        return trained

    prefix = "original_module."
    return {key[len(prefix) :] if key.startswith(prefix) else key: value for key, value in state.items()}


def _collect_state(adapter) -> dict:
    """取出 Adapter 权重，兼容 DeepSpeed ZeRO-3 的参数分片。

    ZeRO-3 会把参数切片存放，此时 state_dict() 返回的是shape 为 [0] 的
    空张量，直接保存会得到一个看似正常、实际无法加载的文件。这里通过
    GatheredParameters 先把分片汇聚成完整张量。
    """
    params = list(adapter.parameters(recurse=True))
    is_sharded = any(getattr(p, "ds_id", None) is not None for p in params)
    if not is_sharded:
        return adapter.state_dict()

    import deepspeed

    with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
        return {key: value.clone() for key, value in adapter.state_dict().items()}


def save_speech_adapter(model, output_dir: str, is_main_process: bool = True) -> str | None:
    """保存 Adapter。

    注意：ZeRO-3 下的参数汇聚是集合通信，所有 rank 都必须进入，
    否则会死锁；但只有主进程负责落盘。
    """
    target = _unwrap(model)
    if target is None:
        raise RuntimeError("未能在模型中定位 speech_adapter，Adapter 未保存。")

    state = _normalize_keys(_collect_state(target.speech_adapter))
    if not is_main_process:
        return None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in state.items()}
    empty = [key for key, value in state.items() if value.numel() == 0]
    if empty:
        raise RuntimeError(f"Adapter 权重为空张量（疑似分片未汇聚）：{empty[:3]}等 {len(empty)} 个。")

    save_file(state, str(out / ADAPTER_WEIGHTS))

    config = target.config
    # 这份 meta 是给「离线加载 adapter 权重」用的，要能唯一确定结构。
    # 按 adapter_type 分开写：Q-Former 的那堆字段对 FSK 没意义，
    # 而 num_speech_tokens 在 FSK 下读了会抛 AttributeError（故意的）。
    adapter_type = getattr(config, "adapter_type", "qformer")
    meta = {
        "adapter_type": adapter_type,
        "encoder_dim": config.encoder_dim,
        "encoder_type": config.encoder_type,
        "llm_dim": config.llm_dim,
    }
    if adapter_type == "qformer":
        meta.update(
            {
                "qformer_hidden": config.qformer_hidden,
                "qformer_layers": config.qformer_layers,
                "qformer_heads": config.qformer_heads,
                "qformer_ffn_dim": config.qformer_ffn_dim,
                "cross_attention_frequency": config.cross_attention_frequency,
                "qformer_dropout": config.qformer_dropout,
                "qformer_layer_norm_eps": config.qformer_layer_norm_eps,
                "query_len": config.query_len,
                "use_pooling": config.use_pooling,
                "pool_stride": config.pool_stride,
                "mlp_hidden": config.mlp_hidden,
                "num_speech_tokens": config.num_speech_tokens,
            }
        )
    else:
        # 变长：token 数由 fsk_size 与音频时长共同决定，没有单一数值可记
        meta["fsk_size"] = config.fsk_size
    (out / ADAPTER_CONFIG).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out / ADAPTER_WEIGHTS)


class SpeechAdapterSaveCallback(TrainerCallback):
    """所有 rank 都要调用（ZeRO-3 汇聚是集合通信），仅主进程落盘。"""

    @staticmethod
    def _skip_for_smoke() -> bool:
        return os.environ.get("A3B_SMOKE_SKIP_ADAPTER_SAVE", "0") == "1"

    def on_save(self, args, state, control, model=None, **kwargs):
        if self._skip_for_smoke():
            return control
        if model is None:
            return

        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        save_speech_adapter(model, str(ckpt), is_main_process=state.is_world_process_zero)

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self._skip_for_smoke():
            if state.is_world_process_zero:
                print(
                    "[smoke] A3B_SMOKE_SKIP_ADAPTER_SAVE=1: skipping Adapter weight export.",
                    flush=True,
                )
            return control
        if model is None:
            return

        save_speech_adapter(model, args.output_dir, is_main_process=state.is_world_process_zero)


class BestAdapterMirrorCallback(TrainerCallback):
    """把当前最优的 adapter 权重镜像到 <output_dir>/best/，有更好的就整体替换。

    动机：想知道"哪个是最优"得去翻 trainer_state.json 的 best_global_step；
    而且训练中途根目录还没有产物（load_best_model_at_end 只在训练结束时回滚）。
    有了固定路径就能随时直接用，也就不必为了保住最优而把 save_total_limit 设很大。

    为什么目录叫 best 而不是 checkpoint-best：
    Trainer._rotate_checkpoints 用 glob("checkpoint-*") 收候选、按 mtime 排序删除
    （trainer.py:4364 与 3370），它"不删最优"的保护只认 state.best_model_checkpoint
    指向的 checkpoint-<step>。叫 checkpoint-best 会被当普通 checkpoint 轮转掉，
    而且它 mtime 最新，收紧 save_total_limit 时还会连带挤掉别的。
    不带 checkpoint- 前缀就完全不在它的视野里。

    只放推理用的小文件，不含 optimizer.pt / rng_state —— 那些只有续训要用，
    而续训应该从 checkpoint-<step> 走。
    """

    COPY_FILES = (
        "speech_adapter.safetensors",
        "speech_adapter_config.json",
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "trainer_state.json",
    )

    def __init__(self, dir_name: str = "best") -> None:
        self.dir_name = dir_name
        self._done_step = None

    def on_save(self, args, state, control, **kwargs):
        # best_global_step 由 _determine_best_metric 在评估后写入，而
        # _maybe_log_save_evaluate 是先 evaluate 再 save，所以这里读到的已是最新值。
        best_step = getattr(state, "best_global_step", None)
        if not state.is_world_process_zero or best_step is None:
            return control
        # 只有"这次存的就是最优"才镜像；最优在别的 step 说明本轮没刷新
        if best_step != state.global_step or self._done_step == best_step:
            return control

        import shutil

        src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        dst = Path(args.output_dir) / self.dir_name
        if not src.is_dir():
            return control

        try:
            tmp = dst.with_name(dst.name + ".tmp")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            for name in self.COPY_FILES:
                f = src / name
                if f.exists():
                    shutil.copy2(f, tmp / name)
            metric = getattr(state, "best_metric", None)
            (tmp / "BEST_INFO.json").write_text(
                json.dumps(
                    {
                        "best_metric": metric,
                        "best_global_step": best_step,
                        "epoch": round(state.epoch or 0.0, 4),
                        "source": str(src),
                        "note": "由 BestAdapterMirrorCallback 镜像，有更好的会整体替换",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            # 先写 .tmp 再原子替换，避免中途崩掉留下半份权重
            shutil.rmtree(dst, ignore_errors=True)
            tmp.rename(dst)
            self._done_step = best_step
            print(
                f"[best] 已更新 {dst}（step {best_step}"
                + (f"，metric {metric:.4f}" if isinstance(metric, float) else "")
                + "）",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - 镜像失败不该影响训练
            print(f"[best] 镜像失败: {exc}", flush=True)

        return control


#断点续训需要的状态文件；只有最新 checkpoint 需要留着它们
RESUME_ONLY_FILES =("optimizer.pt", "scheduler.pt", "scaler.pt")
RESUME_ONLY_GLOBS = ("rng_state*.pth",)


class PruneOptimizerStateCallback(TrainerCallback):
    """只在最新 checkpoint 里保留优化器状态，旧的删掉以省空间。

    每个 checkpoint 的模型权重全部保留（便于挑选/对比），但
    optimizer.pt 占体积的一半以上（Adapter 场景约 111MB vs 55MB 权重），
    而断点续训只需要最新那一个，旧的纯属浪费。

    Trainer 自带的 save_total_limit 只能整体删除 checkpoint，
    做不到"留权重、删优化器状态"，所以用回调补上。
    """

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        output_dir = Path(args.output_dir)
        current = output_dir / f"checkpoint-{state.global_step}"
        freed = 0
        for ckpt in sorted(output_dir.glob("checkpoint-*")):
            if ckpt == current or not ckpt.is_dir():
                continue

            targets = [ckpt / name for name in RESUME_ONLY_FILES]
            for pattern in RESUME_ONLY_GLOBS:
                targets.extend(ckpt.glob(pattern))

            for path in targets:
                if path.is_file():
                    freed += path.stat().st_size
                    path.unlink()

        if freed:
            print(f"[prune] 已清理旧 checkpoint 的优化器状态，释放 {freed / 1e6:.0f} MB")


class EarlyStopCallback(TrainerCallback):
    """带「改善阈值」的早停：连续若干次评估没有明显下降（或反而变高）就停。

    不直接用 LlamaFactory 的 early_stopping_steps：它内部构造 HF 的
    EarlyStoppingCallback 时没有传 early_stopping_threshold（tuner.py:82），
    只能用默认值 0.0。而 eval_loss 后期常常每次只降 0.0005，
    threshold=0 意味着这也算「有改善」，早停几乎永远不会触发。

    「最优权重」不由本回调负责：开 load_best_model_at_end 后，Trainer 会在
    训练循环结束、on_train_end 之前把模型回滚成 best checkpoint
    （trainer.py:2811），因此 SpeechAdapterSaveCallback 存下的就是最优权重。
    本回调只负责发停止信号并把最优出现的位置打进日志。
    """

    def __init__(
        self,
        patience: int,
        threshold: float = 0.0,
        metric: str = "eval_loss",
        greater_is_better: bool | None = None,
    ) -> None:
        self.patience = patience
        self.threshold = threshold
        self.metric = metric if metric.startswith("eval_") else f"eval_{metric}"
        # 未显式指定时按名字推断，规则与 HF TrainingArguments 一致：
        # loss 类越小越好，其余（准确率之类）越大越好。
        if greater_is_better is None:
            greater_is_better = not self.metric.lower().endswith("_loss")
        self.greater = bool(greater_is_better)
        self.best = None
        self.best_step = None
        self.bad = 0
        self.stopped = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.metric not in metrics:
            return control

        # 训练结束后 Trainer 还会跑一次最终评估，那次不该再累加计数
        if self.stopped:
            return control

        value = metrics[self.metric]
        # 只有「比历史最优好出 threshold 以上」才算真的改善；
        # 因此「基本没变化」和「反而变差」都会累加计数。
        if self.greater:
            improved = self.best is None or value > self.best + self.threshold
        else:
            improved = self.best is None or value < self.best - self.threshold
        if improved:
            self.best, self.best_step, self.bad = value, state.global_step, 0
            return control

        self.bad += 1
        if state.is_world_process_zero:
            worse = value < self.best if self.greater else value > self.best
            trend = "变差" if worse else "无明显改善"
            print(
                f"[early-stop] {self.metric}={value:.4f} {trend}"
                f"（最优 {self.best:.4f} @ step {self.best_step}，阈值 {self.threshold}）"
                f" {self.bad}/{self.patience}"
            )

        if self.bad >= self.patience:
            self.stopped = True
            control.should_training_stop = True
            if state.is_world_process_zero:
                print(
                    f"[early-stop] 连续 {self.patience} 次评估无显著改善，停止训练。\n"
                    f"[early-stop] 历史最优 {self.metric}={self.best:.4f} 出现在 step {self.best_step}。"
                )
                if getattr(args, "load_best_model_at_end", False):
                    print("[early-stop] 已开启 load_best_model_at_end，最终产物取该最优 checkpoint。")
                else:
                    print("[early-stop] 未开启 load_best_model_at_end，最终产物是最后一个 checkpoint。")

        return control


class GenEvalMetricCallback(TrainerCallback):
    """在 on_evaluate 里跑生成式评测，把字符准确率注入 metrics。

    为什么能这样注入：Trainer 传给回调的 metrics 就是 evaluate() 返回的
    同一个 dict 对象（trainer.py 里 on_evaluate 拿到的和 return 的是一个），
    而 _maybe_log_save_evaluate 是先 metrics=self._evaluate(...) 再
    _determine_best_metric(metrics)，所以这里写进去的键能被最优 checkpoint
    选择和早停读到。

    为什么不用 compute_metrics：那里只能拿到 teacher-forcing 的 logits，
    做不了自回归生成；而 predict_with_generate 又与本模型的
    generate_from_speech 不兼容（见 eval_gen 模块说明）。

    注意：Trainer 的 self.log(metrics) 发生在 on_evaluate 之前，
    所以这里注入的指标不会自动进 trainer_log.jsonl / TensorBoard，
    只能靠下面的 print 落到 train.log。
    """

    # 按任务分开的指标键。转写报 acc、翻译报 BLEU —— 两个任务混用一个指标
    # 会让翻译被严重低估（同义改写在字符级比对下算错）。
    # asr_acc / mt_bleu 是各自的均值；bleu / mean_acc 是兼容别名，
    # 让历史配置里的 BEST_METRIC=eval_bleu、eval_mean_acc 继续可用。
    METRIC_KEYS = (
        "asr_zh_acc", "asr_en_acc", "asr_zh_exact", "asr_en_exact", "asr_acc",
        "mt_en2zh_bleu", "mt_zh2en_bleu", "mt_bleu",
        "bleu", "mean_acc",
    )

    def __init__(
        self,
        samples_path: str,
        media_dir: str,
        variant_name: str,
        composite_dir: str,
        batch_size: int = 16,
        max_new_tokens: int = 110,
        prefix: str = "eval",
        langs: str = "",
        eval_dir: str = "",
    ) -> None:
        from s2s_train.common.eval_gen import GenEvalRunner

        self.prefix = prefix
        # 留空则沿用旧行为（写在 output_dir 下）。给了就把评测产物单独放，
        # 免得 gen_eval.jsonl / gen_eval_cases/ 和几 GB 的 checkpoint 混在一层。
        self.eval_dir = str(eval_dir) if eval_dir else ""
        self.runner = GenEvalRunner(
            composite_dir=composite_dir,
            media_dir=media_dir,
            variant_name=variant_name,
            samples_path=samples_path,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            langs=langs,
        )
        self.last = None
        self._best_acc = None

    def _out_dir(self, args):
        """评测产物的落地目录：优先 eval_dir，否则回退 output_dir。"""
        from pathlib import Path as _Path

        return _Path(self.eval_dir) if self.eval_dir else _Path(args.output_dir)

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics is None or model is None:
            return control

        import torch
        import torch.distributed as dist

        from s2s_train.common.eval_gen import aggregate

        distributed = dist.is_available() and dist.is_initialized()
        world = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        # 所有 rank 都必须走到这里：下面的 all_gather 是集合通信，
        # 任何一个 rank 提前返回都会让其余 rank 永久挂住；
        # 而且 _determine_best_metric 在每个 rank 上都跑，
        # 谁的 metrics 里缺这个键谁就 KeyError。
        try:
            with torch.inference_mode():
                local = self.runner.run(model, rank=rank, world=world)
            local["ok"] = True
            # BLEU 只能在语料级算（brevity penalty 与 n-gram 命中都是跨句累计），
            # 所以把本 rank 的 ref/hyp 原文一起 gather，到 aggregate 里统一算一次。
            # 体量可控：1500 条译文约 100KB，与 case 的 1MB 不同级。
            local["texts"] = getattr(self.runner, "texts", {})
            # case 单独走一次 all_gather：它比指标大得多（1500 条约 1MB），
            # 和指标混在一个 payload 里会让每次评测都传一遍，没必要。
            local_cases = self.runner.cases
        except Exception as exc:  # noqa: BLE001 - 评测失败不该杀掉多天的训练
            print(f"[gen-eval] rank {rank} 评测失败: {type(exc).__name__}: {exc}", flush=True)
            local = {"ok": False}
            local_cases = []

        if world > 1:
            gathered: list = [None] * world
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]

        # 任一 rank 失败就整体回退，保证所有 rank 注入完全相同的值
        if all(s and s.get("ok") for s in gathered):
            stats = aggregate(gathered)
            self.last = stats
        elif self.last is not None:
            stats = self.last
            if state.is_world_process_zero:
                print("[gen-eval] 本次评测有 rank 失败，沿用上一次的指标值", flush=True)
        else:
            # 首次就失败：注入哨兵值而不是不注入，否则 _determine_best_metric 直接 KeyError
            stats = {k: -1.0 for k in self.METRIC_KEYS}
            if state.is_world_process_zero:
                print("[gen-eval] 首次评测即失败，注入哨兵值 -1.0", flush=True)

        for key in self.METRIC_KEYS:
            if key in stats:
                metrics[f"{self.prefix}_{key}"] = stats[key]

        if state.is_world_process_zero:
            # 按任务打，而不是按语种。之前按语种打的坑：<zh><zh>（中文转写）
            # 与 <en><zh>（英译中）的 label 都是中文，两种任务被并进一个
            # zh_acc 里，转写 0.97 和翻译的字符级 0.3 平均成一个没法解读的数。
            # 只打实际存在的任务：纯翻译训练时没有 asr 桶，反之亦然。
            parts = []
            for task in ("asr_zh", "asr_en"):
                if f"{task}_acc" in stats:
                    parts.append(
                        f"{task}_acc={stats[f'{task}_acc']:.4f}"
                        f"(完全一致 {stats.get(f'{task}_exact', 0):.1%})"
                    )
            for task in ("mt_en2zh", "mt_zh2en"):
                if f"{task}_bleu" in stats:
                    parts.append(f"{task}_bleu={stats[f'{task}_bleu']:.2f}")
            summary = []
            if "asr_acc" in stats:
                summary.append(f"asr_acc={stats['asr_acc']:.4f}")
            if "mt_bleu" in stats:
                summary.append(f"mt_bleu={stats['mt_bleu']:.2f}")
            print(
                f"[gen-eval] step {state.global_step} "
                + " ".join(summary + parts),
                flush=True,
            )
            self._write_log(args, state, stats, metrics.get(f"{self.prefix}_loss"))

        # 逐条 case：光有聚合数字看不出错在哪（幻觉？漏词？数字格式？），
        # 排查必须要原文。只在最优被刷新时落盘，避免每次评测都写一份 1MB。
        # 用什么当「最优」：有翻译桶就看 mt_bleu（翻译才是目标任务），
        # 否则退回 asr_acc。不能只看 mean_acc —— 纯翻译训练时它不存在，
        # get 到 -1 会让每次评测都判为「刷新」，白写一遍。
        primary = stats.get("mt_bleu", stats.get("asr_acc", -1))
        is_best = primary >= (self._best_acc if self._best_acc is not None else -1)
        if is_best:
            self._best_acc = primary
        # 正常只在刷新 best 时落 case（省磁盘）。但调试期恰恰相反 ——
        # 指标越差越需要看模型到底吐了什么，而那时永远不是 best。
        # GEN_EVAL_ALWAYS_DUMP_CASES=1 强制每次评测都写。
        import os

        force_dump = os.environ.get("GEN_EVAL_ALWAYS_DUMP_CASES", "") == "1"
        if (is_best or force_dump) and all(s and s.get("ok") for s in gathered):
            all_cases = local_cases
            if world > 1:
                buf: list = [None] * world
                dist.all_gather_object(buf, local_cases)
                all_cases = [c for part in buf if part for c in part]
            if state.is_world_process_zero:
                self._write_cases(args, state, all_cases, stats)

        # 落进 log_history，这样 trainer_state.json 里也能查到历史 acc
        # （LogCallback 只认自己白名单里的键，写不进 trainer_log.jsonl，
        #  所以另外维护 gen_eval.jsonl；Trainer.log 存进 log_history 的是副本，
        #  在 on_evaluate 里改 metrics 影响不到它，必须显式补写。）
        if state.log_history:
            for key in self.METRIC_KEYS:
                if key in stats:
                    state.log_history[-1][f"{self.prefix}_{key}"] = stats[key]

        # 主动上报到 swanlab / tensorboard。
        # 必须自己报：Trainer 的 self.log(metrics) 在 on_evaluate 之前就跑完了，
        # 这里注入 metrics 的时机已经错过那次上报，acc / bleu 不会出现在曲线里，
        # 只有 eval_loss 有。所以显式再推一次，指标名与 metrics 保持一致。
        if state.is_world_process_zero:
            self._report(args, state, stats)

        return control

    def _report(self, args, state, stats: dict) -> None:
        """把生成式指标推给 swanlab 和 tensorboard，失败不影响训练。"""
        payload = {
            f"{self.prefix}_{k}": stats[k] for k in self.METRIC_KEYS if k in stats
        }
        if not payload:
            return

        # swanlab：进程内已由 SwanLabCallback 初始化过，直接 log 到当前 run
        try:
            import swanlab

            if getattr(swanlab, "get_run", lambda: None)() is not None:
                swanlab.log(payload, step=state.global_step)
        except Exception:  # noqa: BLE001 - 可视化不该影响训练
            pass

        # tensorboard：复用 Trainer 的 logging_dir，事件文件与 loss 曲线同目录
        try:
            if "tensorboard" in (args.report_to or []):
                from torch.utils.tensorboard import SummaryWriter

                if not hasattr(self, "_tb"):
                    self._tb = SummaryWriter(log_dir=args.logging_dir)
                for key, value in payload.items():
                    self._tb.add_scalar(key, value, state.global_step)
                self._tb.flush()
        except Exception:  # noqa: BLE001
            pass

    def _write_cases(self, args, state, cases: list[dict], stats: dict) -> None:
        """把逐条预测写到 <eval_dir 或 output_dir>/gen_eval_cases/。

        只保留最新一份（best_cases.jsonl）加一份按 err_rate 排序的最差样本，
        不按 step 累积 —— 1500 条约 1MB，几十个评估点会白占几十 MB，
        而排查时真正要看的是当前最优模型错在哪。

        worst 样本按任务各取一份：混合训练时翻译的字符错误率天然比转写高，
        统一排序会让最差列表被翻译占满，转写退化的迹象一条都看不到。
        """
        import json as _json

        from s2s_train.common.eval_gen import ALL_TASKS, ASR_TASKS

        try:
            d = self._out_dir(args) / "gen_eval_cases"
            d.mkdir(parents=True, exist_ok=True)
            order = {t: i for i, t in enumerate(ALL_TASKS)}
            cases = sorted(
                cases, key=lambda c: (order.get(c.get("task", ""), 9), -c["err_rate"])
            )

            # 强制 dump 模式带上 step 号，多个评估点各留一份才能看趋势；
            # 正常模式仍用固定名，只保留当前 best（省磁盘，见 docstring）
            import os as _os

            name = (
                f"cases_step{state.global_step}.jsonl"
                if _os.environ.get("GEN_EVAL_ALWAYS_DUMP_CASES", "") == "1"
                else "best_cases.jsonl"
            )
            with (d / name).open("w", encoding="utf-8") as f:
                f.write(
                    _json.dumps(
                        {"_step": state.global_step, "_stats": stats}, ensure_ascii=False
                    )
                    + "\n"
                )
                for c in cases:
                    f.write(_json.dumps(c, ensure_ascii=False) + "\n")

            # 每个任务各出「最差 25 + 随机 5 条正常的」纯文本，方便快速扫。
            # 光看最差会误判整体水平，所以夹几条中位样本作参照。
            with (d / "worst_by_task.txt").open("w", encoding="utf-8") as f:
                f.write(f"step {state.global_step}\n{_json.dumps(stats, ensure_ascii=False)}\n\n")
                for task in ALL_TASKS:
                    sub = [c for c in cases if c.get("task") == task]
                    if not sub:
                        continue
                    metric = (
                        f"acc={stats.get(f'{task}_acc', -1):.4f}"
                        if task in ASR_TASKS
                        else f"bleu={stats.get(f'{task}_bleu', -1):.2f}"
                    )
                    f.write(f"===== {task}  n={len(sub)}  {metric} =====\n\n")
                    picked = sub[:25] + sub[len(sub) // 2 : len(sub) // 2 + 5]
                    for c in picked:
                        f.write(
                            f"[{task}] err {c['errors']}/{c['ref_chars']} "
                            f"({c['err_rate']:.2%})  {c['audio']}\n"
                            f"  REF: {c['ref']}\n  HYP: {c['hyp']}\n\n"
                        )
        except Exception as exc:  # noqa: BLE001 - 写 case 失败不该影响训练
            print(f"[gen-eval] 写 case 失败: {exc}", flush=True)

    def _write_log(self, args, state, stats: dict, eval_loss=None) -> None:
        """把生成式指标追加到 <eval_dir 或 output_dir>/gen_eval.jsonl。

        单独一个文件而不是塞进 trainer_log.jsonl：后者由 LogCallback 用线程池
        异步追加，插队写同一个文件会有交错风险；而且它只写自己白名单里的键。
        """
        import json as _json

        row = {"step": state.global_step, "epoch": round(state.epoch or 0.0, 4)}
        row.update({k: stats[k] for k in self.METRIC_KEYS if k in stats})
        for task in ("asr_zh", "asr_en", "mt_en2zh", "mt_zh2en"):
            if f"{task}_samples" in stats:
                row[f"{task}_samples"] = stats[f"{task}_samples"]
        if eval_loss is not None:
            row["eval_loss"] = round(float(eval_loss), 6)

        try:
            path = self._out_dir(args) / "gen_eval.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(row, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001 - 写日志失败不该影响训练
            print(f"[gen-eval] 写 gen_eval.jsonl 失败: {exc}", flush=True)


class ResetBestMetricCallback(TrainerCallback):
    """续训时清掉上一个口径留下的 best_metric。

    换 metric_for_best_model 的口径（比如 eval_loss -> eval_mean_acc）时，
    checkpoint 的 trainer_state.json 里存的还是旧口径的数值。
    Trainer 的 _determine_best_metric 只在 best_metric 为 None 时才用
    ∓inf 初始化，于是 np.greater(准确率 0.65, 旧 loss 0.92) 恒为假，
    best_global_step 永远不更新，load_best_model_at_end 最后会把模型
    回滚到续训起点 —— 白跑几天。而 HF 的
    compare_trainer_and_checkpoint_args 不检查这个字段，全程无警告。

    best_model_checkpoint 也必须一起清：否则即使始终没有新最优，
    _load_best_model 仍会按它指向的旧路径回滚。
    """

    def __init__(self, expected_metric: str = "", greater_is_better: bool = False) -> None:
        self.expected_metric = expected_metric
        self.greater_is_better = greater_is_better

    def on_train_begin(self, args, state, control, **kwargs):
        if state.best_metric is None:
            return control

        old = state.best_metric
        # 所有 rank 都要改：state 是各进程独立的，_determine_best_metric 也在每个 rank 上跑
        state.best_metric = None
        state.best_global_step = None
        state.best_model_checkpoint = None
        if state.is_world_process_zero:
            print(
                f"[reset-best] 最优指标口径已改为 {self.expected_metric}"
                f"（greater_is_better={self.greater_is_better}），"
                f"清空 checkpoint 里的旧值 best_metric={old}。首次评估后重新确立最优。",
                flush=True,
            )

        return control
