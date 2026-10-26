"""hy_v3 (Hy-MT2-30B-A3B, MoE) 在 ZeRO-3 下训练所需的运行时补丁。

必须在**建模型之前** import 本模块（补丁改的是类方法）。
launcher 里已按 variant 自动调用 apply_hy_v3_patches()，一般不用手动管。

三个补丁的来源与依据
--------------------
补丁 1/2 移植自官方 train/llama_factory_support/hy_v3_patches.py
（Hy-MT2-30B-A3B 的 HF repo 自带，给 LlamaFactory 用的适配）。
补丁 3 是我们自己加的，官方没做 —— 理由见下。

1. Router gate 的 dtype 对齐（官方 Patch 2）—— 不打这个补丁会**直接崩**
   transformers 的 HYV3TopKRouter.forward 里写的是
       F.linear(hidden_states.float(), self.weight.float())
   而 DeepSpeed ZeRO-3 把 F.linear 换成了 zero3_linear_wrap，其内部做
   input.matmul(weight.t()) 时**不对齐 dtype**。ZeRO-3 下 gate 权重是 bf16、
   输入被 .float() 成 fp32 -> RuntimeError: expected scalar type ...
   修法：先把输入 cast 到 weight.dtype 再 linear，输出再回 fp32
   （sigmoid 需要 fp32 保证数值稳定）。

2. gradient_checkpointing 的 use_reentrant=True（官方 Patch 3）
   PyTorch 默认 use_reentrant=False 会对重算张量做严格 metadata 检查，
   而 ZeRO-3 下参数在第一次 forward 时 all-gather、重算时可能已被
   partition 回去 -> CheckpointError。
   我们在 YAML 里已经写了 gradient_checkpointing_kwargs.use_reentrant: true，
   这里作为兜底再查一次（配置被覆盖时仍能生效）。

3. z3_leaf_modules（官方**没做**，我们可选做）
   注意：这一项与我最初的判断不同，先说清楚以免误导 ——
   set_z3_leaf_modules 解决的是「MoE 每次 forward 的专家计算顺序不同，
   打乱 ZeRO-3 依赖的顺序追踪、导致预取失效」，**不是减少通信量**。
   transformers 5.6 的 HYV3Experts 把 128 个专家存成两个大 3D nn.Parameter
   （gate_up_proj (128,1536,2048) / down_proj (128,2048,768)），而不是
   per-expert ModuleList（官方补丁注释里写着 "Patch 5 (Removed) -- was
   per-expert ModuleList, now using native 3D Parameters"）。
   ZeRO-3 按 Parameter 粒度分片，这两个张量无论设不设 leaf 都是整体
   all-gather，所以**省不了通信量**；可能的收益只在 hook 开销与预取顺序上。
   官方没做很可能就是因为换成 3D Parameter 后必要性下降。
   默认关闭，用 S2S_HY_V3_Z3_LEAF=1 打开做 A/B 实测。
"""

import logging
import os

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_router_patched = False


def _patch_router_dtype() -> bool:
    """把 HYV3TopKRouter.forward 的 gate 输入 cast 到权重 dtype。

    返回是否成功打上（模型未安装/版本不符时返回 False 并告警，不抛异常 ——
    非 hy_v3 的训练不该被这个补丁影响）。
    """
    global _router_patched
    if _router_patched:
        return True

    try:
        from transformers.models.hy_v3.modeling_hy_v3 import HYV3TopKRouter
    except ImportError:
        logger.warning(
            "导入 HYV3TopKRouter 失败（需要 transformers>=5.6），router dtype 补丁未应用。"
        )
        return False

    def _forward(self, hidden_states: torch.Tensor, e_score_correction_bias: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # 关键差异：不硬编码 .float()，而是对齐到权重 dtype。
        # ZeRO-3 下 weight 是 bf16，用 fp32 输入会让 zero3_linear_wrap 里的
        # matmul 报 dtype 不匹配。
        weight_dtype = self.weight.dtype
        router_logits = F.linear(
            hidden_states.to(weight_dtype), self.weight.to(weight_dtype)
        )
        # sigmoid 前回到 fp32：路由权重对数值精度敏感
        router_logits = router_logits.to(torch.float32)
        routing_weights = torch.sigmoid(router_logits)

        scores_for_choice = routing_weights + e_score_correction_bias
        _, top_k_index = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)
        top_k_weights = routing_weights.gather(1, top_k_index)

        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-20)
        top_k_weights = top_k_weights * self.router_scaling_factor

        return router_logits, top_k_weights, top_k_index

    HYV3TopKRouter.forward = _forward
    _router_patched = True
    logger.info("hy_v3 补丁：HYV3TopKRouter.forward 已对齐 dtype（ZeRO-3 必需）")
    return True


def _patch_decode_experts_switch() -> bool:
    """禁掉 generate 时把 MoE 专家实现从 grouped_mm 切到 batched_mm 的优化。

    transformers 5.x 的 GenerationMixin._optimize_model_for_decode()
    （generation/utils.py:2077）在解码前会切成 batched_mm，理由是它对小输入
    更快 —— 代价是**复制被选中的专家权重**
    （源码注释：duplicating selected experts weights）。

    在 ZeRO-3 下这条路走不通：
      - 复制专家权重要触及全部 128 个专家的分片参数，触发隐式 all-gather；
        它在 generate 的上下文管理器里执行，各 rank 进入时机不同 ->
        **全体永久挂死**（实测 8 个节点的日志都停在
        "We will be switching to 'batched_mm'" 这一行，GPU 100% 零进展 8 小时）
      - 而如果反过来把 batched_mm 设成常驻（我最初的错误修法），
        那份复制的专家权重会在训练期间一直占着显存 -> CUDA OOM
        （实测 Tried to allocate 11.25 GiB / 25.50 GiB）

    所以正解是**保持 grouped_mm 不变**，只把那个上下文管理器换成空操作：
    训练和生成都走同一套实现，既不挂死也不多占显存。
    代价是解码速度略慢于 batched_mm —— dev 评测每 N 步才跑一次，可以接受。
    """
    try:
        from transformers.generation.utils import GenerationMixin
    except Exception:  # noqa: BLE001 - 版本不同时静默跳过
        return False

    if getattr(GenerationMixin, "_s2s_decode_switch_patched", False):
        return True

    from contextlib import contextmanager

    @contextmanager
    def _noop_optimize(self):  # noqa: ANN001, ANN202
        # 原实现会 set_experts_implementation("batched_mm")，ZeRO-3 下不可用
        yield

    GenerationMixin._optimize_model_for_decode = _noop_optimize
    GenerationMixin._s2s_decode_switch_patched = True
    return True


def apply_hy_v3_patches() -> None:
    """应用建模层补丁。必须在实例化模型之前调用。"""
    _patch_router_dtype()
    _patch_decode_experts_switch()


def ensure_reentrant_checkpointing(training_args) -> None:
    """兜底把 gradient_checkpointing_kwargs.use_reentrant 设为 True。

    YAML 里已经配了，这里防的是配置被别处覆盖的情况。
    只在同时开了 gradient_checkpointing 与 deepspeed 时才动 —— 单卡
    非 ZeRO-3 场景下 use_reentrant=False 是更好的默认（支持更多特性）。
    """
    if not getattr(training_args, "gradient_checkpointing", False):
        return
    if not getattr(training_args, "deepspeed", None):
        return

    kwargs = getattr(training_args, "gradient_checkpointing_kwargs", None) or {}
    if kwargs.get("use_reentrant") is not True:
        kwargs["use_reentrant"] = True
        training_args.gradient_checkpointing_kwargs = kwargs
        logger.info("hy_v3 补丁：gradient_checkpointing use_reentrant=True（ZeRO-3 必需）")


def maybe_set_z3_leaf_modules(model) -> bool:
    """可选：把 HYV3MoE 标为 ZeRO-3 的 leaf module。

    默认不做（官方也没做）。设 S2S_HY_V3_Z3_LEAF=1 打开，用于 A/B 实测。

    为什么默认关：见模块顶部第 3 条 —— HYV3Experts 是 3D Parameter 而非
    per-expert ModuleList，设 leaf 省不了 all-gather 的通信量，收益不明确。
    与其凭猜测开着，不如留开关、用实测数据决定。
    """
    if os.environ.get("S2S_HY_V3_Z3_LEAF", "0") != "1":
        return False

    try:
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.hy_v3.modeling_hy_v3 import HYV3MoE
    except ImportError as exc:
        logger.warning("z3_leaf 未应用（导入失败：%s）", exc)
        return False

    try:
        hit = set_z3_leaf_modules(model, [HYV3MoE])
    except Exception as exc:  # noqa: BLE001 - 加速项失败不该杀掉训练
        logger.warning("z3_leaf 应用失败：%s", exc)
        return False

    logger.info("hy_v3 补丁：已将 %d 个 HYV3MoE 设为 ZeRO-3 leaf module", len(hit or []))
    return True
