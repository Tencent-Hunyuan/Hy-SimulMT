"""语音到文本组合模型的通用配置。

与具体 LLM 尺寸无关：Adapter 维度、语音token 数等都由参数决定，
LLM 相关的内容（词表占位符、chat 模板、hidden 维度）由各变体提供。

当前训练只支持 Hy-SimulMT 声学编码器，encoder_dim 必须与其真实输出
维度一致。
"""

from transformers import PretrainedConfig

# 各声学编码器的关键规格。
#
# 为什么要集中登记：encoder_dim 必须等于编码器最后一层的真实输出维度，
# 写错了 Adapter 的 fc_in 会静默按错的维度建（cross-attention 的
# encoder_hidden_size 对不上直接报错，或更糟——维度凑巧能过但语义错位）。
# 把规格写在一处并在 launcher 里用 assert 校验，比散落在脚本里安全。
#
#   dim           送进 Adapter cross-attention 的隐藏维度
#   mel_bins      特征提取器的 mel 通道数
#   frames_30s    30 秒音频编码后的帧数，决定 Adapter 要压缩多少倍
#   note          维度来源，改动前先核对 config
ENCODER_SPECS: dict[str, dict] = {
    "hy_simulmt_audio": {
        # 注意不是 audio_config.d_model(1024)：audio tower 末尾有
        # proj1(d_model->d_model) -> proj2(d_model->output_dim) -> conv_out，
        # 输出维度是 audio_config.output_dim=2048（已用真实权重前向实测确认）。
        # 用 1024 会让 cross-attention 的 encoder_hidden_size 差一倍。
        "dim": 2048,
        "mel_bins": 128,
        "frames_30s": 1500,
        "note": "audio_config.output_dim=2048（经 proj1/proj2 投影后），非 d_model=1024",
        # 输入 (128, ΣT) 沿时间轴拼接，输出 (ΣT', 2048)，随后按长度拆包。
        "packed_io": True,
    },
}


class SpeechLLMConfig(PretrainedConfig):
    """声学 Encoder + Speech Adapter + LLM 组合模型配置。

    model_type 由 Hy-SimulMT 变体子类覆盖，以便 LlamaFactory 的冻结与
    LoRA 保护逻辑生效。

    ---- 为什么字段用类注解声明（transformers 5.x 必需）----

    transformers 5.x 把 PreTrainedConfig 变成了「严格 dataclass」
    （configuration_utils.py 顶部 @strict(accept_kwargs=True)，来自
    huggingface_hub.dataclasses），基类只声明了 10 个 dataclass 字段。
    在 __init__ 里写 `self.xxx = ...` 设的属性会在实例化后消失 ——
    实测 5.8.0 下本类 19 个字段有 15 个丢失，症状是
        AttributeError: '...Config' object has no attribute 'use_pooling'
    （num_speech_tokens 读 use_pooling 时崩）。

    绕不过去的两种错法（都试过）：
      - super().__init__() 之后再赋值一遍：仍然丢
      - object.__setattr__ 硬写：仍然丢（问题不在 __setattr__）
      - 把字段塞进 kwargs 传给 super()：仍然丢

    正解是把字段声明成**类注解**，让它们成为 dataclass 字段。
    这种写法在 4.57 上等价于普通类属性（默认值），完全向后兼容 ——
    两个环境都实测通过。

    改动这些字段时：直接改下面的注解，不要再写 __init__。
    """

    model_type = "speech_llm"

    # ---- 路径 ----
    # 声学编码器路径。
    encoder_path: str = ""
    llm_path: str = ""
    speech_adapter_path: str | None = None
    # Non-empty only for the exported HYV3SpeechForCausalLM checkpoint.  It is
    # always treated as read-only; generated views/checkpoints live in output_dir.
    merged_model_path: str = ""

    # ---- 编码器 ----
    encoder_dim: int = 2048
    encoder_type: str = "hy_simulmt_audio"
    # 是否训练声学编码器。默认 False（历史行为：编码器全程冻结）。
    #
    # 设为 True 时模型侧会做两件事，缺一件编码器都训不起来：
    #   1. 不调 speech_encoder.requires_grad_(False)
    #   2. _encode_speech 不套 torch.no_grad() —— 这条更关键，
    #      套了 no_grad 即使 requires_grad=True 梯度也传不回编码器
    # 另外 state_dict() 会改为保留编码器权重，否则训了也存不下来。
    #
    # 注意还要同时让 LlamaFactory 别冻它：freeze_vision_tower 默认 True
    #（register.py 把 speech_encoder 注册成了 vision_model_keys）。
    # run.sh 里用 LLM_MODE=full_with_encoder 会一并处理。
    train_encoder: bool = False

    # ---- Adapter 结构 ----
    # "qformer"      Q-Former + 固定 query_len 个输出（定长，历史默认）
    # "fsk"  相邻 fsk_size 帧拼接 + MLP（变长，输出随时长变）
    #
    # 两者的关键差别是**输出长度**：
    #   qformer      恒为 num_speech_tokens，与音频时长无关。3 秒和 30 秒
    #                的音频都出 query_len 个 token —— 短的被扩张、长的被压缩。
    #   fsk          = ceil(编码器输出帧数 / fsk_size)，随时长变。
    #                无 padding，短音频不浪费、长音频不丢信息。
    adapter_type: str = "qformer"
    # fsk 的时间压缩倍数。声学编码器输出约 13 帧/秒（77 ms/帧），
    # k=2 -> 6.5 帧/秒（154 ms/token）。
    # 实测时长分布（3000 条抽样）：p50 3.22s / p90 12.14s / max 45.24s
    #   k=1  p50 41 / p90 157 token   零损失但长音频撑爆序列
    #   k=2  p50 20 / p90  78 token   当前选择
    #   k=4  p50 10 / p90  39 token   p10(1.16s) 只剩 3 个 token，短句可能不够
    fsk_size: int = 2

    # ---- Q-Former（仅 adapter_type="qformer" 用）----
    qformer_hidden: int = 768
    qformer_layers: int = 8
    qformer_heads: int = 12
    qformer_ffn_dim: int = 3072
    # 每 cross_attention_frequency 层插一个 cross-attention（BLIP-2 默认 2）
    cross_attention_frequency: int = 2
    # BLIP-2 的 hidden/attention dropout，正则化用，参考实现为 0.1
    qformer_dropout: float = 0.1
    # BLIP-2 沿用 BERT 的 1e-12，与 PyTorch 默认的 1e-5 不同
    qformer_layer_norm_eps: float = 1e-12
    query_len: int = 150
    use_pooling: bool = False
    pool_stride: int = 5
    mlp_hidden: int = 2560

    # ---- LLM 侧 ----
    llm_dim: int = 2048
    audio_token_id: int | None = None
    # False 时，若 LLM 被冻结则不把它写进 checkpoint（省约 3.6GB 冗余拷贝）
    save_frozen_llm: bool = True

    @property
    def encoder_spec(self) -> dict:
        """当前编码器的规格。未登记的类型直接报错，不猜。"""
        if self.encoder_type not in ENCODER_SPECS:
            raise ValueError(
                f"未登记的 encoder_type={self.encoder_type!r}，"
                f"可选 {sorted(ENCODER_SPECS)}。新增时请在 ENCODER_SPECS 登记规格，"
                f"并在 modeling_speech_llm._load_speech_encoder 里加载分支。"
            )
        return ENCODER_SPECS[self.encoder_type]

    def validate_encoder_dim(self) -> None:
        """校验 encoder_dim 与所选编码器一致。

        为什么要显式校验：维度写错时 Adapter 仍能建起来（cross-attention 的
        encoder_hidden_size 只在 forward 才对齐），最坏情况是训练跑起来、
        loss 也降，但语音表征是错的。宁可在启动时就崩。
        """
        expected = self.encoder_spec["dim"]
        if self.encoder_dim != expected:
            raise ValueError(
                f"encoder_dim={self.encoder_dim} 与 encoder_type={self.encoder_type!r} "
                f"不匹配，应为 {expected}（{self.encoder_spec['note']}）"
            )

    @property
    def num_speech_tokens(self) -> int:
        """Adapter 输出的语音 token 数，需与数据侧占位符展开长度一致。

        仅对**定长** Adapter（Q-Former）有意义：关闭 pooling 时等于 query_len，
        开启时再按 pool_stride 压缩。

        变长 Adapter（adapter_type="fsk"）没有这个概念 —— 输出长度
        由每条音频的时长决定，读这个属性说明调用方还在假定定长，直接报错
        比返回一个会导致占位符数量对不上的数字要好。
        """
        if self.adapter_type != "qformer":
            raise AttributeError(
                f"adapter_type={self.adapter_type!r} 的输出长度随音频时长变化，"
                "没有固定的 num_speech_tokens。请改用 "
                "speech_tokens_for(mel_frames) 按条计算。"
            )

        if self.use_pooling:
            return self.query_len // self.pool_stride

        return self.query_len

    def speech_tokens_for(self, mel_frames: int) -> int:
        """给定 mel 帧数，算出这条音频最终占多少个语音 token。

        两级压缩：
          1. 编码器约 7.7x（每满 100 mel 帧出 13 帧，余数走三层卷积）
          2. Adapter：fsk 再按 fsk_size 拼接

        向上取整 —— 与 FSKAdapter 里的补齐行为一致（尾部不足 k 帧
        也要产出一个 token，不能丢）。两边必须用同一套算法，否则占位符
        数量与 Adapter 输出对不上，_merge_speech_into_text 会直接报错。
        """
        if self.encoder_type != "hy_simulmt_audio":
            raise ValueError(f"未实现的 encoder_type={self.encoder_type!r}")

        from s2s_train.vendor_qwen3_asr.modeling_qwen3_asr import (
            _get_feat_extract_output_lengths,
        )

        import torch

        enc_frames = int(_get_feat_extract_output_lengths(torch.tensor([mel_frames]))[0])

        if self.adapter_type == "qformer":
            return self.num_speech_tokens

        k = self.fsk_size
        return (enc_frames + k - 1) // k

    def speech_tokens_for_samples(self, n_samples: int, pad_to_max: bool = True) -> int:
        """给定波形采样点数，算出这条音频占多少个语音 token。

        这是调用方该用的**唯一入口** —— 采样数到 mel 帧的换算只在这里做。

        `pad_to_max` 必须与调用方给 feature_extractor 的 padding 一致：

            padding="max_length"  -> pad_to_max=True   取整用 ceil
            padding="longest"     -> pad_to_max=False  取整用 floor

        **这两种 padding 的取整规则真的不同**，用合成波形实测（同一条波形、
        只改 padding 参数）：

            采样 80001:  longest -> 500 (floor)   max_length -> 501 (ceil)
            采样 81408:  longest -> 508 (floor)   max_length -> 509 (ceil)
            采样 80000:  两者都是 500（整除时无差别，所以容易漏掉）

        踩过两次坑，都是「改对一边、弄坏另一边」：
          - 先按训练路径（max_length）写 ceil -> 评测路径（longest）报
            "占位符 23 个，Adapter 输出 22 个"
          - 改成 floor -> 训练路径又报 "占位符 18 个，Adapter 输出 19 个"
        单一取整方式无解，必须按 padding 分。

        30 秒上限：feature_extractor 的 nb_max_frames=3000
        （chunk_length=30 × 100 帧/秒），更长的音频被截断，mask 恒为 3000。
        实测 33s / 37.3s / 59.5s 都是 3000。本数据集 349 条（0.010%）超 30 秒。
        """
        hop_length = 160  # 16 kHz、hop 160
        max_mel_frames = 3000  # = feature_extractor.nb_max_frames

        if pad_to_max:
            mel_frames = -(-n_samples // hop_length)  # ceil
        else:
            mel_frames = n_samples // hop_length  # floor

        return self.speech_tokens_for(min(mel_frames, max_mel_frames))

    @property
    def hidden_size(self) -> int:
        """DeepSpeed ZeRO-3 需要用它推导 auto 配置项（bucket size 等）。

        对外暴露 LLM 的隐藏维度，与实际参与反向传播的主体保持一致。
        """
        return self.llm_dim
