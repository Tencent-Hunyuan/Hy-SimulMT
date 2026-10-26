"""Hy-SimulMT Audio Encoder + Speech Adapter + LLM 组合模型。

与具体 LLM 无关，可用于任意尺寸的decoder-only 模型。

关键点：
1. 声学编码器默认冻结，且不写入 checkpoint。
2. Adapter 输出按 `<audio>` 占位 token 位置替换文本 embedding，
   再以 `inputs_embeds` 交给 LLM，不改动 LLM 内部实现。
3. 使用 feature_attention_mask 屏蔽 30 秒补零区域。
"""

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_speech_llm import SpeechLLMConfig
from .modeling_speech_adapter import build_speech_adapter


def _load_qwen3_asr_encoder(
    model_path: str, dtype: torch.dtype, attn_impl: str | None = None
) -> nn.Module:
    """加载 Qwen3-ASR 的 audio_tower 作为声学编码器。

    checkpoint 里的键带 `thinker.audio_tower.` 前缀，
    不剥掉的话 load_state_dict 会全部 missing 而只打警告，模型照样能跑但输出噪声。
    所以手动剥前缀并校验没有残留 meta 张量。

    建模代码来自 s2s_train.vendor_qwen3_asr（vendored 自 pip 包 qwen-asr）——
    transformers 4.57.6 不识别 model_type='qwen3_asr'，模型目录里也没有
    modeling_*.py，只能自带。

    维度要点：输出是 config.output_dim=2048，不是 d_model=1024。
    audio_tower 末尾有 proj1(d_model->d_model) -> proj2(d_model->output_dim)。
    ENCODER_SPECS 里按 2048 登记，launcher 会校验。
    """
    import glob
    import json
    from pathlib import Path

    from safetensors.torch import load_file

    from s2s_train.vendor_qwen3_asr import Qwen3ASRAudioEncoder, Qwen3ASRAudioEncoderConfig

    model_dir = Path(model_path)
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    try:
        audio_cfg = raw["thinker_config"]["audio_config"]
    except KeyError as exc:
        raise RuntimeError(
            f"{model_dir}/config.json 里找不到 thinker_config.audio_config，"
            "确认这是 Qwen3-ASR checkpoint。"
        ) from exc

    # 过滤 _name_or_path 这类内部字段，避免 PretrainedConfig 报未知参数
    config = Qwen3ASRAudioEncoderConfig(
        **{k: v for k, v in audio_cfg.items() if not k.startswith("_")}
    )
    if attn_impl:
        config._attn_implementation = attn_impl

    # 先按 meta device 建骨架，避免白白初始化一遍再被覆盖
    with torch.device("meta"):
        encoder = Qwen3ASRAudioEncoder(config)

    prefix = "thinker.audio_tower."
    state: dict[str, torch.Tensor] = {}

    shard_files = sorted(model_dir.glob("model-*.safetensors"))
    single = model_dir / "model.safetensors"
    if single.is_file():
        shard_files = [single]
    if not shard_files:
        raise FileNotFoundError(f"{model_dir} 下找不到 model*.safetensors")

    for shard in shard_files:
        for key, tensor in load_file(str(shard)).items():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = tensor.to(dtype)

    if not state:
        raise RuntimeError(
            f"未能从 {model_dir} 提取到任何 '{prefix}*' 权重，"
            "请确认这是标准的 Qwen3-ASR checkpoint。"
        )

    missing, unexpected = encoder.load_state_dict(state, strict=False, assign=True)
    # 剩下的 meta 张量说明真的没加载上，必须报错而不是放过
    still_meta = [name for name, p in encoder.named_parameters() if p.is_meta]
    if still_meta:
        raise RuntimeError(
            f"Qwen3-ASR encoder 有 {len(still_meta)} 个权重未加载"
            f"（示例：{still_meta[:5]}），missing={len(missing)} unexpected={len(unexpected)}"
        )

    # 非持久 buffer（正弦位置编码）不在 state_dict 里，meta 上建出来是空壳，
    # 必须按原公式重算 —— 不能用 zeros_like 顶替。
    #
    # 踩过的坑（会静默毁掉训练，不报错）：
    # 原先这里写 `setattr(mod, leaf, torch.zeros_like(buf, device="cpu"))`，
    # 两个后果：
    #   1. 位置编码全为 0，等于编码器完全没有位置信息；
    #   2. 钉在 CPU 上，而 modeling_qwen3_asr.py:713 只 .to(dtype) 不 .to(device)
    #      （它假设 buffer 已随模型 .cuda()，但手动 setattr 的 buffer 不会被
    #      register 之后的 .to() 追踪到吗 —— 会被追踪，问题是 DeepSpeed 初始化
    #      时机在此之后，实际表现为 forward 里
    #      "Expected all tensors to be on the same device, cuda:0 and cpu"）。
    # 第 2 点让它崩了所以被发现；第 1 点才是真正危险的，因为它不会报错。
    #
    # 正解：重新实例化持有该 buffer 的子模块（在真实 device 上），让
    # SinusoidsPositionEmbedding.__init__ 自己把正弦值算出来。
    from s2s_train.vendor_qwen3_asr.modeling_qwen3_asr import SinusoidsPositionEmbedding

    for module in encoder.modules():
        if isinstance(module, SinusoidsPositionEmbedding):
            buf = module.positional_embedding
            if not buf.is_meta:
                continue
            length, channels = buf.shape
            # 在 CPU 上重算（值与 device 无关），后续 .to(device) 会带走它
            rebuilt = SinusoidsPositionEmbedding(length, channels)
            module.register_buffer(
                "positional_embedding",
                rebuilt.positional_embedding.to(dtype),
                persistent=False,
            )

    # 兜底：还有 meta buffer 说明出现了未知的非持久 buffer，宁可报错也不要
    # 用零值糊过去（前车之鉴见上）。
    left = [name for name, b in encoder.named_buffers() if b.is_meta]
    if left:
        raise RuntimeError(
            f"Qwen3-ASR encoder 仍有 meta buffer 未实体化: {left}。"
            "不要用 zeros 填充 —— 请按其原始定义重算。"
        )

    return encoder


def _load_speech_encoder(
    config: SpeechLLMConfig, dtype: torch.dtype, attn_impl: str | None = None
) -> nn.Module:
    """按 config.encoder_type 分派到具体编码器的加载函数。

    新增编码器时：在 configuration_speech_llm.ENCODER_SPECS 登记规格，
    再在这里加一个分支。不要在别处零散判断 encoder_type。

    attn_impl 是外层 SpeechLLMConfig 上的 _attn_implementation。必须显式
    往下传：子模块的 config 是这里新建的，不传就默认 sdpa，外层设了
    flash_attention_2 也只是空转。
    """
    if getattr(config, "merged_model_path", ""):
        from .merged_checkpoint import load_merged_audio_encoder

        return load_merged_audio_encoder(config.merged_model_path, dtype, attn_impl)
    encoder_type = getattr(config, "encoder_type", "hy_simulmt_audio")
    if encoder_type != "hy_simulmt_audio":
        raise ValueError(f"未实现的 encoder_type={encoder_type!r}")
    return _load_qwen3_asr_encoder(config.encoder_path, dtype, attn_impl)


class SpeechLLMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SpeechLLMConfig
    base_model_prefix = "language_model"
    supports_gradient_checkpointing = True
    # transformers 5.x 把这个标记从 _supports_flash_attn_2 改名成
    # _supports_flash_attn，只声明旧名会在 __init__ 里直接抛
    # "does not support Flash Attention 2 yet"。两个名字都留：4.57 认旧名。
    # 声明为 True 是合法的 —— 本类自己不做 attention，三个子模块
    # (audio encoder / HYV3 language model)
    # 都已声明支持 FA2 与 SDPA，attn_implementation 会透传给它们。
    _supports_flash_attn = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
        self,
        config: SpeechLLMConfig,
        speech_encoder: nn.Module | None = None,
        language_model: PreTrainedModel | None = None,
    ) -> None:
        super().__init__(config)
        if speech_encoder is None:
            # 只建骨架、不灌权重（from_config 路径，权重由调用方后续 load）。
            # 建空骨架后由调用方灌入声学编码器权重。
            speech_encoder = self._build_empty_encoder()

        if language_model is None:
            from transformers import AutoConfig

            llm_config = AutoConfig.from_pretrained(config.llm_path, local_files_only=True)
            attn_impl = getattr(config, "_attn_implementation", None)
            if attn_impl:
                llm_config._attn_implementation = attn_impl
            language_model = AutoModelForCausalLM.from_config(llm_config)

        self.speech_encoder = speech_encoder
        self.speech_adapter = build_speech_adapter(config)
        self.language_model = language_model

        # 默认冻结编码器；config.train_encoder=true 时放开。
        # 放开时还要保证 _encode_speech 不套 no_grad —— 那边读的是同一个开关。
        if not getattr(config, "train_encoder", False):
            self.speech_encoder.requires_grad_(False)
        # patch_model 会读取 generation_config，复用 LLM 的即可
        self.generation_config = self.language_model.generation_config

    def _build_empty_encoder(self) -> nn.Module:
        """按 encoder_type 建对应编码器的空骨架（不灌权重）。

        供 from_config 路径使用（离线评测脚本会先 from_config 再逐部分灌权重）。
        权重由调用方负责，这里只保证结构对得上。
        """
        encoder_type = getattr(self.config, "encoder_type", "hy_simulmt_audio")
        attn_impl = getattr(self.config, "_attn_implementation", None)
        if encoder_type == "hy_simulmt_audio":
            import json
            from pathlib import Path

            from s2s_train.vendor_qwen3_asr import (
                Qwen3ASRAudioEncoder,
                Qwen3ASRAudioEncoderConfig,
            )

            raw = json.loads(
                (Path(self.config.encoder_path) / "config.json").read_text(encoding="utf-8")
            )
            audio_cfg = raw["thinker_config"]["audio_config"]
            cfg = Qwen3ASRAudioEncoderConfig(
                **{k: v for k, v in audio_cfg.items() if not k.startswith("_")}
            )
            if attn_impl:
                cfg._attn_implementation = attn_impl
            return Qwen3ASRAudioEncoder(cfg)

        raise ValueError(f"未实现的 encoder_type={encoder_type!r}")

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        from pathlib import Path

        from transformers import AutoConfig

        config = kwargs.pop("config", None)
        if config is None:
            config = SpeechLLMConfig.from_pretrained(pretrained_model_name_or_path)

        dtype = kwargs.pop("torch_dtype", None) or kwargs.pop("dtype", None)
        if dtype == "auto" or dtype is None:
            dtype = torch.bfloat16

        # 与 LlamaFactory 传入的 HF 通用参数解耦，只保留本模型需要的
        for key in ("trust_remote_code", "cache_dir", "revision", "token", "pretrained_model_name_or_path"):
            kwargs.pop(key, None)

        # attn_impl 要手动往下传给两个子模型：它们的 config 是在这里新建的，
        # 不继承外层 SpeechLLMConfig 的 _attn_implementation，
        # 不传就静默回落 sdpa（yaml 里的 flash_attn: fa2 等于没设）。
        attn_impl = getattr(config, "_attn_implementation", None)
        speech_encoder = _load_speech_encoder(config, dtype, attn_impl)

        llm_config = AutoConfig.from_pretrained(config.llm_path, local_files_only=True)
        language_model = AutoModelForCausalLM.from_pretrained(
            config.llm_path,
            config=llm_config,
            dtype=dtype,
            local_files_only=True,
            **({"attn_implementation": attn_impl} if attn_impl else {}),
        )

        model = cls(config, speech_encoder=speech_encoder, language_model=language_model)
        model.speech_adapter.to(dtype=dtype)

        if getattr(config, "merged_model_path", ""):
            from .merged_checkpoint import load_merged_adapter

            load_merged_adapter(model.speech_adapter, config.merged_model_path, dtype)
            # Do not return here. A checkpoint-local speech_adapter_path must
            # override the adapter embedded in the original merged export.
            # This is how adapter-only outputs and best/ mirrors are restored.

        adapter_path = config.speech_adapter_path
        if adapter_path:
            weights = Path(adapter_path)
            if weights.is_dir():
                weights = weights / "speech_adapter.safetensors"

            from safetensors.torch import load_file

            state = load_file(str(weights))
            missing, unexpected = model.speech_adapter.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(f"Adapter 权重不匹配: missing={missing}, unexpected={unexpected}")

        return model

    # ------------------------------------------------------------------
    # 与 LLM 对齐的常用接口
    # ------------------------------------------------------------------
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.language_model.gradient_checkpointing_enable(**kwargs)

    # ------------------------------------------------------------------
    # 语音编码
    # ------------------------------------------------------------------
    def _encode_speech(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """把 mel 特征编码成 Adapter 输出的语音 embedding。

        编码器使用拼接的 2D 输入并返回拼接的 2D 输出；这里将其统一成
        「(B, T, encoder_dim) + padding_mask(True=屏蔽)」再交给 Adapter。

        返回值的形状取决于 adapter_type：
          qformer      (B, num_speech_tokens, llm_dim)  每条一样长
          fsk          (ΣN_i, llm_dim)  已剔除 padding 的拼接张量

        变长时必须在这里剔除 padding：下游 _merge_speech_into_text 是
        `inputs_embeds[mask] = speech_embeds.reshape(-1, D)`，按行优先平铺，
        padding 位不剔掉就会被当成有效 embedding 写进去（数量还会对不上）。
        """
        speech_states, padding_mask = self._encode_speech_qwen3_asr(
            input_features, feature_attention_mask
        )

        speech_embeds = self.speech_adapter(speech_states, padding_mask)

        if getattr(self.config, "adapter_type", "qformer") == "qformer":
            return speech_embeds

        # fsk：按每条的真实输出长度筛出有效帧后拼接。
        # padding_mask 是编码器输出粒度 (B, T)，而 Adapter 做了 k 帧拼接，
        # 所以有效长度要按同样的向上取整换算，不能直接复用 mask。
        if padding_mask is None:
            return speech_embeds.reshape(-1, speech_embeds.size(-1))

        k = self.config.fsk_size
        valid_frames = (~padding_mask).sum(dim=1)           # 每条的编码器输出帧数
        valid_tokens = (valid_frames + k - 1) // k          # 与 speech_tokens_for 同源
        positions = torch.arange(speech_embeds.size(1), device=speech_embeds.device)
        keep = positions.unsqueeze(0) < valid_tokens.unsqueeze(1)
        return speech_embeds[keep]

    def _encoder_grad_ctx(self):
        """编码器前向的梯度上下文。

        train_encoder=false 时用 no_grad 省激活值内存；true 时必须用
        nullcontext —— 否则即使 requires_grad=True，no_grad 也会切断计算图，
        编码器永远收不到梯度（这种错不报任何异常，只是白训）。
        """
        from contextlib import nullcontext

        if getattr(self.config, "train_encoder", False):
            return nullcontext()
        return torch.no_grad()

    def _encode_speech_qwen3_asr(
        self,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Qwen3-ASR 的 packed I/O 适配。

          输入   (mel_bins, ΣT) 沿时间轴拼接、不 padding
                 —— 内部按 n_window 切块，要求总长恰好等于 sum(feature_lens)
          长度   feature_lens (B,) 每条真实 mel 帧数，而非 attention_mask
          输出   (ΣT', output_dim) 同样是拼接的，需按每条的输出长度拆回

        输出帧数随时长变（压缩比约 7.7x，
        实测 1700 mel 帧 -> 221 帧），所以拆回后要 pad 到本 batch 最长。
        """
        target_dtype = self.speech_adapter.fc_in.weight.dtype

        if feature_attention_mask is None:
            # 没给 mask 说明整段都有效
            batch, _, frames = input_features.shape
            feature_lens = torch.full(
                (batch,), frames, dtype=torch.long, device=input_features.device
            )
        else:
            feature_lens = feature_attention_mask.sum(dim=-1).to(torch.long)

        # 逐条裁掉 padding 再沿时间轴拼接
        packed = torch.cat(
            [input_features[i, :, : int(n)] for i, n in enumerate(feature_lens)], dim=-1
        )

        encoder_dtype = self.speech_encoder.conv2d1.weight.dtype
        with self._encoder_grad_ctx():
            out = self.speech_encoder(
                packed.to(dtype=encoder_dtype), feature_lens=feature_lens
            )
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out
        hidden = hidden.to(dtype=target_dtype)

        # 每条的输出帧数：用编码器自己的换算函数，别自己推导 stride
        from s2s_train.vendor_qwen3_asr.modeling_qwen3_asr import (
            _get_feat_extract_output_lengths,
        )

        out_lens = _get_feat_extract_output_lengths(feature_lens).to(torch.long)
        total = int(out_lens.sum())
        if hidden.size(0) != total:
            raise RuntimeError(
                f"Qwen3-ASR 输出帧数 {hidden.size(0)} 与换算值 {total} 不一致"
                f"（feature_lens={feature_lens.tolist()}），拆包会错位。"
            )

        # 拆回每条并 pad 到本 batch 最长，得到 Adapter 期望的 (B, T, D)
        chunks = list(torch.split(hidden, out_lens.tolist(), dim=0))
        speech_states = torch.nn.utils.rnn.pad_sequence(chunks, batch_first=True)

        # padding_mask 沿用本项目约定：True = 需屏蔽
        positions = torch.arange(speech_states.size(1), device=speech_states.device)
        padding_mask = positions.unsqueeze(0) >= out_lens.unsqueeze(1).to(
            speech_states.device
        )

        return speech_states, padding_mask

    def _merge_speech_into_text(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        speech_embeds: torch.Tensor,
    ) -> torch.Tensor:
        audio_token_id = self.config.audio_token_id
        if audio_token_id is None:
            raise ValueError("config.audio_token_id 未设置，无法定位语音占位符。")

        placeholder_mask = input_ids == audio_token_id
        # speech_embeds 可能是 (B, N, D)（定长）或 (ΣN_i, D)（变长，已剔 padding）
        expected = speech_embeds.numel() // speech_embeds.size(-1)
        found = int(placeholder_mask.sum())
        if found != expected:
            raise ValueError(
                f"语音占位符数量不匹配：占位符 {found} 个，Adapter 输出 {expected} 个。"
                "变长模式下这通常意味着 mm_plugin 的 speech_tokens_for 与 Adapter "
                "的输出长度算法不同源（比如取整方式不一致），两边必须都走 "
                "config.speech_tokens_for()。"
            )

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[placeholder_mask] = speech_embeds.reshape(-1, speech_embeds.size(-1)).to(
            inputs_embeds.dtype
        )
        return inputs_embeds

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        input_features: torch.Tensor | None = None,
        feature_attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            if input_features is not None:
                speech_embeds = self._encode_speech(input_features, feature_attention_mask)
                inputs_embeds = self._merge_speech_into_text(input_ids, inputs_embeds, speech_embeds)

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_from_speech(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        input_features: torch.Tensor,
        feature_attention_mask: torch.Tensor | None = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """语音 + 文本提示 -> 生成 token。

        不能直接复用 forward 走GenerationMixin：开启 KV cache 后
        第二步之后的 input_ids 只剩 1 个 token，占位符数量校验会失败。
        这里先一次性融合好 inputs_embeds，再交给 LLM 自己的 generate。
        """
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if input_features is not None:
            speech_embeds = self._encode_speech(input_features, feature_attention_mask)
            inputs_embeds = self._merge_speech_into_text(input_ids, inputs_embeds, speech_embeds)

        return self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    # ------------------------------------------------------------------
    # 保存：冻结的声学编码器（可选还有 LLM）不写入 checkpoint
    # ------------------------------------------------------------------
    def _omitted_state_prefixes(self) -> tuple[str, ...]:
        """state_dict() 会刻意剔掉哪些前缀。

        抽成一处是因为 load_state_dict() 必须容忍**完全相同**的这几类缺失。
        两边各写一份的话迟早漂移 —— 2026-08-30 就踩过：先只在加载侧放宽了
        speech_encoder.，结果续训改报 language_model.* 缺失，又崩一次。
        """
        prefixes: list[str] = []
        # 编码器冻结时不写入，
        # 推理时从 encoder_path 现加载即可。
        # train_encoder=true 时**必须保留** —— 否则训完的编码器权重丢了，
        # 推理还是加载原始权重，等于这一阶段白训（而且不会有任何报错）。
        if not getattr(self.config, "train_encoder", False):
            prefixes.append("speech_encoder.")
        # LLM 也冻结时（阶段 A），它在 checkpoint 里只是原始权重的拷贝，
        # 约 3.6GB 纯冗余。用 config.save_frozen_llm=false 可以一并剔除，
        # 此时产物只有 speech_adapter.safetensors，推理照常从 llm_path 加载 LLM。
        if not getattr(self.config, "save_frozen_llm", True):
            prefixes.append("language_model.")
        return tuple(prefixes)

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        state = super().state_dict(*args, **kwargs)
        drop = self._omitted_state_prefixes()
        if "language_model." in drop:
            # 只有真冻结才剔 LLM：解冻着还剔就等于把训练成果丢掉
            if any(p.requires_grad for p in self.language_model.parameters()):
                drop = tuple(p for p in drop if p != "language_model.")
        if drop:
            state = {key: value for key, value in state.items() if not key.startswith(drop)}
        return state

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):  # type: ignore[override]
        """与 state_dict() 配对：容忍**且仅容忍**上面刻意剔掉的那几类缺失。

        为什么需要：续训时 HF 走
        `deepspeed_load_checkpoint(engine, path, load_module_strict=True)`
        （transformers/integrations/deepspeed.py:646），strict 检查会因
            Missing key(s) in state_dict: "speech_encoder.layers.0..."
        直接 RuntimeError，整个续训起不来。2026-08-30 实测：8 机 64 卡从
        checkpoint-14000 续训，全 rank 都在加载阶段崩掉。

        这些权重不会丢：编码器与 LLM 在 from_pretrained 阶段已从 encoder_path /
        llm_path 加载；ZeRO-3 的冻结参数另有一份 frozen_param_fragments
        （checkpoint 里那 57GB 就是它），DeepSpeed 会单独恢复。

        只放宽这几类 —— Adapter 的 key 缺失、或出现任何多余 key，仍然照常报错。
        否则权重错配会被静默吞掉：那种 bug 不报错，只让指标莫名其妙地差，极难查。
        """
        incompatible = super().load_state_dict(state_dict, strict=False, assign=assign)
        if not strict:
            return incompatible

        allowed = self._omitted_state_prefixes()
        real_missing = [
            key for key in incompatible.missing_keys
            if not (allowed and key.startswith(allowed))
        ]
        if real_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"加载 {self.__class__.__name__} 的 state_dict 失败：\n"
                f"  缺失: {real_missing[:8]}{' ...' if len(real_missing) > 8 else ''}\n"
                f"  多余: {list(incompatible.unexpected_keys)[:8]}"
                f"{' ...' if len(incompatible.unexpected_keys) > 8 else ''}\n"
                f"（{', '.join(allowed) or '无'} 这几类缺失是正常的、已忽略；"
                "上面列的是别的问题）"
            )
        return incompatible._replace(missing_keys=[])
