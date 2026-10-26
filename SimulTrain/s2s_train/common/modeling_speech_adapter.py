"""MCAT (arXiv:2512.01512) 风格的语音 Adapter。

结构：Q-Former -> 可选AvgPool -> MLP

    Encoder hidden (B, T, encoder_dim)
      -> Q-Former  (B, query_len, qformer_hidden)
      -> AvgPool(B, query_len // pool_stride, qformer_hidden)
      -> MLP       (B, num_speech_tokens, llm_dim)

Q-Former 直接用 HF 官方的 `Blip2QFormerModel`，与参考实现
(https://github.com/yxduir/m2m-70)逐层对齐：post-norm、每
cross_attention_frequency 层一个 cross-attention、dropout 0.1、
LayerNorm eps 1e-12。自己手写容易在这些细节上偏离，进而偏离论文结论。

注意两件事：
1. mask 语义。本模块对外沿用 PyTorch 的约定（True=屏蔽），
   但 Blip2QFormerModel 要的是 HF 约定（1=保留），forward 里会翻转。
2. 参数命名。官方类内部叫 query/key/value/dense，正好落在
   LlamaFactory `lora_target: all` 的叶子名收集范围内，因此必须靠
   composite 注册里的 lora_conflict_keys 来隔离，不能只依赖命名规避。
"""

import torch
from torch import nn
from transformers import Blip2QFormerConfig, Blip2QFormerModel

from .configuration_speech_llm import SpeechLLMConfig


def build_qformer_config(config: SpeechLLMConfig) -> Blip2QFormerConfig:
    """把本框架的配置映射成 Blip2QFormerConfig。"""
    if config.qformer_hidden % config.qformer_heads != 0:
        raise ValueError(
            f"qformer_hidden({config.qformer_hidden}) 必须能被 "
            f"qformer_heads({config.qformer_heads}) 整除"
        )

    return Blip2QFormerConfig(
        hidden_size=config.qformer_hidden,
        num_hidden_layers=config.qformer_layers,
        num_attention_heads=config.qformer_heads,
        intermediate_size=config.qformer_ffn_dim,
        cross_attention_frequency=config.cross_attention_frequency,
        # cross-attention 的 key/value 来自声学编码器
        encoder_hidden_size=config.encoder_dim,
        hidden_dropout_prob=config.qformer_dropout,
        attention_probs_dropout_prob=config.qformer_dropout,
        layer_norm_eps=config.qformer_layer_norm_eps,
        # 纯语音场景没有文本输入分支，关掉可省下一半 FFN 参数
        use_qformer_text_input=False,
    )


class SpeechAdapter(nn.Module):
    """Q-Former + 可选平均池化 + MLP 对齐到 LLM embedding 空间。"""

    def __init__(self, config: SpeechLLMConfig) -> None:
        super().__init__()
        self.config = config

        qformer_config = build_qformer_config(config)
        self.query = nn.Parameter(torch.zeros(1, config.query_len, config.qformer_hidden))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(qformer_config)

        self.fc_in = nn.Linear(config.qformer_hidden, config.mlp_hidden)
        self.act = nn.ReLU()
        self.fc_out = nn.Linear(config.mlp_hidden, config.llm_dim)
        self.out_norm = nn.LayerNorm(config.llm_dim, eps=1e-5)

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        stride = self.config.pool_stride
        batch_size, seq_len, dim = hidden_states.size()
        remainder = seq_len % stride
        if remainder > 0:  # 先裁掉不能整除的尾部
            hidden_states = hidden_states[:, : seq_len - remainder, :]
            seq_len = hidden_states.size(1)

        hidden_states = hidden_states.reshape(batch_size, seq_len // stride, stride, dim)
        return hidden_states.mean(dim=2)

    def forward(
        self,
        speech_states: torch.Tensor,
        speech_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """speech_states: (B, T, encoder_dim)，speech_padding_mask 中 True 表示需屏蔽。"""
        query_states = self.query.expand(speech_states.size(0), -1, -1).to(speech_states.dtype)

        # 语义翻转：外部True=屏蔽 -> BLIP-2 需要 1=保留
        encoder_attention_mask = None
        if speech_padding_mask is not None:
            encoder_attention_mask = (~speech_padding_mask).to(dtype=speech_states.dtype)

        hidden_states = self.qformer(
            query_embeds=query_states,
            encoder_hidden_states=speech_states,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=True,
        ).last_hidden_state

        # Blip2QFormerModel 内部按 fp32 维护LayerNorm，输出可能被提升精度
        hidden_states = hidden_states.to(dtype=self.fc_in.weight.dtype)

        if self.config.use_pooling:
            hidden_states = self._pool(hidden_states)

        hidden_states = self.fc_out(self.act(self.fc_in(hidden_states)))
        return self.out_norm(hidden_states)


class FSKAdapter(nn.Module):
    """相邻帧拼接 + MLP 的变长 Adapter。

        编码器输出 (B, T, encoder_dim)
          -> 相邻 k 帧沿特征维拼接   (B, ceil(T/k), encoder_dim * k)
          -> Linear(encoder_dim*k -> llm_dim)
          -> SiLU
          -> Dropout
          -> Linear(llm_dim -> llm_dim)
          -> LayerNorm(llm_dim)

    与 Q-Former 的本质区别：**输出长度随输入变化**。Q-Former 的 query 是
    固定数量的 nn.Parameter，输出长度与输入解耦 —— 3 秒和 30 秒的音频都出
    query_len 个 token，短的被无谓扩张（p50 3.22s 的 41 帧被摊到 150），
    长的被压缩丢信息（p90 12.14s 的 157 帧压到 150）。这里没有 query，
    输出条数直接由帧数决定。

    MLP 部分（Linear -> SiLU -> Dropout -> Linear -> LayerNorm）对齐
    Kimi-Audio 的音频 adaptor 结构。两处刻意的选择：
      - 第一层直接降到 llm_dim，不设更宽的中间层。参数量 encoder_dim*k*llm_dim
        已经是主要开销（k=2 时 4096×2048≈8.4M），再加宽收益不明显。
      - LayerNorm 放在最后而不是最前。输出要直接充当 LLM 的 input embedding，
        分布得与词嵌入可比；先归一化再线性变换起不到这个作用。

    尾部不足 k 帧时**补零而不是丢弃**：丢掉会让最后 77~154 ms 的语音消失，
    而这段往往是词尾。补零后仍产出一个 token，与 config.speech_tokens_for()
    的向上取整保持一致 —— 两边算法必须同源，否则占位符数量对不上。
    """

    def __init__(self, config: SpeechLLMConfig) -> None:
        super().__init__()
        self.config = config
        self.stack_size = config.fsk_size
        if self.stack_size < 1:
            raise ValueError(f"fsk_size 必须 >= 1，当前 {self.stack_size}")

        in_dim = config.encoder_dim * self.stack_size
        self.layers = nn.Sequential(
            nn.Linear(in_dim, config.llm_dim, bias=True),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(config.llm_dim, config.llm_dim, bias=True),
            nn.LayerNorm(config.llm_dim, eps=1e-5, bias=True),
        )

    @property
    def fc_in(self) -> nn.Linear:
        """兼容访问：外部用 adapter.fc_in.weight.dtype 探测计算精度。

        Q-Former 版本有一个真的 fc_in，调用方（_encode_speech_* 与
        callbacks）都靠它拿 dtype。这里暴露第一层 Linear，保持那些代码
        不用按 adapter_type 分叉。
        """
        return self.layers[0]

    def _stack_frames(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, ceil(T/k), D*k)，尾部补零。"""
        batch, seq_len, dim = hidden_states.shape
        k = self.stack_size
        if k == 1:
            return hidden_states

        remainder = seq_len % k
        if remainder:
            pad = k - remainder
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad))
            seq_len += pad

        return hidden_states.reshape(batch, seq_len // k, dim * k)

    def forward(
        self,
        speech_states: torch.Tensor,
        speech_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """speech_states: (B, T, encoder_dim)。

        speech_padding_mask 只用来把 padding 位清零，不参与计算 —— 本模块
        逐帧独立映射，没有跨帧 attention，所以 padding 不会污染有效帧。
        清零是为了让下游按掩码取用时拿到确定值（而非 LayerNorm 后的噪声）。
        """
        if speech_padding_mask is not None:
            speech_states = speech_states.masked_fill(
                speech_padding_mask.unsqueeze(-1), 0.0
            )

        stacked = self._stack_frames(speech_states)
        return self.layers(stacked.to(dtype=self.fc_in.weight.dtype))


def build_speech_adapter(config: SpeechLLMConfig) -> nn.Module:
    """按 config.adapter_type 建 Adapter。

    新增类型时在这里加分支，不要在别处零散判断 adapter_type。
    """
    builders = {
        "qformer": SpeechAdapter,
        "fsk": FSKAdapter,
    }
    if config.adapter_type not in builders:
        raise ValueError(
            f"未实现的 adapter_type={config.adapter_type!r}，可选 {sorted(builders)}"
        )
    return builders[config.adapter_type](config)
