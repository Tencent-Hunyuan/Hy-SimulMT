"""Reuse a CUDA graph for the fixed 15 residual codebook predictions per frame.

The text/first-codebook talker, sampling settings, speaker prompt and speech
history remain owned by Qwen's original generation implementation.
"""
from types import SimpleNamespace

import torch
from transformers import StaticCache


class PredictorCache(StaticCache):
    """Use preallocated storage but expose the same key length as DynamicCache.

    The fixed Python slice is captured once for each of the 15 codebooks.
    Keeping the original attention shape also preserves bf16 kernel numerics.
    """
    active_length = 2

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        return keys[:, :, :self.active_length], values[:, :, :self.active_length]


class CodePredictorGraph:
    def __init__(self, predictor, talker_hidden_size, top_k=50, temperature=0.8, sample=True):
        self.predictor = predictor
        self.groups = predictor.config.num_code_groups - 1
        self.top_k = top_k
        self.temperature = temperature
        self.sample = sample
        weight = next(predictor.parameters())
        self.device = weight.device
        self.input = torch.zeros((1, 2, talker_hidden_size), dtype=weight.dtype, device=self.device)
        self.cache = PredictorCache(config=predictor.config, max_cache_len=self.groups + 1)
        self.positions = [torch.arange(2, device=self.device)] + [
            torch.tensor([i + 1], device=self.device) for i in range(1, self.groups)
        ]
        self.masks = [None] * self.groups
        self.graph = torch.cuda.CUDAGraph()
        # Lazy cache tensors must be allocated before capture.
        with torch.inference_mode():
            self._forward()
            torch.cuda.synchronize(self.device)
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._forward()
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            with torch.cuda.graph(self.graph, stream=stream):
                self.output = self._forward()

    def _forward(self):
        model = self.predictor
        embeddings = self.input
        tokens = []
        for i in range(self.groups):
            self.cache.active_length = i + 2
            if i:
                embeddings = model.get_input_embeddings()[i - 1](tokens[-1])
            projected = model.small_to_mtp_projection(embeddings)
            hidden = model.model(
                inputs_embeds=projected,
                attention_mask={"full_attention": self.masks[i]},
                position_ids=self.positions[i].unsqueeze(0),
                cache_position=self.positions[i],
                past_key_values=self.cache,
                use_cache=True, output_hidden_states=False, return_dict=True,
            ).last_hidden_state[:, -1:, :]
            logits = model.lm_head[i](hidden).float()
            if self.sample:
                scaled = logits[:, -1] / self.temperature
                if self.top_k:
                    threshold = torch.topk(scaled, self.top_k, dim=-1).values[:, -1:]
                    scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
                token = torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1)
            else:
                token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens.append(token)
        return torch.cat(tokens, dim=-1)

    def run(self, embeddings):
        self.input.copy_(embeddings)
        self.graph.replay()
        return self.output.clone()


def accelerate_code_predictor(model):
    """Install only the production single-item sampling path; retain fallback."""
    predictor = model.model.talker.code_predictor
    graph = CodePredictorGraph(predictor, model.model.config.talker_config.hidden_size)
    original = predictor.generate

    def generate(*args, **kwargs):
        embeddings = kwargs.get("inputs_embeds")
        compatible = (
            not args and embeddings is not None and tuple(embeddings.shape) == tuple(graph.input.shape)
            and kwargs.get("max_new_tokens") == graph.groups
            and kwargs.get("do_sample") is True
            and kwargs.get("top_k") == graph.top_k
            and kwargs.get("top_p") == 1.0
            and kwargs.get("temperature") == graph.temperature
        )
        if not compatible:
            return original(*args, **kwargs)
        return SimpleNamespace(sequences=graph.run(embeddings))

    predictor.generate = generate
    return graph
