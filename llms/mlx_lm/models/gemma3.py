# Copyright © 2025 Apple Inc.

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map

from .base import (  # scaled_dot_product_attention adjusted locally
    BaseModelArgs,
    create_attention_mask,
)
from .cache import QuantizedKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma3"
    hidden_size: int = 2304
    num_hidden_layers: int = 26
    intermediate_size: int = 9216
    num_attention_heads: int = 8
    head_dim: int = 256
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262_208
    num_key_value_heads: int = 4
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    rope_traditional: bool = False
    rope_scaling: Optional[dict] = None
    attn_logit_softcapping: Optional[float] = None
    final_logit_softcapping: Optional[float] = None
    query_pre_attn_scalar: float = 256.0
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    max_position_embeddings: int = 131_072

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.rope_scaling and "factor" not in self.rope_scaling:
            raise ValueError("rope_scaling must include 'factor'")


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)


class Gemma3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        max_position_embeddings: int = 131072,
        traditional: bool = False,
        scaling: Optional[dict] = None,
    ):
        super().__init__()
        self.dims = dims
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional
        factor = scaling.get("factor", 1.0) if scaling else 1.0
        freqs = 1.0 / (base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims))
        self.freqs = freqs * factor

    @mx.compile
    def __call__(self, x, offset: int = 0):
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self.freqs,
        )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, is_sliding: bool, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.repeats = self.n_heads // self.n_kv_heads
        self.head_dim = args.head_dim
        self.is_sliding = is_sliding
        self.sliding_window = args.sliding_window if is_sliding else None
        self.scale = args.query_pre_attn_scalar**-0.5
        self.attn_logit_softcapping = args.attn_logit_softcapping

        dim = args.hidden_size
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)

        # Per-head normalization
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        # Dual RoPE: global and local
        self.rope_global = Gemma3RotaryEmbedding(
            self.head_dim,
            base=args.rope_theta,
            max_position_embeddings=args.max_position_embeddings,
            traditional=args.rope_traditional,
            scaling=args.rope_scaling,  # {"factor": 8.0}
        )
        self.rope_local = Gemma3RotaryEmbedding(
            self.head_dim,
            base=args.rope_local_base_freq,
            max_position_embeddings=args.max_position_embeddings,
            traditional=args.rope_traditional,
            scaling=None,
        )

    def create_sliding_mask(
        self, seq_len: int, offset: int = 0, window_size: int = 1024
    ) -> mx.array:
        rinds = mx.arange(offset, offset + seq_len, dtype=mx.int32)
        linds = rinds[:, None]
        rinds = rinds[None, :]
        causal_mask = linds <= rinds
        window_mask = (linds > rinds - window_size) & causal_mask
        return mx.where(window_mask, 0.0, -1e9).astype(mx.float32)

    @mx.compile
    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        B, L, D = x.shape
        queries = (
            self.q_proj(x)
            .reshape(B, L, self.n_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        keys = (
            self.k_proj(x)
            .reshape(B, L, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        values = (
            self.v_proj(x)
            .reshape(B, L, self.n_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        queries = self.q_norm(queries)
        keys = self.k_norm(keys)

        # Select RoPE based on sliding window status
        rope = self.rope_local if self.is_sliding else self.rope_global
        offset = cache.offset if cache is not None else 0
        queries = rope(queries, offset=offset)
        keys = rope(keys, offset=offset)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if self.is_sliding and mask is None:
            mask = self.create_sliding_mask(
                L, offset=offset, window_size=self.sliding_window
            )

        # Attention computation with quantization support
        if isinstance(cache, QuantizedKVCache):
            # Quantized case (adapted from quantized_scaled_dot_product_attention)
            queries = queries * self.scale
            if self.repeats > 1:  # Handle GQA
                queries = mx.reshape(
                    queries, (B, self.n_kv_heads, self.repeats, L, self.head_dim)
                )
                keys = tree_map(lambda x: mx.expand_dims(x, axis=-3), keys)
                values = tree_map(lambda x: mx.expand_dims(x, axis=-3), values)

            scores = mx.quantized_matmul(
                queries,
                keys,
                transpose=True,
                group_size=cache.group_size,
                bits=cache.bits,
            )
            if mask is not None:
                scores = scores + mask
            # Remove unless weights expect it
            # if self.attn_logit_softcapping is not None:
            #     scores = mx.tanh(scores / self.attn_logit_softcapping) * self.attn_logit_softcapping
            scores = mx.softmax(scores, axis=-1, precise=True)
            output = mx.quantized_matmul(
                scores,
                values,
                transpose=False,
                group_size=cache.group_size,
                bits=cache.bits,
            )
            if self.repeats > 1:
                output = mx.reshape(output, (B, self.n_heads, L, self.head_dim))
        else:
            # Non-quantized case
            scores = queries @ keys.transpose(0, 1, 3, 2) * self.scale
            if mask is not None:
                scores = scores + mask
            # Remove unless weights expect it
            # if self.attn_logit_softcapping is not None:
            #     scores = mx.tanh(scores / self.attn_logit_softcapping) * self.attn_logit_softcapping
            scores = mx.softmax(scores, axis=-1, precise=True)
            output = scores @ values

        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    @mx.compile
    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_sliding = bool((layer_idx + 1) % args.sliding_window_pattern)
        self.self_attn = Attention(args, self.is_sliding, layer_idx)
        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def create_causal_mask(self, seq_len: int, offset: int = 0) -> mx.array:
        rinds = mx.arange(offset, offset + seq_len, dtype=mx.int32)
        linds = rinds[:, None]
        rinds = rinds[None, :]
        mask = linds <= rinds
        return mx.where(mask, 0.0, -1e9).astype(mx.float32)

    @mx.compile
    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        # Create mask if not provided
        if mask is None:
            offset = cache.offset if cache else 0
            mask = (
                self.create_causal_mask(L, offset=offset)
                if not self.is_sliding
                else self.self_attn.create_sliding_mask(
                    L, offset=offset, window_size=self.self_attn.sliding_window
                )
            )

        # Pre-attention residual
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, mask, cache)
        h = self.post_attention_layernorm(h)
        h = residual + h

        # Pre-feedforward residual
        residual = h
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        return residual + h


class Gemma3Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [TransformerBlock(args, i) for i in range(args.num_hidden_layers)]
        self.norm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    @mx.compile
    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array]]]:
        h = self.embed_tokens(inputs)
        h = h * (self.args.hidden_size**0.5)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        hidden_states = () if output_hidden_states else None
        for layer, c in zip(self.layers, cache):
            if output_hidden_states:
                hidden_states += (h,)
            h = layer(h, mask, c)

        h = self.norm(h)
        return (h, hidden_states) if output_hidden_states else h


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.model_type = args.model_type
        self.final_logit_softcapping = args.final_logit_softcapping
        self.model = Gemma3Model(args)
        self.lm_head = self.model.embed_tokens.as_linear  # Reuse embedding weights
        self.args = args

    def load_weights(self, weights):
        """Load the full checkpoint weights without filtering."""
        super().load_weights(weights)  # Let MLX handle mapping to defined modules

    @mx.compile
    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array]]]:
        out = self.model(inputs, mask, cache, output_hidden_states)
        if output_hidden_states:
            out, hidden_states = out
        else:
            hidden_states = None
        out = self.lm_head(out)
        if self.final_logit_softcapping is not None:
            out = (
                mx.tanh(out / self.final_logit_softcapping)
                * self.final_logit_softcapping
            )
        return (out, hidden_states) if output_hidden_states else out

    @property
    def layers(self):
        return self.model.layers
