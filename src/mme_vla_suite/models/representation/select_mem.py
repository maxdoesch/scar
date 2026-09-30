from collections.abc import Callable

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp

from mme_vla_suite.models.representation.utils import kernel_init
import openpi.shared.array_typing as at


def _attention_entropy_hook(attn_weights: at.Array) -> dict[str, at.Array]:
    eps = jnp.finfo(attn_weights.dtype).tiny
    entropy = -jnp.sum(attn_weights * jnp.log(jnp.clip(attn_weights, eps, 1.0)), axis=-1)
    return {
        "attention_entropy": jnp.mean(entropy),
    }


def dot_product_attention_with_score_hook(
    query: at.Array,
    key: at.Array,
    value: at.Array,
    bias: at.Array | None = None,
    mask: at.Array | None = None,
    score_hook: Callable[[at.Array], dict[str, at.Array]] | None = None,
) -> tuple[at.Array, dict[str, at.Array] | None]:
    """Copied attention core with an optional hook on normalized attention weights."""
    assert key.ndim == query.ndim == value.ndim, "q, k, v must have same rank."
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], "q, k, v batch dims must match."
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], "q, k, v num_heads must match."
    assert key.shape[-3] == value.shape[-3], "k, v lengths must match."

    dtype = query.dtype
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    attn_weights = jnp.einsum("...qhd,...khd->...hqk", query, key)

    if bias is not None:
        attn_weights = attn_weights + bias
    if mask is not None:
        big_neg = jnp.finfo(dtype).min
        attn_weights = jnp.where(mask, attn_weights, big_neg)

    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

    metrics = None
    if score_hook is not None:
        metrics = score_hook(attn_weights)

    output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
    return output, metrics


class SelfAttention(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        self.config = config
        self.dtype = dtype
        self.attn_head_dim = config.selective_memory.attn_head_dim
        self.attn_heads = config.selective_memory.attn_heads

        inner_dim = self.attn_heads * self.attn_head_dim

        self.w_in = nnx.Linear(
            in_features=config.memory_token_dim,
            out_features=3 * inner_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.w_out = nnx.Linear(
            in_features=inner_dim,
            out_features=config.memory_token_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )

    def __call__(self, x: at.Float[at.Array, "b l d"]):
        qkv = self.w_in(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        q = einops.rearrange(q, "b l (h d) -> b l h d", h=self.attn_heads)
        k = einops.rearrange(k, "b l (h d) -> b l h d", h=self.attn_heads)
        v = einops.rearrange(v, "b l (h d) -> b l h d", h=self.attn_heads)

        out = nnx.dot_product_attention(q, k, v)
        out = einops.rearrange(out, "b l h d -> b l (h d)")
        return self.w_out(out)


class CrossAttention(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        self.config = config
        self.dtype = dtype
        self.attn_head_dim = config.selective_memory.attn_head_dim
        self.attn_heads = config.selective_memory.attn_heads

        inner_dim = self.attn_heads * self.attn_head_dim

        self.w_q = nnx.Linear(
            in_features=config.memory_token_dim,
            out_features=inner_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.w_kv = nnx.Linear(
            in_features=config.memory_token_dim,
            out_features=2 * inner_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.w_out = nnx.Linear(
            in_features=inner_dim,
            out_features=config.memory_token_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )

    def __call__(
        self,
        x_q: at.Float[at.Array, "b lq d"],
        x_kv: at.Float[at.Array, "b lk d"],
        mask: at.Bool[at.Array, "b lq lk"] | None = None,
        return_metrics: bool = False,
    ):
        q = self.w_q(x_q)
        kv = self.w_kv(x_kv)
        k, v = jnp.split(kv, 2, axis=-1)

        q = einops.rearrange(q, "b lq (h d) -> b lq h d", h=self.attn_heads)
        k = einops.rearrange(k, "b lk (h d) -> b lk h d", h=self.attn_heads)
        v = einops.rearrange(v, "b lk (h d) -> b lk h d", h=self.attn_heads)

        out, metrics = dot_product_attention_with_score_hook(
            q,
            k,
            v,
            mask=mask,
            score_hook=_attention_entropy_hook,
        )
        out = einops.rearrange(out, "b lq h d -> b lq (h d)")
        out = self.w_out(out)
        if return_metrics:
            return out, metrics
        return out


class FeedForward(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        self.config = config
        self.dtype = dtype

        self.w_1 = nnx.Linear(
            in_features=config.memory_token_dim,
            out_features=config.selective_memory.mlp_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )
        self.w_2 = nnx.Linear(
            in_features=config.selective_memory.mlp_dim,
            out_features=config.memory_token_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )

    def __call__(self, x: at.Float[at.Array, "b l d"]):
        return self.w_2(jax.nn.gelu(self.w_1(x)))


class Block(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        self.self_attn = SelfAttention(config, rngs, dtype)
        self.cross_attn = CrossAttention(config, rngs, dtype)
        self.mlp = FeedForward(config, rngs, dtype)

        self.norm_sa = nnx.RMSNorm(config.memory_token_dim, rngs=rngs, dtype=dtype)
        self.norm_ca_q = nnx.RMSNorm(config.memory_token_dim, rngs=rngs, dtype=dtype)
        self.norm_ca_kv = nnx.RMSNorm(config.memory_token_dim, rngs=rngs, dtype=dtype)
        self.norm_ff = nnx.RMSNorm(config.memory_token_dim, rngs=rngs, dtype=dtype)

    def __call__(
        self,
        x_q: at.Float[at.Array, "b lq d"],
        x_kv: at.Float[at.Array, "b lk d"],
        mask: at.Bool[at.Array, "b lq lk"] | None = None,
        return_metrics: bool = False,
    ):
        x_q = x_q + self.self_attn(self.norm_sa(x_q))
        cross_out, cross_metrics = self.cross_attn(
            self.norm_ca_q(x_q),
            self.norm_ca_kv(x_kv),
            mask,
            True,
        )
        x_q = x_q + cross_out
        x_q = x_q + self.mlp(self.norm_ff(x_q))
        if return_metrics:
            return x_q, cross_metrics
        return x_q


class SelectiveMemory(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        assert config.model_token_budget == config.token_per_image * (1 + config.selective_memory.query_reg_len), (
            f"SelectiveMemory requires model_token_budget == token_per_image * (1 + query_reg_len), "
            f"got {config.model_token_budget} vs {config.token_per_image * (1 + config.selective_memory.query_reg_len)}"
        )
        self.config = config
        self.dtype = dtype

        self.temp_spatial_pos_dim = config.memory_feature.pos.input_dim
        self.spatial_pos_dim = 2 * config.memory_feature.pos.input_dim // 3
        self.image_input_dim = config.memory_feature.img.input_dim
        self.memory_token_dim = config.memory_token_dim
        self.token_per_image = config.token_per_image

        self.mem_sink_len = config.selective_memory.mem_sink_len
        self.query_reg_len = config.selective_memory.query_reg_len

        if self.config.selective_memory.mask_or_pad not in ["mask", "pad"]:
            raise ValueError(f"Invalid mask_or_pad: {self.config.selective_memory.mask_or_pad}")
        self.mask_or_pad = self.config.selective_memory.mask_or_pad

        self.mem_sink = (
            nnx.Param(kernel_init(rngs.params(), (1, self.token_per_image * self.mem_sink_len, self.memory_token_dim)))
            if self.mem_sink_len > 0
            else None
        )

        self.query_reg = (
            nnx.Param(kernel_init(rngs.params(), (1, self.token_per_image * self.query_reg_len, self.memory_token_dim)))
            if self.query_reg_len > 0
            else None
        )

        self.padding_token = (
            nnx.Param(kernel_init(rngs.params(), (1, 1, self.memory_token_dim))) if self.mask_or_pad == "pad" else None
        )

        self.img_proj = nnx.Linear(
            self.image_input_dim,
            self.memory_token_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )

        self.temp_spatial_pos_proj = nnx.Linear(
            self.temp_spatial_pos_dim,
            self.memory_token_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )

        self.spatial_pos_proj = nnx.Linear(
            self.spatial_pos_dim,
            self.memory_token_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )

        @nnx.split_rngs(splits=self.config.selective_memory.attn_blocks)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def create_blocks(rngs_):
            return Block(config, rngs_, dtype)

        self.blocks = create_blocks(rngs)

        self.out_proj = nnx.Linear(
            in_features=self.memory_token_dim,
            out_features=self.memory_token_dim,
            rngs=rngs,
            dtype=dtype,
            kernel_init=kernel_init,
        )

    def __call__(
        self,
        static_image_emb: at.Float[at.Array, "b l d_img"],
        static_pos_emb: at.Float[at.Array, "b l d_pos"],
        static_mask: at.Bool[at.Array, "b l"],
    ):
        b, l, _ = static_image_emb.shape
        assert l == self.config.dataset_token_budget
        assert static_pos_emb.shape[:2] == (b, l)
        assert self.config.dataset_token_budget % self.token_per_image == 0

        static_image_emb = self.img_proj(static_image_emb)
        if self.mask_or_pad == "pad":
            static_image_emb = jnp.where(
                static_mask[:, :, None],
                static_image_emb,
                jnp.broadcast_to(self.padding_token.value, static_image_emb.shape),
            )
        spatial_pos_emb = static_pos_emb[:, -self.token_per_image :, self.temp_spatial_pos_dim // 3 :]
        query_tokens = static_image_emb[:, -self.token_per_image :] + self.spatial_pos_proj(spatial_pos_emb)

        if self.query_reg is not None:
            register = jnp.broadcast_to(
                self.query_reg.value, (b, self.token_per_image * self.query_reg_len, self.memory_token_dim)
            )
            query_tokens = jnp.concatenate(
                [
                    register,
                    query_tokens,
                ],
                axis=1,
            )

        memory_tokens = static_image_emb + self.temp_spatial_pos_proj(static_pos_emb)

        if self.mem_sink is not None:
            sink = jnp.broadcast_to(
                self.mem_sink.value, (b, self.mem_sink_len * self.token_per_image, self.memory_token_dim)
            )
            memory_tokens = jnp.concatenate([sink, memory_tokens], axis=1)

        static_mask = jnp.concatenate(
            [jnp.ones((b, self.token_per_image * self.mem_sink_len), dtype=static_mask.dtype), static_mask], axis=-1
        )

        if self.mask_or_pad == "mask":
            cross_attn_mask = einops.repeat(static_mask, "b lk -> b 1 lq lk", lq=query_tokens.shape[1])
        else:
            cross_attn_mask = None

        def block_scan_step(block, x_q, x_kv, mask):
            return block(x_q, x_kv, mask, True)

        query_tokens, block_metrics = nnx.scan(
            block_scan_step,
            in_axes=(nnx.StateAxes({nnx.Param: 0, ...: None}), nnx.Carry, None, None),
            out_axes=(nnx.Carry, 0),
        )(
            self.blocks,
            query_tokens,
            memory_tokens,
            cross_attn_mask,
        )

        hidden_states = self.out_proj(query_tokens)
        metrics = None
        if block_metrics is not None:
            metrics = {
                "selective_memory/attention_entropy": jnp.mean(block_metrics["attention_entropy"]),
                "selective_memory/attention_entropy_per_block": block_metrics["attention_entropy"],
            }
        out_mask = jnp.ones((b, hidden_states.shape[1]), dtype=jnp.bool_)
        return hidden_states, out_mask, metrics
