import math

import einops
import flax.nnx as nnx
import jax.numpy as jnp

from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder
from mme_vla_suite.models.representation.utils import kernel_init
import openpi.shared.array_typing as at


class TemporalConvBlock(nnx.Module):
    def __init__(self, kernel_size: int, strides: int, dim: int, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        self.dtype = dtype

        self.conv1 = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=(1,),
            strides=(strides,),
            padding="SAME",
            kernel_init=kernel_init,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

        self.conv2 = nnx.Conv(
            in_features=dim,
            out_features=dim,
            kernel_size=(kernel_size,),
            strides=(strides,),
            padding="SAME",
            kernel_init=kernel_init,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

        self.linear = nnx.Linear(
            in_features=dim,
            out_features=dim,
            kernel_init=kernel_init,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

        self.norm = nnx.LayerNorm(num_features=dim, dtype=dtype, rngs=rngs)

    def __call__(self, x):
        """x: (B*num_tokens, T, D)"""

        residual = self.conv1(x)

        y = self.norm(x)
        y = self.conv2(y)
        y = nnx.gelu(y)
        y = self.linear(y)

        return residual + y


class TemporalConv(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        assert config.dataset_token_budget % (2 * config.model_token_budget) == 0, (
            f"dataset_token_budget ({config.dataset_token_budget}) must be divisible by "
            f"2*model_token_budget ({2 * config.model_token_budget})"
        )

        self.config = config
        self.dtype = dtype
        self.reduction_factor = config.dataset_token_budget // config.model_token_budget

        assert self.reduction_factor & (self.reduction_factor - 1) == 0, (
            f"reduction_factor ({self.reduction_factor}) must be a power of two"
        )
        num_blocks = self.reduction_factor.bit_length() - 1

        self.feature_encoder = FeatureEncoder(
            rngs=rngs,
            dtype=dtype,
            image_input_dim=self.config.memory_feature.img.input_dim,
            pos_input_dim=self.config.memory_feature.pos.input_dim,
            state_input_dim=self.config.memory_feature.state.input_dim,
            pos_output_dim=self.config.memory_feature.pos.hidden_dim,
            state_output_dim=self.config.memory_feature.state.hidden_dim,
            output_dim_for_percep=self.config.memory_token_dim,
            use_pos_emb=self.config.use_pos_emb,
            use_state_emb=self.config.use_state_emb,
        )

        d_img = self.config.memory_feature.img.input_dim

        perceptual_cfg = getattr(config, "perceptual_memory", None)
        kernel_size = getattr(perceptual_cfg, "kernel_size", 5) if perceptual_cfg is not None else 5

        self.temporal_blocks = {
            f"block_{i}": TemporalConvBlock(kernel_size=kernel_size, strides=2, dim=d_img, rngs=rngs, dtype=dtype)
            for i in range(num_blocks)
        }

    def __call__(
        self,
        static_image_emb: at.Float[at.Array, "b l d1"],
        static_pos_emb: at.Float[at.Array, "b l d2"],
        static_state_emb: at.Float[at.Array, "b l d3"],
        static_mask: at.Bool[at.Array, "b l"],
    ) -> tuple[at.Float[at.Array, "b l d"], at.Bool[at.Array, "b l"], None]:
        assert static_image_emb.shape[1] == self.config.dataset_token_budget
        assert static_mask.shape == (static_image_emb.shape[0], self.config.dataset_token_budget)

        h_times_w = self.config.token_per_image
        hw = math.isqrt(h_times_w)
        assert hw * hw == h_times_w, f"token_per_image must be square, got {h_times_w}"
        assert hw in (2, 4, 8), (
            f"TemporalConv expects token_per_image in {{4, 16, 64}} (2x2, 4x4, 8x8), got {h_times_w}"
        )
        num_frames = static_image_emb.shape[1] // h_times_w
        assert num_frames % self.reduction_factor == 0, (
            f"T must be divisible by {self.reduction_factor}, got {num_frames}"
        )

        # (b, T*H*W, d) -> (b, T, H, W, d)
        static_image_emb = einops.rearrange(static_image_emb, "b (t h w) d -> b t h w d", h=hw, w=hw)
        static_pos_emb = einops.rearrange(static_pos_emb, "b (t h w) d -> b t h w d", h=hw, w=hw)
        static_state_emb = einops.rearrange(static_state_emb, "b (t h w) d -> b t h w d", h=hw, w=hw)
        static_mask = einops.rearrange(static_mask, "b (t h w) -> b t h w", h=hw, w=hw)

        b = static_image_emb.shape[0]

        # Merge spatial dims into batch: (b*H*W, T, d)
        static_image_emb = einops.rearrange(static_image_emb, "b t h w d -> (b h w) t d")

        for block in self.temporal_blocks.values():
            static_image_emb = block(static_image_emb)

        # Restore batch dim: (b*H*W, T/reduction_factor, d) -> (b, T/reduction_factor, H, W, d)
        static_image_emb = einops.rearrange(static_image_emb, "(b h w) t d -> b t h w d", b=b, h=hw, w=hw)

        # Subsample pos/state/mask by reduction_factor in the temporal dimension
        static_pos_emb = static_pos_emb[:, :: self.reduction_factor, :, :, :]
        static_state_emb = static_state_emb[:, :: self.reduction_factor, :, :, :]
        static_mask = static_mask[:, :: self.reduction_factor, :, :]

        # Flatten back to sequence: (b, T/reduction_factor*H*W, d)
        static_image_emb = einops.rearrange(static_image_emb, "b t h w d -> b (t h w) d")
        static_pos_emb = einops.rearrange(static_pos_emb, "b t h w d -> b (t h w) d")
        static_state_emb = einops.rearrange(static_state_emb, "b t h w d -> b (t h w) d")
        static_mask = einops.rearrange(static_mask, "b t h w -> b (t h w)")

        hidden_states = self.feature_encoder.encode_perceptual_memory(
            static_image_emb,
            static_pos_emb,
            static_state_emb,
        )

        return hidden_states, static_mask, None
