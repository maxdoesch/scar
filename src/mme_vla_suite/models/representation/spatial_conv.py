import math

import einops
import flax.nnx as nnx
import jax.numpy as jnp

from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder
from mme_vla_suite.models.representation.utils import kernel_init
import openpi.shared.array_typing as at


class SpatialConv(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        assert config.model_token_budget == config.dataset_token_budget // 4, (
            f"SpatialConv requires model_token_budget == dataset_token_budget // 4, "
            f"got {config.model_token_budget} vs {config.dataset_token_budget}"
        )

        self.config = config
        self.dtype = dtype

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
        self.norm = nnx.LayerNorm(d_img, dtype=dtype, rngs=rngs)

        self.conv_type = getattr(self.config.perceptual_memory, "conv_type", None)
        assert self.conv_type in ["single", "two_layer"], f"Unknown conv_type: {self.conv_type}"

        if self.conv_type == "single":
            # 8x8 -> 4x4
            self.conv = nnx.Conv(
                in_features=d_img,
                out_features=d_img,
                kernel_size=(2, 2),
                strides=(2, 2),
                padding="VALID",
                kernel_init=kernel_init,
                use_bias=False,
                dtype=dtype,
                rngs=rngs,
            )

        elif self.conv_type == "two_layer":
            # 8x8 -> 8x8 local mixing
            self.conv_mix = nnx.Conv(
                in_features=d_img,
                out_features=d_img,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding="SAME",
                kernel_init=kernel_init,
                use_bias=False,
                dtype=dtype,
                rngs=rngs,
            )

            # 8x8 -> 4x4 downsampling
            self.conv_down = nnx.Conv(
                in_features=d_img,
                out_features=d_img,
                kernel_size=(2, 2),
                strides=(2, 2),
                padding="VALID",
                kernel_init=kernel_init,
                use_bias=False,
                dtype=dtype,
                rngs=rngs,
            )

        else:
            raise ValueError(f"Unknown memory_spatial_conv_type: {self.conv_type}")

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
        assert hw == 8, f"This SpatialConv version expects 8x8 tokens, got {hw}x{hw}"

        static_image_emb = einops.rearrange(
            static_image_emb,
            "b (t h w) d -> b t h w d",
            h=hw,
            w=hw,
        )

        static_pos_emb = einops.rearrange(
            static_pos_emb,
            "b (t h w) d -> b t h w d",
            h=hw,
            w=hw,
        )

        static_state_emb = einops.rearrange(
            static_state_emb,
            "b (t h w) d -> b t h w d",
            h=hw,
            w=hw,
        )

        static_mask = einops.rearrange(
            static_mask,
            "b (t h w) -> b t h w",
            h=hw,
            w=hw,
        )

        static_image_emb = self.norm(static_image_emb)

        if self.conv_type == "single":
            static_image_emb = self.conv(static_image_emb)

        elif self.conv_type == "two_layer":
            static_image_emb = self.conv_mix(static_image_emb)
            static_image_emb = nnx.gelu(static_image_emb)
            static_image_emb = self.conv_down(static_image_emb)

        static_image_emb = nnx.gelu(static_image_emb)

        # Match pos/state/mask tokens to 8x8 -> 4x4.
        # We keep the top-left representative of each 2x2 cell.
        static_pos_emb = static_pos_emb[:, :, ::2, ::2, :]
        static_state_emb = static_state_emb[:, :, ::2, ::2, :]
        static_mask = static_mask[:, :, ::2, ::2]

        static_image_emb = einops.rearrange(
            static_image_emb,
            "b t h w d -> b (t h w) d",
        )

        static_pos_emb = einops.rearrange(
            static_pos_emb,
            "b t h w d -> b (t h w) d",
        )

        static_state_emb = einops.rearrange(
            static_state_emb,
            "b t h w d -> b (t h w) d",
        )

        static_mask = einops.rearrange(
            static_mask,
            "b t h w -> b (t h w)",
        )

        hidden_states = self.feature_encoder.encode_perceptual_memory(
            static_image_emb,
            static_pos_emb,
            static_state_emb,
        )

        return hidden_states, static_mask, None
