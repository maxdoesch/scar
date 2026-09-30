import flax.nnx as nnx
import jax.numpy as jnp

from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder
from mme_vla_suite.models.representation.utils import kernel_init
import openpi.shared.array_typing as at


class TemporalConvIdentityBlock(nnx.Module):
    """Parameter-matched MLP block for TemporalConvBlock without temporal compression or mixing.

    Replaces the 1D temporal convolution with standard pointwise linear layers (MLP).
    To match the exact parameter count of TemporalConvBlock ((K + 2)*D^2 + 2D), the intermediate
    hidden dimension is expanded to:
        hidden_dim = ((kernel_size + 1) / 2) * dim
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 5,
        hidden_dim: int | None = None,
        rngs: nnx.Rngs = None,
        dtype: at.DTypeLike = jnp.float32,
    ):
        self.dtype = dtype

        if hidden_dim is None:
            hidden_dim = int(((kernel_size + 1) / 2) * dim)

        self.linear_skip = nnx.Linear(
            in_features=dim,
            out_features=dim,
            kernel_init=kernel_init,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

        self.linear1 = nnx.Linear(
            in_features=dim,
            out_features=hidden_dim,
            kernel_init=kernel_init,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

        self.linear2 = nnx.Linear(
            in_features=hidden_dim,
            out_features=dim,
            kernel_init=kernel_init,
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )

        self.norm = nnx.LayerNorm(num_features=dim, dtype=dtype, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "... d"]) -> at.Float[at.Array, "... d"]:
        """Apply pointwise MLP transformations without cross-token (temporal) mixing.

        x: (..., D) -> transformed (..., D)
        """
        residual = self.linear_skip(x)

        y = self.norm(x)
        y = self.linear1(y)
        y = nnx.gelu(y)
        y = self.linear2(y)

        return residual + y


class TemporalConvIdentity(nnx.Module):
    """Parameter-matched ablation module for TemporalConv without temporal compression or temporal mixing.

    Used for ICLR ablation experiments to isolate the effect of temporal compression & temporal mixing
    from the sheer number of learnable parameters added by the temporal convolution blocks.

    - Parameter-matched: Uses pointwise linear layers with expanded hidden dimension to match
      the exact parameter count of TemporalConv.
    - No temporal compression: `model_token_budget == dataset_token_budget` (stride=1, no temporal downsampling).
    - No temporal mixing: Every token is processed strictly independently via pointwise linear transformations.
    """

    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        assert config.model_token_budget == config.dataset_token_budget, (
            f"TemporalConvIdentity requires model_token_budget == dataset_token_budget "
            f"(no temporal compression), got {config.model_token_budget} vs {config.dataset_token_budget}"
        )

        self.config = config
        self.dtype = dtype

        perceptual_cfg = getattr(config, "perceptual_memory", None)
        kernel_size = getattr(perceptual_cfg, "kernel_size", 5) if perceptual_cfg is not None else 5
        num_blocks = getattr(perceptual_cfg, "num_blocks", 2) if perceptual_cfg is not None else 2

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

        # String-keyed dictionary to maintain compatibility with flax flatten_dict
        self.temporal_blocks = {
            f"block_{i}": TemporalConvIdentityBlock(dim=d_img, kernel_size=kernel_size, rngs=rngs, dtype=dtype)
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

        for block in self.temporal_blocks.values():
            static_image_emb = block(static_image_emb)

        hidden_states = self.feature_encoder.encode_perceptual_memory(
            static_image_emb,
            static_pos_emb,
            static_state_emb,
        )

        return hidden_states, static_mask, None
