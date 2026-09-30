import flax.nnx as nnx
import jax
import jax.numpy as jnp

from mme_vla_suite.models.representation.mem_encoder import FeatureEncoder
from mme_vla_suite.models.representation.utils import kernel_init
import openpi.shared.array_typing as at


class GatedPerceptualMemory(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs, dtype: at.DTypeLike = jnp.float32):
        assert config.model_token_budget == config.dataset_token_budget, (
            f"GatedPerceptualMemory requires model_token_budget == dataset_token_budget, "
            f"got {config.model_token_budget} vs {config.dataset_token_budget}"
        )

        self.config = config
        self.dtype = dtype

        self.mem_type = config.perceptual_memory.type

        self.alpha = nnx.Param(
            -3.0
            * jnp.ones(
                self.config.model_token_budget,
                dtype=jnp.float32,
            )
        )

        self.memory_prefix = nnx.Param(
            kernel_init(rngs.params(), (self.config.model_token_budget, self.config.memory_token_dim), dtype)
        )

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

    def __call__(
        self,
        static_image_emb: at.Float[at.Array, "b l d1"],
        static_pos_emb: at.Float[at.Array, "b l d2"],
        static_state_emb: at.Float[at.Array, "b l d3"],
        static_mask: at.Bool[at.Array, "b l"],
    ) -> tuple[at.Float[at.Array, "b l d"], at.Bool[at.Array, "b l"], dict]:
        assert static_image_emb.shape[1] == self.config.dataset_token_budget
        assert static_mask.shape == (static_image_emb.shape[0], self.config.dataset_token_budget)

        hidden_states = self.feature_encoder.encode_perceptual_memory(
            static_image_emb, static_pos_emb, static_state_emb
        )

        gate = nnx.sigmoid(self.alpha[:, None]).astype(self.dtype)
        memory_tokens = (1 - gate) * self.memory_prefix + gate * hidden_states

        stats = {
            "alpha": jnp.mean(jax.nn.sigmoid(self.alpha.value)),
            "alpha_std": jnp.std(jax.nn.sigmoid(self.alpha.value)),
            "prefix_norm": jnp.mean(jnp.linalg.norm(self.memory_prefix.value, axis=-1)),
            "prefix_std": jnp.std(jnp.linalg.norm(self.memory_prefix.value, axis=-1)),
        }

        return memory_tokens, static_mask, stats
