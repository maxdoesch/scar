import os
import types

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax.numpy as jnp

from mme_vla_suite.models.representation.temporal_conv import TemporalConv


def _cfg(dataset_token_budget=128, model_token_budget=32, token_per_image=16):
    # token_per_image=16 (4x4 per frame); 128 dataset tokens => 8 frames.
    # reduction_factor = 128 / 32 = 4, so 8 frames -> 2 frames => 32 model tokens.
    return types.SimpleNamespace(
        dataset_token_budget=dataset_token_budget,
        model_token_budget=model_token_budget,
        token_per_image=token_per_image,
        memory_token_dim=16,
        use_pos_emb=True,
        use_state_emb=False,
        memory_feature=types.SimpleNamespace(
            img=types.SimpleNamespace(input_dim=12),
            pos=types.SimpleNamespace(input_dim=8, hidden_dim=8),
            state=types.SimpleNamespace(input_dim=4, hidden_dim=4),
        ),
    )


def test_temporal_conv_output_shapes():
    cfg = _cfg()
    m = TemporalConv(cfg, rngs=nnx.Rngs(0), dtype=jnp.float32)

    b = 2
    img = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.img.input_dim), dtype=jnp.float32)
    pos = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.pos.input_dim), dtype=jnp.float32)
    state = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.state.input_dim), dtype=jnp.float32)
    static_mask = jnp.ones((b, cfg.dataset_token_budget), dtype=jnp.bool_)

    tokens, out_mask, extra = m(img, pos, state, static_mask)

    assert tokens.shape == (b, cfg.model_token_budget, cfg.memory_token_dim)
    assert out_mask.shape == (b, cfg.model_token_budget)
    assert out_mask.dtype == jnp.bool_
    assert extra is None


def test_temporal_conv_rejects_mismatched_budgets():
    # dataset_token_budget must be divisible by 2 * model_token_budget.
    cfg = _cfg(dataset_token_budget=128, model_token_budget=96)
    with pytest.raises(AssertionError):
        TemporalConv(cfg, rngs=nnx.Rngs(1), dtype=jnp.float32)
