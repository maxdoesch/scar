import os
import types

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax.numpy as jnp

from mme_vla_suite.models.representation.percep_mem import PerceptualMemory


def _cfg(dataset_token_budget=8, model_token_budget=8):
    return types.SimpleNamespace(
        dataset_token_budget=dataset_token_budget,
        model_token_budget=model_token_budget,
        memory_token_dim=16,
        use_pos_emb=True,
        use_state_emb=False,
        perceptual_memory=types.SimpleNamespace(type="frame_sampling"),
        memory_feature=types.SimpleNamespace(
            img=types.SimpleNamespace(input_dim=12),
            pos=types.SimpleNamespace(input_dim=8, hidden_dim=8),
            state=types.SimpleNamespace(input_dim=4, hidden_dim=4),
        ),
    )


def test_perceptual_memory_returns_tokens_and_passthrough_mask():
    cfg = _cfg()
    m = PerceptualMemory(cfg, rngs=nnx.Rngs(0), dtype=jnp.float32)

    b = 2
    img = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.img.input_dim), dtype=jnp.float32)
    pos = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.pos.input_dim), dtype=jnp.float32)
    state = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.state.input_dim), dtype=jnp.float32)
    static_mask = jnp.concatenate(
        [jnp.ones((b, cfg.dataset_token_budget - 2), dtype=jnp.bool_), jnp.zeros((b, 2), dtype=jnp.bool_)],
        axis=1,
    )

    tokens, out_mask, stats = m(img, pos, state, static_mask)

    assert tokens.shape == (b, cfg.model_token_budget, cfg.memory_token_dim)
    assert out_mask.shape == (b, cfg.model_token_budget)
    assert jnp.array_equal(out_mask, static_mask)
    assert stats is None


def test_perceptual_memory_rejects_mismatched_budgets():
    cfg = _cfg(dataset_token_budget=8, model_token_budget=16)
    with pytest.raises(AssertionError):
        PerceptualMemory(cfg, rngs=nnx.Rngs(1), dtype=jnp.float32)


def test_perceptual_memory_rejects_wrong_mask_shape():
    cfg = _cfg()
    m = PerceptualMemory(cfg, rngs=nnx.Rngs(2), dtype=jnp.float32)

    b = 2
    img = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.img.input_dim), dtype=jnp.float32)
    pos = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.pos.input_dim), dtype=jnp.float32)
    state = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.state.input_dim), dtype=jnp.float32)
    bad_mask = jnp.ones((b, cfg.dataset_token_budget - 1), dtype=jnp.bool_)

    with pytest.raises(AssertionError):
        m(img, pos, state, bad_mask)
