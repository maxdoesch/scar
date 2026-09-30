import os
import types

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax.numpy as jnp

from mme_vla_suite.models.representation.spatial_conv import SpatialConv


def _cfg(conv_type: str, dataset_token_budget=128, model_token_budget=32):
    # 128 dataset tokens at 8x8 per frame = 2 frames; 32 model tokens at 4x4 per frame = 2 frames.
    return types.SimpleNamespace(
        dataset_token_budget=dataset_token_budget,
        model_token_budget=model_token_budget,
        token_per_image=64,
        memory_token_dim=16,
        kernel_init_scale=0.02,
        use_pos_emb=True,
        use_state_emb=False,
        perceptual_memory=types.SimpleNamespace(type="frame_sampling", conv_type=conv_type),
        memory_feature=types.SimpleNamespace(
            img=types.SimpleNamespace(input_dim=12),
            pos=types.SimpleNamespace(input_dim=8, hidden_dim=8),
            state=types.SimpleNamespace(input_dim=4, hidden_dim=4),
        ),
    )


@pytest.mark.parametrize("conv_type", ["single", "two_layer"])
def test_spatial_conv_downsamples_mask(conv_type):
    cfg = _cfg(conv_type)
    m = SpatialConv(cfg, rngs=nnx.Rngs(0), dtype=jnp.float32)

    b = 2
    img = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.img.input_dim), dtype=jnp.float32)
    pos = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.pos.input_dim), dtype=jnp.float32)
    state = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.state.input_dim), dtype=jnp.float32)
    # First frame: all valid. Second frame: all invalid.
    static_mask = jnp.concatenate(
        [jnp.ones((b, 64), dtype=jnp.bool_), jnp.zeros((b, 64), dtype=jnp.bool_)],
        axis=1,
    )

    tokens, out_mask, _ = m(img, pos, state, static_mask)

    assert tokens.shape == (b, cfg.model_token_budget, cfg.memory_token_dim)
    assert out_mask.shape == (b, cfg.model_token_budget)
    # First-frame model-side tokens are valid, second-frame are not.
    expected = jnp.concatenate(
        [jnp.ones((b, 16), dtype=jnp.bool_), jnp.zeros((b, 16), dtype=jnp.bool_)],
        axis=1,
    )
    assert jnp.array_equal(out_mask, expected)


def test_spatial_conv_rejects_mismatched_budgets():
    cfg = _cfg("single", dataset_token_budget=128, model_token_budget=64)  # not /4
    with pytest.raises(AssertionError):
        SpatialConv(cfg, rngs=nnx.Rngs(1), dtype=jnp.float32)
