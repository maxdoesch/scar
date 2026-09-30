import os
import types

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from mme_vla_suite.models.representation.temporal_conv import TemporalConv
from mme_vla_suite.models.representation.temporal_conv_identity import TemporalConvIdentity
from mme_vla_suite.models.representation.temporal_conv_identity import TemporalConvIdentityBlock


def _cfg_temporal_conv(dataset_token_budget=128, model_token_budget=32, token_per_image=16, kernel_size=5):
    # reduction_factor = 128 / 32 = 4 -> 2 blocks
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
        perceptual_memory=types.SimpleNamespace(
            type="frame_sampling_every_n",
            encoder_type="temporal_conv",
            kernel_size=kernel_size,
        ),
    )


def _cfg_identity(token_budget=32, token_per_image=16, kernel_size=5, num_blocks=2):
    return types.SimpleNamespace(
        dataset_token_budget=token_budget,
        model_token_budget=token_budget,
        token_per_image=token_per_image,
        memory_token_dim=16,
        use_pos_emb=True,
        use_state_emb=False,
        memory_feature=types.SimpleNamespace(
            img=types.SimpleNamespace(input_dim=12),
            pos=types.SimpleNamespace(input_dim=8, hidden_dim=8),
            state=types.SimpleNamespace(input_dim=4, hidden_dim=4),
        ),
        perceptual_memory=types.SimpleNamespace(
            type="frame_sampling_every_n",
            encoder_type="temporal_conv_identity",
            kernel_size=kernel_size,
            num_blocks=num_blocks,
        ),
    )


def test_temporal_conv_identity_output_shapes():
    cfg = _cfg_identity()
    m = TemporalConvIdentity(cfg, rngs=nnx.Rngs(0), dtype=jnp.float32)

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


def test_temporal_conv_identity_parameter_parity():
    # Verify that TemporalConv (with 2 blocks) and TemporalConvIdentity (with 2 blocks)
    # have 100% identical total parameter counts when using matching hidden_dim = ((kernel_size + 1) / 2) * dim.
    cfg_tc = _cfg_temporal_conv(dataset_token_budget=128, model_token_budget=32, kernel_size=5)
    cfg_id = _cfg_identity(token_budget=32, kernel_size=5, num_blocks=2)

    m_tc = TemporalConv(cfg_tc, rngs=nnx.Rngs(0), dtype=jnp.float32)
    m_id = TemporalConvIdentity(cfg_id, rngs=nnx.Rngs(0), dtype=jnp.float32)

    total_tc = sum(v.value.size for v in nnx.state(m_tc, nnx.Param).flat_state().values() if v.value is not None)
    total_id = sum(v.value.size for v in nnx.state(m_id, nnx.Param).flat_state().values() if v.value is not None)
    assert total_tc == total_id


def test_temporal_conv_identity_no_temporal_mixing():
    # Modifying a single token in the input must ONLY affect the corresponding token in output,
    # with 0.0 change to all other tokens across the entire batch and sequence.
    cfg = _cfg_identity(token_budget=32)
    m = TemporalConvIdentity(cfg, rngs=nnx.Rngs(0), dtype=jnp.float32)

    b = 2
    img = jax.random.normal(jax.random.key(0), (b, cfg.dataset_token_budget, cfg.memory_feature.img.input_dim))
    pos = jax.random.normal(jax.random.key(1), (b, cfg.dataset_token_budget, cfg.memory_feature.pos.input_dim))
    state = jax.random.normal(jax.random.key(2), (b, cfg.dataset_token_budget, cfg.memory_feature.state.input_dim))
    static_mask = jnp.ones((b, cfg.dataset_token_budget), dtype=jnp.bool_)

    base_out, _, _ = m(img, pos, state, static_mask)

    target_token_idx = 7
    img_pert = img.at[0, target_token_idx, :].add(10.0)
    pert_out, _, _ = m(img_pert, pos, state, static_mask)

    diff = jnp.abs(pert_out - base_out)

    # Changed token must have changed output
    assert float(diff[0, target_token_idx].sum()) > 0.0
    # All other tokens in batch 0 must have exactly zero change
    other_tokens_diff = jnp.delete(diff[0], target_token_idx, axis=0)
    assert float(other_tokens_diff.max()) == 0.0
    # Other batch elements must have exactly zero change
    assert float(diff[1].max()) == 0.0


def test_temporal_conv_identity_all_params_active_gradients():
    # Ensure all parameters in the block receive non-zero gradients
    block = TemporalConvIdentityBlock(dim=12, kernel_size=5, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(42), (2, 32, 12))

    def loss(b, data):
        return jnp.sum(b(data))

    grad = nnx.grad(loss)(block, x)

    assert bool(jnp.all(grad.linear_skip.kernel.value != 0))
    assert bool(jnp.all(grad.linear1.kernel.value != 0))
    assert bool(jnp.all(grad.linear2.kernel.value != 0))
    assert bool(jnp.all(grad.norm.scale.value != 0))


def test_temporal_conv_identity_rejects_budget_mismatch():
    # dataset_token_budget must equal model_token_budget (no compression allowed)
    cfg = types.SimpleNamespace(
        dataset_token_budget=128,
        model_token_budget=32,
        token_per_image=16,
        memory_token_dim=16,
        use_pos_emb=True,
        use_state_emb=False,
        memory_feature=types.SimpleNamespace(
            img=types.SimpleNamespace(input_dim=12),
            pos=types.SimpleNamespace(input_dim=8, hidden_dim=8),
            state=types.SimpleNamespace(input_dim=4, hidden_dim=4),
        ),
    )
    with pytest.raises(AssertionError):
        TemporalConvIdentity(cfg, rngs=nnx.Rngs(1), dtype=jnp.float32)
