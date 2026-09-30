import os
import types

import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax.numpy as jnp

from mme_vla_suite.models.representation.select_mem import Block
from mme_vla_suite.models.representation.select_mem import CrossAttention
from mme_vla_suite.models.representation.select_mem import FeedForward
from mme_vla_suite.models.representation.select_mem import SelectiveMemory
from mme_vla_suite.models.representation.select_mem import SelfAttention


def _cfg(mask_or_pad: str = "mask", mem_sink_len: int = 1, query_reg_len: int = 0):
    token_per_image = 2
    return types.SimpleNamespace(
        selective_memory=types.SimpleNamespace(
            attn_head_dim=4,
            attn_heads=4,
            mlp_dim=16,
            attn_blocks=2,
            mask_or_pad=mask_or_pad,
            mem_sink_len=mem_sink_len,
            query_reg_len=query_reg_len,
        ),
        memory_token_dim=8,
        dataset_token_budget=6,
        model_token_budget=token_per_image * (1 + query_reg_len),
        token_per_image=token_per_image,
        memory_feature=types.SimpleNamespace(
            pos=types.SimpleNamespace(input_dim=6),
            img=types.SimpleNamespace(input_dim=5),
        ),
    )


def test_self_attention_forward_and_jittable():
    cfg = _cfg()
    m = SelfAttention(cfg, rngs=nnx.Rngs(0), dtype=jnp.float32)
    x = jnp.ones((3, 4, cfg.memory_token_dim), dtype=jnp.float32)

    y = m(x)
    y_jit = nnx.jit(lambda mod, x_: mod(x_))(m, x)

    assert y.shape == (3, 4, cfg.memory_token_dim)
    assert jnp.allclose(y, y_jit, atol=1e-5)


def test_cross_attention_forward_and_jittable():
    cfg = _cfg()
    m = CrossAttention(cfg, rngs=nnx.Rngs(1), dtype=jnp.float32)
    x_q = jnp.ones((3, 3, cfg.memory_token_dim), dtype=jnp.float32)
    x_kv = jnp.ones((3, 5, cfg.memory_token_dim), dtype=jnp.float32)
    mask = jnp.ones((3, 1, 3, 5), dtype=jnp.bool_)  # (b, 1, lq, lk)

    y = m(x_q, x_kv, mask)
    y_jit = nnx.jit(lambda mod, q, kv, m_: mod(q, kv, m_))(m, x_q, x_kv, mask)

    assert y.shape == (3, 3, cfg.memory_token_dim)
    assert jnp.allclose(y, y_jit, atol=1e-5)


def test_feedforward_forward_and_jittable():
    cfg = _cfg()
    m = FeedForward(cfg, rngs=nnx.Rngs(2), dtype=jnp.float32)
    x = jnp.ones((3, 4, cfg.memory_token_dim), dtype=jnp.float32)

    y = m(x)
    y_jit = nnx.jit(lambda mod, x_: mod(x_))(m, x)

    assert y.shape == (3, 4, cfg.memory_token_dim)
    assert jnp.allclose(y, y_jit, atol=1e-5)


def test_block_forward_and_jittable():
    cfg = _cfg()
    m = Block(cfg, rngs=nnx.Rngs(3), dtype=jnp.float32)
    x_q = jnp.ones((3, 2, cfg.memory_token_dim), dtype=jnp.float32)
    x_kv = jnp.ones((3, cfg.dataset_token_budget, cfg.memory_token_dim), dtype=jnp.float32)
    mask = jnp.ones((3, 1, 2, cfg.dataset_token_budget), dtype=jnp.bool_)  # (b, 1, lq, lk)

    y = m(x_q, x_kv, mask)
    y_jit = nnx.jit(lambda mod, q, kv, m_: mod(q, kv, m_))(m, x_q, x_kv, mask)

    assert y.shape == (3, 2, cfg.memory_token_dim)
    assert jnp.allclose(y, y_jit, atol=1e-5)


@pytest.mark.parametrize("mask_or_pad", ["mask", "pad"])
def test_selective_memory_mask_or_pad(mask_or_pad):
    cfg = _cfg(mask_or_pad=mask_or_pad)
    m = SelectiveMemory(cfg, rngs=nnx.Rngs(10), dtype=jnp.float32)

    b = 3
    static_image_emb = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.img.input_dim), dtype=jnp.float32)
    static_pos_emb = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.pos.input_dim), dtype=jnp.float32)
    # partial mask: last token is invalid
    static_mask = jnp.concatenate(
        [jnp.ones((b, cfg.dataset_token_budget - 1), dtype=jnp.bool_), jnp.zeros((b, 1), dtype=jnp.bool_)], axis=1
    )

    y, out_mask, metrics = m(static_image_emb, static_pos_emb, static_mask)
    assert y.shape == (b, cfg.model_token_budget, cfg.memory_token_dim)
    assert out_mask.shape == (b, cfg.model_token_budget)
    assert jnp.all(out_mask)
    assert metrics is not None

    if mask_or_pad == "pad":
        assert m.padding_token is not None
        assert m.padding_token.value.shape == (1, 1, cfg.memory_token_dim)
    else:
        assert m.padding_token is None


def test_selective_memory_invalid_mask_or_pad():
    cfg = _cfg(mask_or_pad="invalid")
    with pytest.raises(ValueError, match="Invalid mask_or_pad"):
        SelectiveMemory(cfg, rngs=nnx.Rngs(11), dtype=jnp.float32)


def test_selective_memory_forward_and_jittable():
    cfg = _cfg()
    m = SelectiveMemory(cfg, rngs=nnx.Rngs(4), dtype=jnp.float32)

    b = 3
    static_image_emb = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.img.input_dim), dtype=jnp.float32)
    static_pos_emb = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.pos.input_dim), dtype=jnp.float32)
    static_mask = jnp.ones((b, cfg.dataset_token_budget), dtype=jnp.bool_)  # (b, l)

    y, out_mask, metrics = m(static_image_emb, static_pos_emb, static_mask)
    y_jit, out_mask_jit, metrics_jit = nnx.jit(lambda mod, a, b_, c: mod(a, b_, c))(
        m, static_image_emb, static_pos_emb, static_mask
    )

    assert y.shape == (b, cfg.model_token_budget, cfg.memory_token_dim)
    assert out_mask.shape == (b, cfg.model_token_budget)
    assert jnp.all(out_mask)
    assert metrics is not None and metrics_jit is not None
    assert "selective_memory/attention_entropy" in metrics
    assert "selective_memory/attention_entropy_per_block" in metrics
    assert metrics["selective_memory/attention_entropy_per_block"].shape == (cfg.selective_memory.attn_blocks,)
    assert jnp.allclose(
        metrics["selective_memory/attention_entropy"],
        metrics_jit["selective_memory/attention_entropy"],
        atol=1e-5,
    )
    assert jnp.allclose(
        metrics["selective_memory/attention_entropy_per_block"],
        metrics_jit["selective_memory/attention_entropy_per_block"],
        atol=1e-5,
    )
    assert jnp.allclose(y, y_jit, atol=1e-5)
    assert jnp.array_equal(out_mask, out_mask_jit)


def test_selective_memory_query_register_length_affects_output_shape():
    query_reg_len = 2
    cfg = _cfg(mask_or_pad="mask", query_reg_len=query_reg_len)
    m = SelectiveMemory(cfg, rngs=nnx.Rngs(12), dtype=jnp.float32)

    b = 2
    static_image_emb = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.img.input_dim), dtype=jnp.float32)
    static_pos_emb = jnp.ones((b, cfg.dataset_token_budget, cfg.memory_feature.pos.input_dim), dtype=jnp.float32)
    static_mask = jnp.ones((b, cfg.dataset_token_budget), dtype=jnp.bool_)

    y, out_mask, metrics = m(static_image_emb, static_pos_emb, static_mask)

    expected_lq = cfg.token_per_image * (1 + query_reg_len)
    assert y.shape == (b, cfg.model_token_budget, cfg.memory_token_dim)
    assert cfg.model_token_budget == expected_lq  # sanity
    assert out_mask.shape == (b, cfg.model_token_budget)
    assert metrics is not None
