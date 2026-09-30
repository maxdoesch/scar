import flax.nnx as nnx
import jax
import jax.numpy as jnp

kernel_init = nnx.initializers.normal(stddev=0.02)
kernel_init_out_proj = nnx.initializers.normal(stddev=0.002)


@jax.jit
def rms_normalize(x):
    dtype = x.dtype
    x = x.astype(jnp.float32)
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    norm_x = x / (norm + 1e-6)
    return norm_x.astype(dtype)
