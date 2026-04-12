# SPDX-License-Identifier: Apache-2.0
"""Quantized matmul kernel."""

import jax
import jax.numpy as jnp
from jax import lax

from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor_simple


def xla_quantized_matmul_local(
    x: jax.Array,
    w_scale: jax.Array,
    x_scale: jax.Array | None = None,
    quantize_activation: bool = True,
    compute_dtype: jnp.dtype | None = None,
) -> jax.Array:
    """
    Local quantized matmul for use inside shard_map.

    All computation (quantize, matmul, dequantize) happens locally on each device.
    If reduce_axis is provided, uses psum to combine partial sums across devices.

    Args:
        x: Activation tensor [batch, n_input_features] (local slice)
        w_q: Quantized weight tensor [n_output_features, n_input_features] (local slice)
        w_scale: Weight quantization scale [n_output_features]
        quantize_activation: Whether to quantize activations
        reduce_axis: Axis name for psum reduction (e.g., "tensor"). None skips reduction.

    Returns:
        Output of the quantized matmul.
    """
    out_dtype = x.dtype
    compute_dtype = jnp.float32 if compute_dtype is None else compute_dtype

    if quantize_activation:
        # Local dequantization
        x = x.astype(compute_dtype)
        x = (
            x * x_scale.astype(compute_dtype) * jnp.expand_dims(w_scale, 0).astype(compute_dtype)
        )
    else:
        # Local matmul without activation quantization
        
        x = x.astype(compute_dtype)
        x = x * jnp.expand_dims(w_scale, 0).astype(compute_dtype)

    x = x.astype(out_dtype)
    # Sum partial results across devices (single all-reduce)

    return x
