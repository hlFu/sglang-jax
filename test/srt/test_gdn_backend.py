"""Unit tests for :class:`GDNAttnBackend`.

The kernels themselves are covered in ``test_gated_delta.py`` and
``test_ragged_gated_delta_rule_ref.py``; this file exercises the backend
glue: ``__init__`` parameter ownership, decode/extend dispatch, conv +
recurrent-rule pipeline, and the contract that ``forward_*`` returns
``(core_attn_out, new_conv, new_rec)`` shaped for
``RecurrentStatePool.write_layer``.

Run with:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m pytest test/srt/test_gdn_backend.py -v
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.layers.attention.linear.gated_delta import (
    jax_causal_conv1d_update,
    ragged_gated_delta_rule_ref,
)
from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_mesh():
    devices = mesh_utils.create_device_mesh((8,))
    return Mesh(devices[:1], ("tensor",), axis_types=(AxisType.Explicit,))


def _make_backend(mesh, n_kq=1, n_v=2, d_k=4, d_v=8, K=3):
    conv_dim = 2 * n_kq * d_k + n_v * d_v
    backend = GDNAttnBackend(
        num_k_heads=n_kq, num_v_heads=n_v,
        head_k_dim=d_k, head_v_dim=d_v,
        conv_dim=conv_dim, conv_kernel_size=K,
        mesh=mesh, dtype=jnp.bfloat16,
    )
    rng = jax.random.split(jax.random.key(0), 3)
    backend.conv1d_weight = nnx.Param(
        jax.random.normal(rng[0], (conv_dim, K), dtype=jnp.bfloat16) * 0.1
    )
    backend.A_log = nnx.Param(jax.random.normal(rng[1], (n_v,)) * 0.3)
    backend.dt_bias = nnx.Param(jax.random.normal(rng[2], (n_v,)) * 0.3)
    return backend, conv_dim, K


class _FakeForwardMode:
    def __init__(self, decode: bool):
        self._decode = decode

    def is_decode(self):
        return self._decode


class _FakeGDNMetadata:
    def __init__(self, cu_seqlens):
        self.cu_seqlens = cu_seqlens


class _FakeForwardBatch:
    def __init__(
        self,
        is_decode: bool,
        mamba_cache_indices,
        cu_seqlens=None,
        extend_prefix_lens=None,
    ):
        self.forward_mode = _FakeForwardMode(is_decode)
        self.mamba_cache_indices = mamba_cache_indices
        self.gdn_metadata = _FakeGDNMetadata(cu_seqlens)
        self.extend_prefix_lens = extend_prefix_lens


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class GDNAttnBackendInitTest(unittest.TestCase):
    def test_param_shapes_match_config(self):
        """Construction allocates conv1d_weight, A_log, dt_bias at the
        shapes that weight loading targets."""
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            n_kq, n_v, d_k, d_v, K = 2, 4, 8, 16, 4
            conv_dim = 2 * n_kq * d_k + n_v * d_v
            backend = GDNAttnBackend(
                num_k_heads=n_kq, num_v_heads=n_v,
                head_k_dim=d_k, head_v_dim=d_v,
                conv_dim=conv_dim, conv_kernel_size=K,
                mesh=mesh, dtype=jnp.bfloat16,
            )
            self.assertEqual(backend.conv1d_weight.value.shape, (conv_dim, K))
            self.assertEqual(backend.A_log.value.shape, (n_v,))
            self.assertEqual(backend.dt_bias.value.shape, (n_v,))


class GDNAttnBackendDispatchTest(unittest.TestCase):
    """``__call__`` should route to forward_decode/forward_extend based on
    forward_mode.is_decode()."""

    def test_decode_dispatch(self):
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            backend, conv_dim, K = _make_backend(mesh)
            B = 2
            mq = jnp.ones((B, conv_dim), dtype=jnp.bfloat16) * 0.1
            cs = jnp.zeros((B + 1, conv_dim, K - 1), dtype=jnp.bfloat16)
            rs = jnp.zeros((B + 1, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim), dtype=jnp.float32)
            b = jnp.zeros((B, backend.num_v_heads), dtype=jnp.bfloat16)
            a = jnp.zeros((B, backend.num_v_heads), dtype=jnp.bfloat16)
            fb = _FakeForwardBatch(
                is_decode=True,
                mamba_cache_indices=jnp.array([1, 2], dtype=jnp.int32),
            )
            out, new_conv, new_rec = backend(fb, mq, cs, rs, b, a)
            # Shapes match the decode contract: B-row outputs.
            self.assertEqual(out.shape, (B, backend.num_v_heads, backend.head_v_dim))
            self.assertEqual(new_conv.shape, (B, conv_dim, K - 1))
            self.assertEqual(new_rec.shape, (B, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim))

    def test_extend_dispatch(self):
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            backend, conv_dim, K = _make_backend(mesh)
            T = 5  # 2 reqs of lengths [3, 2]
            mq = jnp.ones((T, conv_dim), dtype=jnp.bfloat16) * 0.1
            cs = jnp.zeros((3, conv_dim, K - 1), dtype=jnp.bfloat16)
            rs = jnp.zeros((3, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim), dtype=jnp.float32)
            b = jnp.zeros((T, backend.num_v_heads), dtype=jnp.bfloat16)
            a = jnp.zeros((T, backend.num_v_heads), dtype=jnp.bfloat16)
            fb = _FakeForwardBatch(
                is_decode=False,
                mamba_cache_indices=jnp.array([1, 2], dtype=jnp.int32),
                cu_seqlens=jnp.array([0, 3, 5], dtype=jnp.int32),
                extend_prefix_lens=jnp.array([0, 0], dtype=jnp.int32),
            )
            out, new_conv, new_rec = backend(fb, mq, cs, rs, b, a)
            # Output is per-token; new_conv/new_rec are per-request (B=2).
            self.assertEqual(out.shape, (T, backend.num_v_heads, backend.head_v_dim))
            self.assertEqual(new_conv.shape, (2, conv_dim, K - 1))
            self.assertEqual(new_rec.shape, (2, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim))


class GDNAttnBackendDecodeFastPathTest(unittest.TestCase):
    """forward_decode (parallel single-step) must produce the same
    numerical result as feeding the same per-token data through
    ragged_gated_delta_rule_ref with cu_seqlens=arange(B+1)."""

    def test_decode_equals_ragged_with_singletons(self):
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            backend, conv_dim, K = _make_backend(mesh)
            B = 4
            rng = jax.random.split(jax.random.key(42), 4)
            mq = jax.random.normal(rng[0], (B, conv_dim), dtype=jnp.bfloat16) * 0.3
            cs = jax.random.normal(rng[1], (B + 1, conv_dim, K - 1), dtype=jnp.bfloat16) * 0.1
            rs = jax.random.normal(rng[2], (B + 1, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim), dtype=jnp.float32) * 0.05
            b = jax.random.normal(rng[3], (B, backend.num_v_heads), dtype=jnp.bfloat16) * 0.5
            a = jax.random.normal(jax.random.key(43), (B, backend.num_v_heads), dtype=jnp.bfloat16) * 0.5
            state_indices = jnp.array([1, 2, 3, 4], dtype=jnp.int32)

            # Path A: backend's forward_decode.
            fb = _FakeForwardBatch(is_decode=True, mamba_cache_indices=state_indices)
            out_d, nc_d, nr_d = backend(fb, mq, cs, rs, b, a)

            # Path B: same conv1d_update + ragged kernel with singleton seqs.
            per_req_conv = cs[state_indices]
            conv_out, nc_ref = jax_causal_conv1d_update(
                mq, per_req_conv, backend.conv1d_weight.value, bias=None, activation="silu",
            )
            nr_r, out_r = ragged_gated_delta_rule_ref(
                conv_out, b, a, rs, backend.A_log.value, backend.dt_bias.value,
                cu_seqlens=jnp.arange(B + 1, dtype=jnp.int32),
                state_indices=state_indices,
                has_initial_state=jnp.ones((B,), dtype=jnp.bool_),
                n_kq=backend.num_k_heads, n_v=backend.num_v_heads,
                d_k=backend.head_k_dim, d_v=backend.head_v_dim,
            )
            np.testing.assert_allclose(out_d, out_r, atol=1e-3, rtol=1e-3)
            np.testing.assert_allclose(nr_d, nr_r, atol=1e-4, rtol=1e-4)
            # Conv state writeback should also match (it's the same kernel).
            np.testing.assert_allclose(nc_d, nc_ref, atol=0)


class GDNAttnBackendExtendStateTest(unittest.TestCase):
    """forward_extend should write back per-request state suitable for
    RecurrentStatePool.write_layer (i.e. shape == (B, ...))."""

    def test_extend_returns_per_request_state_shape(self):
        mesh = _make_mesh()
        with jax.set_mesh(mesh):
            backend, conv_dim, K = _make_backend(mesh)
            # 3 reqs, lengths [4, 2, 1] = T=7.
            lens = [4, 2, 1]
            T = sum(lens)
            B = len(lens)
            rng = jax.random.split(jax.random.key(50), 4)
            mq = jax.random.normal(rng[0], (T, conv_dim), dtype=jnp.bfloat16) * 0.3
            cs = jax.random.normal(rng[1], (B + 1, conv_dim, K - 1), dtype=jnp.bfloat16) * 0.1
            rs = jax.random.normal(rng[2], (B + 1, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim), dtype=jnp.float32) * 0.05
            b = jax.random.normal(rng[3], (T, backend.num_v_heads), dtype=jnp.bfloat16) * 0.5
            a = jax.random.normal(jax.random.key(51), (T, backend.num_v_heads), dtype=jnp.bfloat16) * 0.5

            fb = _FakeForwardBatch(
                is_decode=False,
                mamba_cache_indices=jnp.array([1, 2, 3], dtype=jnp.int32),
                cu_seqlens=jnp.array([0, 4, 6, 7], dtype=jnp.int32),
                extend_prefix_lens=jnp.array([0, 0, 0], dtype=jnp.int32),
            )
            out, new_conv, new_rec = backend(fb, mq, cs, rs, b, a)
            self.assertEqual(out.shape, (T, backend.num_v_heads, backend.head_v_dim))
            self.assertEqual(new_conv.shape, (B, conv_dim, K - 1))
            self.assertEqual(new_rec.shape, (B, backend.num_v_heads, backend.head_k_dim, backend.head_v_dim))
            self.assertTrue(bool(jnp.all(jnp.isfinite(out))))
            self.assertTrue(bool(jnp.all(jnp.isfinite(new_rec))))


if __name__ == "__main__":
    unittest.main()
