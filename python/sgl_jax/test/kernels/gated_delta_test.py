"""Unit tests for gated_delta kernel and causal conv1d helpers.

Reference implementations are pure-numpy loops that follow the HuggingFace
``torch_recurrent_gated_delta_rule`` math exactly. We compare the JAX
implementation to these loops on small shapes with tight tolerances.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.gated_delta import (
    causal_conv1d_prefill,
    causal_conv1d_update,
    fused_recurrent_gated_delta,
    recurrent_gated_delta_step,
)


def _ref_gated_delta_dense(q, k, v, g, beta, initial_state=None, use_qk_l2norm=False):
    """Reference pure-numpy gated delta recurrence, dense [B,T,H,*] layout."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    q = q.astype(np.float32)
    k = k.astype(np.float32)
    v = v.astype(np.float32)
    g = g.astype(np.float32)
    beta = beta.astype(np.float32)
    if use_qk_l2norm:
        q = q / np.sqrt((q**2).sum(-1, keepdims=True) + 1e-6)
        k = k / np.sqrt((k**2).sum(-1, keepdims=True) + 1e-6)
    scale = K**-0.5
    q = q * scale

    S = (
        np.zeros((B, H, K, V), dtype=np.float32)
        if initial_state is None
        else initial_state.astype(np.float32).copy()
    )
    out = np.zeros((B, T, H, V), dtype=np.float32)
    for t in range(T):
        decay = np.exp(g[:, t])[..., None, None]
        S = S * decay
        kv_mem = (S * k[:, t][..., :, None]).sum(axis=-2)
        delta = (v[:, t] - kv_mem) * beta[:, t][..., None]
        S = S + k[:, t][..., :, None] * delta[..., None, :]
        out[:, t] = (S * q[:, t][..., :, None]).sum(axis=-2)
    return out, S


def _ref_gated_delta_packed(q, k, v, g, beta, cu_seqlens, initial_state):
    """Reference for the packed [1,T,H,*] + cu_seqlens path."""
    # q, k, v, g, beta: [1, T_total, H, *] or [1, T_total, H]
    N = len(cu_seqlens) - 1
    B, T_total, H, K = q.shape
    V = v.shape[-1]
    assert B == 1

    out = np.zeros((1, T_total, H, V), dtype=np.float32)
    final_states = initial_state.astype(np.float32).copy()
    for s in range(N):
        start = int(cu_seqlens[s])
        end = int(cu_seqlens[s + 1])
        if end == start:
            continue
        q_s = q[:, start:end]
        k_s = k[:, start:end]
        v_s = v[:, start:end]
        g_s = g[:, start:end]
        beta_s = beta[:, start:end]
        out_s, final_s = _ref_gated_delta_dense(
            q_s, k_s, v_s, g_s, beta_s, initial_state=initial_state[s : s + 1]
        )
        out[:, start:end] = out_s
        final_states[s] = final_s[0]
    return out, final_states


def _ref_causal_conv1d(x, weight, bias=None, initial_state=None):
    """Reference depthwise causal conv1d with state, numpy loop."""
    B, T, D = x.shape
    K = weight.shape[1]
    x = x.astype(np.float32)
    w = weight.astype(np.float32)

    if initial_state is None:
        state = np.zeros((B, D, K - 1), dtype=np.float32)
    else:
        state = initial_state.astype(np.float32).copy()
    y = np.zeros((B, T, D), dtype=np.float32)
    # history buffer of length K per batch/channel, rolling
    for t in range(T):
        # window = [state..., x_t]
        hist = np.concatenate([state, x[:, t : t + 1].transpose(0, 2, 1)], axis=-1)
        y[:, t] = (hist * w[None]).sum(-1)
        # shift state
        state = hist[..., 1:]
    if bias is not None:
        y = y + bias[None, None, :].astype(np.float32)
    return y, state


class TestGatedDelta(unittest.TestCase):
    def _rand(self, key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype=dtype)

    def test_decode_single_step(self):
        rng = jax.random.PRNGKey(0)
        B, H, K, V = 2, 4, 8, 16
        k1, k2, k3, k4, k5, k6 = jax.random.split(rng, 6)
        q = self._rand(k1, (B, 1, H, K))
        k = self._rand(k2, (B, 1, H, K))
        v = self._rand(k3, (B, 1, H, V))
        g = self._rand(k4, (B, 1, H)) * 0.1 - 0.1  # ensure decay ≤ 1
        beta = jax.nn.sigmoid(self._rand(k5, (B, 1, H)))
        init = self._rand(k6, (B, H, K, V))

        out, state = fused_recurrent_gated_delta(q, k, v, g, beta, initial_state=init)
        ref_out, ref_state = _ref_gated_delta_dense(
            np.asarray(q),
            np.asarray(k),
            np.asarray(v),
            np.asarray(g),
            np.asarray(beta),
            initial_state=np.asarray(init),
        )
        np.testing.assert_allclose(np.asarray(out), ref_out, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.asarray(state), ref_state, rtol=1e-4, atol=1e-5)

    def test_prefill_dense_various_lengths(self):
        rng = jax.random.PRNGKey(1)
        B, H, K, V = 2, 3, 8, 16
        for T in (1, 7, 64, 120):
            keys = jax.random.split(rng, 5)
            q = self._rand(keys[0], (B, T, H, K))
            k = self._rand(keys[1], (B, T, H, K))
            v = self._rand(keys[2], (B, T, H, V))
            g = self._rand(keys[3], (B, T, H)) * 0.05 - 0.1
            beta = jax.nn.sigmoid(self._rand(keys[4], (B, T, H)))

            out, state = fused_recurrent_gated_delta(q, k, v, g, beta)
            ref_out, ref_state = _ref_gated_delta_dense(
                np.asarray(q),
                np.asarray(k),
                np.asarray(v),
                np.asarray(g),
                np.asarray(beta),
            )
            # Long scans accumulate float32 rounding; allow ~1e-3 rel drift.
            np.testing.assert_allclose(np.asarray(out), ref_out, rtol=2e-3, atol=1e-4)
            np.testing.assert_allclose(np.asarray(state), ref_state, rtol=2e-3, atol=1e-4)

    def test_prefill_packed_varlen(self):
        rng = jax.random.PRNGKey(2)
        seq_lens = [3, 5, 2, 8]
        cu = np.array([0, *np.cumsum(seq_lens)], dtype=np.int32)
        T = int(cu[-1])
        N, H, K, V = len(seq_lens), 2, 6, 12

        keys = jax.random.split(rng, 6)
        q = self._rand(keys[0], (1, T, H, K))
        k = self._rand(keys[1], (1, T, H, K))
        v = self._rand(keys[2], (1, T, H, V))
        g = self._rand(keys[3], (1, T, H)) * 0.05 - 0.1
        beta = jax.nn.sigmoid(self._rand(keys[4], (1, T, H)))
        init = self._rand(keys[5], (N, H, K, V))

        out, state = fused_recurrent_gated_delta(
            q, k, v, g, beta, initial_state=init, cu_seqlens=cu
        )
        ref_out, ref_state = _ref_gated_delta_packed(
            np.asarray(q),
            np.asarray(k),
            np.asarray(v),
            np.asarray(g),
            np.asarray(beta),
            cu_seqlens=cu,
            initial_state=np.asarray(init),
        )
        np.testing.assert_allclose(np.asarray(out), ref_out, rtol=2e-4, atol=1e-5)
        np.testing.assert_allclose(np.asarray(state), ref_state, rtol=2e-4, atol=1e-5)

    def test_l2norm_option(self):
        rng = jax.random.PRNGKey(3)
        B, T, H, K, V = 1, 5, 2, 4, 4
        keys = jax.random.split(rng, 5)
        q = self._rand(keys[0], (B, T, H, K)) * 3
        k = self._rand(keys[1], (B, T, H, K)) * 3
        v = self._rand(keys[2], (B, T, H, V))
        g = self._rand(keys[3], (B, T, H)) * 0.05 - 0.1
        beta = jax.nn.sigmoid(self._rand(keys[4], (B, T, H)))

        out, _ = fused_recurrent_gated_delta(q, k, v, g, beta, use_qk_l2norm=True)
        ref_out, _ = _ref_gated_delta_dense(
            np.asarray(q),
            np.asarray(k),
            np.asarray(v),
            np.asarray(g),
            np.asarray(beta),
            use_qk_l2norm=True,
        )
        np.testing.assert_allclose(np.asarray(out), ref_out, rtol=5e-4, atol=1e-5)

    def test_decode_wrapper_matches_sequence_version(self):
        rng = jax.random.PRNGKey(4)
        B, H, K, V = 2, 3, 6, 8
        keys = jax.random.split(rng, 6)
        q = self._rand(keys[0], (B, H, K))
        k = self._rand(keys[1], (B, H, K))
        v = self._rand(keys[2], (B, H, V))
        g = self._rand(keys[3], (B, H)) * 0.05 - 0.1
        beta = jax.nn.sigmoid(self._rand(keys[4], (B, H)))
        state = self._rand(keys[5], (B, H, K, V))

        out_a, state_a = recurrent_gated_delta_step(q, k, v, g, beta, state)
        out_b, state_b = fused_recurrent_gated_delta(
            q[:, None],
            k[:, None],
            v[:, None],
            g[:, None],
            beta[:, None],
            initial_state=state,
        )
        np.testing.assert_allclose(np.asarray(out_a), np.asarray(out_b[:, 0]), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(state_a), np.asarray(state_b), rtol=1e-6, atol=1e-6)


class TestCausalConv1d(unittest.TestCase):
    def test_prefill_fresh_state(self):
        rng = jax.random.PRNGKey(10)
        B, T, D, K = 2, 9, 7, 4
        k1, k2, k3 = jax.random.split(rng, 3)
        x = jax.random.normal(k1, (B, T, D))
        w = jax.random.normal(k2, (D, K))
        b = jax.random.normal(k3, (D,))
        y, new_state = causal_conv1d_prefill(x, w, b)
        ref_y, ref_state = _ref_causal_conv1d(np.asarray(x), np.asarray(w), np.asarray(b))
        np.testing.assert_allclose(np.asarray(y), ref_y, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(new_state), ref_state, rtol=1e-5, atol=1e-6)

    def test_prefill_with_initial_state(self):
        rng = jax.random.PRNGKey(11)
        B, T, D, K = 2, 5, 4, 4
        keys = jax.random.split(rng, 4)
        x = jax.random.normal(keys[0], (B, T, D))
        w = jax.random.normal(keys[1], (D, K))
        b = jax.random.normal(keys[2], (D,))
        init = jax.random.normal(keys[3], (B, D, K - 1))
        y, new_state = causal_conv1d_prefill(x, w, b, initial_state=init)
        ref_y, ref_state = _ref_causal_conv1d(
            np.asarray(x),
            np.asarray(w),
            np.asarray(b),
            initial_state=np.asarray(init),
        )
        np.testing.assert_allclose(np.asarray(y), ref_y, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(np.asarray(new_state), ref_state, rtol=1e-5, atol=1e-6)

    def test_update_matches_prefill_one_step(self):
        """Running prefill[T=1] starting from S must match causal_conv1d_update."""
        rng = jax.random.PRNGKey(12)
        B, D, K = 3, 5, 4
        keys = jax.random.split(rng, 4)
        x_token = jax.random.normal(keys[0], (B, D))
        state = jax.random.normal(keys[1], (B, D, K - 1))
        w = jax.random.normal(keys[2], (D, K))
        b = jax.random.normal(keys[3], (D,))
        y_update, state_update = causal_conv1d_update(x_token, state, w, b)

        # Equivalent prefill of one token.
        y_prefill, state_prefill = causal_conv1d_prefill(
            x_token[:, None, :],
            w,
            b,
            initial_state=state,
        )
        np.testing.assert_allclose(
            np.asarray(y_update), np.asarray(y_prefill[:, 0]), rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            np.asarray(state_update), np.asarray(state_prefill), rtol=1e-6, atol=1e-6
        )

    def test_activation_silu(self):
        rng = jax.random.PRNGKey(13)
        B, T, D, K = 1, 4, 3, 4
        keys = jax.random.split(rng, 3)
        x = jax.random.normal(keys[0], (B, T, D))
        w = jax.random.normal(keys[1], (D, K))
        b = jax.random.normal(keys[2], (D,))
        y_act, _ = causal_conv1d_prefill(x, w, b, activation="silu")
        y_raw, _ = causal_conv1d_prefill(x, w, b)
        np.testing.assert_allclose(
            np.asarray(y_act),
            np.asarray(jax.nn.silu(y_raw)),
            rtol=1e-6,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
