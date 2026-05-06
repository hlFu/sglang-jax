"""Gated Delta-Rule recurrence and causal conv1d helpers for Qwen3-Next.

The math matches HuggingFace ``torch_recurrent_gated_delta_rule``
(``transformers/models/qwen3_next/modeling_qwen3_next.py``): each step

    scale     = 1 / sqrt(K)
    q_t       = q_t * scale
    S_t       = S_{t-1} * exp(g_t)
    kv_mem_t  = (S_t * k_t[..., None]).sum(axis=-2)
    delta_t   = (v_t - kv_mem_t) * beta_t
    S_t       = S_t + k_t[..., None] * delta_t[..., None, :]
    o_t       = (S_t * q_t[..., None]).sum(axis=-2)

with S stored in ``[K, V]`` order per-head. The implementation is pure JAX
(``jax.lax.scan``); a Pallas kernel can be plugged in later behind the same API.

Two entry points:

* :func:`fused_recurrent_gated_delta` — sequence-level forward (prefill + decode).
  Handles both dense ``[B, T, H, K]`` batches and packed ``[1, T, H, K]`` inputs
  with ``cu_seqlens``.
* :func:`recurrent_gated_delta_step` — single-token decode step; a thin wrapper
  that forwards to the sequence version with ``T=1``.

Two helpers for the short causal conv1d that front-runs the delta rule:

* :func:`causal_conv1d_prefill` — depthwise conv1d over a padded sequence
  batch, returning both the conv output and the new conv state.
* :func:`causal_conv1d_update` — single-step update using a carried state.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _l2norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    norm = jnp.sqrt((x.astype(jnp.float32) ** 2).sum(axis=-1, keepdims=True) + eps)
    return (x.astype(jnp.float32) / norm).astype(x.dtype)


def _gated_delta_step(
    state: jax.Array,  # [B, H, K, V] float32
    q_t: jax.Array,  # [B, H, K]
    k_t: jax.Array,  # [B, H, K]
    v_t: jax.Array,  # [B, H, V]
    g_t: jax.Array,  # [B, H] log-decay
    beta_t: jax.Array,  # [B, H]
) -> tuple[jax.Array, jax.Array]:
    """Single gated delta step. Returns (new_state [B,H,K,V], out [B,H,V])."""
    decay = jnp.exp(g_t.astype(jnp.float32))[..., None, None]  # [B,H,1,1]
    state = state * decay
    kv_mem = (state * k_t[..., :, None]).sum(axis=-2)  # [B,H,V]
    delta = (v_t - kv_mem) * beta_t[..., None]  # [B,H,V]
    state = state + k_t[..., :, None] * delta[..., None, :]  # [B,H,K,V]
    out = (state * q_t[..., :, None]).sum(axis=-2)  # [B,H,V]
    return state, out


def fused_recurrent_gated_delta(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    g: jax.Array,
    beta: jax.Array,
    initial_state: jax.Array | None = None,
    cu_seqlens: jax.Array | np.ndarray | None = None,
    scale: float | None = None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = False,
) -> tuple[jax.Array, jax.Array | None]:
    """Gated delta-rule forward over a sequence axis.

    Args:
        q: ``[B, T, H, K]`` queries.
        k: ``[B, T, H, K]`` keys (same shape as ``q``).
        v: ``[B, T, H, V]`` values.
        g: ``[B, T, H]`` per-token log-decay (``S = S * exp(g)``).
        beta: ``[B, T, H]`` delta-rule update coefficient.
        initial_state: optional ``[N, H, K, V]`` state carried in. For the
            dense path ``N == B``. For the packed path (``cu_seqlens`` set,
            which requires ``B == 1``) ``N == len(cu_seqlens) - 1``.
        cu_seqlens: optional ``[N+1]`` int cumulative sequence lengths for
            packed mode. If given, ``B`` must be 1 and ``T == cu_seqlens[-1]``.
        scale: query scaling factor. Defaults to ``K ** -0.5``.
        output_final_state: if True, return the per-sequence final state.
        use_qk_l2norm: if True, L2-normalise q and k along the head dim before
            the scan (matches HF's ``use_qk_l2norm_in_kernel``).

    Returns:
        ``(output, final_state)`` where ``output`` has shape ``[B, T, H, V]``
        in ``v.dtype`` and ``final_state`` has shape ``[N, H, K, V]`` in fp32
        (or ``None`` when ``output_final_state`` is False).
    """
    assert q.ndim == 4 and k.shape == q.shape
    assert v.ndim == 4 and v.shape[:3] == q.shape[:3]
    assert g.shape == q.shape[:3] and beta.shape == q.shape[:3]

    B, T, H, K = q.shape
    V = v.shape[-1]

    if cu_seqlens is not None:
        assert B == 1, f"cu_seqlens requires B=1, got B={B}"
        cu = jnp.asarray(cu_seqlens, dtype=jnp.int32)
        N = int(cu.shape[0]) - 1
    else:
        N = B

    if scale is None:
        scale = K**-0.5

    if use_qk_l2norm:
        q = _l2norm(q)
        k = _l2norm(k)

    in_dtype = v.dtype
    q_f = q.astype(jnp.float32) * float(scale)
    k_f = k.astype(jnp.float32)
    v_f = v.astype(jnp.float32)
    g_f = g.astype(jnp.float32)
    beta_f = beta.astype(jnp.float32)

    if initial_state is None:
        h0 = jnp.zeros((N, H, K, V), dtype=jnp.float32)
    else:
        assert initial_state.shape == (
            N,
            H,
            K,
            V,
        ), f"initial_state shape {initial_state.shape} != expected {(N, H, K, V)}"
        h0 = initial_state.astype(jnp.float32)

    if cu_seqlens is None:
        # Dense path: independent scan per batch element.
        # Rearrange to scan axis first: [T, B, H, *].
        q_t = jnp.transpose(q_f, (1, 0, 2, 3))
        k_t = jnp.transpose(k_f, (1, 0, 2, 3))
        v_t = jnp.transpose(v_f, (1, 0, 2, 3))
        g_t = jnp.transpose(g_f, (1, 0, 2))
        beta_t = jnp.transpose(beta_f, (1, 0, 2))

        def body(state, inputs):
            qi, ki, vi, gi, bi = inputs
            state, out = _gated_delta_step(state, qi, ki, vi, gi, bi)
            return state, out

        final_state, outs = jax.lax.scan(body, h0, (q_t, k_t, v_t, g_t, beta_t))
        out = jnp.transpose(outs, (1, 0, 2, 3))  # [B, T, H, V]
        out = out.astype(in_dtype)
        return out, (final_state if output_final_state else None)

    # Packed path: B == 1. The scan walks T and resets state at seq boundaries.
    # We tag each token with its sequence id and use that to load/save state.
    seq_id = jnp.searchsorted(cu[1:], jnp.arange(T, dtype=jnp.int32), side="right")

    q_s = q_f[0]  # [T, H, K]
    k_s = k_f[0]
    v_s = v_f[0]
    g_s = g_f[0]
    beta_s = beta_f[0]

    # Token position within its sequence (0 at each seq start).
    token_offset = jnp.arange(T, dtype=jnp.int32) - cu[:-1][seq_id]
    is_first_token = token_offset == 0

    def body(carry, inputs):
        # carry: (current_state [H,K,V], final_states [N,H,K,V])
        state, final_states = carry
        qi, ki, vi, gi, bi, sid, first = inputs
        # At sequence boundary, reload state from initial_state[sid].
        state = jnp.where(first, h0[sid], state)
        new_state, out = _gated_delta_step(
            state[None], qi[None], ki[None], vi[None], gi[None], bi[None]
        )
        new_state = new_state[0]  # drop the B=1 axis
        # Record current state into final_states[sid] every token; the last
        # write per sid wins (i.e. end-of-sequence state).
        final_states = final_states.at[sid].set(new_state)
        return (new_state, final_states), out[0]  # out[0]: [H, V]

    init_carry = (jnp.zeros((H, K, V), dtype=jnp.float32), h0)
    (_, final_states), outs = jax.lax.scan(
        body, init_carry, (q_s, k_s, v_s, g_s, beta_s, seq_id, is_first_token)
    )
    out = outs[None].astype(in_dtype)  # [1, T, H, V]
    return out, (final_states if output_final_state else None)


def recurrent_gated_delta_step(
    q: jax.Array,  # [B, H, K]
    k: jax.Array,  # [B, H, K]
    v: jax.Array,  # [B, H, V]
    g: jax.Array,  # [B, H]
    beta: jax.Array,  # [B, H]
    state: jax.Array,  # [B, H, K, V] float32
    scale: float | None = None,
    use_qk_l2norm: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Single decode step of gated delta rule. Returns (output [B,H,V], new_state)."""
    out, new_state = fused_recurrent_gated_delta(
        q[:, None],
        k[:, None],
        v[:, None],
        g[:, None],
        beta[:, None],
        initial_state=state,
        cu_seqlens=None,
        scale=scale,
        output_final_state=True,
        use_qk_l2norm=use_qk_l2norm,
    )
    return out[:, 0], new_state


# ---------------------------------------------------------------------------
# Causal conv1d (depthwise, kernel_size=K, stride=1, dilation=1)
# ---------------------------------------------------------------------------

def jax_causal_conv1d_prefill(
    x: jax.Array,  # [D, T]  activations
    weight: jax.Array,  # [D, kernel_size]  depthwise weight
    query_start_loc: jax.Array,
    bias: jax.Array | None = None,  # [D] optional
    initial_state: jax.Array | None = None,  # [B, D, kernel_size-1] carried in
    activation: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Depthwise causal conv1d over a ragged-batched packed sequence.

    Sequences are concatenated along the token axis. Boundaries are given by
    ``query_start_loc`` (``[0, len_0, len_0+len_1, ...]``). Each output
    position only mixes inputs from its own request — boundary lookbacks are
    served from ``initial_state`` if provided, else zero.

    Returns ``(y [D, T], final_state [B, D, K-1])``. ``final_state`` holds the
    last ``K-1`` logical tokens of each request, with left-padding (zero or
    from ``initial_state``) when the request is shorter than ``K-1``.
    """
    if activation not in (None, "silu"):
        raise ValueError(f"Unsupported causal conv1d activation: {activation}")

    D, T = x.shape
    K = int(weight.shape[1])
    B = int(query_start_loc.shape[0]) - 1
    assert weight.shape == (D, K), f"weight {weight.shape} vs x {x.shape}"
    if initial_state is not None:
        assert initial_state.shape == (B, D, K - 1)

    starts = query_start_loc[:-1]  # [B] inclusive
    ends = query_start_loc[1:]  # [B] exclusive
    seq_lens = ends - starts  # [B]

    # Map each packed token index to its request id and intra-request position.
    t_idx = jnp.arange(T)
    seq_idx = jnp.searchsorted(query_start_loc, t_idx, side="right") - 1  # [T]
    pos = t_idx - starts[seq_idx]  # [T]

    # Build the depthwise window. For each lookback o in [0, K-1] the source
    # logical position is p' = pos[t] - o; in-request when p' >= 0, otherwise
    # served from initial_state slot (K-1) + p' (newest at K-2).
    o = jnp.arange(K)
    src_t = t_idx[:, None] - o[None, :]  # [T, K]
    in_seq = src_t >= starts[seq_idx][:, None]  # [T, K]
    src_t_safe = jnp.clip(src_t, 0, T - 1)
    x_gathered = x[:, src_t_safe]  # [D, T, K]

    if initial_state is not None and K > 1:
        p_prime = pos[:, None] - o[None, :]  # [T, K]
        is_idx = jnp.clip((K - 1) + p_prime, 0, K - 2)  # [T, K]
        init_pulled = initial_state[seq_idx[:, None], :, is_idx]  # [T, K, D]
        init_pulled = jnp.transpose(init_pulled, (2, 0, 1))  # [D, T, K]
        x_gathered = jnp.where(in_seq[None], x_gathered, init_pulled)
    elif K > 1:
        x_gathered = jnp.where(in_seq[None], x_gathered, jnp.zeros_like(x_gathered))

    # weight[d, K-1-o] is the coefficient for lookback o.
    w_flipped = weight[:, ::-1].astype(x.dtype)  # [D, K]
    y = jnp.sum(x_gathered * w_flipped[:, None, :], axis=-1)  # [D, T]
    if bias is not None:
        y = y + bias.astype(x.dtype)[:, None]
    if activation == "silu":
        y = jax.nn.silu(y)

    # Final state: the K-1 most-recent logical tokens of each request.
    if K > 1:
        j = jnp.arange(K - 1)[None, :]  # [1, K-1]
        logical_idx = seq_lens[:, None] - (K - 1) + j  # [B, K-1]
        take_from_x = logical_idx >= 0
        src_t_end_safe = jnp.clip(starts[:, None] + logical_idx, 0, T - 1)
        from_x = jnp.transpose(x[:, src_t_end_safe], (1, 0, 2))  # [B, D, K-1]
        if initial_state is not None:
            is_slot = jnp.clip((K - 1) + logical_idx, 0, K - 2)  # [B, K-1]
            b_idx = jnp.arange(B)[:, None]
            from_init = initial_state[b_idx, :, is_slot]  # [B, K-1, D]
            from_init = jnp.transpose(from_init, (0, 2, 1))  # [B, D, K-1]
            final_state = jnp.where(take_from_x[:, None, :], from_x, from_init)
        else:
            final_state = jnp.where(
                take_from_x[:, None, :], from_x, jnp.zeros_like(from_x)
            )
    else:
        final_state = jnp.zeros((B, D, 0), dtype=x.dtype)

    return y, final_state


def jax_causal_conv1d_update(
    x: jax.Array,  # [B, D]  one new token per batch element
    state: jax.Array,  # [B, D, kernel_size-1]
    weight: jax.Array,  # [D, kernel_size]
    bias: jax.Array | None = None,  # [D]
    activation: str | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Single-token causal conv1d update.

    Returns ``(y [B, D], new_state [B, D, kernel_size-1])``.
    """
    assert x.ndim == 2, f"x must be [B, D], got shape {x.shape}"
    B, D = x.shape
    kernel = int(weight.shape[1])
    assert state.shape == (B, D, kernel - 1)

    # Rolling buffer: [state(kernel-1), x_new] → window of length kernel.
    window = jnp.concatenate([state, x[..., None]], axis=-1)  # [B, D, K]
    y = jnp.sum(window * weight[None, :, :].astype(x.dtype), axis=-1)  # [B, D]
    if bias is not None:
        y = y + bias.astype(x.dtype)[None, :]
    if activation == "silu":
        y = jax.nn.silu(y)
    elif activation is None:
        pass
    else:
        raise ValueError(f"Unsupported causal conv1d activation: {activation}")
    new_state = window[..., 1:]  # drop oldest
    return y, new_state
