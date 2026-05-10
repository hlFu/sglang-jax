import jax
import jax.numpy as jnp
from flax import nnx

from sgl_jax.srt.layers.attention.linear.gated_delta import (
    decode_gated_delta_rule_ref,
    jax_causal_conv1d_prefill,
    jax_causal_conv1d_update,
    ragged_gated_delta_rule_ref,
)
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch


class GDNAttnBackend(nnx.Module):
    """Gated-DeltaNet attention backend.

    Owns the conv1d weight and the gated-delta-rule parameters (``A_log``,
    ``dt_bias``); the parent layer hands in the full per-layer
    ``conv_state`` / ``recurrent_state`` tables plus ``mixed_qkv``, ``b``,
    ``a``. The kernels gather per-request state internally (via
    ``forward_batch.mamba_cache_indices``) and the backend returns
    ``(core_attn_out, new_conv, new_rec)`` where ``new_conv`` and
    ``new_rec`` are per-request — the format
    ``RecurrentStatePool.write_layer`` expects.
    """

    def __init__(
        self,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_dim: int,
        conv_kernel_size: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.conv_dim = conv_dim
        self.conv_kernel_size = conv_kernel_size
        self.mesh = mesh

        # Depthwise conv1d weight (HF stores [conv_dim, 1, K]; we squeeze).
        self.conv1d_weight = nnx.Param(
            jnp.zeros((conv_dim, conv_kernel_size), dtype=dtype)
        )
        # Delta-rule params (fp32 for numerical stability).
        self.A_log = nnx.Param(jnp.zeros((num_v_heads,), dtype=jnp.float32))
        self.dt_bias = nnx.Param(jnp.ones((num_v_heads,), dtype=jnp.float32))

    def __call__(
        self,
        forward_batch: ForwardBatch,
        mixed_qkv: jax.Array,
        conv_state_in: jax.Array,
        recurrent_state_in: jax.Array,
        b: jax.Array,
        a: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                forward_batch, mixed_qkv, conv_state_in, recurrent_state_in, b, a
            )
        return self.forward_extend(
            forward_batch, mixed_qkv, conv_state_in, recurrent_state_in, b, a
        )

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        mixed_qkv: jax.Array,
        conv_state_in: jax.Array,
        recurrent_state_in: jax.Array,
        b: jax.Array,
        a: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Decode-only fast path.

        One token per request — no cross-token dependencies — so the
        recurrence becomes a single ``_gated_delta_step`` parallelised
        across the batch axis (via ``decode_gated_delta_rule_ref``)
        instead of running ``ragged_gated_delta_rule_ref`` with
        ``cu_seqlens = arange(B+1)`` (which would serialise B independent
        steps as a ``T=B`` scan).
        """
        state_indices = forward_batch.mamba_cache_indices
        per_req_conv_state = conv_state_in[state_indices]
        conv_out, new_conv = jax_causal_conv1d_update(
            mixed_qkv,
            per_req_conv_state,
            self.conv1d_weight.value,
            bias=None,
            activation="silu",
        )
        new_rec, core_attn_out = decode_gated_delta_rule_ref(
            conv_out,
            b,
            a,
            recurrent_state_in,
            self.A_log.value,
            self.dt_bias.value,
            state_indices,
            n_kq=self.num_k_heads,
            n_v=self.num_v_heads,
            d_k=self.head_k_dim,
            d_v=self.head_v_dim,
        )
        return core_attn_out, new_conv, new_rec

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        mixed_qkv: jax.Array,
        conv_state_in: jax.Array,
        recurrent_state_in: jax.Array,
        b: jax.Array,
        a: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        cu_seqlens = forward_batch.gdn_metadata.cu_seqlens
        state_indices = forward_batch.mamba_cache_indices
        # jax_causal_conv1d_prefill expects activations as [D, T] and gathers
        # per-seq prior state from the full per-layer table internally.
        conv_out_dt, new_conv = jax_causal_conv1d_prefill(
            x=mixed_qkv.T,
            weight=self.conv1d_weight.value,
            bias=None,
            cu_seqlens=cu_seqlens,
            conv_state=conv_state_in,
            state_indices=state_indices,
            activation="silu",
        )
        conv_out = conv_out_dt.T  # [T, D]

        # has_initial_state[i] is True iff request i already has computed
        # tokens before this extend window — chunked-prefill continuation
        # or prefix-cache hit. False for brand-new prefills.
        has_initial_state = forward_batch.extend_prefix_lens > 0
        new_rec, core_attn_out = ragged_gated_delta_rule_ref(
            conv_out,
            b,
            a,
            recurrent_state_in,
            self.A_log.value,
            self.dt_bias.value,
            cu_seqlens=cu_seqlens,
            state_indices=state_indices,
            has_initial_state=has_initial_state,
            n_kq=self.num_k_heads,
            n_v=self.num_v_heads,
            d_k=self.head_k_dim,
            d_v=self.head_v_dim,
        )
        return core_attn_out, new_conv, new_rec
