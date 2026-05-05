import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.layers.attention.gated_delta import (
    jax_causal_conv1d_prefill,
    jax_causal_conv1d_update,
    fused_recurrent_gated_delta,
)

causal_conv1d_prefill = jax_causal_conv1d_prefill
causal_conv1d_update = jax_causal_conv1d_update

class GDNAttnBackend(nnx.Module):
    def __init__(self):
        super().__init__()

    def _prepare(self, conv_out, b, a):
        seq_len = conv_out.shape[0]

        # Split conv output back into Q/K/V.
        q_mix = conv_out[:, : self.key_dim]
        k_mix = conv_out[:, self.key_dim : 2 * self.key_dim]
        v_mix = conv_out[:, 2 * self.key_dim :]
        q_mix = jax.lax.reshape(
            q_mix,
            (seq_len, self.num_k_heads, self.head_k_dim),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor", None)),
        )
        k_mix = jax.lax.reshape(
            k_mix,
            (seq_len, self.num_k_heads, self.head_k_dim),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor", None)),
        )
        v_mix = jax.lax.reshape(
            v_mix,
            (seq_len, self.num_v_heads, self.head_v_dim),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor", None)),
        )

        # Repeat Q/K from num_k_heads up to num_v_heads.
        if self.num_v_heads != self.num_k_heads:
            repeat = self.num_v_heads // self.num_k_heads
            out_sh = NamedSharding(self.mesh, P(None, "tensor", None))
            q_mix = jnp.repeat(q_mix, repeat, axis=1, out_sharding=out_sh)
            k_mix = jnp.repeat(k_mix, repeat, axis=1, out_sharding=out_sh)

        # Compute beta and g.
        beta = jax.nn.sigmoid(b.astype(jnp.float32))  # [seq_len, num_v_heads]
        a_f32 = a.astype(jnp.float32)
        A = jnp.exp(self.A_log.value)
        g = -A[None] * jax.nn.softplus(a_f32 + self.dt_bias.value[None])  # [seq_len, H_v]

        return

    def __call__(self, 
                  forward_batch: ForwardBatch,
                  mixed_qkv: jax.Array, 
                  conv_state_in: jax.Array,
                  recurrent_state_in: jax.Array,
                  b: jax.Array,
                  a: jax.Array,
                  ):
        if forward_batch.forward_mode.is_decode():
            core_attn_out = self.forward_decode(forward_batch, mixed_qkv, conv_state_in, recurrent_state_in)
        else:
            core_attn_out = self.forward_extend(forward_batch, mixed_qkv, conv_state_in, recurrent_state_in)
        
        return core_attn_out

    def forward_decode(self,
                       forward_batch: ForwardBatch,
                       mixed_qkv: jax.Array, 
                       conv_state_in: jax.Array,
                       recurrent_state_in: jax.Array,
                  ):
        # [seq_len, conv_dim] where seq_len == req_size (one token per request).
        conv_out, new_conv_state = causal_conv1d_update(
            mixed_qkv,
            conv_state_in,
            self.conv1d_weight.value,
            bias=None,
            activation="silu",
        )

        self._prepare()

        # One token per request.  Feed [req_size, 1, H, K/V] to the kernel.
        q_in = q_mix[:, None]
        k_in = k_mix[:, None]
        v_in = v_mix[:, None]
        g_in = g[:, None]
        beta_in = beta[:, None]
        out_bt, new_rec_full = fused_recurrent_gated_delta(
            q_in,
            k_in,
            v_in,
            g_in,
            beta_in,
            initial_state=recurrent_state_in,
            use_qk_l2norm=True,
        )
        core_attn_out = out_bt[:, 0]  # [seq_len, H_v, V]
        new_rec = new_rec_full  # already [req_size, H, K, V]

        return core_attn_out

    def forward_extend(self,
            forward_batch: ForwardBatch,
            mixed_qkv: jax.Array, 
            conv_state_in: jax.Array,
            recurrent_state_in: jax.Array,
            ):
         # Prefill / extend: current implementation treats the packed batch as
            # a single logical sequence (req_size=1). Take the first slot's conv state;
            # we'll pad the new state back to the original req_size at the end.
        conv_in_b1 = mixed_qkv[None]  # [1, seq_len, conv_dim]
        init_state_b1 = conv_state_in[:1]
        conv_out_b1, new_conv_b1 = causal_conv1d_prefill(
            conv_in_b1,
            self.conv1d_weight.value,
            bias=None,
            initial_state=init_state_b1,
            activation="silu",
        )
        conv_out = conv_out_b1[0]  # [seq_len, conv_dim]
        # Pad updates back to the full [req_size, conv_dim, K-1] shape (other slots unchanged).
        new_conv_full = conv_state_in.at[:1].set(new_conv_b1)
        new_conv_state = new_conv_full

        self._prepare()

        # Prefill: packed [1, seq_len, H, *]. For v1 we require a single-request
        # prefill: take slot 0's state, scan, and write the update back into
        # slot 0 of the full state tensor. Multi-request prefill needs the
        # recurrence to reset at sequence boundaries — handled by the kernel's
        # cu_seqlens path but requires the state batch dim to match N.
        q_in = q_mix[None]
        k_in = k_mix[None]
        v_in = v_mix[None]
        g_in = g[None]
        beta_in = beta[None]
        init_b1 = recurrent_state_in[:1]
        out_bt, new_rec_b1 = fused_recurrent_gated_delta(
            q_in,
            k_in,
            v_in,
            g_in,
            beta_in,
            initial_state=init_b1,
            cu_seqlens=None,
            use_qk_l2norm=True,
        )
        core_attn_out = out_bt[0]  # [seq_len, H_v, V]
        new_rec = recurrent_state_in.at[:1].set(new_rec_b1)

        return core_attn_out