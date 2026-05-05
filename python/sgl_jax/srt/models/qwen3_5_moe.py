"""Qwen3-Next (aka Qwen3.5) — hybrid linear + full attention sparse MoE.

Every 4th layer uses standard grouped-query attention with QK-GemmaRMSNorm and
partial RoPE; the other layers use Gated DeltaNet linear attention backed by
a per-request recurrent state and a short causal conv1d state. Every layer is
a sparse MoE (routed experts + shared expert).

This module is the Flax NNX port of
``transformers.models.qwen3_next.modeling_qwen3_next``. The kernels live in
``sgl_jax.srt.kernels.gated_delta``; the recurrent state pool lives in
``sgl_jax.srt.mem_cache.recurrent_state_pool``.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from transformers import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.kernels.gated_delta import (
    causal_conv1d_prefill,
    causal_conv1d_update,
    fused_recurrent_gated_delta,
)
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, get_rope
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.layernorm import GemmaRMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK, create_moe_weights_mapping
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache, RecurrentStatePool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen3 import Qwen3MLP
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def _layer_types(config) -> list[str]:
    """Return the per-layer type list, filling in from ``full_attention_interval``
    when ``layer_types`` is absent. The result is always exactly
    ``config.num_hidden_layers`` long so overrides (e.g. trimming to 4 layers)
    don't leak stale 48-entry lists from the base config."""
    n = config.num_hidden_layers
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None and len(layer_types) >= n:
        return list(layer_types)[:n]
    interval = getattr(config, "full_attention_interval", 4)
    return ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(n)]


def linear_attention_layer_ids(config) -> list[int]:
    return [i for i, t in enumerate(_layer_types(config)) if t == "linear_attention"]


def full_attention_layer_ids(config) -> list[int]:
    return [i for i, t in enumerate(_layer_types(config)) if t == "full_attention"]


class Qwen3_5FullAttention(nnx.Module):
    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        layer_id: int,
        kv_layer_id: int,  # dense index into the (full-attn-only) KV pool
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.kv_layer_id = kv_layer_id
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.q_head_num = config.num_attention_heads
        self.kv_head_num = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        bias = getattr(config, "attention_bias", False)
        # Qwen3-Next full attention has a gated output: q_proj emits 2×heads
        # worth of features (first half = queries, second half = gate which
        # modulates attn_output via sigmoid after attention).
        self.q_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.q_head_num * self.head_dim * 2,
            use_bias=bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="q_proj",
        )
        self.k_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.kv_head_num * self.head_dim,
            use_bias=bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="k_proj",
        )
        self.v_proj = LinearBase(
            input_size=self.hidden_size,
            output_size=self.kv_head_num * self.head_dim,
            use_bias=bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="v_proj",
        )
        self.o_proj = LinearBase(
            input_size=self.q_head_num * self.head_dim,
            output_size=self.hidden_size,
            use_bias=bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="o_proj",
        )
        self.q_norm = GemmaRMSNorm(
            self.head_dim,
            epsilon=config.rms_norm_eps,
            scope_name="q_norm",
        )
        self.k_norm = GemmaRMSNorm(
            self.head_dim,
            epsilon=config.rms_norm_eps,
            scope_name="k_norm",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
            base=config.rope_theta,
            is_neox_style=True,
            partial_rotary_factor=getattr(config, "partial_rotary_factor", 1.0),
            dtype=dtype,
        )
        # RadixAttention uses layer_id to index into the KV pool. The pool is
        # sized to the full-attn layer count only, so pass the dense index.
        self.attn = RadixAttention(
            num_heads=self.q_head_num,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=self.kv_head_num,
            layer_id=kv_layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        qg, _ = self.q_proj(hidden_states)  # [seq_len, q_heads * 2 * head_dim]
        k, _ = self.k_proj(hidden_states)  # [seq_len, kv_heads*head_dim]
        v, _ = self.v_proj(hidden_states)
        T = hidden_states.shape[0]
        # HF layout (Qwen3_5Attention.forward):
        #     self.q_proj(x).view(*input_shape, -1, head_dim * 2)   -> (seq_len, q_heads, 2*head_dim)
        #     query_states, gate = torch.chunk(..., 2, dim=-1)
        # i.e. for each head h the q_proj emits [q_h(head_dim), gate_h(head_dim)]
        # contiguously. So the correct reshape is (seq_len, q_heads, head_dim*2) and
        # the split is along the last (head_dim) axis, NOT along the head axis.
        qg = qg.reshape(T, self.q_head_num, 2 * self.head_dim)
        q, gate = jnp.split(qg, 2, axis=-1)
        k = k.reshape(T, self.kv_head_num, self.head_dim)
        v = v.reshape(T, self.kv_head_num, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)
        # Gated output: sigmoid(gate) elementwise over all heads, flattened
        # back to (seq_len, q_heads*head_dim).
        gate_flat = gate.reshape(T, self.q_head_num * self.head_dim)
        attn_output = attn_output * jax.nn.sigmoid(gate_flat)
        out, _ = self.o_proj(attn_output)
        return out, kv_fused


class Qwen3_5GatedDeltaNet(nnx.Module):
    """Gated DeltaNet linear-attention layer — shape-correct port of HF ref.

    Weight shapes follow HuggingFace ``Qwen3_5GatedDeltaNet``:

        in_proj_qkvz:   [hidden, 2*key_dim + 2*value_dim]
        in_proj_ba:     [hidden, 2*num_v_heads]
        conv1d.weight:  [conv_dim, kernel_size]  (HF stores [conv_dim, 1, K], we squeeze)
        conv1d.bias:    [conv_dim]
        A_log:          [num_v_heads]
        dt_bias:        [num_v_heads]
        norm.weight:    [head_v_dim]  (GemmaRMSNormGated)
        out_proj:       [value_dim, hidden]
    """

    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        layer_id: int,
        mamba_layer_id: int,  # dense index into RecurrentStatePool (0..num_linear-1)
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.mamba_layer_id = mamba_layer_id
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.eps = config.rms_norm_eps

        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_dim = 2 * self.key_dim + self.value_dim

        projection_size_qkvz = 2 * self.key_dim + 2 * self.value_dim
        projection_size_ba = 2 * self.num_v_heads

        self.in_proj_qkvz = LinearBase(
            input_size=self.hidden_size,
            output_size=projection_size_qkvz,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="in_proj_qkvz",
        )
        self.in_proj_ba = LinearBase(
            input_size=self.hidden_size,
            output_size=projection_size_ba,
            use_bias=False,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="in_proj_ba",
        )
        self.out_proj = LinearBase(
            input_size=self.value_dim,
            output_size=self.hidden_size,
            use_bias=False,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="out_proj",
        )

        # Depthwise conv1d weights: [conv_dim, kernel_size] (HF uses bias=False).
        self.conv1d_weight = nnx.Param(
            jnp.zeros((self.conv_dim, self.conv_kernel_size), dtype=dtype)
        )
        # Delta-rule params (fp32 for numerical stability).
        self.A_log = nnx.Param(jnp.zeros((self.num_v_heads,), dtype=jnp.float32))
        self.dt_bias = nnx.Param(jnp.ones((self.num_v_heads,), dtype=jnp.float32))
        # Gated GemmaRMSNorm (applied per head along head_v_dim).
        self.rms_scale = nnx.Param(jnp.ones((self.head_v_dim,), dtype=jnp.float32))

    # ----- helpers ----------------------------------------------------------
    def _split_qkvz_ba(self, qkvz: jax.Array, ba: jax.Array):
        """Split the fused ``in_proj_qkvz`` / ``in_proj_ba`` outputs into
        ``(q, k, v, z, b, a)`` matching the HF reshape convention.

        Input shapes:
            qkvz: ``[seq_len, 2*key_dim + 2*value_dim]``
            ba:   ``[seq_len, 2*num_v_heads]``
        """
        T = qkvz.shape[0]
        v_per_k = self.num_v_heads // self.num_k_heads
        per_k_head = 2 * self.head_k_dim + 2 * v_per_k * self.head_v_dim

        # Group along num_k_heads, then split per-group components.
        # The "tensor" shard sits on the num_k_heads dim after the reshape.
        qkvz = jax.lax.reshape(
            qkvz,
            (T, self.num_k_heads, per_k_head),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor", None)),
        )
        ba = jax.lax.reshape(
            ba,
            (T, self.num_k_heads, 2 * v_per_k),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor", None)),
        )

        split_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            v_per_k * self.head_v_dim,
            v_per_k * self.head_v_dim,
        ]
        q = qkvz[..., : split_qkvz[0]]
        k = qkvz[..., split_qkvz[0] : split_qkvz[0] + split_qkvz[1]]
        v = qkvz[
            ...,
            split_qkvz[0] + split_qkvz[1] : split_qkvz[0] + split_qkvz[1] + split_qkvz[2],
        ]
        z = qkvz[..., split_qkvz[0] + split_qkvz[1] + split_qkvz[2] :]

        b = ba[..., :v_per_k]
        a = ba[..., v_per_k:]
        # Expand the grouped v/z back to per-head shape: [seq_len, num_v_heads, head_v_dim].
        v = jax.lax.reshape(
            v,
            (T, self.num_v_heads, self.head_v_dim),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor", None)),
        )
        z = jax.lax.reshape(
            z,
            (T, self.num_v_heads, self.head_v_dim),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor", None)),
        )
        b = jax.lax.reshape(
            b,
            (T, self.num_v_heads),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor")),
        )
        a = jax.lax.reshape(
            a,
            (T, self.num_v_heads),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor")),
        )
        return q, k, v, z, b, a

    def _rms_gate(self, core_attn_out: jax.Array, z: jax.Array) -> jax.Array:
        """Equivalent of HF ``Qwen3_5GemmaRMSNormGated(norm(core) * silu(z))``.

        Input ``core_attn_out`` and ``z`` have shape ``[seq_len, num_v_heads, head_v_dim]``.
        """
        x = core_attn_out.astype(jnp.float32)
        rms = jnp.sqrt((x * x).mean(axis=-1, keepdims=True) + self.eps)
        x = x / rms
        x = x * self.rms_scale.value.astype(jnp.float32)
        # z gate — HF form: out = norm(core) * silu(z).
        gated = x * jax.nn.silu(z.astype(jnp.float32))
        return gated.astype(core_attn_out.dtype)

    def forward_in_proj(self, hidden_states: jax.Array):
        seq_len = hidden_states.shape[0]

        qkvz, _ = self.in_proj_qkvz(hidden_states)
        ba, _ = self.in_proj_ba(hidden_states)
        q, k, v, z, b, a = self._split_qkvz_ba(qkvz, ba)

        # Build mixed_qkv in the layout the conv1d expects: [seq_len, conv_dim].
        # HF does concat(query, key, value) along the feature axis, each flattened.
        # Tensor sharding sits on the heads axis, so the flattened dim is sharded.
        q_flat = jax.lax.reshape(
            q,
            (seq_len, self.num_k_heads * self.head_k_dim),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor")),
        )
        k_flat = jax.lax.reshape(
            k,
            (seq_len, self.num_k_heads * self.head_k_dim),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor")),
        )
        v_flat = jax.lax.reshape(
            v,
            (seq_len, self.num_v_heads * self.head_v_dim),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor")),
        )
        mixed_qkv = jnp.concatenate([q_flat, k_flat, v_flat], axis=-1)  # [seq_len, conv_dim]

        return mixed_qkv, z, b, a

    

    # ----- forward ----------------------------------------------------------
    def __call__(
        self,
        hidden_states: jax.Array,  # [seq_len, hidden]
        forward_batch: ForwardBatch,
        conv_state_in: jax.Array,  # [req_size, conv_dim, kernel-1]
        recurrent_state_in: jax.Array,  # [req_size, num_v_heads, head_k_dim, head_v_dim] fp32
    ):
        """Returns ``(output [seq_len, hidden], new_conv [req_size, conv_dim, K-1], new_rec [req_size, H, K, V])``.

        ``req_size`` is the number of in-flight requests; the kernel expects one state
        per request. For decode ``seq_len == req_size`` (one token per request). For prefill,
        each request's tokens are laid out contiguously; the per-request
        boundaries come from ``forward_batch.extend_seq_lens`` and the linear-
        attention metadata already on ``forward_batch``.
        """
        seq_len = hidden_states.shape[0]

        mixed_qkv, z, b, a = self.forward_in_proj(hidden_states)

        core_attn_out = self.attention(forward_batch, mixed_qkv, conv_state_in, recurrent_state_in, b, a)      

        core_attn_out = self._rms_gate(core_attn_out, z)  # [seq_len, H_v, V]
        core_attn_out = jax.lax.reshape(
            core_attn_out,
            (seq_len, self.num_v_heads * self.head_v_dim),
            out_sharding=NamedSharding(self.mesh, P(None, "tensor")),
        )
        output, _ = self.out_proj(core_attn_out)
        return output

# ---------------------------------------------------------------------------
# Sparse MoE block with shared expert
# ---------------------------------------------------------------------------


class Qwen3_5SparseMoeBlock(nnx.Module):
    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        layer_id: int,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.mesh = mesh
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok

        self.moe_gate = GateLogit(
            input_size=config.hidden_size,
            num_experts=config.num_experts,
            weight_dtype=dtype,
        )
        self.topk = TopK(
            topk=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
        )

        moe_backend = getattr(config, "moe_backend", "epmoe")
        self.use_fused = moe_backend == "fused"
        self.moe_backend = moe_backend

        ep_size = getattr(config, "ep_size", 1)
        if self.use_fused:
            self.experts = FusedEPMoE(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                intermediate_dim=config.moe_intermediate_size,
                mesh=mesh,
                activation="silu",
                ep_size=ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
                renormalize_topk_logits=config.norm_topk_prob,
            )
        else:
            self.experts = EPMoE(
                hidden_size=config.hidden_size,
                num_experts=config.num_experts,
                num_experts_per_tok=config.num_experts_per_tok,
                intermediate_dim=config.moe_intermediate_size,
                mesh=mesh,
                ep_size=ep_size,
                weight_dtype=dtype,
                dtype=dtype,
                layer_id=layer_id,
            )

        # Shared expert (always applied, gated by a scalar sigmoid).
        self.shared_expert = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            layer_id=layer_id,
            dtype=dtype,
            mesh=mesh,
        )
        self.shared_expert_gate = LinearBase(
            input_size=config.hidden_size,
            output_size=1,
            use_bias=False,
            kernel_axes=(None, None),
            params_dtype=dtype,
            mesh=mesh,
            scope_name="shared_expert_gate",
        )

    def __call__(self, hidden_states: jax.Array, forward_batch: ForwardBatch, dispatch_info=None):
        router_logits = self.moe_gate(hidden_states)
        topk_weights, topk_ids = self.topk(router_logits, dispatch_info=dispatch_info)

        if self.use_fused:
            token_valid_mask = forward_batch.get_token_valid_mask(hidden_states.shape[0])
            if token_valid_mask is not None:
                topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)

        routed = self.experts(hidden_states, topk_weights, topk_ids)
        shared = self.shared_expert(hidden_states)
        gate, _ = self.shared_expert_gate(hidden_states)
        shared = shared * jax.nn.sigmoid(gate)
        return routed + shared, topk_ids


# ---------------------------------------------------------------------------
# Decoder layer — branches on layer type
# ---------------------------------------------------------------------------


class Qwen3_5DecoderLayer(nnx.Module):
    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        layer_id: int,
        mamba_layer_id: int | None,  # dense mamba index for linear layers, else None
        kv_layer_id: int | None,  # dense kv index for full layers, else None
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.is_linear = _layer_types(config)[layer_id] == "linear_attention"
        self.mamba_layer_id = mamba_layer_id
        self.kv_layer_id = kv_layer_id

        if self.is_linear:
            assert mamba_layer_id is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                config,
                layer_id,
                mamba_layer_id,
                mesh,
                dtype,
            )
            self.self_attn = None
        else:
            assert kv_layer_id is not None
            self.self_attn = Qwen3_5FullAttention(
                config,
                layer_id,
                kv_layer_id,
                mesh,
                dtype,
            )
            self.linear_attn = None

        self.mlp = Qwen3_5SparseMoeBlock(config, layer_id, mesh, dtype)

        self.input_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            scope_name="input_layernorm",
        )
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            scope_name="post_attention_layernorm",
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        recurrent_state_pool: RecurrentStatePool | None,
        residual: jax.Array | None = None,
        dispatch_info=None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states = hidden_states + residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        kv_fused = None
        new_conv = new_rec = None
        if self.is_linear:
            slot_idx = forward_batch.mamba_cache_indices
            assert (
                recurrent_state_pool is not None and slot_idx is not None
            ), "Qwen3_5 linear layers require recurrent_state_pool and mamba_cache_indices"
            conv_in, rec_in = recurrent_state_pool.get_layer(self.mamba_layer_id, slot_idx)
            hidden_states, new_conv, new_rec = self.linear_attn(
                hidden_states,
                forward_batch,
                conv_in,
                rec_in,
            )
        else:
            hidden_states, kv_fused = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                token_to_kv_pool=token_to_kv_pool,
            )

        hidden_states = hidden_states + residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, topk_ids = self.mlp(hidden_states, forward_batch, dispatch_info)

        return hidden_states, residual, kv_fused, topk_ids, new_conv, new_rec


# ---------------------------------------------------------------------------
# Model + ForCausalLM
# ---------------------------------------------------------------------------


class Qwen3_5Model(nnx.Module):
    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        layer_types = _layer_types(config)
        # Build dense index maps:
        #   full-attn layers → 0, 1, ..., N_full-1  (used as RadixAttention layer_id)
        #   linear layers    → 0, 1, ..., N_lin-1   (used as RecurrentStatePool layer index)
        full_dense = -1
        lin_dense = -1
        kv_ids: list[int | None] = []
        mamba_ids: list[int | None] = []
        for lt in layer_types:
            if lt == "full_attention":
                full_dense += 1
                kv_ids.append(full_dense)
                mamba_ids.append(None)
            else:
                lin_dense += 1
                mamba_ids.append(lin_dense)
                kv_ids.append(None)
        self.num_full_attn_layers = full_dense + 1
        self.num_linear_layers = lin_dense + 1

        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )
        self.layers = nnx.data(
            [
                Qwen3_5DecoderLayer(
                    config,
                    layer_id=i,
                    mamba_layer_id=mamba_ids[i],
                    kv_layer_id=kv_ids[i],
                    mesh=mesh,
                    dtype=dtype,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = GemmaRMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        recurrent_state_pool: RecurrentStatePool | None = None,
    ):
        if forward_batch.input_embedding:
            hidden_states = forward_batch.input_embedding
        else:
            hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        # Only full-attention layers emit kv_fused; the outer pool is sized to
        # num_full_attn_layers and `replace_kv_buffer` expects a dense list.
        layers_kv_fused: list = []
        layers_topk_ids: list = []

        for layer_id, layer in enumerate(self.layers):
            hidden_states, residual, kv_fused, topk_ids, new_conv, new_rec = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                recurrent_state_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )

            if (
                forward_batch.deepstack_visual_embedding is not None
                and layer_id < forward_batch.deepstack_visual_embedding.shape[0]
            ):
                hidden_states += forward_batch.deepstack_visual_embedding[layer_id].astype(
                    hidden_states.dtype
                )

            if not layer.is_linear:
                layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)

            if layer.is_linear and recurrent_state_pool is not None:
                recurrent_state_pool.write_layer(
                    layer.mamba_layer_id,
                    forward_batch.mamba_cache_indices,
                    new_conv,
                    new_rec,
                )

        if residual is not None:
            hidden_states = hidden_states + residual
        hidden_states = self.norm(hidden_states)
        return hidden_states, layers_kv_fused, layers_topk_ids, recurrent_state_pool


class Qwen3_5ForCausalLM(nnx.Module):
    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        self.model = Qwen3_5Model(config, dtype=dtype, mesh=mesh)
        if not getattr(config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
        recurrent_state_pool: RecurrentStatePool | None = None,
    ):
        hidden_states, layers_kv_fused, layers_topk_ids, new_recurrent_state_pool = self.model(
            forward_batch,
            token_to_kv_pool,
            recurrent_state_pool,
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        # Mirror the Qwen3Moe / other model return contract:
        # (output, layers_kv_fused, needs_write_kv_flag, layers_topk_ids)
        # Plus the new recurrent_state_pool pytree tail for hybrid-linear models.
        return output, layers_kv_fused, True, layers_topk_ids, new_recurrent_state_pool

    # ---- weight loading ----------------------------------------------------
    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        mappings = self._create_weight_mappings()
        loader.load_weights_from_safetensors(mappings)
        logger.info("Qwen3_5 weights loaded successfully!")

    def _create_weight_mappings(self) -> dict:
        c = self.config
        mappings: dict = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }
        if not getattr(c, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        layer_types = _layer_types(c)
        for i, lt in enumerate(layer_types):
            if lt == "linear_attention":
                mappings.update(self._linear_layer_mappings(i))
            else:
                mappings.update(self._full_layer_mappings(i))
            mappings.update(self._moe_layer_mappings(i))

        return mappings

    @staticmethod
    def _ln_mappings(prefix: str, target_prefix: str) -> dict:
        return {
            f"{prefix}.input_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.input_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
            f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                target_path=f"{target_prefix}.post_attention_layernorm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }

    def _full_layer_mappings(self, i: int) -> dict:
        prefix = f"model.layers.{i}"
        m = self._ln_mappings(prefix, prefix)
        m.update(
            {
                f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                    target_path=f"{prefix}.self_attn.q_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                    target_path=f"{prefix}.self_attn.k_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                    target_path=f"{prefix}.self_attn.v_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                    target_path=f"{prefix}.self_attn.o_proj.weight",
                    sharding=("tensor", None),
                    transpose=True,
                ),
                f"{prefix}.self_attn.q_norm.weight": WeightMapping(
                    target_path=f"{prefix}.self_attn.q_norm.scale",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.self_attn.k_norm.weight": WeightMapping(
                    target_path=f"{prefix}.self_attn.k_norm.scale",
                    sharding=(None,),
                    transpose=False,
                ),
            }
        )
        return m

    def _linear_layer_mappings(self, i: int) -> dict:
        prefix = f"model.layers.{i}"
        m = self._ln_mappings(prefix, prefix)
        la = "linear_attn"  # HF name in the checkpoint
        ours = "linear_attn"
        m.update(
            {
                f"{prefix}.{la}.in_proj_qkvz.weight": WeightMapping(
                    target_path=f"{prefix}.{ours}.in_proj_qkvz.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{prefix}.{la}.in_proj_ba.weight": WeightMapping(
                    target_path=f"{prefix}.{ours}.in_proj_ba.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                # Conv1d weight in HF has shape [conv_dim, 1, kernel]; we store
                # [conv_dim, kernel]. Use the loader's reshape to drop the group axis.
                f"{prefix}.{la}.conv1d.weight": WeightMapping(
                    target_path=f"{prefix}.{ours}.conv1d_weight",
                    sharding=("tensor", None),
                    transpose=False,
                    reshape=(
                        (
                            2 * self.config.linear_num_key_heads * self.config.linear_key_head_dim
                            + self.config.linear_num_value_heads * self.config.linear_value_head_dim
                        ),
                        self.config.linear_conv_kernel_dim,
                    ),
                ),
                f"{prefix}.{la}.A_log": WeightMapping(
                    target_path=f"{prefix}.{ours}.A_log",
                    sharding=("tensor",),
                    transpose=False,
                ),
                f"{prefix}.{la}.dt_bias": WeightMapping(
                    target_path=f"{prefix}.{ours}.dt_bias",
                    sharding=("tensor",),
                    transpose=False,
                ),
                f"{prefix}.{la}.norm.weight": WeightMapping(
                    target_path=f"{prefix}.{ours}.rms_scale",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.{la}.out_proj.weight": WeightMapping(
                    target_path=f"{prefix}.{ours}.out_proj.weight",
                    sharding=("tensor", None),
                    transpose=True,
                ),
            }
        )
        return m

    def _moe_layer_mappings(self, i: int) -> dict:
        prefix = f"model.layers.{i}"
        m: dict = {}
        # Routing gate.
        m[f"{prefix}.mlp.gate.weight"] = WeightMapping(
            target_path=f"{prefix}.mlp.moe_gate.kernel",
            sharding=(None, None),
            transpose=True,
        )
        # Shared expert.
        for name in ("gate_proj", "up_proj"):
            m[f"{prefix}.mlp.shared_expert.{name}.weight"] = WeightMapping(
                target_path=f"{prefix}.mlp.shared_expert.{name}.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
        m[f"{prefix}.mlp.shared_expert.down_proj.weight"] = WeightMapping(
            target_path=f"{prefix}.mlp.shared_expert.down_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        m[f"{prefix}.mlp.shared_expert_gate.weight"] = WeightMapping(
            target_path=f"{prefix}.mlp.shared_expert_gate.weight",
            sharding=(None, None),
            transpose=True,
        )
        # Routed experts via helper.
        moe_backend = getattr(self.config, "moe_backend", "epmoe")
        # HF stores experts at `model.layers.{i}.mlp.experts.{j}.{g,u,d}_proj.weight`;
        # our module path is `model.layers.{i}.mlp.experts.{wi_0,wi_1,wo}`. Use the
        # helper with moe_path="mlp.experts" and bare-index source pattern so both
        # target and source paths match.
        m.update(
            create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=prefix,
                num_experts=self.config.num_experts,
                moe_backend=moe_backend,
                moe_path="mlp.experts",
                source_expert_pattern="{i}",
            )
        )
        return m
