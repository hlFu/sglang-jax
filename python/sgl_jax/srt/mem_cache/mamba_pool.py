"""Recurrent-state cache pool for hybrid linear-attention models (e.g. Qwen3-Next).

Qwen3-Next replaces the KV cache on its linear-attention layers with a pair of
per-request tensors: a short conv1d state (``kernel_size - 1`` positions) and a
recurrent-state matrix (``[heads, head_k_dim, head_v_dim]``). Both live in this
pool, indexed by **request pool index** rather than per-token.

One :class:`MambaPool` instance holds all linear layers' states in a single
allocation to keep JAX pytree leaves minimal. A dedicated :class:`HybridReqToTokenPool`
wraps :class:`ReqToTokenPool` (for the full-attention token-to-slot mapping)
and routes alloc / free to both.

The first entry (index 0) of each buffer is a **null slot**. Any request that
did not allocate a real slot reads/writes through index 0; this lets the model
forward unconditionally gather from the pool without a branch.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool

logger = logging.getLogger(__name__)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


@register_pytree_node_class
class MambaPool:
    """Per-request recurrent + conv state pool for linear-attention layers.

    Attributes:
        size: number of real request slots (slot 0 is always the null slot).
        num_layers: number of linear-attention layers this pool serves.
        conv_state:     ``[num_layers, size + 1, conv_dim, kernel_size - 1]``.
        recurrent_state:``[num_layers, size + 1, num_heads, head_k_dim, head_v_dim]``.
    """

    def __init__(
        self,
        size: int,
        num_layers: int,
        conv_dim: int,
        conv_state_len: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        dtype: jnp.dtype = jnp.bfloat16,
        mesh: jax.sharding.Mesh | None = None,
    ):
        self.size = size
        self.num_layers = num_layers
        self.conv_dim = conv_dim
        self.conv_state_len = conv_state_len
        self.num_heads = num_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.dtype = dtype
        self.mesh = mesh

        # Shard the two big pool axes along "tensor" so scatters from
        # tensor-sharded activations don't require a collective reshape.
        conv_sh = NamedSharding(mesh, P(None, None, "tensor", None)) if mesh is not None else None
        rec_sh = (
            NamedSharding(mesh, P(None, None, "tensor", None, None)) if mesh is not None else None
        )
        self.conv_state = jnp.zeros(
            (num_layers, size + 1, conv_dim, conv_state_len),
            dtype=dtype,
            out_sharding=conv_sh,
        )
        # Recurrent state is always fp32 for numerical stability.
        self.recurrent_state = jnp.zeros(
            (num_layers, size + 1, num_heads, head_k_dim, head_v_dim),
            dtype=jnp.float32,
            out_sharding=rec_sh,
        )
        # host-side free list; slot 0 is reserved as the null slot.
        self.free_slots = list(range(1, size + 1))

    # --- allocation ---------------------------------------------------------
    def alloc(self, need_size: int) -> list[int] | None:
        if need_size > len(self.free_slots):
            return None
        picked = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]
        return picked

    def free(self, slots):
        if isinstance(slots, int):
            slots = [slots]
        slots = list(slots)
        # Zero the freed slots so a subsequent alloc starts clean. The pool
        # arrays are sharded on the tensor axis, so the scatter must run inside
        # the corresponding mesh context.
        if slots:
            idx = jnp.asarray(slots, dtype=jnp.int32)
            # API moved between jax versions; use whichever is available.
            if self.mesh is not None:
                _set_mesh = getattr(jax.sharding, "use_mesh", None) or getattr(
                    jax, "set_mesh", None
                )
                ctx = _set_mesh(self.mesh) if _set_mesh is not None else _NullCtx()
            else:
                ctx = _NullCtx()
            with ctx:
                self.conv_state = self.conv_state.at[:, idx].set(0)
                self.recurrent_state = self.recurrent_state.at[:, idx].set(0)
        self.free_slots.extend(slots)

    def clear(self):
        self.conv_state = jnp.zeros_like(self.conv_state)
        self.recurrent_state = jnp.zeros_like(self.recurrent_state)
        self.free_slots = list(range(1, self.size + 1))

    def available_size(self) -> int:
        return len(self.free_slots)

    # --- per-layer access ---------------------------------------------------
    def get_layer(self, layer_id: int, slot_indices: jax.Array):
        """Gather the conv and recurrent state for the given slots, one layer.

        Args:
            layer_id: integer layer index into the pool.
            slot_indices: ``[B]`` int32 array of slot ids (0 = null slot).

        Returns:
            ``(conv_in [B, conv_dim, kernel-1], rec_in [B, H, K, V])``.
        """
        conv = jnp.take(self.conv_state[layer_id], slot_indices, axis=0)
        rec = jnp.take(self.recurrent_state[layer_id], slot_indices, axis=0)
        return conv, rec

    def write_layer(
        self,
        layer_id: int,
        slot_indices: jax.Array,
        new_conv: jax.Array,
        new_rec: jax.Array,
    ) -> MambaPool:
        """Write updated states back. Returns a new pool (JAX functional style).

        When running under an explicit-sharding mesh, the scatter cannot infer
        its output sharding, so we pass the pool's own sharding explicitly.
        """
        new_conv = new_conv.astype(self.conv_state.dtype)
        new_rec = new_rec.astype(self.recurrent_state.dtype)
        # Two-step: scatter into the per-layer slice (static layer_id), then
        # write the slice back with dynamic_update_slice. Keeps the scatter
        # indexer on a single advanced axis, avoiding a JAX tracer edge case
        # when mixing int + tracer indices under jit.
        # Step 1: scatter slot updates into the per-layer slice (single
        # advanced-index axis; safe under jit). Step 2: write the slice back
        # with dynamic_update_slice at the static layer axis.
        if self.mesh is not None:
            conv_layer_sh = NamedSharding(self.mesh, P(None, "tensor", None))
            rec_layer_sh = NamedSharding(self.mesh, P(None, "tensor", None, None))
            conv_layer = self.conv_state[layer_id]
            conv_layer = conv_layer.at[slot_indices].set(
                new_conv,
                out_sharding=conv_layer_sh,
            )
            rec_layer = self.recurrent_state[layer_id]
            rec_layer = rec_layer.at[slot_indices].set(
                new_rec,
                out_sharding=rec_layer_sh,
            )
        else:
            conv_layer = self.conv_state[layer_id].at[slot_indices].set(new_conv)
            rec_layer = self.recurrent_state[layer_id].at[slot_indices].set(new_rec)
        self.conv_state = jax.lax.dynamic_update_slice(
            self.conv_state,
            conv_layer[None],
            (layer_id, 0, 0, 0),
        )
        self.recurrent_state = jax.lax.dynamic_update_slice(
            self.recurrent_state,
            rec_layer[None],
            (layer_id, 0, 0, 0, 0),
        )
        return self

    # --- pytree -------------------------------------------------------------
    def tree_flatten(self):
        children = (self.conv_state, self.recurrent_state)
        aux_data = {
            "size": self.size,
            "num_layers": self.num_layers,
            "conv_dim": self.conv_dim,
            "conv_state_len": self.conv_state_len,
            "num_heads": self.num_heads,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "dtype": self.dtype,
            "free_slots": self.free_slots,
            "mesh": self.mesh,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        for key in (
            "size",
            "num_layers",
            "conv_dim",
            "conv_state_len",
            "num_heads",
            "head_k_dim",
            "head_v_dim",
            "dtype",
            "free_slots",
            "mesh",
        ):
            setattr(obj, key, aux_data.get(key))
        obj.conv_state, obj.recurrent_state = children
        return obj


@register_pytree_node_class
class HybridReqToTokenPool(ReqToTokenPool):
    """``ReqToTokenPool`` + ``MambaPool`` side-by-side.

    Every successful ``alloc`` returns a request pool index *and* a mamba slot
    index; they have matching positions in the returned lists. The internal
    ``req_to_mamba`` host array records the 1-to-1 mapping so the model runner
    can turn a ``forward_batch.req_pool_indices`` device array into the
    corresponding ``mamba_cache_indices`` before JIT.
    """

    def __init__(
        self,
        size: int,
        max_context_len: int,
        num_linear_layers: int,
        conv_dim: int,
        conv_state_len: int,
        num_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        state_dtype: jnp.dtype = jnp.bfloat16,
        dtype: np.dtype = np.int32,
        mesh: jax.sharding.Mesh | None = None,
    ):
        super().__init__(size=size, max_context_len=max_context_len, dtype=dtype)
        self.mamba_pool = MambaPool(
            size=size,
            num_layers=num_linear_layers,
            conv_dim=conv_dim,
            conv_state_len=conv_state_len,
            num_heads=num_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            dtype=state_dtype,
            mesh=mesh,
        )
        # req_pool_idx -> mamba slot id (0 = null). Host-side, never on device.
        self.req_to_mamba = np.zeros(size, dtype=np.int32)

    def alloc(self, need_size: int = 1) -> list[int] | None:
        req_slots = super().alloc(need_size)
        if req_slots is None:
            return None
        mamba_slots = self.mamba_pool.alloc(need_size)
        if mamba_slots is None:
            # Roll back req allocation so the caller's retry path is clean.
            super().free(req_slots)
            return None
        for r, m in zip(req_slots, mamba_slots):
            self.req_to_mamba[r] = m
        return req_slots

    def free(self, free_index):
        if isinstance(free_index, int):
            free_index = [free_index]
        mamba_slots = [int(self.req_to_mamba[r]) for r in free_index]
        for r in free_index:
            self.req_to_mamba[r] = 0
        self.mamba_pool.free(mamba_slots)
        super().free(free_index)

    def clear(self):
        super().clear()
        self.mamba_pool.clear()
        self.req_to_mamba[:] = 0

    def get_mamba_indices(self, req_pool_indices) -> np.ndarray:
        """Translate request pool indices to mamba slot indices. Host-side."""
        idx = np.asarray(req_pool_indices, dtype=np.int32)
        return self.req_to_mamba[idx].astype(np.int32)

    # --- pytree -------------------------------------------------------------
    def tree_flatten(self):
        children = (self.req_to_token, self.mamba_pool)
        aux_data = {
            "size": self.size,
            "max_context_len": self.max_context_len,
            "dtype": self.dtype,
            "free_slots": self.free_slots,
            "req_to_mamba": self.req_to_mamba,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.size = aux_data["size"]
        obj.max_context_len = aux_data["max_context_len"]
        obj.dtype = aux_data["dtype"]
        obj.free_slots = aux_data["free_slots"]
        obj.req_to_mamba = aux_data["req_to_mamba"]
        obj.req_to_token, obj.mamba_pool = children
        return obj
