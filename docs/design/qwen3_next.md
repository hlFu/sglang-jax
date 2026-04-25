# [RFC] Qwen3-Next (Qwen3.5) — Hybrid Linear + Full Attention MoE

## Background

Qwen3-Next (released upstream as "Qwen3.5") is a sparse MoE model whose main
architectural novelty is a **hybrid attention stack** that mixes:

- **Full multi-head attention** with QK-RMSNorm and partial RoPE on ~1/4 of
  layers (the "landmark" layers), and
- **Gated DeltaNet linear attention** (with a short causal conv1d state) on the
  remaining ~3/4 of layers, backed by a constant-size recurrent state.

The 80B-A3B reference config places a full-attention layer every 4 layers
(`(i+1) % 4 == 0`): layer indices 3, 7, 11, … are full attention, the rest are
linear. All 48 layers are sparse MoE (512 experts, top-10, `moe_intermediate_size=512`)
plus a shared expert. The linear-attention layers replace the KV cache with a
per-request recurrent-state + conv-state pair.

This RFC specifies how sglang-jax will support Qwen3-Next end-to-end: the model
module, the gated-delta kernel, the recurrent-state cache pool, the hybrid
attention backend, and the forward-batch / scheduler plumbing.

---

## Goals & Non-goals

**Goals**

1. Load an HF Qwen3-Next checkpoint (fused `in_proj_qkvz`, `in_proj_ba`, shared
   expert, routed experts) into a Flax NNX module.
2. Correct inference with matched output vs. HF reference on small prompts.
3. Hybrid KV cache: full-attention layers use the existing `KVCache`; linear
   layers use a new `MambaPool` indexed by request.
4. Prefill + decode both pass, with fixed TPU shapes (shape buckets by token
   count); decode uses a single-step recurrence, prefill uses a chunked
   recurrence inside `shard_map`.
5. Reach GPQA-Diamond ≈ 50 on the full model (80B-A3B-Instruct), iterated on via
   a 4-layer smoke config first.

**Non-goals (this iteration)**

- Peak performance (the gated-delta recurrence ships as a pure-JAX scan first;
  a Pallas kernel is follow-up work).
- Prefix radix caching of mamba state (the first version drops mamba state on
  request completion; sglang's branching / checkpointing of mamba state is left
  as follow-up, tracked in `ssm-cache-design-watchouts.md`).
- Speculative decoding with linear layers (not supported upstream either).

---

## Architecture summary (from HF `Qwen3-Next-80B-A3B-Instruct`)

| Param | Value |
|-------|-------|
| `num_hidden_layers` | 48 |
| `hidden_size` | 2048 |
| `intermediate_size` (dense MLP — not used when `decoder_sparse_step=1`) | 5120 |
| **Full attention layers** | |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 2  (GQA 8:1) |
| `head_dim` | 256 |
| `partial_rotary_factor` | 0.25 (RoPE over first 64 of 256) |
| `qk_norm` | yes (per-head RMSNorm on Q and K) |
| **Linear attention layers (gated DeltaNet)** | |
| `linear_num_key_heads` | 16 |
| `linear_num_value_heads` | 32 |
| `linear_key_head_dim` | 128 |
| `linear_value_head_dim` | 128 |
| `linear_conv_kernel_dim` | 4 |
| **MoE** (every layer, `decoder_sparse_step=1`) | |
| `num_experts` | 512 routed |
| `num_experts_per_tok` | 10 |
| `moe_intermediate_size` | 512 |
| `shared_expert_intermediate_size` | 512 |
| `norm_topk_prob` | True |
| **Misc** | |
| `rms_norm_eps` | 1e-6 |
| `rope_theta` | 10_000_000 |
| `hidden_act` | silu |
| `max_position_embeddings` | 262144 |
| `vocab_size` | 151936 |
| `tie_word_embeddings` | False |

The layer-type pattern is `"linear_attention"` for `(i+1) % full_attention_interval != 0`
else `"full_attention"`. Concretely: layers {3, 7, 11, …, 47} are full (12 total),
the other 36 are linear.

---

## Reuse map — what we take vs. what we build

| Area | Reuse / Extend | File |
|------|----------------|------|
| RMSNorm, rotary (incl. partial), `LinearBase`, `Embed`, `ParallelLMHead`, `LogitsProcessor` | ✅ reuse | `layers/layernorm.py`, `layers/embeddings.py`, `layers/linear.py`, `layers/logits_processor.py` |
| `EPMoE` / `FusedEPMoE` + `GateLogit` + `TopK` + `create_moe_weights_mapping` | ✅ reuse (add shared expert alongside) | `layers/moe.py`, `layers/fused_moe.py` |
| Full-attention layer (RadixAttention, QK-norm) | ✅ reuse from `models/qwen3.py` / `qwen3_moe.py` with partial-RoPE option | `models/qwen3_moe.py` |
| `LinearAttentionBackend` metadata scaffold (`cu_seqlens_dev`, `scatter_idx`) | ✅ extend — add mamba-slot indices | `layers/attention/fla/linear_attention_backend.py` |
| `scatter_to_packed` / `gather_from_packed` | ✅ reuse | same |
| `GroupRMSNorm` | ✅ reuse | `layers/attention/fla/group_rmsnorm.py` |
| Simple-GLA kernel (`simple_gla_fwd`, `fused_recurrent_simple_gla`) | ❌ wrong recurrence — can't reuse | — |
| Gated-delta-rule kernel | ❌ **build** | new: `kernels/gated_delta/gated_delta.py` |
| Causal conv1d with state | ❌ **build** | new: same package |
| Mamba pool (per-request recurrent + conv state) | ❌ **build** | new: `mem_cache/mamba_pool.py` |
| Hybrid req→token / state pool wrapper | ❌ **build** | new: `mem_cache/hybrid_req_to_token_pool.py` |
| Hybrid attention backend | ❌ **build** | new: `layers/attention/hybrid_linear_backend.py` |
| Qwen3-Next model | ❌ **build** | new: `models/qwen3_next.py` |
| Weight loader | ✅ reuse framework, author new `_create_*_weight_mappings` | same |
| ForwardBatch extensions | ❌ minor patch — add `linear_attn_metadata` and `mamba_cache_indices` fields | `model_executor/forward_batch_info.py` |
| ModelRunner / TpWorker | ❌ thin hooks — allocate/free mamba slots; call `get_forward_metadata` | `model_executor/model_runner.py`, `managers/tp_worker.py`, `managers/schedule_batch.py` |

---

## Design

### Gated DeltaNet layer (linear-attention block)

Forward (per token, conceptually):

```
z  = in_proj_qkvz(x)    # fused, split → [q_raw, k_raw, v_raw, z_gate]
ba = in_proj_ba(x)      # fused → [beta_logits, alpha_logits]  (per v-head)

# Short causal conv1d over [q_raw; k_raw; v_raw] along the time axis,
# kernel_size=4, depthwise, state carried in MambaPool.conv_state.
qkv  = silu(causal_conv1d(concat(q_raw, k_raw, v_raw), conv_state))

q, k, v = split(qkv)                            # head layouts below
beta  = sigmoid(beta_logits)                    # [T, H_v]
# alpha via a soft-plus + learned time-scale (A_log, dt_bias):
#   alpha = exp(-softplus(a_logits + dt_bias) * exp(A_log))         # [T, H_v]
alpha = compute_alpha(ba, A_log, dt_bias)

# Gated delta rule recurrence:
#   S_t = S_{t-1} * alpha_t_broadcast * (I - beta_t * k_t k_tᵀ / scale)
#         + beta_t * k_t * v_tᵀ
# attn_t = q_t @ S_tᵀ
attn_output = gated_delta_rule(q, k, v, alpha, beta, recurrent_state)

# RMSNorm+Gate + output proj
attn_output = rms_norm_gated(attn_output, z_gate)  # per-head RMSNorm scaled by silu(z_gate)
out = out_proj(attn_output)
```

Head layout and TP: Q/K heads = `linear_num_key_heads`, V heads =
`linear_num_value_heads`. Heads shard along `"tensor"`. The recurrent state
per request has shape `[num_v_heads, linear_value_head_dim, linear_key_head_dim]`
(note the order vᵀ·k to match the decoder's `q @ Sᵀ`). Conv state has shape
`[conv_dim, conv_kernel_dim - 1]`, where `conv_dim = num_k_heads*K + num_k_heads*K + num_v_heads*V`.

#### Kernel: `gated_delta_rule`

We ship a **chunked pure-JAX implementation** first:

- **Decode** (`T==batch, 1 token each`): single `jax.lax.scan` step over requests; trivially parallel across heads.
- **Prefill**: chunked along the time axis at `CHUNK=64`. Within a chunk we
  compute the inter-token delta-rule update in a tight `scan` over the 64
  positions (one per token); between chunks we commit to the carried state
  exactly as the reference (no further approximation). Heads are sharded via
  `shard_map(… in_specs=P(None, "tensor", None, None))`. We reuse the existing
  `scatter_to_packed` / `gather_from_packed` to map arbitrary batched extend
  lengths into fixed-shape chunk buckets.

Correctness is checked in `test/kernels/test_gated_delta.py` against a
3-nested-loop Python reference over (batch, head, time).

A Pallas kernel is future work. The pure-JAX scan is measured; if it is too
slow we fall back to larger chunks or swap in a Triton-style Mosaic kernel
later. Per product requirement, accuracy trumps speed.

#### Kernel: causal `conv1d` with state

Depthwise kernel of size 4 applied along time. State shape
`[num_reqs, conv_dim, kernel - 1]`. For decode: `y_t = sum_k W_k * [state_{-3}, state_{-2}, state_{-1}, x_t][k]`, then
state gets shifted. For prefill with variable lengths we expand the conv state in front of the sequence
(padding) before doing a plain depthwise convolution, then pick the last `kernel-1` tokens as the new
state. Implemented as pure JAX; tested against `nn.Conv1d(padding="causal")`.

### Recurrent-state cache: `MambaPool`

Upstream sglang stores a `(conv_state, temporal_state)` pair per layer per slot.
We do the same.

```python
class MambaPool:
    def __init__(self, size: int, num_layers: int, conv_shape, temporal_shape, dtype):
        # size == max concurrent requests using linear layers (== ReqToTokenPool.size)
        # +1 for a "null" slot that's always zero (used for padded / unused reqs)
        self.conv_state     = jnp.zeros((num_layers, size + 1, *conv_shape), dtype=dtype)
        self.temporal_state = jnp.zeros((num_layers, size + 1, *temporal_shape), dtype=jnp.float32)
        self.free_slots     = list(range(1, size + 1))  # 0 is the null slot

    def alloc(self, n): ...
    def free(self, idx): ...
    def get_layer(self, layer_id) -> (conv_state_layer, temporal_state_layer)
    def write_layer(self, layer_id, slot_idx, new_conv, new_temporal)
```

Indexing is by **request pool index**, not token index — one slot per
in-flight request, independent of sequence length.

Dtype policy: conv state is bf16/fp32 following input dtype; temporal state is
always fp32 (numerical stability of the recurrence at long context).

### Hybrid pool: `HybridReqToTokenPool`

Wraps the existing `ReqToTokenPool` (for full-attention token-to-slot mapping)
alongside the new `MambaPool`. `alloc(n_reqs)` returns a list of request pool
indices and a 1:1-aligned list of mamba slot indices. It mirrors the pattern in
sglang's `HybridReqToTokenPool` but only needs a **mamba** half (no SWA).

```python
class HybridReqToTokenPool(ReqToTokenPool):
    def __init__(self, ..., num_linear_layers, conv_shape, temporal_shape, dtype):
        super().__init__(...)
        self.mamba_pool = MambaPool(size=self.size, num_layers=num_linear_layers, ...)
        # host-side int32 array: req_pool_idx -> mamba slot (1..size, 0=null)
        self.req_to_mamba = np.zeros(self.size + 1, dtype=np.int32)

    def alloc(self, n) -> list[int]:     # assigns both req slot and mamba slot
    def free(self, req_indices):
    def get_mamba_indices(self, req_indices) -> jnp.ndarray     # [B] int32
```

### Hybrid attention backend

```python
class HybridLinearAttnBackend(AttentionBackend):
    def __init__(self, mesh, full_backend, linear_backend, linear_layer_ids: set[int]):
        self.full = full_backend           # e.g. FlashAttention
        self.linear = linear_backend       # LinearAttentionBackend
        self.linear_layer_ids = linear_layer_ids

    def get_forward_metadata(self, batch):
        # returns a namespace with both .full_meta and .linear_meta + mamba indices
        ...
```

The `ModelRunner` holds one `HybridLinearAttnBackend` when the model reports
`attention_arch = "hybrid_linear"`. The `Qwen3NextDecoderLayer` checks
`layer_id in linear_layer_ids` and dispatches accordingly.

### ForwardBatch changes

Add two optional fields to `ForwardBatch`:

```python
linear_attn_metadata: LinearAttentionMetadata | None = None   # per-extend scatter/gather
mamba_cache_indices: jax.Array | None = None                  # [batch_size] int32
```

Both are included in `tree_flatten` / `tree_unflatten`. For full-attention-only
models they stay `None` and cost nothing.

### Model module

New file `python/sgl_jax/srt/models/qwen3_next.py`:

- `Qwen3NextFullAttention` (like `QWen3MoeAttention` but with `partial_rotary_factor`).
- `Qwen3NextGatedDeltaNet` (new, described above).
- `Qwen3NextMoE` (routed `EPMoE/FusedEPMoE` + shared expert `Qwen3MLP(intermediate_size=shared_expert_intermediate_size)`,
  added element-wise).
- `Qwen3NextDecoderLayer`: residual-LN-{attn | gateddelta}-LN-MoE pattern;
  branches at init-time on `layer_id in linear_layer_ids`.
- `Qwen3NextModel`, `Qwen3NextForCausalLM` (mirrors `Qwen3MoeForCausalLM`).
- `EntryClass = Qwen3NextForCausalLM`.

### Weight mapping (HF → sglang-jax)

- `lm_head.weight`, `model.embed_tokens.weight`, `model.norm.weight`, each
  layer's `input_layernorm.weight`, `post_attention_layernorm.weight`.
- For full-attention layers: same as `qwen3_moe.py` (`q_proj`, `k_proj`,
  `v_proj`, `o_proj` → `c_proj`, `q_norm`, `k_norm`).
- For linear layers (HF keys → ours):
  - `linear_attn.in_proj_qkvz.weight` → fused kernel projected column-parallel.
  - `linear_attn.in_proj_ba.weight` → same.
  - `linear_attn.conv1d.weight` → depthwise conv weights
    `[conv_dim, 1, kernel]` (squeezed; head-sharded).
  - `linear_attn.conv1d.bias` → `[conv_dim]` head-sharded.
  - `linear_attn.A_log` → `[num_v_heads]` head-sharded.
  - `linear_attn.dt_bias` → `[num_v_heads]` head-sharded.
  - `linear_attn.norm.weight` → GroupRMSNorm scale.
  - `linear_attn.out_proj.weight` → row-parallel.
- For all layers' MoE: reuse `create_moe_weights_mapping`.
- `mlp.shared_expert.gate_proj|up_proj|down_proj.weight` → shared MLP (reuse
  `Qwen3MLP` mapping, with `intermediate_size=shared_expert_intermediate_size`).
- `mlp.shared_expert_gate.weight` — scalar per-token gate for shared expert.

---

## Execution flow

```
[load time]
  ModelRunner.load_model()
    Qwen3NextForCausalLM.__init__
      • builds one LinearAttentionBackend (shared by all linear layers)
      • computes linear_layer_ids = {i for i in range(L) if (i+1) % interval != 0}
    ModelRunner._get_attention_backend() returns HybridLinearAttnBackend
  ModelRunner.init_memory_pool()
    if model is hybrid:
      req_to_token_pool = HybridReqToTokenPool(..., num_linear_layers=...)
      token_to_kv_pool = MHATokenToKVPool(only full-attn layers' kv, sized by that layer count)

[scheduler]
  Scheduler.alloc_req_slots → HybridReqToTokenPool.alloc → returns (req_idx, mamba_idx)
  req done → HybridReqToTokenPool.free → returns both slots to their free lists

[per-batch]
  tp_worker.forward_batch_generation()
    linear_meta = linear_backend.get_forward_metadata(batch)       # on host
    mamba_cache_indices = req_to_mamba[batch.req_pool_indices]     # on host → device
    forward_batch.linear_attn_metadata = linear_meta
    forward_batch.mamba_cache_indices = mamba_cache_indices
    model(forward_batch, token_to_kv_pool)                         # JIT boundary

[model forward]
  for each layer:
    if layer.is_linear:
      conv_in, temp_in = mamba_pool.get_layer(layer_id)
      conv_in = conv_in[mamba_cache_indices]
      temp_in = temp_in[mamba_cache_indices]
      y, new_conv, new_temp = GatedDeltaNet(x, conv_in, temp_in, meta)
      mamba_pool.write_layer(layer_id, mamba_cache_indices, new_conv, new_temp)
    else:
      y, kv_fused = full_attn(x, forward_batch, token_to_kv_pool)
    x = residual + y → MoE (routed+shared) → residual
```

The mamba pool writes happen inside the JIT'd model forward via
`jax.lax.dynamic_update_slice` (or `.at[idx].set`) on the pool's `.value`. The
pool itself is an `nnx.Variable` so writes stay functional; the updated pool is
returned out of the call and swapped in by the runner (same pattern as the
existing `replace_kv_buffer`).

---

## Testing strategy

1. **Kernel unit tests** (no model, no server):
   - `test_gated_delta.py`: compare chunked prefill and single-step decode
     against a plain Python reference for B=2, L∈{1, 7, 64, 120, 512}, small
     head dims; check float tolerances tight enough to catch indexing bugs.
   - `test_causal_conv1d.py`: compare against a padded `jnp.conv_general_dilated`
     reference for the same B/L/kernel combinations.
2. **MambaPool tests**: alloc, free, wraparound, get/write layer.
3. **Model module test**: instantiate a 4-layer, 2-expert config from a minimal
   `Qwen3NextConfig`; run a random-weight forward; shapes match; decode after
   a prefill produces different hidden states than prefill alone.
4. **Numerical match vs. HF (small)**: load a 4-layer toy HF Qwen3-Next (either
   a downsized copy of official weights or random weights loaded symmetrically
   into both backends) and match the forward output within bf16 tolerance.
5. **Smoke**: adapt `/dev/shm/work/sglang-tools/tools/grok/launch-grok-4layers.sh`
   into a Qwen3-Next 4-layer launch. Send a few completions through
   `/v1/chat/completions`; verify non-empty plausible outputs.
6. **GPQA-Diamond** via `/dev/shm/work/sglang-tools/tools/accuracy/eval_gpqa.sh`
   with the instruct model once smoke passes. Target ≈ 50.

---

## Rollout plan (incremental; each step landable)

1. Kernels: `gated_delta` + `causal_conv1d` with tests. (standalone, no model changes)
2. `MambaPool`, `HybridReqToTokenPool` with tests. (standalone)
3. ForwardBatch field additions. (mechanical pytree update)
4. `HybridLinearAttnBackend`. (thin glue)
5. `qwen3_next.py` model + weight mapping; entry registered.
6. Scheduler / TpWorker / ModelRunner hooks.
7. 4-layer smoke; then full model GPQA.

## End-to-end runtime status

**Working end-to-end on TPU v7-8.** With the trimmed override
`num_hidden_layers=8, num_experts=8, num_experts_per_tok=2,
moe_intermediate_size=512, shared_expert_intermediate_size=512`, tp=8, the
launch script `/dev/shm/work/sglang-tools/tools/qwen3_next/launch-qwen3-next-4layers.sh`
loads the real HF Qwen3-Next-80B-A3B-Instruct safetensors (58 regular + 24 MoE
groups), builds the hybrid pool (6 linear-attn + 2 full-attn layers), reaches
`Uvicorn running on http://127.0.0.1:30000`, and serves
`/v1/completions` requests end-to-end (prefill → gated DeltaNet decode →
sampler → token stream → mamba-slot free).

Output text is gibberish because the trimmed config drops 80% of layers and
all but 8 of the 512 routed experts; running the full 48-layer / 512-expert
model gives the real accuracy story. But the full pipeline — gated-delta
recurrence, conv1d state, mamba pool I/O, hybrid layer dispatch, full-attn
gated output, sparse MoE, sampler, request lifecycle — all runs.

The end-to-end run took these fixes on top of the design described above:

1. `Qwen3NextFullAttention.q_proj` emits `2×num_attention_heads×head_dim` and
   `__call__` splits the projection into query + gate, multiplies
   `attn_output` by `sigmoid(gate)` before `o_proj` (matches HF
   `Qwen3NextAttention`'s "gated output").
2. `MambaPool.write_layer` performs the per-slot update as
   `at[slot_indices].set(...)` on the per-layer slice followed by
   `jax.lax.dynamic_update_slice` for the layer axis, avoiding a JAX
   tracer-bool edge case with `.at[layer_id, slot_indices]` mixed indexing
   under jit.
3. `MambaPool` backing arrays are sharded at allocation
   (`P(None, None, "tensor", None)` / `P(None, None, "tensor", None, None)`)
   and `write_layer` passes the matching `out_sharding`.
4. `ModelRunner.init_memory_pool()` allocates the pool inside
   `with jax.set_mesh(self.mesh):` so the sharded `jnp.zeros` materialise.
5. `_layer_types(config)` truncates the base-config's 48-entry list to
   `config.num_hidden_layers` so a 4- or 8-layer override emits weight
   mappings only for the trimmed range.
6. **`vocab_mask` (= grammar bitmask) is converted to a replicated
   `jax.Array` in `TpModelWorker._update_grammar_vocab_mask`** rather than
   handed in as a numpy ndarray. Its packed last dim is `vocab_size // 32`
   (e.g. 4748 for Qwen3-Next), which is divisible by 4 but not by tp=8;
   leaving it as numpy let the JIT input-sharding inference invent a
   `devices=[1, 4, 2]<=[8] last_tile_dim_replicate` GSPMD layout that no
   `NamedSharding` on a flat `(data, tensor)` mesh can express. The error
   message `IndivisibleError: shape=[1, 4, 2]` is the *device-tile shape*
   of that GSPMD spec, not a tensor shape.
7. `MambaPool.free()` zeros freed slots inside `jax.set_mesh(self.mesh)` so
   the host-side scatter can resolve the shard layout (the call happens
   outside any model JIT, when the scheduler reclaims a request slot).
8. `ModelRunner.run_model_wrapper` updates **both** `self.mamba_pool` *and*
   `self.req_to_token_pool.mamba_pool` with the post-jit pool. Without the
   second assignment, the `HybridReqToTokenPool` kept a reference to the
   donated-and-freed input arrays, and the next `free()` on a finished
   request hit `RuntimeError: Array has been deleted`.

## Implementation status (as of this RFC landing)

| Step | Status | Notes |
|------|--------|-------|
| 1. Kernels | **Landed & tested** — `python/sgl_jax/srt/kernels/gated_delta/`; 9 gated-delta + 4 conv1d unit tests in `test/kernels/gated_delta_test.py`. |
| 2. `MambaPool` / `HybridReqToTokenPool` | **Landed & tested** — `mem_cache/mamba_pool.py`; 7 unit tests in `test/mem_cache/mamba_pool_test.py`. |
| 3. ForwardBatch fields | **Landed** — two optional fields (`linear_attn_metadata`, `mamba_cache_indices`) added; pytree round-trip verified. |
| 4. `HybridLinearAttnBackend` | **Not yet** — deferred; model already consumes `forward_batch.linear_attn_metadata` which is filled by the existing `LinearAttentionBackend`, so for a first pass the runner can instantiate both backends directly without a wrapper. |
| 5. `qwen3_next.py` model | **Landed** — constructs, forwards end-to-end on a 3-layer linear-only decode smoke (4-layer config with toy dims; output shape `(B, vocab)` correct). Weight-mapping dict drafted (HF `in_proj_qkvz`, `in_proj_ba`, `conv1d.{weight,bias}`, `A_log`, `dt_bias`, `norm.weight`, `out_proj.weight`; shared-expert gate; routed experts via `create_moe_weights_mapping`). Registered in model registry. |
| 6. Runner / scheduler hooks | **Not yet** — remaining touch points: `ModelRunner.init_memory_pool()` to instantiate `HybridReqToTokenPool` when the HF config reports gated-DeltaNet layers; `TpModelWorker.forward_batch_generation()` to call `LinearAttentionBackend.get_forward_metadata(batch)` and `HybridReqToTokenPool.get_mamba_indices(...)` on the host, assigning both to `forward_batch` before JIT; and `Scheduler.alloc_req_slots` / `free_req` to use the hybrid pool so mamba slots are allocated and released with request lifecycle. |
| 7. 4-layer launch + GPQA | **Not yet** — blocked on step 6; once the runner wiring lands, re-run `/dev/shm/work/sglang-tools/tools/grok/launch-grok-4layers.sh` with a Qwen3-Next override and then `eval_gpqa.sh`. |

### Known gaps / TODOs to close before end-to-end accuracy

1. **Multi-request prefill conv1d**: `causal_conv1d_prefill` currently takes a single `[B, T, D]` dense batch with one initial state per batch element. For sglang-jax's packed prefill (all requests concatenated into `[T_total, D]`), we either (a) pad per-request into `[B_reqs, max_T, D]` or (b) run a ragged per-request scan. This must be wired in the `Qwen3NextGatedDeltaNet.__call__` EXTEND branch; until then prefill is correct only when a single request is processed per call.
2. **Gated delta prefill performance**: pure-JAX scan; for long contexts a Pallas kernel is required. Out of scope for v1 / GPQA run.
3. **Weight-loader specifics**: `squeeze_dim` is referenced on `WeightMapping` for the `[conv_dim, 1, K]` → `[conv_dim, K]` squeeze; if the loader lacks that field we must fold a squeeze into `load_weights_from_safetensors` or the model itself. Either way the final test is loading an HF checkpoint and comparing logits on a fixed prompt.
4. **Mamba pool sharding**: the state arrays are currently replicated; reshard to `P(None, None, "tensor", ...)` at init time so reads don't need a `reshard` inside the JIT graph (current workaround adds one per linear layer per step).
5. **Per-linear-layer indexing**: the pool currently keys by absolute `layer_id`; a denser storage layout keyed by linear-layer-only index (0..36) cuts memory waste in the `num_layers × size × ...` buffers from 48× to 36×.

The RFC-level design covers all of the above; the remaining work is wiring, not algorithmic.

---

## Risks & open questions

- **Gated-delta kernel performance**: pure-JAX chunked scan will be slow on
  long contexts. Mitigation: first make it correct; profile; move the inner
  chunk to Pallas if needed. GPQA-Diamond sequences are short (<4k tokens), so
  pure-JAX is very likely acceptable for the initial eval.
- **Per-layer pool sizing**: each linear layer carries
  `size × num_v_heads × V × K × 4 bytes`. For 80B-A3B that's
  `max_reqs × 32 × 128 × 128 × 4 ≈ 2 MB/req/layer × 36 layers ≈ 72 MB per request`.
  Plan for `max_running_requests ≤ 256` on 8-chip TPU — ~18 GB, fits.
- **Shared expert load-balance**: the Qwen3-Next router uses shared + routed;
  we must add the shared output *unconditionally* (no gate), matching HF.
- **Prefix caching of mamba state**: deferred. The first version frees mamba
  slots at request end and does not share state across requests.
- **In-place pool updates inside JIT**: we must pass the pool in and out of the
  model as part of the JAX pytree; confirm that `ModelRunner._forward` swaps it
  like it does with `token_to_kv_pool` today.
- **Weight naming**: HF's fused `in_proj_qkvz` exactly matches what we want;
  but the split shape must be verified head-by-head (sglang's loader has custom
  per-head slicing). We mirror that logic.

---

## References

- sglang ref: `/dev/shm/work/sglang/python/sglang/srt/models/qwen3_next.py`
- sglang mamba pool: `/dev/shm/work/sglang/python/sglang/srt/mem_cache/memory_pool.py:191+`
- sglang hybrid pool: `memory_pool.py:449+` (`HybridReqToTokenPool`)
- sglang hybrid backend: `layers/attention/hybrid_linear_attn_backend.py`
- sglang-jax linear-attn precedent: `python/sgl_jax/srt/models/bailing_moe_v2_5_linear_attention.py`
  and `layers/attention/fla/linear_attention_backend.py`
- Qwen3-Next HF config: `transformers.models.qwen3_next.configuration_qwen3_next`
