import logging

import jax
from flax import nnx
from jax import numpy as jnp
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_hf_text_config
from sgl_jax.srt.layers.embeddings import ParallelLMHead
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import create_moe_weights_mapping
from sgl_jax.srt.mem_cache.memory_pool import KVCache, RecurrentStatePool
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen3_5_moe import Qwen3_5Model, _layer_types
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


class Qwen3_5ForConditionalGeneration(nnx.Module):
    """Multimodal wrapper around the hybrid-linear Qwen3.5 language backbone.

    Owns the inner ``Qwen3_5Model`` directly (not the text-only ``Qwen3_5ForCausalLM``)
    so that ``lm_head`` and ``logits_processor`` live exactly once on this wrapper —
    matching the omni precedent. The wrapper translates HF's ``thinker.…`` checkpoint
    prefix and forwards the recurrent state pool through to the inner model so the
    linear-attention layers can read/write per-request mamba state.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = get_hf_text_config(config) or config
        self.dtype = dtype
        self.model = Qwen3_5Model(self.config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(self.config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_qwen3_5_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen3_5 (multimodal) weights loaded successfully!")

    def _create_qwen3_5_weight_mappings(self) -> dict:
        mappings: dict = {
            "thinker.model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "thinker.model.norm.weight": WeightMapping(
                target_path="model.norm.scale",
                sharding=(None,),
                transpose=False,
            ),
        }
        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["thinker.lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding",
                sharding=("tensor", None),
                transpose=False,
            )

        # Qwen3.5 interleaves linear-attention and full-attention layers; mappings differ.
        for layer_idx, lt in enumerate(_layer_types(self.config)):
            if lt == "linear_attention":
                mappings.update(self._create_linear_attn_layer_mappings(layer_idx))
            else:
                mappings.update(self._create_full_attn_layer_mappings(layer_idx))
            mappings.update(self._create_moe_layer_mappings(layer_idx))

        return mappings

    @staticmethod
    def _create_layernorm_mappings(prefix: str, target_prefix: str) -> dict:
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

    def _create_full_attn_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"thinker.model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"
        mappings = self._create_layernorm_mappings(prefix, target_prefix)
        mappings.update(
            {
                f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.k_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.v_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.o_proj.weight",
                    sharding=("tensor", None),
                    transpose=True,
                ),
                f"{prefix}.self_attn.q_norm.weight": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_norm.scale",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.self_attn.k_norm.weight": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.k_norm.scale",
                    sharding=(None,),
                    transpose=False,
                ),
            }
        )
        return mappings

    def _create_linear_attn_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"thinker.model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"
        la = "linear_attn"  # HF and ours share this attribute name
        mappings = self._create_layernorm_mappings(prefix, target_prefix)
        mappings.update(
            {
                f"{prefix}.{la}.in_proj_qkvz.weight": WeightMapping(
                    target_path=f"{target_prefix}.{la}.in_proj_qkvz.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{prefix}.{la}.in_proj_ba.weight": WeightMapping(
                    target_path=f"{target_prefix}.{la}.in_proj_ba.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                # HF stores conv1d as [conv_dim, 1, K]; we keep [conv_dim, K] — drop axis 1.
                f"{prefix}.{la}.conv1d.weight": WeightMapping(
                    target_path=f"{target_prefix}.{la}.conv1d_weight",
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
                    target_path=f"{target_prefix}.{la}.A_log",
                    sharding=("tensor",),
                    transpose=False,
                ),
                f"{prefix}.{la}.dt_bias": WeightMapping(
                    target_path=f"{target_prefix}.{la}.dt_bias",
                    sharding=("tensor",),
                    transpose=False,
                ),
                f"{prefix}.{la}.norm.weight": WeightMapping(
                    target_path=f"{target_prefix}.{la}.rms_scale",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.{la}.out_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.{la}.out_proj.weight",
                    sharding=("tensor", None),
                    transpose=True,
                ),
            }
        )
        return mappings

    def _create_moe_layer_mappings(self, layer_idx: int) -> dict:
        prefix = f"thinker.model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"
        mappings: dict = {}
        # Routing gate (lives under mlp.moe_gate in Qwen3_5SparseMoeBlock).
        mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
            target_path=f"{target_prefix}.mlp.moe_gate.kernel",
            sharding=(None, None),
            transpose=True,
        )
        # Shared expert.
        for name in ("gate_proj", "up_proj"):
            mappings[f"{prefix}.mlp.shared_expert.{name}.weight"] = WeightMapping(
                target_path=f"{target_prefix}.mlp.shared_expert.{name}.weight",
                sharding=(None, "tensor"),
                transpose=True,
            )
        mappings[f"{prefix}.mlp.shared_expert.down_proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}.mlp.shared_expert.down_proj.weight",
            sharding=("tensor", None),
            transpose=True,
        )
        mappings[f"{prefix}.mlp.shared_expert_gate.weight"] = WeightMapping(
            target_path=f"{target_prefix}.mlp.shared_expert_gate.weight",
            sharding=(None, None),
            transpose=True,
        )
        # Routed experts.
        moe_backend = getattr(self.config, "moe_backend", "epmoe")
        mappings.update(
            create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target_prefix,
                num_experts=self.config.num_experts,
                moe_backend=moe_backend,
                moe_path="mlp.experts",
                source_expert_pattern="{i}",
            )
        )
        return mappings

    def get_embed_and_head(self):
        return (
            self.model.embed_tokens.embedding.value,
            self.lm_head.embedding.value,
        )

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        if embed_weight is not None:
            self.model.embed_tokens.embedding.value = embed_weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

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
            output = self.logits_processor(
                hidden_states, self.model.embed_tokens, logits_metadata
            )
        return output, layers_kv_fused, True, layers_topk_ids, new_recurrent_state_pool
