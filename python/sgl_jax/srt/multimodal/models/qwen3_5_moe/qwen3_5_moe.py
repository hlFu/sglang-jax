import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from transformers import PretrainedConfig
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.eplb.expert_location import ExpertLocationMetadata
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead, RotaryEmbedding
from sgl_jax.srt.layers.fused_moe import FusedEPMoE
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.layers.moe import EPMoE, GateLogit, TopK, create_moe_weights_mapping
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen3 import Qwen3MLP
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping
from qwen3_5_vision_encoder import Qwen3_5VisionEncoder

logger = logging.getLogger(__name__)

init_fn = nnx.initializers.uniform()


class QWen3MoeAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int,
        mesh: jax.sharding.Mesh,
        rope_theta: float = 10000,
        rope_scaling: dict[str, Any] | None = None,
        head_dim: int | None = None,
        rms_norm_eps: float = None,
        layer_id: int = 0,
        attention_bias: bool = False,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        assert num_heads % num_kv_heads == 0

        self.head_dim = head_dim or hidden_size // num_heads
        self.q_head_num = num_heads
        self.kv_head_num = num_kv_heads

        self.q_size = num_heads * self.head_dim
        self.kv_size = num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.q_norm = RMSNorm(
            self.head_dim,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            epsilon=rms_norm_eps,
            param_dtype=dtype,
        )

        self.q_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.k_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.v_proj = LinearBase(
            input_size=hidden_size,
            output_size=num_kv_heads * self.head_dim,
            use_bias=attention_bias,
            kernel_axes=(None, "tensor"),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.c_proj = LinearBase(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            use_bias=attention_bias,
            kernel_axes=("tensor", None),
            params_dtype=dtype,
            mesh=mesh,
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
            dtype=dtype,
        )
        self.attn = RadixAttention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scaling=self.scaling,
            num_kv_heads=num_kv_heads,
            layer_id=layer_id,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, jax.Array]:
        q, _ = self.q_proj(hidden_states)
        k, _ = self.k_proj(hidden_states)
        v, _ = self.v_proj(hidden_states)

        q = q.reshape(-1, self.q_head_num, self.head_dim)
        k = k.reshape(-1, self.kv_head_num, self.head_dim)
        v = v.reshape(-1, self.kv_head_num, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q, k = self.rotary_emb(positions, q, k)
        attn_output, kv_fused = self.attn(q, k, v, forward_batch, token_to_kv_pool)

        output, _ = self.c_proj(attn_output)
        return output, kv_fused


class QWen3MoeDecoderLayer(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        layer_id: int = 0,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.mesh = mesh
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 40960)
        head_dim = getattr(config, "head_dim", None)

        self.self_attn = QWen3MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            rms_norm_eps=config.rms_norm_eps,
            layer_id=layer_id,
            attention_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            mesh=mesh,
        )

        mlp_only_layers = getattr(config, "mlp_only_layers", [])

        if layer_id in mlp_only_layers:
            self.mlp = Qwen3MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                layer_id=layer_id,
                dtype=dtype,
                mesh=mesh,
            )
            self.is_moe_layer = False
            self.moe_gate = None
        else:
            num_experts = getattr(config, "num_experts", 128)
            num_experts_per_tok = getattr(config, "num_experts_per_tok", 8)
            moe_intermediate_size = getattr(config, "moe_intermediate_size", 768)

            self.moe_backend = getattr(config, "moe_backend", "epmoe")
            self.use_fused = self.moe_backend == "fused"

            self.moe_gate = GateLogit(
                input_size=config.hidden_size,
                num_experts=num_experts,
                weight_dtype=dtype,
            )

            self.topk = TopK(
                topk=num_experts_per_tok,
                renormalize=config.norm_topk_prob,
                layer_id=layer_id,
            )

            if self.use_fused:
                self.mlp = FusedEPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    intermediate_dim=moe_intermediate_size,
                    mesh=mesh,
                    activation="silu",
                    ep_size=config.ep_size,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    renormalize_topk_logits=config.norm_topk_prob,
                    quantization_config=getattr(config, "quantization_config", None),
                )
            else:
                self.mlp = EPMoE(
                    hidden_size=config.hidden_size,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    intermediate_dim=moe_intermediate_size,
                    mesh=mesh,
                    ep_size=config.ep_size,
                    weight_dtype=dtype,
                    dtype=dtype,
                    layer_id=layer_id,
                    quantization_config=getattr(config, "quantization_config", None),
                )
            self.is_moe_layer = True

        self.input_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        positions: jax.Array,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        residual: jax.Array | None = None,
        dispatch_info: ExpertLocationMetadata | None = None,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states += residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

        hidden_states, kv_fused = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
            token_to_kv_pool=token_to_kv_pool,
        )

        hidden_states += residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.is_moe_layer:
            router_logits = self.moe_gate(hidden_states)
            topk_weights, topk_ids = self.topk(router_logits, dispatch_info=dispatch_info)

            if self.use_fused:
                token_valid_mask = forward_batch.get_token_valid_mask(hidden_states.shape[0])
                topk_ids = jnp.where(token_valid_mask[:, None], topk_ids, -1)
            else:
                pass
            hidden_states = self.mlp(hidden_states, topk_weights, topk_ids)
        else:
            hidden_states = self.mlp(hidden_states)
            topk_ids = None

        return hidden_states, residual, kv_fused, topk_ids


class QWen3MoeModel(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

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
                QWen3MoeDecoderLayer(
                    config=config,
                    layer_id=i,
                    dtype=dtype,
                    mesh=mesh,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
        )

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ) -> tuple[jax.Array, list[jax.Array]]:
        hidden_states = self.embed_tokens(forward_batch.input_ids)
        residual = None
        layers_kv_fused = []
        layers_topk_ids = []
        for layer in self.layers:
            hidden_states, residual, kv_fused, topk_ids = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
                dispatch_info=forward_batch.expert_location_metadata,
            )
            layers_kv_fused.append(kv_fused)
            layers_topk_ids.append(topk_ids)

        if residual is not None:
            hidden_states += residual
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states, layers_kv_fused, layers_topk_ids


class Qwen3MoeForCausalLM(nnx.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype
        logger.info("QWen3MoeForCausalLMModel config dtype: %s", self.dtype)
        self.model = QWen3MoeModel(config, dtype=self.dtype, mesh=mesh)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=self.dtype,
                param_dtype=self.dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = self._create_qwen3_moe_weight_mappings()
        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen3Moe weights loaded successfully!")

    def _create_qwen3_moe_weight_mappings(self) -> dict:
        mappings = {
            "model.embed_tokens.weight": WeightMapping(
                target_path="model.embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.config.num_hidden_layers
        mlp_only_layers = getattr(self.config, "mlp_only_layers", [])

        for layer_idx in range(num_layers):
            layer_mappings = self._create_moe_layer_mappings(
                layer_idx, layer_idx in mlp_only_layers
            )
            mappings.update(layer_mappings)

        return mappings

    def _create_moe_layer_mappings(self, layer_idx: int, is_mlp_layer: bool) -> dict:
        prefix = f"model.layers.{layer_idx}"
        target_prefix = f"model.layers.{layer_idx}"

        mappings = {
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
                target_path=f"{target_prefix}.self_attn.c_proj.weight",
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

        if getattr(self.config, "attention_bias", False):
            bias_mappings = {
                f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.q_proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
                f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.k_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.v_proj.bias",
                    sharding=(None,),
                    transpose=False,
                    kv_head_padding=True,
                ),
                f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                    target_path=f"{target_prefix}.self_attn.c_proj.bias",
                    sharding=(None,),
                    transpose=False,
                ),
            }
            mappings.update(bias_mappings)

        if is_mlp_layer:
            mlp_mappings = {
                f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.mlp.gate_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{prefix}.mlp.up_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.mlp.up_proj.weight",
                    sharding=(None, "tensor"),
                    transpose=True,
                ),
                f"{prefix}.mlp.down_proj.weight": WeightMapping(
                    target_path=f"{target_prefix}.mlp.down_proj.weight",
                    sharding=("tensor", None),
                    transpose=True,
                ),
            }
            mappings.update(mlp_mappings)
        else:
            mappings[f"{prefix}.mlp.gate.weight"] = WeightMapping(
                target_path=f"{target_prefix}.moe_gate.kernel",
                sharding=(None, None),
                transpose=True,
            )

            moe_backend = getattr(self.config, "moe_backend", "epmoe")
            num_experts = getattr(self.config, "num_experts", 128)

            # Get physical to logical mapping for redundant experts
            from sgl_jax.srt.eplb.expert_location import (
                get_global_expert_location_metadata,
            )

            metadata = get_global_expert_location_metadata()
            phy_to_log = None
            num_physical_experts = num_experts
            if metadata is not None:
                num_physical_experts = metadata.num_physical_experts
                physical_to_logical_map = np.array(jax.device_get(metadata.physical_to_logical_map))
                phy_to_log = physical_to_logical_map[layer_idx]
                sample = phy_to_log[: min(10, phy_to_log.shape[0])].tolist()
                logger.info(
                    "Layer %s: logical=%s, physical=%s, redundancy=%.2fx",
                    layer_idx,
                    num_experts,
                    num_physical_experts,
                    num_physical_experts / num_experts,
                )
                logger.info(
                    "Layer %s EPLB map: size=%s min=%s max=%s sample=%s",
                    layer_idx,
                    phy_to_log.shape[0],
                    int(phy_to_log.min()),
                    int(phy_to_log.max()),
                    sample,
                )

            moe_mappings = create_moe_weights_mapping(
                prefix=prefix,
                target_prefix=target_prefix,
                num_experts=num_experts,
                moe_backend=moe_backend,
                moe_path="mlp",
                source_expert_pattern="experts.{i}",
                physical_to_logical_map=phy_to_log,
            )
            mappings.update(moe_mappings)

        return mappings

    def get_embed_and_head(self):
        return (
            self.transformer.embed_tokens.embedding.value,
            self.lm_head.embedding.value,
        )

    def set_embed_and_head(
        self,
        embed_weight: jax.Array | None = None,
        head_weight: jax.Array | None = None,
    ) -> None:
        """Set word embedding and LM Head weights.

        Args:
            embed_weight: Embedding matrix with shape [vocab_size, hidden_size].
            head_weight:  LM Head matrix with shape [vocab_size, hidden_size].
        """

        # Set embedding weight
        if embed_weight is not None:
            self.transformer.embed_tokens.embedding.value = embed_weight

        # Set LM Head weight
        if head_weight is not None:
            self.lm_head.embedding.value = head_weight

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states, layers_kv_fused, layers_topk_ids = self.model(
            forward_batch,
            token_to_kv_pool,
        )
        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(hidden_states, self.lm_head, logits_metadata)
        else:
            output = self.logits_processor(hidden_states, self.model.embed_tokens, logits_metadata)
        return output, layers_kv_fused, True, layers_topk_ids


class Qwen3MoeForConditionalGeneration(nnx.Module):

    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        *,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        if rngs is None:
            rngs = nnx.Rngs(0)

        self.mesh = mesh
        self.dtype = dtype
        self.config = config
        self.visual = Qwen3_5VisionEncoder(
            config.vision_config, mesh=mesh, dtype=dtype, rngs=rngs
        )
        self.text_embed_tokens = Embed(
            num_embeddings=config.text_config.vocab_size,
            features=config.text_config.hidden_size,
            dtype=dtype,
            param_dtype=dtype,
            kernel_axes=("tensor", None),
            mesh=mesh,
        )

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling
        
        self.logits_processor = LogitsProcessor(self.config.vocab_size, mesh=self.mesh)
        

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )

        weight_mappings = {
            **self._create_audio_tower_weight_mappings(self.config.audio_config),
            **self._create_visual_weight_mappings(self.config.vision_config),
            **self._create_text_embed_tokens_mappings(self.config.text_config),
        }

        loader.load_weights_from_safetensors(weight_mappings)
        logger.info("Qwen3OmniMoeThinkerEmbedding weights loaded successfully!")

    @staticmethod
    def _create_audio_tower_weight_mappings(config: Qwen3OmniMoeAudioEncoderConfig) -> dict:
        mappings = {}
        prefix = "thinker.audio_tower"
        target_prefix = "audio_tower"

        # 1. conv2d layer: (conv2d1, conv2d2, conv2d3)
        for i in range(1, 4):
            mappings[f"{prefix}.conv2d{i}.weight"] = WeightMapping(
                target_path=f"{target_prefix}.conv2d{i}.kernel",
                transpose_axes=(2, 3, 1, 0),  # PT [O, I, H, W] -> JAX [H, W, I, O]
                sharding=(None, None, None, None),
            )
            mappings[f"{prefix}.conv2d{i}.bias"] = WeightMapping(
                target_path=f"{target_prefix}.conv2d{i}.bias",
                sharding=(None,),
            )

        # 2. conv_out layer:
        mappings[f"{prefix}.conv_out.weight"] = WeightMapping(
            target_path=f"{target_prefix}.conv_out.weight",
            transpose=True,  # PT [O, I] -> JAX [I, O]
            sharding=(None, None),
        )

        # 3. Transformer layer: (0-31)
        for i in range(config.num_hidden_layers):
            l_pre = f"{prefix}.layers.{i}"
            l_targ = f"{target_prefix}.layers.{i}"

            # Self Attention: q_proj, k_proj, v_proj, out_proj
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"]:
                mappings[f"{l_pre}.self_attn.{proj}.weight"] = WeightMapping(
                    target_path=f"{l_targ}.self_attn.{proj}.weight",
                    transpose=True,  # PT [O, I] -> JAX [I, O]
                    sharding=(None, None),
                )
                mappings[f"{l_pre}.self_attn.{proj}.bias"] = WeightMapping(
                    target_path=f"{l_targ}.self_attn.{proj}.bias",
                    sharding=(None,),
                )

            # Attention LayerNorm (weight -> scale)
            mappings[f"{l_pre}.self_attn_layer_norm.weight"] = WeightMapping(
                target_path=f"{l_targ}.self_attn_layer_norm.scale",
                sharding=(None,),
            )
            mappings[f"{l_pre}.self_attn_layer_norm.bias"] = WeightMapping(
                target_path=f"{l_targ}.self_attn_layer_norm.bias",
                sharding=(None,),
            )

            # MLP layer: (fc1, fc2)
            for fc in ["fc1", "fc2"]:
                mappings[f"{l_pre}.{fc}.weight"] = WeightMapping(
                    target_path=f"{l_targ}.{fc}.weight",
                    transpose=True,  # PT [O, I] -> JAX [I, O]
                    sharding=(None, None),
                )
                mappings[f"{l_pre}.{fc}.bias"] = WeightMapping(
                    target_path=f"{l_targ}.{fc}.bias",
                    sharding=(None,),
                )

            # Final LayerNorm (weight -> scale)
            mappings[f"{l_pre}.final_layer_norm.weight"] = WeightMapping(
                target_path=f"{l_targ}.final_layer_norm.scale",
                sharding=(None,),
            )
            mappings[f"{l_pre}.final_layer_norm.bias"] = WeightMapping(
                target_path=f"{l_targ}.final_layer_norm.bias",
                sharding=(None,),
            )

        # 4. post process: (ln_post, proj1, proj2)
        # ln_post (weight -> scale)
        mappings[f"{prefix}.ln_post.weight"] = WeightMapping(
            target_path=f"{target_prefix}.ln_post.scale",
            sharding=(None,),
        )
        mappings[f"{prefix}.ln_post.bias"] = WeightMapping(
            target_path=f"{target_prefix}.ln_post.bias",
            sharding=(None,),
        )

        # proj1 & proj2
        for p in ["proj1", "proj2"]:
            mappings[f"{prefix}.{p}.weight"] = WeightMapping(
                target_path=f"{target_prefix}.{p}.weight",
                transpose=True,  # PT [O, I] -> JAX [I, O]
                sharding=(None, None),
            )
            mappings[f"{prefix}.{p}.bias"] = WeightMapping(
                target_path=f"{target_prefix}.{p}.bias",
                sharding=(None,),
            )

        return mappings

    @staticmethod
    def _create_visual_weight_mappings(config: Qwen3OmniMoeVisionEncoderConfig) -> dict:
        prefix: str = "thinker.visual."
        target_prefix = "visual."
        mappings = {}

        # Helper functions to reduce repetition
        def add_linear(src: str, dst: str, tp_col: bool = False, tp_row: bool = False):
            """Add linear layer mapping with optional TP sharding."""
            w_sharding = (
                (None, "tensor") if tp_col else ("tensor", None) if tp_row else (None, None)
            )
            b_sharding = ("tensor",) if tp_col else (None,)
            mappings[f"{prefix}{src}.weight"] = WeightMapping(
                target_path=f"{target_prefix}{dst}.weight", sharding=w_sharding, transpose=True
            )
            mappings[f"{prefix}{src}.bias"] = WeightMapping(
                target_path=f"{target_prefix}{dst}.bias", sharding=b_sharding, transpose=False
            )

        def add_layernorm(src: str, dst: str):
            """Add layernorm mapping."""
            mappings[f"{prefix}{src}.weight"] = WeightMapping(
                target_path=f"{target_prefix}{dst}.scale", sharding=(None,), transpose=False
            )
            mappings[f"{prefix}{src}.bias"] = WeightMapping(
                target_path=f"{target_prefix}{dst}.bias", sharding=(None,), transpose=False
            )

        # ==================== Patch Embedding ====================
        # Conv3d: PyTorch (out, in, T, H, W) -> JAX (T, H, W, in, out)
        mappings[f"{prefix}patch_embed.proj.weight"] = WeightMapping(
            target_path=f"{target_prefix}patch_embed.proj.kernel",
            sharding=(None, None, None, None, None),
            transpose=False,
            transpose_axes=(2, 3, 4, 1, 0),
        )
        mappings[f"{prefix}patch_embed.proj.bias"] = WeightMapping(
            target_path=f"{target_prefix}patch_embed.proj.bias", sharding=(None,), transpose=False
        )

        # ==================== Position Embedding ====================
        mappings[f"{prefix}pos_embed.weight"] = WeightMapping(
            target_path=f"{target_prefix}pos_embed.embedding",
            sharding=(None, None),
            transpose=False,
        )

        # ==================== Transformer Blocks ====================
        for i in range(config.depth):
            block = f"blocks.{i}"

            # LayerNorm
            add_layernorm(f"{block}.norm1", f"{block}.norm1")
            add_layernorm(f"{block}.norm2", f"{block}.norm2")

            # Attention: QKV (column-wise TP), Output (row-wise TP)
            add_linear(f"{block}.attn.qkv", f"{block}.attn.qkv_proj", tp_col=True)
            add_linear(f"{block}.attn.proj", f"{block}.attn.o_proj", tp_row=True)

            # MLP: fc1 (column-wise TP), fc2 (row-wise TP)
            add_linear(f"{block}.mlp.linear_fc1", f"{block}.mlp.fc1", tp_col=True)
            add_linear(f"{block}.mlp.linear_fc2", f"{block}.mlp.fc2", tp_row=True)

        # ==================== Final Merger ====================
        add_layernorm("merger.ln_q", "merger.ln_q")
        # Merger MLP: [0]=Linear, [1]=GELU, [2]=Linear
        add_linear("merger.mlp.0", "merger.mlp_fc1", tp_col=True)
        add_linear("merger.mlp.2", "merger.mlp_fc2", tp_row=True)

        # ==================== Deepstack Mergers ====================
        deepstack_indexes = getattr(config, "deepstack_visual_indexes", [8, 16, 24])
        for idx in range(len(deepstack_indexes)):
            src = f"merger_list.{idx}"
            dst = f"deepstack_mergers.{idx}"

            add_layernorm(f"{src}.ln_q", f"{dst}.ln_q")
            add_linear(f"{src}.mlp.0", f"{dst}.mlp_fc1", tp_col=True)
            add_linear(f"{src}.mlp.2", f"{dst}.mlp_fc2", tp_row=True)

        return mappings

    @staticmethod
    def _create_text_embed_tokens_mappings(config: Qwen3OmniMoeAudioEncoderConfig) -> dict:
        mappings = {}
        prefix = "thinker.model"
        target_prefix = "text_embed_tokens"

        mappings[f"{prefix}.embed_tokens.weight"] = WeightMapping(
            target_path=f"{target_prefix}.embedding",
            sharding=("tensor", None),
        )

        return mappings

    def get_placeholder_mask(
        self,
        input_ids: jnp.ndarray,
        input_embeds: jnp.ndarray,
        image_features: jnp.ndarray | None = None,
        video_features: jnp.ndarray | None = None,
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `input_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id
        special_audio_mask = input_ids == self.config.audio_token_id

        n_image_tokens = jnp.sum(special_image_mask)
        special_image_mask = jnp.broadcast_to(
            jnp.expand_dims(special_image_mask, axis=-1), input_embeds.shape
        )
        if (
            image_features is not None
            and input_embeds[special_image_mask].size != image_features.size
        ):
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
            )

        n_video_tokens = jnp.sum(special_video_mask)
        special_video_mask = jnp.broadcast_to(
            jnp.expand_dims(special_video_mask, axis=-1), input_embeds.shape
        )
        if (
            video_features is not None
            and input_embeds[special_video_mask].size != video_features.size
        ):
            raise ValueError(
                f"Videos features and video tokens do not match: tokens: {n_video_tokens}, features {video_features.shape[0]}"
            )

        special_audio_mask = jnp.broadcast_to(
            jnp.expand_dims(special_audio_mask, axis=-1), input_embeds.shape
        )

        return special_image_mask, special_video_mask, special_audio_mask

    def __call__(
        self,
        input_ids: jax.Array,
        input_features=None,
        audio_feature_lengths=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
    ):
        """
        Encodes audios into continuous embeddings that can be forwarded to the language model.

        Args:
            input_features (`torch.FloatTensor`):
                The tensors corresponding to the input audios.
            audio_feature_lengths (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
        """

        # 1. Extract the input embeddings
        input_embeds = self.text_embed_tokens(input_ids)

        visual_embeds_multiscale = None
        visual_pos_masks = None

        audio_embeds = None
        image_embeds = None
        video_embeds = None

        # Merge text , audios , image and video
        if input_features is not None:
            audio_embeds = self.audio_tower(
                input_features.astype(self.dtype),
                feature_lens=audio_feature_lengths,
            )

        if pixel_values is not None:
            image_features = self.visual(pixel_values.astype(self.dtype), image_grid_thw)
            image_embeds, image_embeds_multiscale = (
                image_features["pooler_output"],
                image_features["deepstack_features"],
            )
            visual_embeds_multiscale = image_embeds_multiscale

        if pixel_values_videos is not None:
            video_features = self.visual(pixel_values_videos.astype(self.dtype), video_grid_thw)
            video_embeds, video_embeds_multiscale = (
                video_features["pooler_output"],
                video_features["deepstack_features"],
            )
            if visual_embeds_multiscale is None:
                visual_embeds_multiscale = video_embeds_multiscale

        image_mask, video_mask, audio_mask = self.get_placeholder_mask(
            input_ids,
            input_embeds=input_embeds,
            image_features=image_embeds,
            video_features=video_embeds,
        )
        if audio_embeds is not None:
            input_embeds = input_embeds.at[audio_mask].set(jnp.ravel(audio_embeds))
        if image_embeds is not None:
            input_embeds = input_embeds.at[image_mask].set(jnp.ravel(image_embeds))
        if video_embeds is not None:
            input_embeds = input_embeds.at[video_mask].set(jnp.ravel(video_embeds))

        # for image and video mask
        if pixel_values is not None and pixel_values_videos is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask | image_mask
            visual_embeds_multiscale_joint = ()
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(image_embeds_multiscale, video_embeds_multiscale):
                embed_joint = jnp.zeros(
                    (visual_pos_masks.sum(), img_embed.shape[-1]), dtype=img_embed.dtype
                )
                embed_joint = embed_joint.at[image_mask_joint, :].set(img_embed)
                embed_joint = embed_joint.at[video_mask_joint, :].set(vid_embed)
                visual_embeds_multiscale_joint = visual_embeds_multiscale_joint + (embed_joint,)
            visual_embeds_multiscale = visual_embeds_multiscale_joint
        elif pixel_values is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
        elif pixel_values_videos is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
        return input_embeds, visual_embeds_multiscale, visual_pos_masks

EntryClass = Qwen3MoeForConditionalGeneration
