import jax
import jax.numpy as jnp
from flax import nnx
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeVisionEncoderConfig,
)

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.multimodal.models.qwen3_VL.vision_encoder import (
    Qwen3VLMoeVisionEncoder,
)


class VisionPatchMerger(nnx.Module):
    """
    Patch merger that projects vision features to language model dimension.

    Projects from hidden_size * (spatial_merge_size^2) to out_hidden_size.

    TP Strategy:
        - mlp_fc1: Column-wise sharding (None, "tensor") - split output dimension
        - mlp_fc2: Row-wise sharding ("tensor", None) - split input dimension, all-reduce output
    """

    def __init__(
        self,
        config: Qwen3OmniMoeVisionEncoderConfig,
        use_postshuffle_norm: bool,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        self.hidden_size = config.hidden_size
        self.spatial_merge_size = config.spatial_merge_size
        self.out_hidden_size = config.out_hidden_size
        self.use_postshuffle_norm = use_postshuffle_norm

        merged_hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        norm_size = merged_hidden_size if use_postshuffle_norm else config.hidden_size
        self.ln_q = nnx.LayerNorm(
            num_features=norm_size,
            epsilon=1e-6,
            param_dtype=dtype,
            use_fast_variance=False,
            rngs=rngs,
        )

        # TP: column-wise sharding
        self.mlp_fc1 = LinearBase(
            input_size=merged_hidden_size,
            output_size=merged_hidden_size,
            use_bias=True,
            kernel_axes=(None, "tensor"),  # TP: column-wise sharding
            params_dtype=dtype,
            mesh=mesh,
        )

        # TP: row-wise sharding
        self.mlp_fc2 = LinearBase(
            input_size=merged_hidden_size,
            output_size=config.out_hidden_size,
            use_bias=True,
            kernel_axes=("tensor", None),  # TP: row-wise sharding
            params_dtype=dtype,
            mesh=mesh,
        )

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        """
        Args:
            hidden_states: (seq_len, hidden_size)

        Returns:
            output: (seq_len, out_hidden_size)
        """
        merged_hidden_size = self.hidden_size * (self.spatial_merge_size**2)

        if self.use_postshuffle_norm:
            hidden_states = hidden_states.reshape(-1, merged_hidden_size)
            hidden_states = self.ln_q(hidden_states)
        else:
            hidden_states = self.ln_q(hidden_states)
            hidden_states = hidden_states.reshape(-1, merged_hidden_size)

        hidden_states, _ = self.mlp_fc1(hidden_states)
        hidden_states = jax.nn.gelu(hidden_states, approximate=False)
        hidden_states, _ = self.mlp_fc2(hidden_states)

        return hidden_states


class Qwen3OmniMoeVisionEncoder(Qwen3VLMoeVisionEncoder):
    """
    Qwen3OmniMoe Vision Encoder for processing images and videos.

    Architecture:
        1. 3D Conv Patch Embedding
        2. Learnable Position Embeddings (interpolated)
        3. 27 Vision Transformer Blocks with 2D RoPE
        4. Deepstack feature extraction at layers 8, 16, 24
        5. Final Patch Merger to language model dimension
    """

    def __init__(
        self,
        config: Qwen3OmniMoeVisionEncoderConfig,
        mesh: jax.sharding.Mesh,
        dtype: jnp.dtype = jnp.bfloat16,
        rngs: nnx.Rngs = None,
    ):
        super().__init__()

        # Deepstack mergers (for intermediate features) - use nnx.List for Flax NNX 0.12.0+ compatibility
        self.deepstack_mergers = nnx.List(
            [
                VisionPatchMerger(
                    config=config,
                    use_postshuffle_norm=True,  # PyTorch uses True for deepstack mergers
                    mesh=mesh,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in config.deepstack_visual_indexes
            ]
        )

        # Final merger
        self.merger = VisionPatchMerger(
            config=config,
            use_postshuffle_norm=False,  # PyTorch uses False for final merger
            mesh=mesh,
            dtype=dtype,
            rngs=rngs,
        )
