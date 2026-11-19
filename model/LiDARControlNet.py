from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import Attention, AttentionProcessor, FusedJointAttnProcessor2_0
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_sd3 import SD3SingleTransformerBlock
from diffusers.models.controlnets.controlnet import BaseOutput, zero_module

logger = logging.get_logger(__name__)

class SpatialAwareControlBlock(nn.Module):

    def __init__(self, dim: int, use_conv: bool = False, spatial_size: int = 64):
        super().__init__()
        self.dim = dim
        self.use_conv = use_conv
        self.spatial_size = spatial_size

        self._register_scanning_priors()

        if use_conv:
            self.control_layer = nn.Sequential(
                nn.Conv1d(dim, dim, 3, padding=1),
                nn.GroupNorm(min(32, dim // 4), dim),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Conv1d(dim, dim, 1)
            )
        else:
            self.control_layer = nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(dim, dim)
            )

        self.weight_mixer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.SiLU(),
            nn.Linear(dim // 2, 2),
            nn.Sigmoid()
        )

        self.learned_attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )

        self.output_scale = nn.Parameter(torch.tensor(0.01))

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        self._improved_init()

    def _register_scanning_priors(self):
        h, w = 8, 64

        vertical_idx = torch.linspace(0, 1, h)
        vertical_prior = 0.3 + 0.7 * torch.sigmoid((vertical_idx - 0.5))
        vertical_prior_2d = vertical_prior.unsqueeze(1).expand(h, w)

        combined_prior = vertical_prior_2d
        combined_prior = 0.5 + 0.5 * (combined_prior - combined_prior.min()) / (
                combined_prior.max() - combined_prior.min() + 1e-8
        )

        combined_prior_1d = combined_prior.flatten()
        self.register_buffer("scanning_prior", combined_prior_1d)

    def _improved_init(self):
        if self.use_conv:
            nn.init.zeros_(self.control_layer[-1].weight)
            nn.init.zeros_(self.control_layer[-1].bias)
            nn.init.xavier_uniform_(self.control_layer[0].weight, gain=0.1)
        else:
            nn.init.zeros_(self.control_layer[-1].weight)
            nn.init.zeros_(self.control_layer[-1].bias)
            nn.init.xavier_uniform_(self.control_layer[0].weight, gain=0.1)
        for module in self.weight_mixer:
            if isinstance(module, nn.Linear):
                if module == self.weight_mixer[-2]:
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    nn.init.zeros_(module.bias)

        for module in self.learned_attention:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def _get_adaptive_spatial_weights(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        B, N, C = x.shape

        if N != len(self.scanning_prior):
            scanning_prior = F.interpolate(
                self.scanning_prior.unsqueeze(0).unsqueeze(0),
                size=N, mode='linear', align_corners=False
            ).squeeze()
        else:
            scanning_prior = self.scanning_prior

        mix_coeffs = self.weight_mixer(x)
        scanning_weight = mix_coeffs[:, :, 0:1]
        learned_weight = mix_coeffs[:, :, 1:2]

        learned_attention = self.learned_attention(x)

        scanning_component = scanning_weight * scanning_prior.unsqueeze(0).unsqueeze(-1)
        learned_component = learned_weight * learned_attention

        spatial_weights = scanning_component + learned_component
        spatial_weights = torch.clamp(spatial_weights, 0.1, 1.0)

        return spatial_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        spatial_weights = self._get_adaptive_spatial_weights(x)

        if self.use_conv and x.dim() == 3:
            x_conv = x.transpose(1, 2)
            control_out = self.control_layer(x_conv)
            control_out = control_out.transpose(1, 2)
        else:
            control_out = self.control_layer(x)

        weighted_out = control_out * spatial_weights * self.output_scale

        output = weighted_out + identity * self.residual_scale
        output = torch.clamp(output, -2.0, 2.0)

        return output



@dataclass
class SD3ControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]


class SD3ControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            sample_size: int = 128,
            patch_size: int = 2,
            in_channels: int = 16,
            num_layers: int = 18,
            attention_head_dim: int = 64,
            num_attention_heads: int = 18,
            joint_attention_dim: int = 4096,
            caption_projection_dim: int = 1152,
            pooled_projection_dim: int = 2048,
            out_channels: int = 16,
            pos_embed_max_size: int = 96,
            extra_conditioning_channels: int = 0,
            dual_attention_layers: Tuple[int, ...] = (),
            qk_norm: Optional[str] = None,
            pos_embed_type: Optional[str] = "sincos",
            use_pos_embed: bool = True,
            force_zeros_for_pooled_projection: bool = True,
    ):
        super().__init__()
        default_out_channels = in_channels
        self.out_channels = out_channels if out_channels is not None else default_out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        if use_pos_embed:
            self.pos_embed = PatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=self.inner_dim,
                pos_embed_max_size=pos_embed_max_size,
                pos_embed_type=pos_embed_type,
            )
        else:
            self.pos_embed = None

        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )

        if joint_attention_dim is not None:
            self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)

            self.transformer_blocks = nn.ModuleList(
                [
                    JointTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        context_pre_only=False,
                        qk_norm=qk_norm,
                        use_dual_attention=True if i in dual_attention_layers else False,
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            self.context_embedder = None
            self.transformer_blocks = nn.ModuleList(
                [
                    SD3SingleTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                    )
                    for _ in range(num_layers)
                ]
            )

        latent_spatial_size = sample_size // patch_size
        self.controlnet_blocks = nn.ModuleList([
            SpatialAwareControlBlock(
                self.inner_dim,
                use_conv=False,
                spatial_size=latent_spatial_size
            )
            for _ in range(len(self.transformer_blocks))
        ])

        pos_embed_input = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels + extra_conditioning_channels,
            embed_dim=self.inner_dim,
            pos_embed_type=None,
        )
        self.pos_embed_input = zero_module(pos_embed_input)

        self.gradient_checkpointing = False

    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def fuse_qkv_projections(self):
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedJointAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @classmethod
    def from_transformer(
            cls,
            transformer,
            num_layers=12,
            num_extra_conditioning_channels=1,
            load_weights_from_transformer=True
    ):
        config = transformer.config
        config["num_layers"] = num_layers or config.num_layers
        config["extra_conditioning_channels"] = num_extra_conditioning_channels
        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)
            controlnet.pos_embed_input = zero_module(controlnet.pos_embed_input)

        return controlnet

    def forward(
            self,
            hidden_states: torch.Tensor,
            controlnet_cond: torch.Tensor,
            conditioning_scale: float = 1.0,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        if self.pos_embed is not None and hidden_states.ndim != 4:
            raise ValueError("hidden_states must be 4D when pos_embed is used")
        elif self.pos_embed is None and hidden_states.ndim != 3:
            raise ValueError("hidden_states must be 3D when pos_embed is not used")

        if self.context_embedder is not None and encoder_hidden_states is None:
            raise ValueError("encoder_hidden_states must be provided when context_embedder is used")
        elif self.context_embedder is None and encoder_hidden_states is not None:
            raise ValueError("encoder_hidden_states should not be provided when context_embedder is not used")

        if self.pos_embed is not None:
            hidden_states = self.pos_embed(hidden_states)

        temb = self.time_text_embed(timestep, pooled_projections)

        if self.context_embedder is not None:
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        hidden_states = hidden_states + self.pos_embed_input(controlnet_cond)

        block_res_samples = ()

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if self.context_embedder is not None:
                    encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                    )
                else:
                    hidden_states = self._gradient_checkpointing_func(block, hidden_states, temb)
            else:
                if self.context_embedder is not None:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb
                    )
                else:
                    hidden_states = block(hidden_states, temb)

            block_res_samples = block_res_samples + (hidden_states,)

        controlnet_block_res_samples = ()
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks):
            block_res_sample = controlnet_block(block_res_sample)
            controlnet_block_res_samples = controlnet_block_res_samples + (block_res_sample,)
        controlnet_block_res_samples = [sample * conditioning_scale for sample in controlnet_block_res_samples]

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_res_samples,)

        return SD3ControlNetOutput(controlnet_block_samples=controlnet_block_res_samples)