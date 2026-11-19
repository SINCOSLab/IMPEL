import os
import numpy as np
import torch
from typing import Union, List, Optional, Dict, Any, Callable
from diffusers import StableDiffusion3ControlNetPipeline

# Global counter for file naming (you may want to handle this differently)
global_count = 0


class RangeImageStableDiffusion3ControlNetPipeline(StableDiffusion3ControlNetPipeline):
    """Custom ControlNet Pipeline for generating and saving range image NPY files"""

    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            prompt_2: Optional[Union[str, List[str]]] = None,
            prompt_3: Optional[Union[str, List[str]]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 28,
            sigmas: Optional[List[float]] = None,
            guidance_scale: float = 7.0,
            clip_skip: Optional[int] = None,
            control_guidance_start: Union[float, List[float]] = 0.0,
            control_guidance_end: Union[float, List[float]] = 1.0,
            control_image=None,
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            controlnet_pooled_projections: Optional[torch.FloatTensor] = None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            negative_prompt_2: Optional[Union[str, List[str]]] = None,
            negative_prompt_3: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            ip_adapter_image=None,
            ip_adapter_image_embeds: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "npy",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            callback_on_step_end: Optional[Callable] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 256,
            # Range image specific parameters
            output_dir: Optional[str] = './controlnet_range_output',
            file_prefix: Optional[str] = "range_image",
            range_scale_factor: float = 40.0,  # Scale factor for range values
    ):
        """
        Generate range images using ControlNet conditioning.

        Args:
            All standard ControlNet parameters plus:
            output_dir: Directory to save NPY files
            file_prefix: Prefix for saved files
            range_scale_factor: Factor to scale final range values
        """
        global global_count

        # Call parent class __call__ method with output_type="pt" to get tensors
        result = super().__call__(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            clip_skip=clip_skip,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            control_image=control_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            controlnet_pooled_projections=controlnet_pooled_projections,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            output_type="pt",  # Force PyTorch tensor output
            return_dict=return_dict,
            joint_attention_kwargs=joint_attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        # Get generated tensors
        images = result.images  # Shape: [batch_size, 3, H, W]

        # Process and save as NPY files
        npy_files = []

        for i, image in enumerate(images):
            # Convert to single channel range image
            # Take mean across channels to get single channel
            range_image = image.mean(dim=0).detach().cpu().numpy()  # Shape: (H, W)

            # Apply scaling - adjust this based on your model's output characteristics
            # If your model outputs values in [0, 1], scale directly
            # If your model outputs values in [-1, 1], you might need: (range_image + 1) / 2.0 * range_scale_factor
            range_image = range_image * range_scale_factor



            # Save as NPY file if output_dir is specified
            # if output_dir and output_type == "npy":
            #     os.makedirs(output_dir, exist_ok=True)
            #     filename = f"{file_prefix}_{global_count:06d}.npy"
            #     filepath = os.path.join(output_dir, filename)
            #     np.save(filepath, range_image)
            #     npy_files.append(filepath)
            #     global_count += 1

        # Return results based on output_type
        if output_type == "npy":
            if return_dict:
                return {
                    "range_image": range_image,
                    "npy_file": npy_files,
                    "original_tensor": images
                }
            else:
                return npy_files
        elif output_type == "pt":
            # Return as PyTorch tensors
            range_tensors = torch.stack([torch.from_numpy(img) for img in range_images])
            if return_dict:
                return {
                    "range_image": range_tensors,
                    "original_tensor": images
                }
            else:
                return range_tensors
        else:
            # For other output types, return processed range images as numpy arrays
            if return_dict:
                return {
                    "range_image": range_images,
                    "original_tensor": images
                }
            else:
                return range_images

    @classmethod
    def from_pretrained_with_lora(
            cls,
            pretrained_model_name_or_path,
            controlnet,
            lora_model_path=None,
            lora_weight_name=None,
            **kwargs
    ):
        """
        Convenience method to create pipeline with LoRA weights loaded.

        Args:
            pretrained_model_name_or_path: Base model path
            controlnet: ControlNet model
            lora_model_path: Path to LoRA weights
            lora_weight_name: Specific LoRA weight file name
        """
        # Create pipeline
        pipeline = cls.from_pretrained(
            pretrained_model_name_or_path,
            controlnet=controlnet,
            **kwargs
        )

        # Load LoRA weights if specified
        if lora_model_path is not None:
            if lora_weight_name:
                pipeline.load_lora_weights(lora_model_path, weight_name=lora_weight_name)
            else:
                pipeline.load_lora_weights(lora_model_path)
            pipeline.fuse_lora()
            print(f"Loaded and fused LoRA weights from {lora_model_path}")

        return pipeline


# Example usage function
def create_range_controlnet_pipeline(
        base_model_path="stabilityai/stable-diffusion-3-medium-diffusers",
        controlnet_path="path/to/your/trained/controlnet",
        lora_path=None,
        device="cuda"
):
    """
    Create a range image ControlNet pipeline with optional LoRA weights.

    Args:
        base_model_path: Path to base SD3 model
        controlnet_path: Path to trained ControlNet
        lora_path: Optional path to LoRA weights
        device: Device to run on
    """
    from diffusers import SD3ControlNetModel

    # Load ControlNet
    controlnet = SD3ControlNetModel.from_pretrained(controlnet_path)

    # Create pipeline with LoRA
    pipeline = RangeImageStableDiffusion3ControlNetPipeline.from_pretrained_with_lora(
        base_model_path,
        controlnet=controlnet,
        lora_model_path=lora_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    pipeline = pipeline.to(device)
    return pipeline


# Usage example:
"""
# Create pipeline
pipeline = create_range_controlnet_pipeline(
    base_model_path="stabilityai/stable-diffusion-3-medium-diffusers",
    controlnet_path="./controlnet_output",
    lora_path="./data3/lhz/diffusers/examples/dreambooth/lora_npy_l1loss_tuneAgain_output/checkpoint-10000",
    device="cuda"
)

# Generate range image with control
from PIL import Image
control_image = Image.open("path/to/control/image.jpg")

result = pipeline(
    prompt="A detailed range map of a mountainous terrain",
    control_image=control_image,
    num_inference_steps=20,
    guidance_scale=7.0,
    controlnet_conditioning_scale=1.0,
    output_type="npy",
    output_dir="./generated_range_images",
    file_prefix="controlled_range",
    range_scale_factor=40.0
)
"""
