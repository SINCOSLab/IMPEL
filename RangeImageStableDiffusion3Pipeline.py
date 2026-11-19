from diffusers import StableDiffusion3Pipeline
import torch
import numpy as np
from PIL import Image
import os
from typing import Optional, Union, List, Callable, Dict, Any
from distance_config import MAX_DISTANCE
global_count = 0
class RangeImageStableDiffusion3Pipeline(StableDiffusion3Pipeline):

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
            ip_adapter_image_embeds: Optional[torch.Tensor] = None,
            output_type: Optional[str] = "npy",
            return_dict: bool = True,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: Optional[int] = None,
            callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            max_sequence_length: int = 256,
            skip_guidance_layers: List[int] = None,
            skip_layer_guidance_scale: float = 2.8,
            skip_layer_guidance_stop: float = 0.2,
            skip_layer_guidance_start: float = 0.01,
            mu: Optional[float] = None,
            output_dir: Optional[str] = '/output_path',
            file_prefix: Optional[str] = "range_image",
    ):
        global global_count
        result = super().__call__(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type="pt",
            return_dict=return_dict,
        )


        images = result.images 

        npy_files = []
        for i, image in enumerate(images):

            range_image = image.mean(dim=0).detach().cpu().numpy()

            range_image = range_image * MAX_DISTANCE

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = f"{file_prefix}_{global_count:06d}.npy"
                filepath = os.path.join(output_dir, filename)
                np.save(filepath, range_image)
                npy_files.append(filepath)
        global_count += 1
        if return_dict:
            return {"range_image": range_image, "images": images}
        else:
            return npy_files