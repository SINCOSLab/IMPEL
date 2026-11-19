import argparse
import contextlib
import copy
import functools
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset, Features, Value, Array2D, Image
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
import torch.nn.functional as F
from torchvision.utils import save_image

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
)

from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.testing_utils import backend_empty_cache
from diffusers.utils.torch_utils import is_compiled_module
from RangeImageStableDiffusion3ControlNetPipeline import *
from model.LiDARControlNet import SD3ControlNetModel
from distance_config import MIN_DISTANCE, MAX_DISTANCE

if is_wandb_available():
    import wandb

    wandb.init(project="project", mode="offline")

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

height = np.array(
    [0.20966667, 0.2092, 0.2078, 0.2078, 0.2078,
     0.20733333, 0.20593333, 0.20546667, 0.20593333, 0.20546667,
     0.20453333, 0.205, 0.2036, 0.20406667, 0.2036,
     0.20313333, 0.20266667, 0.20266667, 0.20173333, 0.2008,
     0.2008, 0.2008, 0.20033333, 0.1994, 0.20033333,
     0.19986667, 0.1994, 0.1994, 0.19893333, 0.19846667,
     0.19846667, 0.19846667, 0.12566667, 0.1252, 0.1252,
     0.12473333, 0.12473333, 0.1238, 0.12333333, 0.1238,
     0.12286667, 0.1224, 0.12286667, 0.12146667, 0.12146667,
     0.121, 0.12053333, 0.12053333, 0.12053333, 0.12006667,
     0.12006667, 0.1196, 0.11913333, 0.11866667, 0.1182,
     0.1182, 0.1182, 0.11773333, 0.11726667, 0.11726667,
     0.1168, 0.11633333, 0.11633333, 0.1154])
zenith = np.array([
    0.03373091, 0.02740409, 0.02276443, 0.01517224, 0.01004049,
    0.00308099, -0.00155868, -0.00788549, -0.01407172, -0.02103122,
    -0.02609267, -0.032068, -0.03853542, -0.04451074, -0.05020488,
    -0.0565317, -0.06180405, -0.06876355, -0.07361411, -0.08008152,
    -0.08577566, -0.09168069, -0.09793721, -0.10398284, -0.11052055,
    -0.11656618, -0.12219002, -0.12725147, -0.13407038, -0.14067839,
    -0.14510716, -0.15213696, -0.1575499, -0.16711043, -0.17568678,
    -0.18278688, -0.19129293, -0.20247031, -0.21146846, -0.21934183,
    -0.22763699, -0.23536977, -0.24528179, -0.25477201, -0.26510582,
    -0.27326038, -0.28232882, -0.28893683, -0.30004392, -0.30953414,
    -0.31993824, -0.32816311, -0.33723155, -0.34447224, -0.352908,
    -0.36282001, -0.37216965, -0.38292524, -0.39164219, -0.39895318,
    -0.40703745, -0.41835542, -0.42777535, -0.43621111
])
incl = -zenith



def latent_edge_preserving_loss(pred, target):
    pred_diff_h = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    target_diff_h = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
    pred_diff_w = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    target_diff_w = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])

    loss_h = F.l1_loss(pred_diff_h, target_diff_h)
    loss_w = F.l1_loss(pred_diff_w, target_diff_w)

    return loss_h + loss_w


def pixel_edge_preserving_loss(pred_latents, target_latents, vae, vae_scaling_factor, vae_shift_factor):
    pred_scaled = (pred_latents / vae_scaling_factor) + vae_shift_factor
    target_scaled = (target_latents / vae_scaling_factor) + vae_shift_factor

    pred_images = vae.decode(pred_scaled.to(vae.dtype), return_dict=False)[0]
    target_images = vae.decode(target_scaled.to(vae.dtype), return_dict=False)[0]


    pred_gray = pred_images.mean(dim=1, keepdim=True).to(torch.float32)
    target_gray = target_images.mean(dim=1, keepdim=True).to(torch.float32)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=vae.device).view(1, 1, 3,
                                                                                                              3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=vae.device).view(1, 1, 3,
                                                                                                              3)
    autocast_ctx = torch.cuda.amp.autocast if hasattr(torch.cuda.amp, "autocast") else None
    if autocast_ctx is not None:
        ctx = autocast_ctx(enabled=False)
    else:
        from contextlib import nullcontext
        ctx = nullcontext()

    with ctx:
        pred_grad_x = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred_gray, sobel_y, padding=1)
        target_grad_x = F.conv2d(target_gray, sobel_x, padding=1)
        target_grad_y = F.conv2d(target_gray, sobel_y, padding=1)

        eps = 1e-6
        pred_edges = torch.sqrt(pred_grad_x * pred_grad_x + pred_grad_y * pred_grad_y + eps)
        target_edges = torch.sqrt(target_grad_x * target_grad_x + target_grad_y * target_grad_y + eps)

    return F.l1_loss(pred_edges, target_edges)


def depth_aware_downsample(range_image):

    img = range_image
    downsampled = F.adaptive_avg_pool2d(img, (64, 512))

    return downsampled.squeeze().squeeze()


def to_pc_torch_fov90(range_images):

    device = range_images.device
    B = range_images.shape[0]

    if range_images.ndim == 4 and range_images.shape[1] == 1:
        r = range_images.squeeze(1)  # [B, W, H]
    else:
        r = range_images  # [B, W, H]

    W, H = r.shape[1], r.shape[2]

    std = MAX_DISTANCE

    range_images = (range_images + 1) / 2

    point_range = range_images * std

    r_true = point_range

    r_true = r_true.permute(0, 2, 1)

    incl_t = torch.from_numpy(incl).to(device=device, dtype=torch.float32).view(1, H, 1)
    height_t = torch.from_numpy(height).to(device=device, dtype=torch.float32).view(1, H, 1)

    z = height_t - r_true * torch.sin(incl_t)  # [B, H, W]
    xy_norm = r_true * torch.cos(incl_t)  # [B, H, W]

    azi = torch.linspace(-np.pi / 4, np.pi / 4, W, device=device)  # [W]
    azi = azi.view(1, 1, W)  # [1,1,W]

    x = (xy_norm * torch.cos(azi)).reshape(B, -1)  # [B, H*W]
    y = (xy_norm * torch.sin(azi)).reshape(B, -1)  # [B, H*W]
    z = z.reshape(B, -1)  # [B, H*W]

    point_cloud = torch.stack([x, y, z], dim=2)  # [B, H*W, 3]

    return point_cloud


def to_pc_torch_fov90_mask(range_images, valid_mask=None):

    device = range_images.device
    B = range_images.shape[0]

    if range_images.ndim == 4 and range_images.shape[1] == 1:
        r = range_images.squeeze(1)  # [B, W, H]
    else:
        r = range_images  # [B, W, H]

    W, H = r.shape[1], r.shape[2]

    range_fill_value = 100.0

    std = MAX_DISTANCE


    range_images = (range_images + 1) / 2

    point_range = range_images * std

    r_true = point_range

    r_true = r_true.permute(0, 2, 1)

    incl_t = torch.from_numpy(incl).to(device=device, dtype=torch.float32).view(1, H, 1)
    height_t = torch.from_numpy(height).to(device=device, dtype=torch.float32).view(1, H, 1)

    z = height_t - r_true * torch.sin(incl_t)  # [B, H, W]
    xy_norm = r_true * torch.cos(incl_t)  # [B, H, W]

    azi = torch.linspace(-np.pi / 4, np.pi / 4, W, device=device)  # [W]
    azi = azi.view(1, 1, W)  # [1,1,W]

    x = (xy_norm * torch.cos(azi)).reshape(B, -1)  # [B, H*W]
    y = (xy_norm * torch.sin(azi)).reshape(B, -1)  # [B, H*W]
    z = z.reshape(B, -1)  # [B, H*W]

    point_cloud = torch.stack([x, y, z], dim=2)  # [B, H*W, 3]

    if valid_mask is not None:
        masked_point_clouds = []
        for b in range(B):
            batch_valid_mask = valid_mask[b]  # [H*W]
            batch_points = point_cloud[b]  # [H*W, 3]
            valid_points = batch_points[batch_valid_mask]  # [N_valid, 3]
            masked_point_clouds.append(valid_points)

        max_valid_points = max(pc.shape[0] for pc in masked_point_clouds)
        if max_valid_points == 0:
            return torch.zeros(B, 1, 3, device=device)

        padded_point_clouds = []
        for pc in masked_point_clouds:
            if pc.shape[0] < max_valid_points:
                if pc.shape[0] > 0:
                    padding = pc[-1:].repeat(max_valid_points - pc.shape[0], 1)

                    padded_pc = torch.cat([pc, padding], dim=0)
                else:
                    padded_pc = torch.zeros(max_valid_points, 3, device=device)
            else:
                padded_pc = pc
            padded_point_clouds.append(padded_pc)

        point_cloud = torch.stack(padded_point_clouds, dim=0)  # [B, max_valid_points, 3]

    return point_cloud

def create_valid_mask(range_images, threshold=-0.8):
    B, W, H = range_images.shape

    range_flat = range_images.permute(0, 2, 1).reshape(B, -1)

    valid_mask = range_flat > threshold
    return valid_mask

def log_validation(controlnet, args, accelerator, weight_dtype, step, is_final_validation=False):
    from PIL import Image
    logger.info("Running validation... ")

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
    else:
        controlnet = SD3ControlNetModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

    pipeline = RangeImageStableDiffusion3ControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=None,
        safety_checker=None,
        transformer=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(torch.device(accelerator.device))
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            validation_prompts,
            prompt_2=None,
            prompt_3=None,
        )

    del pipeline
    gc.collect()
    # backend_empty_cache(accelerator.device.type)

    pipeline = RangeImageStableDiffusion3ControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        safety_checker=None,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    ).to(accelerator.device)
    if args.lora_model_path is not None:
        pipeline.load_lora_weights(args.lora_model_path)
        pipeline.fuse_lora()

    # pipeline.enable_model_cpu_offload(device=accelerator.device.type)
    pipeline.set_progress_bar_config(disable=True)

    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast(accelerator.device.type)

    train_resize = transforms.Resize((128, 1024), interpolation=transforms.InterpolationMode.NEAREST)

    for i, validation_image in enumerate(validation_images):
        validation_image = Image.open(validation_image).convert("RGB")
        validation_image = train_resize(validation_image)

        validation_prompt = validation_prompts[i]

        images = []

        for _ in range(args.num_validation_images):
            with inference_ctx:
                image = pipeline(
                    prompt_embeds=prompt_embeds[i].unsqueeze(0),
                    negative_prompt_embeds=negative_prompt_embeds[i].unsqueeze(0),
                    pooled_prompt_embeds=pooled_prompt_embeds[i].unsqueeze(0),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[i].unsqueeze(0),
                    control_image=validation_image,
                    num_inference_steps=30,
                    generator=generator,
                    height=128,
                    width=1024
                )['range_image']

            range_image = torch.from_numpy(image).float()
            range_image = range_image.unsqueeze(0)
            gray_recon = range_image.permute(0, 2, 1)
            range_images = ((gray_recon / 40)).clamp(0, 1)  #
            image = range_images.permute(0, 2, 1)
            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                tracker.writer.add_image(
                    "RGB conditioning", np.asarray([validation_image]), step, dataformats="NHWC"
                )

                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="RGB conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    free_memory()

    if not is_final_validation:
        controlnet.to(accelerator.device)

    return image_logs


# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# SD3 controlnet-{repo_id} for Range Image Generation

These are controlnet weights trained on {base_model} with new type of conditioning.

"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "sd3",
        "sd3-diffusers",
        "controlnet",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help=""
    )
    parser.add_argument(
        "--num_extra_conditioning_channels",
        type=int,
        default=0,
        help="Number of extra conditioning channels for controlnet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(""
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--upcast_vae",
        action="store_true",
        help="Whether or not to upcast vae to fp32",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
             "model `target` is calculated.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(""
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(""
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            ""
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(""
        ),
    )
    parser.add_argument(
        "--rgb_column", type=str, default="conditioning_image",
        help="The column of the dataset containing the RGB conditioning image."
    )
    parser.add_argument(
        "--range_column", type=str, default="range_image",
        help="The column of the dataset containing the target range image (NPY file)."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--dataset_preprocess_batch_size", type=int, default=1000, help="Batch size for preprocessing dataset."
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
        ),
    )
    parser.add_argument(
        "--lora_model_path",
        type=str,
        default=None,
        help="Path to the LoRA model directory or checkpoint to load into the transformer before training.",
    )
    parser.add_argument(
        "--enable_edge_loss",
        action="store_true",
        help="Enable edge loss for decoded range image"
    )
    parser.add_argument(
        "--enable_3d_loss",
        action="store_true",
        help="Enable 3D loss for low noise samples"
    )
    parser.add_argument(
        "--low_noise_threshold",
        type=float,
        default=0.1,
        help="Sigma threshold for low noise samples"
    )
    parser.add_argument(
        "--point_alignment_loss_interval",
        type=int,
        default=100,
        help="Compute 3D loss every N global steps"
    )
    parser.add_argument(
        "--point_alignment_start_step",
        type=int,
        default=1000,
        help="Compute 3D loss every N global steps"
    )
    parser.add_argument(
        "--mask_noise_points",
        default=True,
        help="Enable masking of noise points (close to -1) in 3D loss calculation"
    )
    parser.add_argument(
        "--noise_threshold",
        type=float,
        default=-0.9,
        help="Threshold for masking noise points in range image (default: -0.8)"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--train_data_dir`")

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
            args.validation_image is not None
            and args.validation_prompt is not None
            and len(args.validation_image) != 1
            and len(args.validation_prompt) != 1
            and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def make_train_dataset(args, tokenizer_one, tokenizer_two, tokenizer_three, accelerator):
    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    features = Features({
        "range_image_file_name": Value("string"),
        "conditioning_rgb": Image(),
        "text": Value("string"),
    })

    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                "imagefolder",
                data_dir=args.train_data_dir,
                cache_dir=args.cache_dir,
                features=features
            )


    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names
    print(f"Original dataset columns: {dataset['train'].column_names}")

    # print(column_names) # ['range_image', 'conditioning_rgb', 'text']

    # 6. Get the column names for input/target.
    if args.rgb_column is None:
        conditioning_image_column = "conditioning_rgb"  # Default based on your metadata format
        logger.info(f"RGB column defaulting to {conditioning_image_column}")
    else: 
        conditioning_image_column = args.rgb_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--rgb_column` value '{args.rgb_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.range_column is None:
        range_column = "range_image"  # Default based on your metadata format
        logger.info(f"range image column defaulting to {range_column}")
    else:
        range_column = args.range_column
        if range_column not in column_names:
            raise ValueError(
                f"`--range_column` value '{args.range_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def process_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        return captions

    rgb_image_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    train_resize = transforms.Resize((128, 1024), interpolation=transforms.InterpolationMode.NEAREST)

    def preprocess_train(examples):
        import torch.nn.functional as F
        from PIL import Image

        rgb_images = []
        for rgb_path in examples[conditioning_image_column]:
            if isinstance(rgb_path, str):
                rgb_image = Image.open(os.path.join(args.train_data_dir, rgb_path)).convert("RGB")
                rgb_image = train_resize(rgb_image)
                rgb_images.append(rgb_image_transforms(rgb_image))
            else:

                rgb_image = rgb_path.convert("RGB")
                rgb_image = train_resize(rgb_image)
                rgb_images.append(rgb_image_transforms(rgb_image))

        range_images = []
        for range_path in examples[range_column]:
            if isinstance(range_path, str):
                # Load NPY file directly
                range_array = np.load(os.path.join(args.train_data_dir, range_path))
                # range_array = np.fliplr(range_array).copy()
                # Convert numpy array to tensor and process similar to your SD3 code
                range_tensor = torch.from_numpy(range_array).float().unsqueeze(0)

                # Resize to target resolution using interpolation
                range_tensor = F.interpolate(range_tensor.unsqueeze(0), size=(128, 1024), mode='nearest')
                range_tensor = range_tensor.squeeze(0)

                # Convert to 3-channel (RGB-like) for VAE compatibility
                range_tensor = range_tensor.repeat(3, 1, 1)

                # Apply normalization to [-1, 1] range
                range_tensor = transforms.Normalize([0.5], [0.5])(range_tensor)

                range_images.append(range_tensor)
            else:
                # Handle other cases if needed
                range_images.append(range_path)

        examples["pixel_values"] = range_images
        examples["conditioning_pixel_values"] = rgb_images
        examples["prompts"] = process_captions(examples)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(
            preprocess_train)  

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
    pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


# Copied from dreambooth sd3 example
def _encode_prompt_with_t5(
        text_encoder,
        tokenizer,
        max_sequence_length,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# Copied from dreambooth sd3 example
def _encode_prompt_with_clip(
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


# Copied from dreambooth sd3 example
def encode_prompt(
        text_encoders,
        tokenizers,
        prompt: str,
        max_sequence_length,
        device=None,
        num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            ""
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":

        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    print('Whether use enable_edge_loss:', args.enable_edge_loss)
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )
    if args.lora_model_path is not None: 
        logger.info(f"Loading LoRA weights from {args.lora_model_path}")
        try:
            # Create a temporary pipeline to load LoRA weights
            temp_pipeline = StableDiffusion3Pipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                transformer=transformer,
                torch_dtype=torch.float32,  # Use float32 for loading
                revision=args.revision,
                variant=args.variant,
            )

            # Load LoRA weights into the pipeline
            temp_pipeline.load_lora_weights(args.lora_model_path)

            # Fuse the LoRA weights into the transformer
            temp_pipeline.fuse_lora()

            # Extract the modified transformer
            transformer = temp_pipeline.transformer

            # Clean up
            del temp_pipeline
            gc.collect()

            logger.info("Successfully loaded and fused LoRA weights into transformer")

        except Exception as e:
            logger.error(f"Failed to load LoRA weights: {e}")
            raise e

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from transformer")
        controlnet = SD3ControlNetModel.from_transformer(
            transformer, num_extra_conditioning_channels=args.num_extra_conditioning_channels
        )
    print('isinstance---------------:', isinstance(controlnet, SD3ControlNetModel))

    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    controlnet.train()

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model


    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = SD3ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    if args.upcast_vae:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    train_dataset = make_train_dataset(args, tokenizer_one, tokenizer_two, tokenizer_three, accelerator)

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    def compute_text_embeddings(batch, text_encoders, tokenizers):
        with torch.no_grad():
            prompt = batch["prompts"]
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

    compute_embeddings_fn = functools.partial(
        compute_text_embeddings,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
    )
    with accelerator.main_process_first():
        from datasets.fingerprint import Hasher

        new_fingerprint = Hasher.hash(args)
        train_dataset = train_dataset.map(
            compute_embeddings_fn,
            batched=True,
            batch_size=args.dataset_preprocess_batch_size,
            new_fingerprint=new_fingerprint,
        )

    del text_encoder_one, text_encoder_two, text_encoder_three
    del tokenizer_one, tokenizer_two, tokenizer_three
    free_memory()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print('num_update_steps_per_epoch:', num_update_steps_per_epoch)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    print('num_train_epochs:', args.num_train_epochs)
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor

    loss_accumulator = []
    pixel_loss_accumulator = []
    edge_loss_accumulator = []
    point_reward_accumulator = []

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_loss = []
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)

                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)


                if (args.enable_3d_loss and
                        global_step % args.point_alignment_loss_interval == 0 and global_step > args.point_alignment_start_step and
                        global_step > 0):
                    num_low_noise = bsz
                    low_noise_sigmas = torch.rand(num_low_noise, device=sigmas.device,
                                                  dtype=sigmas.dtype) * args.low_noise_threshold

                    sigmas = low_noise_sigmas.view_as(sigmas)

                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Get the text embedding for conditioning
                prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=weight_dtype)

                # controlnet(s) inference
                rgb_conditioning_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)


                rgb_conditioning_latents = vae.encode(rgb_conditioning_image).latent_dist.sample()
                rgb_conditioning_latents = rgb_conditioning_latents * vae.config.scaling_factor


                control_block_res_samples = controlnet(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    controlnet_cond=rgb_conditioning_latents,
                    return_dict=False,
                )[0]
                control_block_res_samples = [sample.to(dtype=weight_dtype) for sample in
                                             control_block_res_samples]  

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_res_samples,
                    return_dict=False,
                )[0]  


                # Preconditioning of the model outputs.
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                
                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()  
                epoch_loss.append(loss.item())

                pixel_loss = F.l1_loss(model_pred.float(), model_input.detach().float(), reduction='mean')
                loss += pixel_loss


                # ------------ pixel edge loss ------------
                if args.enable_edge_loss:
                    pixel_edge_loss = pixel_edge_preserving_loss(model_pred,model_input,vae,vae_config_scaling_factor,vae_config_shift_factor)
                    loss += 0.05*pixel_edge_loss.mean()
                # ------------ pixel edge loss ------------


                # -----------3d loss-------------#
                if (args.enable_3d_loss and global_step % args.point_alignment_loss_interval == 0 and global_step > args.point_alignment_start_step and global_step > 0):
                    def get_3d_loss_weight(global_step, start_step=1000, warmup_steps=2000):
                        if global_step < start_step:
                            return 0.0
                        if global_step < start_step + warmup_steps:

                            progress = (global_step - start_step) / warmup_steps
                            return 0.0005 * progress 

                        return 0.0005

                    with torch.no_grad():
                        original_scaled = (model_input / vae_config_scaling_factor) + vae_config_shift_factor
                        original_range_images = vae.decode(original_scaled.to(vae.dtype), return_dict=False)[0]
                    reconstructed_scaled = (model_pred / vae_config_scaling_factor) + vae_config_shift_factor
                    reconstructed_range_images = vae.decode(reconstructed_scaled.to(vae.dtype), return_dict=False)[0]

                    gray_recon = reconstructed_range_images.mean(dim=1)  # [B,128,1024]
                    gray_raw = original_range_images.mean(dim=1)  # [B,128,1024]
                    gray_recon = depth_aware_downsample(gray_recon)  # [B,64,512]
                    gray_raw = depth_aware_downsample(gray_raw)  # [B,64,512]
                    gray_raw = gray_raw.permute(0, 2, 1)  # [B,512,64]
                    gray_recon = gray_recon.permute(0, 2, 1)  # [B,512,64]

                    if args.mask_noise_points:

                        valid_mask = create_valid_mask(gray_raw, threshold=args.noise_threshold)
                        pred_pcs = to_pc_torch_fov90_mask(gray_recon, valid_mask=valid_mask)
                        raw_pcs = to_pc_torch_fov90_mask(gray_raw, valid_mask=valid_mask)

                        if global_step % 1000 == 0:
                            valid_points_count = valid_mask.sum(dim=1).float().mean()
                            total_points = valid_mask.shape[1]
                            logger.info(
                                f"Valid points ratio: {valid_points_count / total_points:.3f} ({valid_points_count:.0f}/{total_points})")
                    else:
                        pred_pcs = to_pc_torch_fov90(gray_recon)
                        raw_pcs = to_pc_torch_fov90(gray_raw)
                    struct_loss = F.l1_loss(pred_pcs, raw_pcs)
                    weight_3d = get_3d_loss_weight(global_step, start_step=args.point_alignment_start_step)
                    loss += weight_3d * struct_loss

                    torch.cuda.empty_cache()

                # --------------------- latent edge loss------------------------------- #
                latent_edge_loss = latent_edge_preserving_loss(model_pred,model_input)
                loss += 0.1 * latent_edge_loss.mean()
                # --------------------- latent edge loss------------------------------- #

                loss_accumulator.append(loss.detach().item())
                pixel_loss_accumulator.append(pixel_loss.detach().item())

                if args.enable_edge_loss:
                    edge_loss_accumulator.append(pixel_edge_loss.detach().item())
                else:
                    edge_loss_accumulator.append(0.0)
                if (
                        args.enable_3d_loss and global_step % args.point_alignment_loss_interval == 0 and global_step > args.point_alignment_start_step and global_step > 0):
                    point_reward_accumulator.append(struct_loss.detach().item())
                else:
                    point_reward_accumulator.append(0.0)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % 1000 == 0 and len(loss_accumulator) > 0:
                    avg_loss = sum(loss_accumulator) / len(loss_accumulator)
                    avg_pixel_loss = sum(pixel_loss_accumulator) / len(pixel_loss_accumulator)
                    avg_edge_loss = sum(edge_loss_accumulator) / len(edge_loss_accumulator)
                    avg_point_loss = sum(point_reward_accumulator) / len(point_reward_accumulator)

                    logger.info(f"Step {global_step} - Average Losses (last 1000 steps):")
                    logger.info(f"  Total Loss: {avg_loss:.6f}")
                    logger.info(f"  Pixel Loss: {avg_pixel_loss:.6f}")
                    logger.info(f"  Edge Loss: {avg_edge_loss:.6f}")
                    logger.info(f"  3D Loss: {avg_point_loss:.6f}")

                    loss_accumulator.clear()
                    pixel_loss_accumulator.clear()
                    edge_loss_accumulator.clear()
                    point_reward_accumulator.clear()

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        print('ready to validate')
                        image_logs = log_validation(
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if accelerator.is_main_process:
            print(f'epoch:{epoch} loss is {np.mean(epoch_loss)}')

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                controlnet=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
