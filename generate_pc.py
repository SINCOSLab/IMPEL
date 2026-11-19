from diffusers import DiffusionPipeline,StableDiffusion3Pipeline
import os, shutil
import numpy as np
import torch
import torchvision.transforms as transforms
from diffusers import AutoencoderKL
from RangeImageStableDiffusion3Pipeline import *
from accelerate import Accelerator
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image
from distance_config import MIN_DISTANCE, MAX_DISTANCE

# KITTI scanning parameters, obtained from Hough transformationRange
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


def to_pc_torch_fov90(range_images):

    incl_t = torch.from_numpy(incl).to(range_images.device, dtype=torch.float32)
    height_t = torch.from_numpy(height).to(range_images.device, dtype=torch.float32)

    batch_size, width_dim, height_dim = range_images.shape

    range_fill_value = 100.0

    point_range = range_images

    r_true = torch.where(point_range < 0, range_fill_value, point_range)

    r_true = r_true.permute(0, 2, 1)

    height_t = height_t.view(1, height_dim, 1)  # [1, 64, 1]

    incl_t = incl_t.view(1, height_dim, 1)  # [1, 64, 1]

    z = height_t - r_true * torch.sin(incl_t)

    xy_norm = r_true * torch.cos(incl_t)

    azi = torch.linspace(np.pi / 4, -np.pi / 4, width_dim, device=range_images.device)
    azi = azi.view(1, 1, width_dim)  # [1, 1, 512]

    x = xy_norm * torch.cos(azi)
    y = xy_norm * torch.sin(azi)

    x_flat = x.reshape(batch_size, -1)
    y_flat = y.reshape(batch_size, -1)
    z_flat = z.reshape(batch_size, -1)

    point_cloud = []
    depth_flat = r_true.reshape(batch_size, -1)
    for b in range(batch_size):
        valid = (depth_flat[b] >= MIN_DISTANCE) &  (depth_flat[b] <= MAX_DISTANCE)
        idx = torch.nonzero(valid).squeeze(1)
        pc = torch.stack([
            x_flat[b, idx],
            y_flat[b, idx],
            z_flat[b, idx]
        ], dim=1)  # [N_valid, 3]
        point_cloud.append(pc)
    # ------------------------------ -#

    return point_cloud

def depth_aware_downsample(range_image):
    img = range_image
    downsampled = F.adaptive_avg_pool2d(img, (64, 256))

    return downsampled.squeeze().squeeze()

accelerator = Accelerator(mixed_precision='fp16')

output_dir = '/output_path'
os.makedirs(output_dir, exist_ok=True)
num_total = 2000

pipe = RangeImageStableDiffusion3Pipeline.from_pretrained("/path_stable-diffusion-3-medium-diffusers").to(accelerator.device)
pipe.load_lora_weights("./weights/pytorch_lora_weights.safetensors")

image_list = []
transform = transforms.ToTensor()
pipe = accelerator.prepare(pipe)

with torch.no_grad():

    for i in tqdm(range(num_total)):
        range_image = pipe("A range image", num_inference_steps=30,guidance_scale=7.0, height=256,width=1024)['range_image']
        range_image = torch.from_numpy(range_image).float()
        range_image = range_image.unsqueeze(0)
        down_recon = depth_aware_downsample(range_image)
        gray_recon = down_recon.unsqueeze(0)
        gray_recon = gray_recon.permute(0, 2, 1)
        pred_pcs = to_pc_torch_fov90(gray_recon)
        pc = pred_pcs[0].cpu().detach().numpy()
        pc.tofile(os.path.join(output_dir, f"{i:06d}.bin"))

        range_images = ((gray_recon / MAX_DISTANCE)).clamp(0, 1)
        range_images = range_images.permute(0, 2, 1)
        save_image(range_images,os.path.join(output_dir, f"{i:06d}.png"))