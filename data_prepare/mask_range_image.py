import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from torchvision.utils import save_image
import torch
import glob
from tqdm import tqdm
from PIL import Image

def mask_range_image_custom_angular_range(range_image, start_angle_deg=0, end_angle_deg=15,
                                          total_fov_deg=90, mask_value=0):

    range_image = np.array(range_image)
    height, width = range_image.shape

    degrees_per_pixel = total_fov_deg / width

    center_col = width // 2

    start_pixels_offset = int(start_angle_deg / degrees_per_pixel)
    end_pixels_offset = int(end_angle_deg / degrees_per_pixel)

    start_col = center_col + start_pixels_offset
    end_col = center_col + end_pixels_offset

    start_col = max(0, start_col)
    end_col = min(width, end_col)

    mask = np.zeros((height, width), dtype=bool)
    if start_col < end_col:
        mask[:, start_col:end_col] = True

    masked_range_image = range_image.copy()
    masked_range_image[mask] = mask_value

    return masked_range_image, mask

if __name__ == "__main__":

    os.makedirs('/inpainting_mask', exist_ok=True)
    npy_files = sorted(glob.glob(os.path.join('/range_image_data_path', "*.npy")))
    for idx, img_path in tqdm(enumerate(npy_files)):
        file_name = os.path.basename(img_path).split(".")[0]
        img = np.load(img_path)
        range_gray = Image.fromarray((img * 255.).astype(np.uint8), mode='L')
        range_gray.save('/inpainting_mask/'+f'{file_name}.png')

        masked_image, mask = mask_range_image_custom_angular_range(img, start_angle_deg=-25, end_angle_deg=-10)
        range_images = torch.from_numpy(masked_image).float()
        save_image(range_images, '/inpainting_mask/'+f'{file_name}_mask.png')
        np.save('/inpainting_mask/'+f'{file_name}_mask.npy', mask.astype(np.uint8))
