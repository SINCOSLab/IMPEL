import numpy as np
import time
from PIL import Image
import torch
import os
import glob
from tqdm import tqdm
from torchvision.utils import save_image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from distance_config import MIN_DISTANCE, MAX_DISTANCE


def save_range_image(range_image, fname):
    range_npy = range_image[:, :, 0].clip(0, MAX_DISTANCE) / MAX_DISTANCE 
    np.save(fname, range_npy)
    range_npy = torch.from_numpy(range_npy).to(dtype=torch.float32)
    png_name = os.path.basename(fname).split('.')[0] + '.png'
    save_image(range_npy, os.path.join("/your_path", png_name))


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
log = False
inverse = False
range_fill_value = np.array([100, 0])


def fill_noise(data, miss_inds, width):
    # data_shift1pxl = data[:, list(range(1, width)) + [0, ], :]  # 360 degree
    data_shift1pxl = data[:, list(range(1, width)) + [width - 1, ], :] # 90 degree
    data[miss_inds, :] = data_shift1pxl[miss_inds, :]
    return data


def process_miss_value(range_image, fill_final=True):
    range_image_mask = range_image[..., 0] > 0
    height, width, _ = range_image.shape
    miss_inds = range_image[:, :, 0] == -1

    range_image = fill_noise(range_image, miss_inds, width)
    range_image_mask = fill_noise(range_image_mask[:, :, None], miss_inds, width).squeeze()

    still_miss_inds = range_image[:, :, 0] == -1

    shift_down_2px = range_image[[height - 2, height - 1] + list(range(height - 2)), :, 0]
    shift_top_2px = range_image[list(range(2, height)) + [0, 1], :, 0]
    shift_right_2px = range_image[:, [width - 2, width - 1] + list(range(width - 2)), 0]
    # shift_left_2px = range_image[:, list(range(2, width)) + [0, 1], 0]  # 360°
    shift_left_2px = range_image[:, list(range(2, width)) + [1, 1], 0] # 90°

    car_window_mask = still_miss_inds & ((shift_down_2px != -1) | (shift_top_2px != -1) |
                                         (shift_right_2px != -1) | (shift_left_2px != -1))

    if fill_final:
        range_image[still_miss_inds, :] = range_fill_value[0]

    return range_image, range_image_mask, car_window_mask

def get_range_image_fov90(pc, incl, height):

    xy_norm = np.linalg.norm(pc[:, :2], axis=1)
    error_list = []
    for i in range(len(incl)):
        theta = incl[i]
        h = height[i]
        error = np.abs(theta - np.arctan2(h - pc[:, 2], xy_norm))
        error_list.append(error)
    all_error = np.stack(error_list, axis=-1)
    row_inds = np.argmin(all_error, axis=-1)  # N,

    azi = np.arctan2(pc[:, 1], pc[:, 0])  # range [-π, π)
    fov_mask = np.logical_and(azi >= -np.pi / 4, azi <= np.pi / 4)

    pc = pc[fov_mask]
    row_inds = row_inds[fov_mask]


    new_width = 512

    col = (azi[fov_mask] + np.pi / 4) / (np.pi / 2) * new_width

    col_inds = np.round(col).astype(int)

    col_inds[col_inds < 0] = 0
    col_inds[col_inds >= new_width] = new_width - 1


    H = len(incl)  # 64
    range_image = np.full((H, new_width, 2), -1, dtype=np.float32)
    point_range = np.linalg.norm(pc[:, :3], axis=1)
    remission = pc[:, 3]

    order = np.argsort(-point_range)
    row_o = row_inds[order]
    col_o = - col_inds[order]
    rng_o = point_range[order]
    rem_o = remission[order]

    # Final fill step
    range_image[row_o, col_o, 0] = rng_o
    range_image[row_o, col_o, 1] = rem_o

    return range_image


if __name__ == '__main__':

    input_folder = "/KITTI360_path"

    output_npy_folder = "/range_image_store_path"
    os.makedirs(output_npy_folder, exist_ok=True)
    bin_files = sorted(glob.glob(os.path.join(input_folder, "*.bin")))
    if len(bin_files) == 0:
        print(f"Can not found .bin file in {input_folder}")
        exit(1)
    outlier_z_threshold = -5.
    max_loss_point = -999
    for bin_path in tqdm(bin_files):
        fname = os.path.os.path.splitext(bin_path)[0]
        file_id = os.path.basename(fname)
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        mask = (points[:, 2] > outlier_z_threshold)

        points = points[mask, :]
        points = points.astype(np.float32)

        depth = np.linalg.norm(points[:, :3], 2, axis=1)
        distance_mask = np.logical_and(depth >= MIN_DISTANCE, depth <= MAX_DISTANCE)
        points = points[distance_mask, :]

        range_image = get_range_image_fov90(points, incl, height)
        range_image, range_image_mask, car_window_mask = process_miss_value(range_image, False)

        npy_output_path = os.path.join(output_npy_folder, file_id + '.npy')
        save_range_image(range_image, npy_output_path)
