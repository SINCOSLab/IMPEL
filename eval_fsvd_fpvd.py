import os
import glob
import math
import argparse

import numpy as np
import torch
import yaml
from tqdm import tqdm
from scipy import linalg

# ========== Print format ==========

OUTPUT_TEMPLATE = 50 * '-' + '\n|' + 16 * ' ' + '{}:{:.4E}' + 17 * ' ' + '|\n' + 50 * '-'

# ========== Constants and configs (from lidm/eval/__init__.py and metric_utils.py) ==========

VOXEL_SIZE = 0.05
NUM_SECTORS = 16
AGG_TYPE = 'depth'
TYPE2DATASET = {'32': 'nuscenes', '64': 'kitti'}

# Voxel/BEV range (same as original code)
DATA_CONFIG = {
    '64': {'x': [-50, 50], 'y': [-50, 50], 'z': [-3, 1]},
    '32': {'x': [-30, 30], 'y': [-30, 30], 'z': [-3, 6]},
}

# Range image parameters for each dataset
DATASET_CONFIG = {
    'kitti': {'size': [64, 1024], 'fov': [3, -25], 'depth_range': [1.0, 56.0], 'depth_scale': 6},
    'nuscenes': {'size': [32, 1024], 'fov': [10, -30], 'depth_range': [1.0, 45.0]},
}

MODALITY2MODEL = {
    'range': 'rangenet',
    'voxel': 'minkowskinet',
    'point_voxel': 'spvcnn',
}

MODAL2BATCHSIZE = {
    'range': 100,
    'voxel': 50,
    'point_voxel': 25,
}

# Root directory of pretrained weights (pretrained_weights/ next to this script)
_DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrained_weights')


# ========== Frechet distance (from fid_score.py) ==========

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute Fréchet Distance between two multivariate Gaussians (FID formula).
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'mean vectors have different dimensions'
    assert sigma1.shape == sigma2.shape, 'covariance matrices have different dimensions'

    diff = mu1 - mu2
    # Numerical stabilization in case the product is nearly singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print('Product of covariance matrices is close to singular; adding eps to the diagonal')
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def compute_fd(reference_logits, sample_logits):
    """
    Frechet distance on feature logits.
    """
    mu1, mu2 = np.mean(reference_logits, axis=0), np.mean(sample_logits, axis=0)
    sigma1, sigma2 = np.cov(reference_logits, rowvar=False), np.cov(sample_logits, rowvar=False)
    distance = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    return distance


from utils.misc_utils import dict2namespace
from modules.rangenet.model import Model as rangenet
from modules.spvcnn.model import Model as spvcnn
from modules.minkowskinet.model import Model as minkowskinet


def build_model(dataset_name, model_name, device='cuda'):
    """
    Load pretrained model from pretrained_weights/{dataset_name}/{model_name}.
    """
    model_folder = os.path.join(_DEFAULT_ROOT, dataset_name, model_name)
    if not os.path.isdir(model_folder):
        raise RuntimeError(f'Pretrained weights not found: {model_folder}')

    config_path = os.path.join(model_folder, 'config.yaml')
    if not os.path.isfile(config_path):
        raise RuntimeError(f'config.yaml not found: {config_path}')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if model_name != 'rangenet':
        config = dict2namespace(config)

    # Build model
    model_cls = eval(model_name)
    model = model_cls(config)

    # Load checkpoint
    if model_name == 'rangenet':
        model.load_pretrained_weights(model_folder)
    else:
        ckpt_path = os.path.join(model_folder, 'model.ckpt')
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f'model.ckpt not found: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)

    model.to(device)
    model.eval()
    return model


# ========== Point cloud preprocessing and logits computation (simplified from metric_utils.py) ==========

from itertools import repeat
from typing import List, Tuple, Union
from torchsparse import SparseTensor, PointTensor
from torchsparse.utils.collate import sparse_collate_fn


def ravel_hash(x: np.ndarray) -> np.ndarray:
    assert x.ndim == 2, x.shape
    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
        h += x[:, k]
        h *= xmax[k + 1]
    h += x[:, -1]
    return h


def sparse_quantize(coords, voxel_size: Union[float, Tuple[float, ...]] = 1,
                    *, return_index: bool = False, return_inverse: bool = False) -> List[np.ndarray]:
    """
    Simplified voxel quantization, based on the original metric_utils.py.
    """
    if isinstance(voxel_size, (float, int)):
        voxel_size = tuple(repeat(voxel_size, coords.shape[1]))
    assert isinstance(voxel_size, tuple) and len(voxel_size) in [2, 3]

    voxel_size = np.array(voxel_size)
    coords = np.floor(coords / voxel_size).astype(np.int32)

    _, indices, inverse_indices = np.unique(
        ravel_hash(coords), return_index=True, return_inverse=True
    )
    coords = coords[indices]

    outputs = [coords]
    if return_index:
        outputs += [indices]
    if return_inverse:
        outputs += [inverse_indices]
    return outputs[0] if len(outputs) == 1 else outputs


def pcd2range(pcd, size, fov, depth_range, remission=None, labels=None, **kwargs):
    # LiDAR parameters
    fov_up = fov[0] / 180.0 * np.pi
    fov_down = fov[1] / 180.0 * np.pi
    fov_range = abs(fov_down) + abs(fov_up)

    depth = np.linalg.norm(pcd, 2, axis=1)

    # Valid range mask
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    depth, pcd = depth[mask], pcd[mask]

    scan_x, scan_y, scan_z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    proj_x = 0.5 * (yaw / np.pi + 1.0)
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov_range

    proj_x *= size[1]
    proj_y *= size[0]

    proj_x = np.maximum(0, np.minimum(size[1] - 1, np.floor(proj_x))).astype(np.int32)
    proj_y = np.maximum(0, np.minimum(size[0] - 1, np.floor(proj_y))).astype(np.int32)

    order = np.argsort(depth)[::-1]
    proj_x, proj_y = proj_x[order], proj_y[order]

    depth = depth[order]
    proj_range = np.full(size, -1, dtype=np.float32)
    proj_range[proj_y, proj_x] = depth

    if remission is not None:
        remission = remission[mask][order]
        proj_feature = np.full(size, -1, dtype=np.float32)
        proj_feature[proj_y, proj_x] = remission
    elif labels is not None:
        labels = labels[mask][order]
        proj_feature = np.full(size, 0, dtype=np.float32)
        proj_feature[proj_y, proj_x] = labels
    else:
        proj_feature = None

    return proj_range, proj_feature


def range2xyz(range_img, fov, depth_range, depth_scale, log_scale=True, **kwargs):
    size = range_img.shape
    fov_up = fov[0] / 180.0 * np.pi
    fov_down = fov[1] / 180.0 * np.pi
    fov_range = abs(fov_down) + abs(fov_up)

    if log_scale:
        depth = (np.exp2(range_img * depth_scale) - 1)
    else:
        depth = range_img

    scan_x, scan_y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    scan_x = scan_x.astype(np.float64) / size[1]
    scan_y = scan_y.astype(np.float64) / size[0]

    yaw = np.pi * (scan_x * 2 - 1)
    pitch = (1.0 - scan_y) * fov_range - abs(fov_down)

    xyz = -np.ones((3, *size))
    xyz[0] = np.cos(yaw) * np.cos(pitch) * depth
    xyz[1] = -np.sin(yaw) * np.cos(pitch) * depth
    xyz[2] = np.sin(pitch) * depth

    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    xyz[:, ~mask] = -1

    return xyz


def pcd2voxel(pcd):
    pcd_voxel = np.round(pcd / VOXEL_SIZE)
    pcd_voxel = pcd_voxel - pcd_voxel.min(0, keepdims=1)
    feat = np.concatenate((pcd, -np.ones((pcd.shape[0], 1))), axis=1)  # -1 is a placeholder for remission
    _, inds, _ = sparse_quantize(pcd_voxel, 1, return_index=True, return_inverse=True)

    feat = torch.FloatTensor(feat[inds])
    pcd_voxel = torch.LongTensor(pcd_voxel[inds])
    lidar = SparseTensor(feat, pcd_voxel)
    output = {'lidar': lidar}
    return output


def preprocess_pcd(pcd, **kwargs):
    depth = np.linalg.norm(pcd, 2, axis=1)
    mask = np.logical_and(depth > kwargs['depth_range'][0], depth < kwargs['depth_range'][1])
    pcd = pcd[mask]
    return pcd


def preprocess_range(pcd, **kwargs):
    depth_img = pcd2range(pcd, **kwargs)[0]
    xyz_img = range2xyz(depth_img, log_scale=False, **kwargs)
    depth_img = depth_img[None]
    img = np.vstack([depth_img, xyz_img])
    return img


def batch2list(batch_dict, agg_type='depth', **kwargs):
    """
    Aggregate sparse voxel backbone output into one feature vector per frame.
    """
    output_list = []
    batch_indices = batch_dict['batch_indices']
    for b_idx in range(batch_indices.max() + 1):
        if agg_type == 'all':
            logits = batch_dict['logits'][batch_indices == b_idx].mean(0)

        elif agg_type == 'sector':
            logits = batch_dict['logits'][batch_indices == b_idx]
            coords = batch_dict['coords'][batch_indices == b_idx].float()
            coords = coords - coords.mean(0)
            angle = torch.atan2(coords[:, 1], coords[:, 0])
            sector_range = torch.linspace(-np.pi - 1e-4, np.pi + 1e-4, NUM_SECTORS + 1)
            logits_list = []
            for i in range(NUM_SECTORS):
                sector_indices = torch.where((angle >= sector_range[i]) & (angle < sector_range[i + 1]))[0]
                sector_logits = logits[sector_indices].mean(0)
                sector_logits = torch.nan_to_num(sector_logits, 0.)
                logits_list.append(sector_logits)
            logits = torch.cat(logits_list)

        elif agg_type == 'depth':
            logits = batch_dict['logits'][batch_indices == b_idx]
            coords = batch_dict['coords'][batch_indices == b_idx].float()
            coords = coords - coords.mean(0)
            bev_depth = torch.norm(coords, dim=-1) * VOXEL_SIZE
            sector_range = torch.linspace(kwargs['depth_range'][0] + 3, kwargs['depth_range'][1], NUM_SECTORS + 1)
            sector_range[0] = 0.
            logits_list = []
            for i in range(NUM_SECTORS):
                sector_indices = torch.where((bev_depth >= sector_range[i]) & (bev_depth < sector_range[i + 1]))[0]
                sector_logits = logits[sector_indices].mean(0)
                sector_logits = torch.nan_to_num(sector_logits, 0.)
                logits_list.append(sector_logits)
            logits = torch.cat(logits_list)

        else:
            raise NotImplementedError

        output_list.append(logits.detach().cpu().numpy())
    return output_list


def compute_logits(data_type, modality, *args):
    """
    Compute feature logits for three modalities: range / voxel / point_voxel.
    data_type: '64' or '32'
    modality: 'range' | 'voxel' | 'point_voxel'
    """
    assert data_type in ['32', '64']
    assert modality in ['range', 'voxel', 'point_voxel']
    is_voxel = 'voxel' in modality
    dataset_name = TYPE2DATASET[data_type]
    dataset_config = DATASET_CONFIG[dataset_name]
    bs = MODAL2BATCHSIZE[modality]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(dataset_name, MODALITY2MODEL[modality], device=device)

    output = tuple()
    for data in args:
        all_logits_list = []
        for i in range(math.ceil(len(data) / bs)):
            batch = data[i * bs:(i + 1) * bs]
            if is_voxel:
                batch = [pcd2voxel(preprocess_pcd(pcd, **dataset_config)) for pcd in batch]
                batch = sparse_collate_fn(batch)
                batch = {
                    k: v.to(device) if isinstance(v, (torch.Tensor, SparseTensor, PointTensor)) else v
                    for k, v in batch.items()
                }
                with torch.no_grad():
                    batch_out = model(batch, return_final_logits=True)
                    batch_out = batch2list(batch_out, AGG_TYPE, **dataset_config)
                    all_logits_list.extend(batch_out)
            else:
                batch = [preprocess_range(pcd, **dataset_config) for pcd in batch]
                batch = torch.from_numpy(np.stack(batch)).float().to(device)
                with torch.no_grad():
                    batch_out = model(batch, return_final_logits=True, agg_type=AGG_TYPE)
                    all_logits_list.append(batch_out)

        if is_voxel:
            all_logits = np.stack(all_logits_list)
        else:
            all_logits = np.vstack(all_logits_list)
        output += (all_logits,)

    del model, batch, batch_out
    if device == 'cuda':
        torch.cuda.empty_cache()

    return output


# ========== FSVD / FPVD computation ==========

def compute_fsvd(reference, samples, data_type):
    """
    FSVD: Fréchet Sparse Volume Distance
    """
    print('Evaluating (FSVD) ...')
    gt_logits, sample_logits = compute_logits(data_type, 'voxel', reference, samples)
    score = compute_fd(gt_logits, sample_logits)
    print(OUTPUT_TEMPLATE.format('FSVD', score))


def compute_fpvd(reference, samples, data_type):
    """
    FPVD: Fréchet Point-based Volume Distance
    """
    print('Evaluating (FPVD) ...')
    gt_logits, sample_logits = compute_logits(data_type, 'point_voxel', reference, samples)
    score = compute_fd(gt_logits, sample_logits)
    print(OUTPUT_TEMPLATE.format('FPVD', score))


def compute_frid(reference, samples, data_type):
    """
    Optional FRID computation; requires range model (rangenet).
    """
    print('Evaluating (FRID) ...')
    gt_logits, sample_logits = compute_logits(data_type, 'range', reference, samples)
    score = compute_fd(gt_logits, sample_logits)
    print(OUTPUT_TEMPLATE.format('FRID', score))


# ========== Point cloud loading (KITTI .bin compatible) ==========

def load_kitti_bin_xyz(file_path):
    """
    KITTI-format .bin: 4 float32 per point (x, y, z, intensity).
    """
    scan = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    points = scan[:, :3].astype(np.float32)
    return points


def load_bin_folder_as_list(folder_path):
    files = sorted(glob.glob(os.path.join(folder_path, '*.bin')))
    if len(files) == 0:
        raise RuntimeError(f'No .bin point cloud files found under {folder_path}')
    pcs = []
    for f in tqdm(files, desc=f'Loading {os.path.basename(folder_path)}'):
        pcs.append(load_kitti_bin_xyz(f))
    return pcs


# ========== CLI & main ==========

DATASET2TYPE = {'kitti': '64', 'nuscenes': '32'}
DATASET2METRICS = {
    'kitti': ['fsvd', 'fpvd'],
    'nuscenes': ['fsvd', 'fpvd'],
}


def parse_args():
    parser = argparse.ArgumentParser(description='Minimal FSVD/FPVD evaluation for point cloud folders')
    parser.add_argument('--samples_folder', type=str, required=True,
                        help='Folder of generated point clouds (.bin)')
    parser.add_argument('--reference_folder', type=str, required=True,
                        help='Folder of ground-truth point clouds (.bin)')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        choices=['kitti', 'nuscenes'],
                        help='Dataset name (kitti / nuscenes)')
    parser.add_argument('-m', '--metrics', type=str, nargs='+',
                        default=None,
                        choices=['fsvd', 'fpvd', 'frid'],
                        help='Metrics to compute (default: all available metrics for the dataset)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print more detailed information')
    return parser.parse_args()


def main():
    opt = parse_args()

    if not os.path.isdir(opt.samples_folder):
        raise RuntimeError(f'Samples folder does not exist: {opt.samples_folder}')
    if not os.path.isdir(opt.reference_folder):
        raise RuntimeError(f'Reference folder does not exist: {opt.reference_folder}')

    print('=' * 80)
    print('Loading point clouds ...')
    print('=' * 80)

    reference = load_bin_folder_as_list(opt.reference_folder)
    samples = load_bin_folder_as_list(opt.samples_folder)

    if len(samples) == 0 or len(reference) == 0:
        raise RuntimeError('No valid point clouds in samples or reference')

    if len(samples) != len(reference):
        print(f'Warning: sample count mismatch: samples={len(samples)}, reference={len(reference)}')
        min_len = min(len(samples), len(reference))
        print(f'Only using the first {min_len} pairs of samples')
        samples = samples[:min_len]
        reference = reference[:min_len]

    if opt.metrics is None:
        metrics = DATASET2METRICS[opt.dataset]
        print(f'Using default metrics for {opt.dataset}: {metrics}')
    else:
        metrics = opt.metrics

    data_type = DATASET2TYPE[opt.dataset]

    if opt.verbose:
        print('\n' + '=' * 80)
        print('Point Cloud Statistics')
        print('=' * 80)
        print(f'Num samples: {len(samples)}')
        npts_s = [p.shape[0] for p in samples]
        npts_r = [p.shape[0] for p in reference]
        print(f'Samples   - min: {min(npts_s)}, max: {max(npts_s)}, avg: {np.mean(npts_s):.1f}')
        print(f'Reference - min: {min(npts_r)}, max: {max(npts_r)}, avg: {np.mean(npts_r):.1f}')
        print('=' * 80 + '\n')

    print('=' * 80)
    print('Start Evaluation')
    print('=' * 80)
    print(f'Metrics: {metrics}')
    print(f'Data type: {data_type}')
    print('=' * 80 + '\n')

    if 'fsvd' in metrics:
        compute_fsvd(reference, samples, data_type)
    if 'fpvd' in metrics:
        compute_fpvd(reference, samples, data_type)
    if 'frid' in metrics:
        compute_frid(reference, samples, data_type)

    print('\n' + '=' * 80)
    print('Evaluation Finished')
    print('=' * 80)


if __name__ == '__main__':
    main()