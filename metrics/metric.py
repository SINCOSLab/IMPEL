import sys

sys.path.append('rangenetpp/lidar_bonnetal_master/train/tasks/semantic')
sys.path.append('rangenetpp/lidar_bonnetal_master/train/')
import argparse
import metrics.histogram.mmd as mmd
import metrics.histogram.jsd as jsd
import os
import rangenetpp.lidar_bonnetal_master.train.tasks.semantic.infer_lib as rangenetpp
import glob
import metrics.fid.lidargen_fid as lidargen_fid
import metrics.iou as lidargen_iou
import metrics.mae as lidargen_mae
import torch
import numpy as np
from tqdm import tqdm


os.environ["KITTI360_DATASET"] = "your_groundtruth_data_path" 


def generate_kitti_fid(folder_fid, folder_segmentations, sample_count,  seed=0):
    # Get dump for KITTI
    os.system("rm -r {folder_segmentations}".format(folder_segmentations=folder_segmentations))
    os.system("rm -r {folder_fid}".format(folder_fid=folder_fid))
    os.system("mkdir {folder_segmentations}".format(folder_segmentations=folder_segmentations))
    os.system("mkdir {folder_fid}".format(folder_fid=folder_fid))
    rangenetpp.main('--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --kitti --output_dir {folder_segmentations} --frd_dir {folder_fid} --kitti_count {kitti_count} --seed {seed}'.format(
        kitti_count=str(sample_count), seed=str(seed), folder_segmentations=folder_segmentations, folder_fid=folder_fid))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--nus", help="Use nuScenes dataset", action="store_true")
    parser.add_argument("--mmd", help="Calculate MMD between samples and KITTI-360", action="store_true")
    parser.add_argument("--jsd", help="Caculate JSD between samples and KITTI-360", action="store_true")
    parser.add_argument("--cd", help="Caculate CD between samples and KITTI-360", action="store_true")
    parser.add_argument("--exp", help="Folder of generated point cloud.")
    parser.add_argument("--fid_folder1", help="Manually provide folder1", type=str, default=None)
    parser.add_argument("--fid_folder2", help="Manually provide folder2", type=str, default=None)
    parser.add_argument("--fid", help="Run generated samples through RangeNet to get FID (KITTI only).", action="store_true")
    args = parser.parse_args()

    if(args.fid):
        folder1 = ""
        folder2 = ""

        if((not (args.fid_folder1 is None)) and (not (args.fid_folder2 is None))):
            folder1 = args.fid_folder1
            folder2 = args.fid_folder2

        elif (not (args.fid_folder1 is None)):
            folder1 = args.fid_folder1
            folder2 = "kitti_fid"
            folder_segmentations = "kitti_seg"
            generate_kitti_fid(folder2, folder_segmentations, 1000, 0)

        else:
            # Get dump for model samples
            os.system("rm -r {exp}/unconditional_fid".format(exp=args.exp))
            os.system("mkdir {exp}/unconditional_fid".format(exp=args.exp))
            os.system("rm -r {exp}/unconditional_segmentations".format(exp=args.exp))
            os.system("mkdir {exp}/unconditional_segmentations".format(exp=args.exp))
            rangenetpp.main(
                '--dataset ignore --model rangenetpp/lidar_bonnetal_master/darknet53-1024/ --dump {exp}/ --output_dir {exp}/unconditional_segmentations --frd_dir {exp}/unconditional_fid --point_cloud'.format(
                    exp=str(args.exp)))
            folder1 = "{exp}/unconditional_fid/".format(exp=str(args.exp))
            # Get dump for model samples

            if not args.fid_folder2:
                sample_count = len(glob.glob("{exp}/unconditional_segmentations/*".format(exp=args.exp))) # 1000
            
                # Get dump for KITTI
                folder_kitti_seg = "kitti_segmentations"
                folder_kitti_fid = "kitti_fid"
                generate_kitti_fid(folder_kitti_fid, folder_kitti_seg, sample_count, 0)

                folder2 = folder_kitti_fid 
            else:
                folder2 = args.fid_folder2
        print(folder1)
        print(folder2)
        fid_score = lidargen_fid.get_fid(folder1, folder2)

        print('-------------------------------------------------')
        print('FID Score: ' + str(fid_score))
        print('-------------------------------------------------')
        
    if (args.jsd):
        jsd_score = jsd.calculate_jsd(args.exp, from_bin=True) \
                    if not args.nus else jsd.calculate_jsd_nus(args.exp, from_bin=True)

        print('-------------------------------------------------')
        print('JSD Score: ' + str(jsd_score))
        print('-------------------------------------------------')

    if (args.mmd):
        mmd_score = mmd.calculate_mmd(args.exp, from_bin=True) \
                    if not args.nus else mmd.calculate_mmd_nus(args.exp, from_bin=True)

        print('-------------------------------------------------')
        print('MMD Score: ' + str(mmd_score))
        print('-------------------------------------------------')

    if (args.cd):
        def chamfer_distance(p1, p2):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            t1 = torch.from_numpy(p1).float().to(device)  # [N1,3]
            t2 = torch.from_numpy(p2).float().to(device)   # [N2,3]
            dist = torch.cdist(t1.unsqueeze(0), t2.unsqueeze(0))[0]  # [B,N,M]
            d1 = dist.min(dim=1).values  # [N1]
            d2 = dist.min(dim=0).values  # [N2]
            return (d1.mean().item() + d2.mean().item()) / 2

        generated_list = glob.glob(f'{args.exp}/*.bin')
        kitti_list = glob.glob(os.environ.get('KITTI360_DATASET') + '/*.bin')
        total_cd = 0
        for i in tqdm(range(len(generated_list))):
            generated_pc = np.fromfile(generated_list[i], dtype=np.float32).reshape(-1, 3)
            kitti_pc = np.fromfile(kitti_list[i], dtype=np.float32).reshape(-1, 4)[:,:3]
            d = chamfer_distance(generated_pc, kitti_pc)
            total_cd += d
        mean_cd = total_cd / len(generated_list)
        print(f"Mean Chamfer Distance: {mean_cd:.6f}")


if __name__ == '__main__':
    main()