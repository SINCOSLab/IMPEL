# IMPEL: Can Pretrained Image Diffusion Enable Realistic LiDAR Point Cloud Generation?

## Create a virtual environment

First create a  python 3.10 virtual environment:

```
conda create -n your_env python=3.10
conda activate your_env
```

Then install dependencies:

```
pip install -r requirements.txt
```

Note that, Our experiments were conducted in a CUDA 12.2 environment.

## Training & Inference & Evaluation

### Unconditional LiDAR generation

To fine-tune the SD3 model, you should first download the pretrained weights of Stable Diffusion 3 Medium from hugging face.

Then, download the KITTI-360 dataset from https://www.cvlibs.net/datasets/kitti-360/download.php and create range-image data from the raw point clouds:

```bash
python data_prepare/create_rangeimage.py
```

The training NPY files are single-channel arrays with shape `64 x 256` (`H x W`). Both fine-tuning stages upsample them to `256 x 1024` before VAE encoding. Accordingly, keep `--height=256` and `--width=1024` in both training scripts.

Please fill in the relevant paths in the `train_firstStage.sh` and `train_secondStage.sh`file as your own.  Then run the following command  to  perform the first stage of fine-tuning:

```
sh ./train_firstStage.sh
```

After the first fine-tuning stage, run the following command  to  perform the second stage of fine-tuning and generate point cloud samples:

```
sh ./train_secondStage.sh
```

To evaluate the generated LiDAR point clouds, please first download the pretrained weights of `RangeNet++` from https://github.com/PRBonn/lidar-bonnetal, and put them in the `metrics/rangenetpp/lidar_bonnetal_master/darknet53-1024`, including the `backbone`, `segmentation_head`, and `segmentation_head`.

Then run the following command from the project root to compute the FRD, JSD, and MMD metrics. `--kitti360_dataset` must point to the directory containing the KITTI-360 ground-truth `.bin` point clouds:

```
python metrics/metric.py \
  --fid --jsd --mmd \
  --exp /generated_samples_path \
  --kitti360_dataset /path/to/kitti360_groundtruth_bin
```

Moreover, we provide built-in scripts to calculate FSVD and FPVD.

In our tests, computing FSVD and FPVD requires a cuda 11.3 environment, and install corresponding dependencies:

```
pip install -r requirements_FSVD_FPVD.txt
```

Then, download the pretrained weights of `minkowskinet` and `spvcnn`, and put them in `FSVD_FPVD_metrics/pretrained_weights/kitti/minkowskinet` and `FSVD_FPVD_metrics/pretrained_weights/kitti/spvcnn`.

Finally, please run the following command:

```
python eval_fsvd_fpvd.py \
  --samples_folder "/path/to/generated_bin" \
  --reference_folder "/path/to/gt_bin" \
  --dataset kitti \
  --metrics fsvd fpvd
```

### Camera-to-LiDAR generation

In our initial Camera-to-LiDAR experiments, we used Canny edges extracted directly from RGB images as the ControlNet condition. We later found that Canny edges extracted from monocular depth maps (depth-Canny) produced better results. Therefore, we provide the depth-Canny setup here. The same training framework also supports RGB-Canny or raw monocular depth maps as alternative conditions, provided that the conditioning data are preprocessed consistently with the target LiDAR range images.

For raw depth conditioning, first align the depth map with the camera view corresponding to the LiDAR range image. Handle invalid depth values, clip or scale the valid depth range consistently across the dataset, normalize it to an image-compatible range, and save it as a three-channel conditioning image.

For RGB-Canny conditioning, first undistort, crop, and resize the RGB image so that its field of view and spatial correspondence match the target LiDAR data. Then convert it to grayscale, apply a fixed Canny configuration consistently to the entire dataset, and save the resulting edge map as a three-channel conditioning image.

Each conditioning modality should have its own metadata paths and should normally be trained or fine-tuned as a separate ControlNet checkpoint, because raw depth, depth-Canny, and RGB-Canny have different input distributions. The directory name is not fixed, but every conditioning image must be paired one-to-one with its target range image in `metadata.jsonl`.

For the depth-Canny setting used in our experiments, prepare the data as follows:

```
controlnet_data/
|-- range/000000.npy ...
|-- depth_canny/000000.png ...
`-- metadata.jsonl
```

Each JSON line should contain the target range-image path, the corresponding depth-Canny path, and the text prompt, for example:

```json
{"range_image_file_name": "range/000000.npy", "conditioning_rgb_file_name": "depth_canny/000000.png", "text": "A range image"}
```

The argument name `--rgb_column=conditioning_rgb` is retained for compatibility with the training code. Depending on the selected experiment, this column may contain a raw depth map, a depth-Canny map, or an RGB-Canny map. The ControlNet training setup used in our depth-Canny experiments resizes both the target and conditioning image to `128 x 1024` (`H x W`).

`train_controlnet.py` uses the custom LiDAR-aware ControlNet in `model/LiDARControlNet.py` and the custom range-image pipeline in `RangeImageStableDiffusion3ControlNetPipeline.py`. Fill in the model, dataset, LoRA, cache, output, and validation-image paths in `train_controlnet.sh`, then run:

```bash
sh train_controlnet.sh
```
