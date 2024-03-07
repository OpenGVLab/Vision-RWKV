# Vision-RWKV for Image Classification(InternImage Codebase)

This folder contains the implementation of the Vision-RWKV (VRWKV) for image classification.

<!-- TOC -->
* [Install](#install)
* [Data Preparation](#data-preparation)
* [Evaluation](#evaluation)
* [Finetune on ImageNet-1K](#finetune-on-imagenet-1k)
* [Manage Jobs with Slurm.](#manage-jobs-with-slurm)
<!-- TOC -->

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/OpenGVLab/Vision-RWKV.git
cd Vision-RWKV
```

- Create a conda virtual environment and activate it:

```bash
conda create -n vrwkv python=3.10 -y
conda activate vrwkv
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install torch==1.12.1 with CUDA==11.3:
```bash
pip install torch==1.12.1+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
```

- Install `timm==0.6.12` and `mmcv-full==1.7.0`:

```bash
pip install -U openmim
mim install mmcv-full==1.7.0
pip install timm==0.6.12
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
```

### Data Preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
   
  ```
- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:
    - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
    - `train.txt`, `val.txt`: which store the relative path in the corresponding zip file and ground truth
      label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip
  
  $ head -n 5 meta_data/val.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516
  
  $ head -n 5 meta_data/train.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```
- For ImageNet-22K dataset, make a folder named `fall11_whole` and move all images to labeled sub-folders in this
  folder. Then download the train-val split
  file ([ILSVRC2011fall_whole_map_train.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_train.txt)
  & [ILSVRC2011fall_whole_map_val.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_val.txt))
  , and put them in the parent directory of `fall11_whole`. The file structure should look like:

  ```bash
    $ tree imagenet22k/
    imagenet22k/
    └── fall11_whole
        ├── n00004475
        ├── n00005787
        ├── n00006024
        ├── n00006484
        └── ...
  ```

### Evaluation

To evaluate a pretrained `VRWKV` on ImageNet-1K val, run:

```bash
sh dist_test_in1k.sh <config-file> <checkpoint> <gpu-num> 
```

For example, to evaluate the `VRWKV-L` with a single GPU:

```bash
sh dist_test_in1k.sh configs/vrwkv_l_22kto1k_384.yaml ./pretrained/vrwkv_l_22kto1k_384.pth 1
```

### Finetune on ImageNet-1K

To finetune an `VRWKV` on ImageNet-1K, run:

```bash
sh dist_train_in1k.sh <config-file> <gpu-num> 
```

For example, to finetune a pretrained `VRWKV-L` on ImageNet-1K with 8 GPU on a single node for 20 epochs, run:

```bash
sh dist_train_in1k.sh configs/vrwkv_l_22kto1k_384.yaml 8
```

### Manage Jobs with Slurm.

For example, to finetune `VRWKV-L` on ImageNet-1K with 16 GPUs on 2 nodes for 20 epochs (total batch size 1024), run:

```bash
GPUS=16 sh slurm_train_in1k.sh <partition> <job-name> configs/vrwkv_l_22kto1k_384.yaml 
```
