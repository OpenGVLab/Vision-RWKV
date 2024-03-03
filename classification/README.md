# Vision-RWKV for Image Classification

This folder contains the implementation of the Vision-RWKV(VRWKV) for image classification.

Our detection code is developed on top of [MMClassification v0.25.0](https://github.com/open-mmlab/mmpretrain/tree/v0.25.0).

<!-- TOC -->
* [Install](#install)
* [Data Preparation](#data-preparation)
* [Evaluation](#evaluation)
* [Training from Scratch on ImageNet-1K](#training-from-scratch-on-imagenet-1k)
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
pip install timm==0.6.12 mmcls==0.25.0
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
```

### Data Preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. The file structure should looks like:
```bash
$ tree data
imagenet/
├── meta/
│   ├── train.txt
│   ├── test.txt
│   └── val.txt
├── train/
│   ├── n01440764
│   │   ├── n01440764_10026.JPEG
│   │   ├── n01440764_10027.JPEG
│   │   ├── n01440764_10029.JPEG
│   │   ├── n01440764_10040.JPEG
│   │   ├── n01440764_10042.JPEG
│   │   ├── n01440764_10043.JPEG
│   │   └── n01440764_10048.JPEG
│   ├── ...
├── val/
│   ├── ILSVRC2012_val_00000001.JPEG
│   ├── ILSVRC2012_val_00000002.JPEG
│   ├── ILSVRC2012_val_00000003.JPEG
│   ├── ILSVRC2012_val_00000004.JPEG
│   ├── ...
```

### Evaluation

To evaluate a pretrained `VRWKV` on ImageNet val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --metrics accuracy
```

For example, to evaluate the `VRWKV-T` with a single GPU:

```bash
python test.py configs/vrwkv/vrwkv_tiny_8xb128_in1k.py checkpoint/vrwkv_t_in1k_224.pth --metrics accuracy
```

For example, to evaluate the `VRWKV-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/vrwkv/vrwkv_base_16xb64_in1k.py checkpoint/vrwkv_b_in1k_224.pth 8 --metrics accuracy
```

### Training from Scratch on ImageNet-1K

To train an `VRWKV` on ImageNet from scratch, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `VRWKV-T` with 8 GPU on a single node for 300 epochs, run:

```bash
sh dist_train.sh configs/vrwkv/vrwkv_tiny_8xb128_in1k.py 8
```

### Manage Jobs with Slurm.

For example, to train `VRWKV-B` with 16 GPU on 2 node (total batch size 1024), run:

```bash
GPUS=16 sh slurm_train.sh <partition> <job-name> configs/vrwkv/vrwkv_base_16xb64_in1k.py work_dirs/vrwkv_base_16xb64_in1k
```