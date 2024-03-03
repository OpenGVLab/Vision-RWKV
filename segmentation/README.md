# Vision-RWKV for Semantic Segmentation

This folder contains the implementation of the Vision-RWKV(VRWKV) for semantic segmentation. 

Our segmentation code is developed on top of [MMSegmentation v0.30.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0).

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


- Install `timm` and `mmcv-full` and `mmsegmentation':

```bash
pip install -U openmim
mim install mmcv-full==1.7.0
mim install mmsegmentation==0.30.0
pip install timm==0.6.11 mmdet==2.28.2
```

- Compile Deformable Attention
```bash
ln -s ../detection/ops ./
cd ./ops
sh ./make.sh
```

### Data Preparation

Prepare datasets according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.


### Evaluation

To evaluate our `VRWKV` on ADE20K val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval mIoU
```

For example, to evaluate the `VRWKV-T` with a single GPU:

```bash
python test.py configs/ade20k/upernet_vrwkv_adapter_tiny_512_160k_ade20k.py checkpoint/upernet_vrwkv_adapter_tiny_512_160k_ade20k.pth --eval mIoU
```

For example, to evaluate the `VRWKV-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/ade20k/upernet_vrwkv_adapter_base_512_160k_ade20k.py checkpoint/upernet_vrwkv_adapter_base_512_160k_ade20k.pth 8 --eval mIoU
```

### Training

To train an `VRWKV` on ADE20K, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `VRWKV-T` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/ade20k/upernet_vrwkv_adapter_tiny_512_160k_ade20k.py 8
```

### Manage Jobs with Slurm

For example, to train `VRWKV-B` with 8 GPU on 1 node (total batch size 16), run:

```bash
GPUS=8 sh slurm_train.sh <partition> <job-name> configs/ade20k/upernet_vrwkv_adapter_base_512_160k_ade20k.py
```
