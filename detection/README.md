# Vision-RWKV for Object Detection

This folder contains the implementation of the Vision-RWKV(VRWKV) for object detection. 

Our detection code is developed on top of [MMDetection v2.28.2](https://github.com/open-mmlab/mmdetection/tree/v2.28.2).


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
pip install timm==0.6.12 mmdet==2.28.2
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
```

- Compile Deformable Attention
```bash
cd ./ops
sh ./make.sh
```

### Data Preparation

Prepare COCO according to the guidelines in [MMDetection v2.28.2](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).


### Evaluation

To evaluate our `VRWKV` on COCO val, run:

```bash
sh dist_test.sh <config-file> <checkpoint> <gpu-num> --eval bbox segm
```

For example, to evaluate the `VRWKV-T` with a single GPU:

```bash
python test.py configs/mask_rcnn/mask_rcnn_vrwkv_adapter_tiny_fpn_1x_coco.py checkpoint/mask_rcnn_vrwkv_adapter_tiny_fpn_1x_coco.pth --eval bbox segm
```

For example, to evaluate the `VRWKV-B` with a single node with 8 GPUs:

```bash
sh dist_test.sh configs/mask_rcnn/mask_rcnn_vrwkv_adapter_base_fpn_1x_coco.py checkpoint/mask_rcnn_vrwkv_adapter_base_fpn_1x_coco.pth 8 --eval bbox segm
```

### Training on COCO

To train an `VRWKV` on COCO, run:

```bash
sh dist_train.sh <config-file> <gpu-num>
```

For example, to train `VRWKV-T` with 8 GPU on 1 node (total batch size 16), run:

```bash
sh dist_train.sh configs/mask_rcnn/mask_rcnn_vrwkv_adapter_tiny_fpn_1x_coco.py 8
```

### Manage Jobs with Slurm

For example, to train `VRWKV-B` with 16 GPU on 2 node (total batch size 16), run:

```bash
GPUS=16 sh slurm_train.sh <partition> <job-name> configs/mask_rcnn/mask_rcnn_vrwkv_adapter_base_fpn_1x_coco.py work_dirs/mask_rcnn_vrwkv_adapter_base_fpn_1x_coco
```
