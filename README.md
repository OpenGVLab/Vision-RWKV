# Vision-RWKV
The official implementation of "[Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures](https://arxiv.org/abs/2403.02308)".

## News🚀🚀🚀
- `2024/04/14`: We support rwkv6 in classification task, higher performance!
- `2024/03/04`: We release the code and models of Vision-RWKV.

## Highlights

- **High-Resolution Efficiency**: Processed high-resolution images smoothly with a global receptive field.
- **Scalability**: Pre-trained with large-scale datasets and posses scale up stablity.
- **Superior Performance**: Achieved a better performance in classfication tasks than ViTs. Surpassed window-based ViTs and comparabled to global attention ViTs with lower flops and higher speed in dense prediction tasks.
- **Efficient Alternative**: Capability to be an alternative backbone to ViT in comprehensive vision tasks.

<img width="1238" alt="image" src="https://github.com/OpenGVLab/Vision-RWKV/assets/23737120/10965279-6542-4f82-aef5-934b8d86b345">


## Overview

<img width="1238" alt="image" src="https://github.com/OpenGVLab/Vision-RWKV/assets/23737120/7521a3d6-6b5a-4a24-9ec8-dfb4abd3fd84">

## Schedule
- [x] Support RWKV6 as VRWKV6
- [x] Release VRWKV-L
- [x] Release VRWKV-T/S/B

## Model Zoo

### Pretrained Models
|  Model  |   Size   |   Pretrain   |       Download       |
|:-------:|:--------:|:------------:|:--------------------:|
| VRWKV-L |    192   | ImageNet-22K | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_l_in22k_192.pth) |

### Image Classification (ImageNet-1K)

|  Model   |   Size   | #Param | #FLOPs |  Top-1 Acc |       Download       |
|:--------:|:--------:| ------:| ------:|:----------:|:--------------------:|
| VRWKV-T  |    224   |   6.2M |   1.2G |    75.1    | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_t_in1k_224.pth)    \| [cfg](classification/configs/vrwkv/vrwkv_tiny_8xb128_in1k.py)        |
| VRWKV-S  |    224   |  23.8M |   4.6G |    80.1    | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_s_in1k_224.pth)    \| [cfg](classification/configs/vrwkv/vrwkv_small_8xb128_in1k.py)       |
| VRWKV-B  |    224   |  93.7M |  18.2G |    82.0    | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_b_in1k_224.pth)    \| [cfg](classification/configs/vrwkv/vrwkv_base_16xb64_in1k.py)        |
| VRWKV-L  |    384   | 334.9M | 189.5G |    86.0    | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_l_22kto1k_384.pth) \| [cfg](classification_internimage/configs/vrwkv_l_22kto1k_384.yaml) |
| VRWKV6-T |    224   |   7.6M |   1.6G |    76.6    | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv6_t_in1k_224.pth)    \| [cfg](classification/configs/vrwkv6/vrwkv6_tiny_8xb128_in1k.py)        |
| VRWKV6-S |    224   |  27.7M |   5.6G |    81.1    | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv6_s_in1k_224.pth)    \| [cfg](classification/configs/vrwkv6/vrwkv6_small_8xb128_in1k.py)       |
| VRWKV6-B |    224   | 104.9M |  20.9G |    82.6    | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv6_b_in1k_224.pth)    \| [cfg](classification/configs/vrwkv6/vrwkv6_base_16xb64_in1k.py)        |

- VRWKV-L is pretrained on ImageNet-22K and then finetuned on ImageNet-1K.
- We train VRWKV-L with the internimage codebase for a higher speed.

### Object Detection with Mask-RCNN head (COCO)


|  Model  | #Param |  #FLOPs | box AP |  mask AP |       Download       |
|:-------:| ------:| -------:|:------:|:--------:|:--------------------:|
| VRWKV-T |   8.4M |   67.9G |  41.7  |   38.0   | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/mask_rcnn_vrwkv_adapter_tiny_fpn_1x_coco.pth)  \| [cfg](detection/configs/mask_rcnn/mask_rcnn_vrwkv_adapter_tiny_fpn_1x_coco.py)  |
| VRWKV-S |  29.3M |  189.9G |  44.8  |   40.2   | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/mask_rcnn_vrwkv_adapter_small_fpn_1x_coco.pth) \| [cfg](detection/configs/mask_rcnn/mask_rcnn_vrwkv_adapter_small_fpn_1x_coco.py) |
| VRWKV-B | 106.6M |  599.0G |  46.8  |   41.7   | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/mask_rcnn_vrwkv_adapter_base_fpn_1x_coco.pth)  \| [cfg](detection/configs/mask_rcnn/mask_rcnn_vrwkv_adapter_base_fpn_1x_coco.py)  |
| VRWKV-L | 351.9M | 1730.6G |  50.6  |   44.9   | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/mask_rcnn_vrwkv_adapter_large_fpn_1x_coco.pth) \| [cfg](detection/configs/mask_rcnn/mask_rcnn_vrwkv_adapter_large_fpn_1x_coco.py) |

- We report the \#Param and \#FLOPs of the backbone in this table.

### Semantic Segmentation with UperNet head (ADE20K)


|  Model  | #Param | #FLOPs |   mIoU   |       Download       |
|:-------:| ------:| ------:|:--------:|:--------------------:|
| VRWKV-T |   8.4M |  16.6G |   43.3   | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/upernet_vrwkv_adapter_tiny_512_160k_ade20k.pth)  \| [cfg](segmentation/configs/ade20k/upernet_vrwkv_adapter_tiny_512_160k_ade20k.py)  |
| VRWKV-S |  29.3M |  46.3G |   47.2   | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/upernet_vrwkv_adapter_small_512_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_vrwkv_adapter_small_512_160k_ade20k.py) |
| VRWKV-B | 106.6M | 146.0G |   49.2   | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/upernet_vrwkv_adapter_base_512_160k_ade20k.pth)  \| [cfg](segmentation/configs/ade20k/upernet_vrwkv_adapter_base_512_160k_ade20k.py)  |
| VRWKV-L | 351.9M | 421.9G |   53.5   | [ckpt](https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/upernet_vrwkv_adapter_large_512_160k_ade20k.pth) \| [cfg](segmentation/configs/ade20k/upernet_vrwkv_adapter_large_512_160k_ade20k.py) |

- We report the \#Param and \#FLOPs of the backbone in this table.

## Citation
If this work is helpful for your research, please consider citing the following BibTeX entry.
```BibTeX
@article{duan2024vrwkv,
  title={Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures},
  author={Duan, Yuchen and Wang, Weiyun and Chen, Zhe and Zhu, Xizhou and Lu, Lewei and Lu, Tong and Qiao, Yu and Li, Hongsheng and Dai, Jifeng and Wang, Wenhai},
  journal={arXiv preprint arXiv:2403.02308},
  year={2024}
}
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Acknowledgement

Vision-RWKV is built with reference to the code of the following projects:  [RWKV](https://github.com/BlinkDL/RWKV-LM), [MMPretrain](https://github.com/open-mmlab/mmpretrain), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMSegmentation](https://github.com/open-mmlab/mmsegmentation), [ViT-Adapter](https://github.com/czczup/ViT-Adapter), [InternImage](https://github.com/OpenGVLab/InternImage). Thanks for their awesome work!