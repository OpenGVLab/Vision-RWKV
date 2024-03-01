# Vision-RWKV
The official implementation of "Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like".

## Abstract
Transformers have revolutionized computer vision and natural language processing, but their high computational complexity limits their application in high-resolution image processing and long-context analysis. This paper introduces Vision-RWKV (VRWKV), a model adapted from the RWKV model used in the NLP field with necessary modifications for vision tasks. Similar to the Vision Transformer (ViT), our model is designed to efficiently handle sparse inputs and demonstrate robust global processing capabilities, while also scaling up effectively, accommodating both large-scale parameters and extensive datasets. Its distinctive advantage lies in its reduced spatial aggregation complexity, which renders it exceptionally adept at processing high-resolution images seamlessly, eliminating the necessity for windowing operations. Our evaluations in image classification demonstrate that VRWKV matches ViT's classification performance with significantly faster speeds and lower memory usage. In dense prediction tasks, it outperforms window-based models, maintaining comparable speeds. These results highlight VRWKV's potential as a more efficient alternative for advanced image analysis tasks.
Code and models shall be available.

## Overview
<div align="center">
<img width="600" alt="image" src="assets/overall_architecture.JPG">
</div>

## Released Models
- Classification

| Model | #Param | FLOPs | Top 1-Acc | Download |
|:------------------------------------------------------------------:|:-------------:|:----------:|:----------:|:----------:|
| [VRWKV-T](https://huggingface.co/duanyuchen/vrwkv_tiny)    |      6.2M       |   1.2G   | 75.1  | [ckpt](https://huggingface.co/duanyuchen/vrwkv_tiny/resolve/main/vrwkv_tiny_in1k_224.pth) \| [cfg](classification/configs/vrwkv/vrwkv_tiny_8xb128_in1k.py) |
| [VRWKV-S](https://huggingface.co/duanyuchen/vrwkv_small)    |     23.8M       |   4.6G   | 80.1  | [ckpt](https://huggingface.co/duanyuchen/vrwkv_small/resolve/main/vrwkv_small_in1k_224.pth) \| [cfg](classification/configs/vrwkv/vrwkv_small_8xb128_in1k.py) |
| [VRWKV-B](https://huggingface.co/duanyuchen/vrwkv_base)    |     93.7M       |  18.2G   | 82.0  | [ckpt](https://huggingface.co/duanyuchen/vrwkv_base/resolve/main/vrwkv_base_in1k_224.pth) \| [cfg](classification/configs/vrwkv/vrwkv_base_16xb64_in1k.py) |
