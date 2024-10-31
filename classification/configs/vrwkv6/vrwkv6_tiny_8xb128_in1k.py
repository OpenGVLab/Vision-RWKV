# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/datasets/imagenet_bs64_swin_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VRWKV6',
        img_size=224,
        patch_size=16,
        embed_dims=192,
        num_heads=3,),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=192,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict()
)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=2,
    save_last=True)
evaluation = dict(interval=1)
# 8 gpus
data = dict(
    samples_per_gpu=128,
)