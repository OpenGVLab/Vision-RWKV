# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

pretrained = 'pretrained/vrwkv_l_in22k_192.pth'
# https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_l_in22k_192.pth

model = dict(
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
        type='VRWKV_Adapter',
        img_size=192,
        patch_size=16,
        embed_dims=1024,
        depth=24,
        pretrained=None,
        init_values=1e-5,
        post_norm=True,
        key_norm=True,
        with_cp=False,
        # adapter param
        drop_path_rate=0.4,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=16,
        cffn_ratio=0.25,
        deform_ratio=0.5,
        interaction_indexes=[[0, 5], [6, 11], [12, 17], [18, 23]],
    ),
    decode_head=dict(num_classes=150, in_channels=[1024, 1024, 1024, 1024]),
    auxiliary_head=dict(num_classes=150, in_channels=1024),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01,
                 constructor='MyLayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.9))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2,
          train=dict(pipeline=train_pipeline),
          val=dict(pipeline=test_pipeline),
          test=dict(pipeline=test_pipeline))
runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=16000, metric='mIoU', save_best='mIoU')
fp16 = dict(loss_scale=dict(init_scale=512))
