# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

pretrained = 'pretrained/vrwkv_s_in1k_224.pth'
# https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_s_in1k_224.pth

model = dict(
    backbone=dict(
        _delete_=True,
        type='VRWKV_Adapter',
        img_size=224,
        patch_size=16,
        embed_dims=384,
        depth=12,
        pretrained=pretrained,
        init_values=1e-5,
        post_norm=True,
        with_cp=False,
        # adapter param
        drop_path_rate=0.2,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        ),
    neck=dict(
        type='FPN',
        in_channels=[384, 384, 384, 384],
        out_channels=256,
        num_outs=5))
# optimizer

# 8 gpus
data = dict(samples_per_gpu=2,
            workers_per_gpu=2)

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.85))
optimizer_config = dict(grad_clip=None)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=2,
    save_last=True,
)