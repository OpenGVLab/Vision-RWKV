# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

pretrained = 'pretrained/vrwkv_l_in22k_192.pth'
# https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_l_in22k_192.pth

model = dict(
    backbone=dict(
        _delete_=True,
        type='VRWKV_Adapter',
        img_size=192,
        patch_size=16,
        embed_dims=1024,
        depth=24,
        pretrained=pretrained,
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
    neck=dict(
        type='FPN',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        num_outs=5))
# optimizer

# 8 gpus
data = dict(samples_per_gpu=2,
            workers_per_gpu=2)

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0001, weight_decay=0.05,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=24, layer_decay_rate=0.85))
# optimizer_config = dict(grad_clip=None)
# fp16 = dict(loss_scale=dict(init_scale=512))
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=1,
    save_last=True,
)