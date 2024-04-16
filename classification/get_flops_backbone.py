# Copyright (c) OpenMMLab. All rights reserved.
import sys
import argparse

import torch
from mmcv import Config
from mmcv.cnn.utils import get_model_complexity_info
from mmcv.cnn.utils.flops_counter import flops_to_string, params_to_string

from mmcls.models import build_classifier
import mmcls_custom

TIME_MIX_EXTRA_DIM = 32
TIME_DECAY_EXTRA_DIM = 64


def parse_args():
    parser = argparse.ArgumentParser(description='Get model flops and params')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[224, 224],
        help='input image size')
    parser.add_argument(
        '--out',
        type=str
    )
    args = parser.parse_args()
    return args


def attn_flops(h, w, dim):
    return 2 * (h * w) * (h * w) * dim


def vrwkv_flops(n, dim):
    return n * dim * 29


def vrwkv6_flops(n, dim, head_size):
    addi_flops = 0
    addi_flops += n * dim * (TIME_MIX_EXTRA_DIM * 10 + TIME_DECAY_EXTRA_DIM * 2 + 7 * head_size + 17)
    addi_flops += n * (TIME_MIX_EXTRA_DIM * 5 + TIME_DECAY_EXTRA_DIM)
    return addi_flops


def get_addi_flops_vit(model, input_shape, cfg):
    _, H, W = input_shape
    try:
        patch_size = cfg.model.backbone.patch_size
    except:
        patch_size = 16
    h, w = H / patch_size, W / patch_size
    emb_dims = model.backbone.embed_dims
    addi_flops = 0
    addi_flops += (len(model.backbone.layers) * attn_flops(h, w, emb_dims))
    print(f"Additional Flops in Vit(Attn)*{len(model.backbone.layers)}: {flops_to_string(addi_flops)}")
    return addi_flops


def get_addi_flops_vrwkv6(model, input_shape, cfg):
    _, H, W = input_shape
    try:
        patch_size = cfg.model.backbone.patch_size
    except:
        patch_size = 16
    h, w = H / patch_size, W / patch_size

    model_name = type(model.backbone).__name__
    embed_dims = model.backbone.embed_dims
    head_size = embed_dims // cfg.model.backbone.num_heads
    print(f"Head Size in VRWKV6: {head_size}")
    num_layers = len(model.backbone.layers)
    addi_flops = 0
    addi_flops += (num_layers * vrwkv6_flops(h*w, embed_dims, head_size))
    print(f"Additional Flops in VRWKV6*{num_layers} layers: {flops_to_string(addi_flops)}")
    return addi_flops


def get_addi_flops_vrwkv(model, input_shape, cfg):
    _, H, W = input_shape
    try:
        patch_size = cfg.model.backbone.patch_size
    except:
        patch_size = 16
    h, w = H / patch_size, W / patch_size

    model_name = type(model.backbone).__name__
    embed_dims = model.backbone.embed_dims
    num_layers = len(model.backbone.layers)
    addi_flops = 0
    addi_flops += (num_layers * vrwkv_flops(h*w, embed_dims))
    print(f"Additional Flops in VRWKV(Attn)*{num_layers} layers: {flops_to_string(addi_flops)}")
    return addi_flops


def get_flops(model, input_shape, cfg, ost):
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False, ost=ost)
    model_name = type(model.backbone).__name__
    if 'VisionTransformer' in model_name:
        add_flops = get_addi_flops_vit(model, input_shape, cfg)
        flops += add_flops
    elif model_name == 'VRWKV':
        add_flops = get_addi_flops_vrwkv(model, input_shape, cfg)
        flops += add_flops
    elif model_name == 'VRWKV6':
        add_flops = get_addi_flops_vrwkv6(model, input_shape, cfg)
        flops += add_flops
    return flops_to_string(flops), params_to_string(params)


def main():
    args = parse_args()

    if len(args.shape) == 1:
        input_shape = (3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (3, ) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    cfg = Config.fromfile(args.config)
    model = build_classifier(cfg.model)
    if torch.cuda.is_available():
        model.cuda()
    else:
        raise AssertionError
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    if args.out is not None:
        f = open(args.out, 'w')
        ost = f
        sys.stdout = f
    else:
        ost = sys.stdout
    flops, params = get_flops(model, input_shape, cfg, ost)

    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
    if args.out is not None:
        f.close()


if __name__ == '__main__':
    main()