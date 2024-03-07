# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import json
from torch import optim as optim
from torch.distributed.optim import ZeroRedundancyOptimizer


def build_optimizer(config, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    # build optimizer with layer-wise lr decay (lrd)
    if config.MODEL.TYPE == 'vrwkv':
        param_groups = param_groups_lrd_vrwkv(model, config.TRAIN.WEIGHT_DECAY,
            no_weight_decay_list=[],
            layer_decay=config.TRAIN.LR_LAYER_DECAY_RATIO if config.TRAIN.LR_LAYER_DECAY else 1,
        )
    else:
        raise NotImplementedError

    opt_lower = config.TRAIN.OPTIMIZER.NAME.lower()
    optimizer = None

    if opt_lower == 'adamw':
        optimizer = optim.AdamW(param_groups,
                                lr=config.TRAIN.BASE_LR)
    else:
        raise NotImplementedError

    return optimizer


def param_groups_lrd_vrwkv(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.75):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
    num_layers = len(model.backbone.layers) + 1

    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if p.ndim == 1 or n in no_weight_decay_list:
            if 'spatial_decay' in n or 'spatial_first' in n: 
                g_decay = 'decay'
                this_decay = weight_decay
            else:
                g_decay = 'no_decay'
                this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
            
        layer_id = get_layer_id_for_vrwkv(n, num_layers)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }

        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))

    return list(param_groups.values())


def get_layer_id_for_vrwkv(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['backbone.cls_token', 'backbone.pos_embed']:
        return 0
    elif name.startswith('backbone.patch_embed'):
        return 0
    elif name.startswith('backbone.blocks') or name.startswith('backbone.layers'):
        return int(name.split('.')[2]) + 1
    else:
        return num_layers
