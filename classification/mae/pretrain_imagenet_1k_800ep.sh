#!/usr/bin/env bash

set -x
PARTITION=$1

# eff bs = 8 * 128 * 4 = 4096
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
MASTER_PORT=30001 srun -p ${PARTITION} \
    --gres=gpu:8 \
    --cpus-per-task=16 \
    --quotatype=spot \
    python -m torch.distributed.launch \
    --nproc_per_node 8 --master_port 30001 main_pretrain.py \
    --dataset imagenet \
    --data_path ./data/imagenet/ \
    --model 'mae_vrwkv_base_patch16' \
    --batch_size 128 \
    --accum_iter 4 \
    --epochs 800 \
    --warmup_epochs 10 \
    --input_size 224 \
    --mask_ratio 0.75 \
    --blr 1.5e-4  \
    --weight_decay 0.05 \
    --gpu_num 8 \
    --output_dir './output_dir/'
