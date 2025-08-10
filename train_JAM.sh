#!/usr/bin/env bash
# -------------------------------------------------- #
GPUS=4                                              #    
# -------------------------------------------------- #
GPUS_PER_NODE=$(($GPUS<8?$GPUS:8))

MASTER_PORT=${MASTER_PORT:-28536}

# export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nproc_per_node=$GPUS_PER_NODE \
        --master_port=$MASTER_PORT \
        JAM/train_JAM.py \
        --batch_size=64 \
        --training_epochs=30 \
        --class_query_topK=1 \
        --class_query_N=64 \
        --train_set=/JAM/waymo_dataset_1_2/processed_train \
        --valid_set=/JAM/waymo_dataset_1_2/processed_valid \
        --save_path=/JAM/jam_log \
        --name=jam \
        --workers=8
