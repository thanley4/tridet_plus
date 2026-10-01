#!/bin/bash

CUDA_VISIBLE_DEVICES=$1 python train.py ./configs/driving.yaml --ckpt-freq 2 --output pretrain
# echo "start testing14..."
CUDA_VISIBLE_DEVICES=$1 python eval.py ./configs/driving.yaml ckpt/driving_pretrain/epoch_008.pth.tar

