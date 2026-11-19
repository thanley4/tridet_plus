#!/bin/bash

# Train
python train.py ./configs/driving.yaml --ckpt-freq 5 --output pretrain

# Evaluate
python eval.py ./configs/driving.yaml ckpt/driving_pretrain/epoch_049.pth.tar

# Evaluate all epochs
python eval_all_epochs.py ./configs/driving.yaml ckpt/driving_pretrain/