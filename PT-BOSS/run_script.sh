#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train.py  --n-labeled 10 --n-epochs 512 --batchsize 32 --mu 7 --thr 0.95 --lam-u 1 --lr 0.04 --weight-decay 5e-4 --momentum 0.85 --seed 6 --balance 4 

