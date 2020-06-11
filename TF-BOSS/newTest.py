#import tensorflow as tf
import numpy as np

train_dir = 'experiments/BOSS/dbg/cifar10p.d.d.d.1@10-1/CTAugment_depth2_th0.80_decay0.990/FixMatch_archresnet_balance2_batch32_confidence0.95_delT0.0_filters32_lr0.04_mom0.88_nclass10_repeat4_scales3_uratio7_wc0.0_wd0.0004_wu1.0'
print(train_dir)

dir = train_dir.replace("BOSS","preds")
print(dir)

exit(1)
