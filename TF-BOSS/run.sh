#!/bin/bash                                                                                                                                                                                                 
export ML_DATA="./data"
export PYTHONPATH=$PYTHONPATH:$PWD

bal=4  #set balance method, 1,2,3 or 4                                                                                                                                                                      
seed=6 #set seed choose number 1 to 7                                                                                                                                                                       
ds=cifar10 #set dataset; either cifar10 or svhn                                                                                                                                                             

iterSize=400 # set to the number of pseudolabeled used for the self-training iteration.

dataset=${ds}p; size=10; valid=1; time=48; arch=resnet
kimg=32768; aug="d.d.d"; mom=0.88; wu=1;  #hyperparameters                                                                                                                                                  

datastring='B'$bal'S'$seed'.npy'
datastring2=${dataset}.${seed}@${size}.npy

if [ $bal -eq 0 ];
then
  delt=0; con=0.95;wd=5e-4;lr=0.03; ratio=7;  batch=64
elif [ $bal -eq 1 ];
then
  delt=0.25; con=0.95;wd=8e-4;lr=0.06; ratio=9;  batch=30
elif [ $bal -eq 2 ];
then
  delt=0; con=0.9;wd=8e-4;lr=0.06; ratio=9;  batch=30
elif [ $bal -eq 3 ];
then
  delt=0; con=0.9;wd=8e-4;lr=0.04; ratio=9;  batch=30
elif [ $bal -eq 4 ];
then
  delt=0.25; con=0.95;wd=8e-4;lr=0.06; ratio=9;  batch=30
else
  delt=0; con=0.95;wd=5e-4;lr=0.03; ratio=7;  batch=64
fi

echo "Running BOSS for balance = " $bal
CUDA_VISIBLE_DEVICES=0 python fixmatch.py --train_kimg $kimg --uratio $ratio --confidence $con --wd $wd --wu $wu  --batch $batch --lr $lr  --arch $arch --filters 32 --scales 3 --repeat 4 --dataset=${dataset}.${seed}@${size}-${valid}  --delT $delt --train_dir experiments/BOSS/ --augment $aug --mom $mom --balance $bal

mv top_probs$datastring ./data/pseudolabeled/top/Probs$datastring2
mv top_labels$datastring ./data/pseudolabeled/top/Labels$datastring2
mv true_labels$datastring ./data/pseudolabeled/top/TrueLabels$datastring2


export ML_DATA="./data/pseudolabeled" #change to pseudolabeled data                                                                                                                                         

echo "Running " ${ds}_iteration.py
python scripts/${ds}_iteration.py --seed=$seed --size=$iterSize --pseudo_file data/pseudolabeled/top/Probs$datastring2 $ML_DATA/SSL2/$dataset $ML_DATA/${dataset}-train.tfrecord

dataset2=${dataset};
iterSize2=$(($size + $iterSize))
bal=0; con=0.95;delt=0;  mom=0.9; ratio=7; wd=5e-4; batch=64; lr=0.03; #these hyperparameters change   
datastring4=${dataset2}.${seed}@${iterSize2}

echo "Running self-training  "
CUDA_VISIBLE_DEVICES=0 python fixmatch.py --train_kimg $kimg --uratio $ratio --confidence $con --wd $wd --wu $wu  --batch $batch --lr $lr  --arch $arch --filters 32 --scales 3 --repeat 4 --dataset=${datastring4}-${valid} --delT $delt --train_dir experiments/BOSS/iter --augment $aug --mom $mom --balance $bal
