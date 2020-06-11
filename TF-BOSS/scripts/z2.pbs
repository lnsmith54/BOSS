#!/bin/bash
#PBS -l walltime=xTime:00:00
#PBS -l select=1:ncpus=12:mpiprocs=12:ngpus=2
#PBS -l place=scatter:excl
#PBS -N TFBOSS2SxSeed0
#PBS -j oe
#PBS -V
#PBS -q standard
#PBS -A ERDCS97260VU3

module load python36
module load openmpi/cuda/64/3.1.4
module load tensorflow-py36-cuda10.1-gcc/1.14.0

cd $PBS_O_WORKDIR
cd ..
pwd
cp -r BOSS /tmp/
cd /tmp/BOSS
export ML_DATA="/tmp/BOSS/data"
export PYTHONPATH=$PYTHONPATH:$PWD

date

echo "FixMatch cifar10.xSeed0@xSize0-xValid0IterxKimg0UxUratio0CxConf0WDxWd0BSxBatch0LRxLr0AxAug0_xRep0"
CUDA_VISIBLE_DEVICES=xRep0 python fixmatch.py --train_kimg xKimg0 --uratio xUratio0 --confidence xConf0 --wd xWd0 --wu 1  --batch xBatch0 --lr xLr0  --filters=32 --scales 3 --repeat 4 --dataset=cifar10.xSeed0@xSize0-xValid0 --train_dir experiments/fixmatch/SxSeed0@xSize0-xValid0IterxKimg0UxUratio0CxConf0WDxWd0BSxBatch0LRxLr0AxAug0_xRep0 --augment xAug0  > $PBS_O_WORKDIR/results/cifar10.xSeed0@xSize0-xValid0IterxKimg0UxUratio0CxConf0WDxWd0BSxBatch0LRxLr0AxAug0_xRep0 &

echo "FixMatch cifar10.xSeed0@xSize0-xValid0IterxKimg0UxUratio0CxConf0WDxWd0BSxBatch0LRxLr0AxAug0_xRep1"
CUDA_VISIBLE_DEVICES=xRep1 python fixmatch.py --train_kimg xKimg0 --uratio xUratio0 --confidence xConf0 --wd xWd0 --wu 1  --batch xBatch0 --lr xLr0  --filters=32 --scales 3 --repeat 4 --dataset=cifar10.xSeed0@xSize0-xValid0 --train_dir experiments/fixmatch/SxSeed0@xSize0-xValid0IterxKimg0UxUratio0CxConf0WDxWd0BSxBatch0LRxLr0AxAug0_xRep1 --augment xAug0  > $PBS_O_WORKDIR/results/cifar10.xSeed0@xSize0-xValid0IterxKimg0UxUratio0CxConf0WDxWd0BSxBatch0LRxLr0AxAug0_xRep1

wait

date

exit
