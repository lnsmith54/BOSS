Seed0=$1

module load python36
module load openmpi/cuda/64/3.1.4
module load tensorflow-py36-cuda10.1-gcc/1.15.2

export ML_DATA="./data/pseudolabeled"
export PYTHONPATH=$PYTHONPATH:$PWD

CUDA_VISIBLE_DEVICES= scripts/iteration1.py --seed=$Seed0 --size=50 --balance $2  --pseudo_file $3  $ML_DATA/SSL2/cifar10p $ML_DATA/cifar10p-train.tfrecord 

CUDA_VISIBLE_DEVICES= scripts/iteration1.py --seed=$Seed0 --size=100 --balance $2  --pseudo_file $3 $ML_DATA/SSL2/cifar10p $ML_DATA/cifar10p-train.tfrecord 

CUDA_VISIBLE_DEVICES= scripts/iteration1.py --seed=$Seed0 --size=200 --balance $2  --pseudo_file $3 $ML_DATA/SSL2/cifar10p $ML_DATA/cifar10p-train.tfrecord 

CUDA_VISIBLE_DEVICES= scripts/iteration1.py --seed=$Seed0 --size=400 --balance $2  --pseudo_file $3 $ML_DATA/SSL2/cifar10p $ML_DATA/cifar10p-train.tfrecord

wait

exit
