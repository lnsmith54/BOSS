
module load python36
module load openmpi/cuda/64/3.1.4
module load tensorflow-py36-cuda10.1-gcc/1.15.2

export ML_DATA="./data"
export PYTHONPATH=$PYTHONPATH:$PWD

date
size=10
for seed in 1 2 3 4 5; do
        CUDA_VISIBLE_DEVICES= scripts/cifar10_prototypes.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10p $ML_DATA/cifar10-train.tfrecord
done

date

exit

