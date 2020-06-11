export ML_DATA="./data"
export PYTHONPATH=$PYTHONPATH:$PWD

cp $ML_DATA/cifar10-test.tfrecord $ML_DATA/cifar10p-test.tfrecord
cp $ML_DATA/cifar10-train.tfrecord $ML_DATA/cifar10p-train.tfrecord

mkdir data/pseudolabeled
mkdir data/pseudolabeled/top
cp data/cifar10p-train.tfrecord data/pseudolabeled/

mkdir data/pseudolabeled/SSL2
cp data/cifar10p-train.tfrecord data/pseudolabeled/
cp data/cifar10p-train.tfrecord data/pseudolabeled/cifar10pB1-train.tfrecord
cp data/cifar10p-test.tfrecord  data/pseudolabeled/cifar10pB1-test.tfrecord
cp data/SSL2/cifar10p-unlabel.* data/pseudolabeled/SSL2/

cp data/pseudolabeled/SSL2/cifar10p-unlabel.json data/pseudolabeled/SSL2/cifar10pB1-unlabel.json
cp data/pseudolabeled/SSL2/cifar10p-unlabel.tfrecord data/pseudolabeled/SSL2/cifar10pB1-unlabel.tfrecord
cp data/pseudolabeled/SSL2/cifar10p-unlabel.json data/pseudolabeled/SSL2/cifar10pB4-unlabel.json
cp data/pseudolabeled/SSL2/cifar10p-unlabel.tfrecord data/pseudolabeled/SSL2/cifar10pB4-unlabel.tfrecord

cp $ML_DATA/SSL2/cifar10-unlabel.json  $ML_DATA/SSL2/cifar10p-unlabel.json
cp $ML_DATA/SSL2/cifar10-unlabel.tfrecord  $ML_DATA/SSL2/cifar10p-unlabel.tfrecord


date
size=10
for seed in 1 2 3 4 5 6 7; do
#        CUDA_VISIBLE_DEVICES= scripts/cifar10_prototypes.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10p $ML_DATA/cifar10-train.tfrecord
	python scripts/cifar10_prototypes.py --seed=$seed --size=$size $ML_DATA/SSL2/cifar10p $ML_DATA/cifar10-train.tfrecord
done

date

exit

