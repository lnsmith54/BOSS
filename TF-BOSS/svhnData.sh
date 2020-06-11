
export ML_DATA="./data"
export PYTHONPATH=$PYTHONPATH:$PWD

date
size=10

cp $ML_DATA/svhn_noextra-test.tfrecord $ML_DATA/svhnp_noextra-test.tfrecord
cp $ML_DATA/svhn-extra.tfrecord $ML_DATA/svhnp-extra.tfrecord
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhnp-test.tfrecord
cp $ML_DATA/svhn-train.tfrecord $ML_DATA/svhnp-train.tfrecord
cp $ML_DATA/svhn-test.tfrecord $ML_DATA/svhnp_noextra-test.tfrecord
cp $ML_DATA/svhn-train.tfrecord $ML_DATA/svhnp_noextra-train.tfrecord

mkdir data/pseudolabeled
mkdir data/pseudolabeled/top
cp data/svhnp-train.tfrecord data/pseudolabeled/

cp $ML_DATA/SSL2/svhn-unlabel.json  $ML_DATA/SSL2/svhnp-unlabel.json
cp $ML_DATA/SSL2/svhn-unlabel.tfrecord  $ML_DATA/SSL2/svhnp-unlabel.tfrecord
cp $ML_DATA/SSL2/svhn_noextra-unlabel.json  $ML_DATA/SSL2/svhnp_noextra-unlabel.json
cp $ML_DATA/SSL2/svhn_noextra-unlabel.tfrecord  $ML_DATA/SSL2/svhnp_noextra-unlabel.tfrecord


size=10

for seed in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES= scripts/svhn_prototypes.py --seed=$seed --size=$size $ML_DATA/SSL2/svhnp $ML_DATA/svhnp-train.tfrecord  $ML_DATA/svhnp-extra.tfrecord 
#    CUDA_VISIBLE_DEVICES= scripts/svhn_prototypes.py --seed=$seed --size=$size $ML_DATA/SSL2/svhnp_noextra $ML_DATA/svhnp-train.tfrecord &
done
wait



date
exit

