dataset=cifar10p
size=10
seed=1
valid=5000

kimg=65536
ratio=7
con=0.95
wd=5e-4
batch=64
lr=0.03
i=0
aug="d.d.d"

#for con  in 0.95 0.9; do
#for ratio in 7 8 16; do
#for batch  in 32 64 128; do
#for lr  in 0.03 0.06 0.01; do
#for wd  in 5e-4 3e-4 2e-4; do
#for kimg in 32768 65536; do
#for aug in "rac.m.rac" "d.m.rac" "aac.m.aac" "d.m.aac" "d.d.aac" "d.d.rac"; do
#for i  in 1 2 3; do
for seed  in 0 1 2 3 4 5; do
    filename="cifar10${dataset}.${seed}@${size}-${valid}Iter${kimg}U${ratio}C${con}WD${wd}BS${batch}LR${lr}Aug${aug}_$i"
    echo $filename
    echo $filename >> history
    sed -e "s/xSeed/${seed}/g" -e "s/xSize/${size}/g" -e "s/xValid/${valid}/g" -e "s/xTime/120/g" -e "s/xKimg/$kimg/g" -e "s/xUratio/$ratio/g" -e "s/xConf/$con/g" -e "s/xWd/$wd/g" -e "s/xBatch/$batch/g" -e "s/xLr/$lr/g" -e "s/xRep/$i/g" -e "s/xAug/$aug/g" z.pbs > Q/$filename
#    exit
    qsub  Q/$filename
    sleep 1
done
#done

exit

for seed in 0 1 2 3 4 5; do
for valid in 1 5000; do
    filename="cifar10${dataset}.${seed}@${size}-${valid}"
    echo "${dataset}.${seed}@${size}-${valid}"
    sed -e "s/xSeed/${seed}/g" -e "s/xSize/${size}/g" -e "s/xValid/${valid}/g" -e "s/xTime/72/g" z.pbs > Q/$filename
#    exit
    qsub  Q/$filename
    sleep 1
done; done
