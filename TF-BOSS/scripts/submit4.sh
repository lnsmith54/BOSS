dataset=cifar10
size=10
seed=1
valid=5000

kimg=32768  # 65536
ratio=7
con=0.95
wd=5e-4
batch=64
lr=0.04
i=3
aug="d.d.d"
#seed=(0 1 2 3 4 5)
seed=1

echo " "  >> history
for seed in 1 5; do
#i2=`expr $i + 1`
#echo $i2
#for con  in 0.99 0.98; do
#for ratio in 7 8 16; do
#for batch  in 64 128 256; do
#for lr  in 0.1 0.06 0.01; do
#for wd  in 7e-4 5e-4 3e-4 1e-4; do
#for kimg in 8192 32768; do
#for aug in "rac.m.rac" "d.m.rac" "aac.m.aac" "d.m.aac"; do
#for i  in 1 2 3; do
    filename="${dataset}.${seed}@${size}-${valid}Iter${kimg}U${ratio}C${con}WD${wd}BS${batch}LR${lr}Aug${aug}_0"
    echo $filename
    echo $filename >> history
    sed -e "s/xSeed0/${seed}/g" -e "s/xSize0/${size}/g" -e "s/xValid0/${valid}/g" -e "s/xTime/120/g" -e "s/xKimg0/$kimg/g" -e "s/xUratio0/$ratio/g" -e "s/xConf0/$con/g" -e "s/xWd0/$wd/g" -e "s/xBatch0/$batch/g" -e "s/xLr0/$lr/g" -e "s/xRep0/0/g"  -e "s/xRep1/1/g" -e "s/xAug0/$aug/g" z2.pbs > Q/$filename
#    exit
    qsub  Q/$filename
    sleep 1
    sed -e "s/xSeed0/${seed}/g" -e "s/xSize0/${size}/g" -e "s/xValid0/${valid}/g" -e "s/xTime/120/g" -e "s/xKimg0/$kimg/g" -e "s/xUratio0/$ratio/g" -e "s/xConf0/$con/g" -e "s/xWd0/$wd/g" -e "s/xBatch0/$batch/g" -e "s/xLr0/$lr/g" -e "s/xRep0/2/g"  -e "s/xRep1/3/g" -e "s/xAug0/$aug/g" z2.pbs > Q/$filename
#    exit
    qsub  Q/$filename
    sleep 1
done
#done

exit
