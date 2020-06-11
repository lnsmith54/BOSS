
# BOSS

## Dataset
## Before training, download cifar-10 dataset: 

    mkdir -p dataset && cd dataset
    wget -c http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz

## Train the model; parameters including balance method and seed, can be set in the train.py flags
    bash run_script.sh 
