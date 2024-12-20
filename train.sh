#!/bin/bash

dt=$(date '+%Y%m%d_%H%M%S')
dataset="mooc"
mode="train"

# training parameters
epochs=500  # Number of epochs
batch_size=4  # Batch size
lr=0.000001  # Learning rate
alpha=0.5  # Alpha value for loss weighting
beta=0.5  # Beta value for loss weighting
seed=25  # Random seed

# output parameters
echo "***** Training *****"
echo "Dataset: $dataset"
echo "Epochs: $epochs"
echo "Batch Size: $batch_size"
echo "Learning Rate: $lr"
echo "Alpha: $alpha"
echo "Beta: $beta"
echo "******************************"

save_dir_name="model_${dataset}_${dt}"
log="logs/${mode}_${dataset}_${save_dir_name}.log.txt"

##### Training ######
python3 -u train.py \
    --epochs $epochs \
    --batch_size $batch_size \
    --lr $lr \
    --alpha $alpha \
    --beta $beta \
    --seed $seed \
    > ${log} 2>&1 &

echo "Log: ${log}"
