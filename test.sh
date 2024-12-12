#!/bin/bash

dt=$(date '+%Y%m%d_%H%M%S')
dataset="mooc"
mode="test"

# testing parameters
batch_size=4  # Batch size
seed=25  # Random seed

# output parameters
echo "***** Testing *****"
echo "Dataset: $dataset"
echo "Batch Size: $batch_size"
echo "******************************"

save_dir_name="model_${dataset}_${dt}"
log="logs/${mode}_${dataset}_${save_dir_name}.log.txt"

##### Testing ######
python3 -u test.py \
    --batch_size $batch_size \
    --seed $seed \
    > ${log} 2>&1 &

echo "Log: ${log}"
