#!/bin/bash

path="src/preprocessing"
upload_users=("deepseek-ai" "deepseek-ai")
model_types=("deepseek-math-7b-instruct" "deepseek-math-7b-rl")
length=${#model_types[@]}
dataset_modes="train test"

for ((i=0; i<$length; i++))
do
    upload_user=${upload_users[$i]}
    model_type=${model_types[$i]}
    python $path/merge_tokenizer.py upload_user=$upload_user model_type=$model_type
    python $path/merge_model.py upload_user=$upload_user model_type=$model_type
    for dataset_mode in $dataset_modes
    do
        python $path/preprocess_dataset.py mode=$dataset_mode upload_user=$upload_user model_type=$model_type
    done
done
