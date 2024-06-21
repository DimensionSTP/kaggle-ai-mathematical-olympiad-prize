#!/bin/bash

is_preprocessed=False
is_tuned="untuned"
strategy="deepspeed_stage_3_offload"
upload_user="meta-math"
model_type="MetaMath-7B-V1.0"
quantization_type="origin"
peft_type="origin"
data_max_length=508
target_max_length=4
precision="bf16"
batch_size=16

python main.py mode=train \
    is_preprocessed=$is_preprocessed \
    is_tuned=$is_tuned \
    strategy=$strategy \
    upload_user=$upload_user \
    model_type=$model_type \
    quantization_type=$quantization_type \
    peft_type=$peft_type \
    data_max_length=$data_max_length \
    target_max_length=$target_max_length \
    precision=$precision \
    batch_size=$batch_size
