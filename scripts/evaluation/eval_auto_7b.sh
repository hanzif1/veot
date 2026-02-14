#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,4,7
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8
export PYTHONPATH="./:$PYTHONPATH"

dataset=$1
split=${2:-"test"}

model_gnd_path="model_zoo/VideoMind-7B"
model_ver_path="model_zoo/VideoMind-7B"
model_pla_path="model_zoo/VideoMind-7B"

pred_path="outputs_7b_new/${dataset}_${split}"

echo -e "\e[1;36mEvaluating:\e[0m $dataset ($split)"

IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python videomind/eval/infer_auto_new.py \
        --dataset $dataset \
        --split $split \
        --pred_path $pred_path \
        --model_gnd_path $model_gnd_path \
        --model_ver_path $model_ver_path \
        --model_pla_path $model_pla_path \
        --chunk $CHUNKS \
        --index $IDX &
done

wait

python videomind/eval/eval_auto.py $pred_path --dataset $dataset
