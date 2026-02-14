#!/bin/bash

set -e

# ==================== 环境变量配置 ====================
# 默认使用 8 张卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_ALLOC_CONF=expandable_segments:True
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8
export PYTHONPATH="./:$PYTHONPATH"

# ==================== 实验参数设置 ====================
# 数据集固定为 mvbench
dataset="mvbench"
split="test"

# 模型路径 (根据你的实际情况修改，这里默认用 2B，如果要用 7B 请改路径)
model_gnd_path="model_zoo/VideoMind-2B"
model_ver_path="model_zoo/VideoMind-2B"
model_pla_path="model_zoo/VideoMind-2B"

# 固定网格大小为 3x3
GRID_SIDE=3

# 定义要测试的迭代深度 (Depth)
# 1: 仅一次全景搜索 (Baseline)
# 2: 全景 -> 放大1次 (Ours Default)
# 3: 全景 -> 放大 -> 再放大 (Stress Test)
DEPTHS=(1 2 3)

# 日志文件
LOG_FILE="ablation_depth_mvbench.log"
echo "=== Ablation Study: Search Depth on MVBench (Grid 3x3) ===" > $LOG_FILE
echo "Time: $(date)" >> $LOG_FILE
echo "Model: $model_gnd_path" >> $LOG_FILE
echo "--------------------------------------------------------" >> $LOG_FILE

echo -e "\e[1;36mStarting Ablation Study on MVBench with 3x3 Grid...\e[0m"

# ==================== 循环运行实验 ====================
for D in "${DEPTHS[@]}"; do
    echo -e "\n\e[1;33m[Running] Depth = $D (Grid 3x3)...\e[0m"
    
    # 定义独立的输出目录，防止覆盖
    pred_path="outputs_ablation/${dataset}_grid${GRID_SIDE}x${GRID_SIDE}_depth${D}"
    
    # 多卡并行推理
    IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
    CHUNKS=${#GPULIST[@]}

    for IDX in $(seq 0 $((CHUNKS-1))); do
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python videomind/eval/infer_auto_spatial_2.py \
            --dataset $dataset \
            --split $split \
            --pred_path $pred_path \
            --model_gnd_path $model_gnd_path \
            --model_ver_path $model_ver_path \
            --model_pla_path $model_pla_path \
            --chunk $CHUNKS \
            --index $IDX \
            --grid_side $GRID_SIDE \
            --search_depth $D & 
    done

    wait # 等待所有 GPU 任务完成

    # 运行评估并记录结果
    echo -e "\e[1;32m[Evaluating] Depth = $D...\e[0m"
    
    echo ">>> Results for Depth $D (Grid 3x3):" >> $LOG_FILE
    # 执行评估，结果同时输出到屏幕和日志
    python videomind/eval/eval_auto.py $pred_path --dataset $dataset | tee -a $LOG_FILE
    
    echo "--------------------------------------------------------" >> $LOG_FILE
done

echo -e "\n\e[1;36mAll Done! Summary saved to $LOG_FILE\e[0m"