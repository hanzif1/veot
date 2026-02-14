#!/bin/bash

set -e

# 设置 GPU (根据你的机器实际情况修改，例如 0,1,2,3,4,5,6,7)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_RT_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export PYTHONPATH="./:$PYTHONPATH"

# 接收命令行参数
dataset=$1
split=${2:-"test"}

# 模型路径 (这里以 7B 为例，如果是 2B 请修改为 model_zoo/VideoMind-2B)
model_gnd_path="model_zoo/VideoMind-2B"
# 注意：如果是纯 Training-Free 的 Grid-VLM，其实不需要 verifier/planner 的路径
# 但为了兼容代码逻辑，这里还是保留
model_ver_path="model_zoo/VideoMind-2B"
model_pla_path="model_zoo/VideoMind-2B"

# ==================== [消融实验核心配置] ====================
# 定义要测试的 Grid Side 列表
# 1 -> 1帧 (1x1)
# 2 -> 4帧 (2x2)
# 3 -> 9帧 (3x3) -> 默认
# 4 -> 16帧 (4x4)
# 8 -> 64帧 (8x8)
# 16 -> 256帧 (16x16)
GRID_SIDES=(1 2 3 4 8 16)

# 结果汇总日志文件
LOG_FILE="ablation_results_${dataset}_${split}.log"
echo "Ablation Study Log - $(date)" > $LOG_FILE
echo "Dataset: $dataset, Split: $split" >> $LOG_FILE
echo "======================================================" >> $LOG_FILE

echo -e "\e[1;36mStarting Ablation Study on Dataset:\e[0m $dataset ($split)"

# ==================== [开始循环实验] ====================
for K in "${GRID_SIDES[@]}"; do
    num_frames=$((K * K))
    echo -e "\n\e[1;33m[Running Experiment] Grid: ${K}x${K} (${num_frames} frames)\e[0m"
    
    # 1. 定义独立的输出目录，防止结果覆盖
    pred_path="outputs_ablation22/${dataset}_${split}_grid${K}x${K}"
    
    # 2. 多卡并行推理
    IFS="," read -ra GPULIST <<< "${CUDA_VISIBLE_DEVICES:-0}"
    CHUNKS=${#GPULIST[@]}

    for IDX in $(seq 0 $((CHUNKS-1))); do
        # 注意：这里调用的是你刚才修改过的 infer_auto_spatial.py
        # 如果你文件名没改，还是 infer_auto.py，请相应修改这里
        CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} ASCEND_RT_VISIBLE_DEVICES=${GPULIST[$IDX]} python videomind/eval/infer_auto_spatial.py \
            --dataset $dataset \
            --split $split \
            --pred_path $pred_path \
            --model_gnd_path $model_gnd_path \
            --model_ver_path $model_ver_path \
            --model_pla_path $model_pla_path \
            --chunk $CHUNKS \
            --index $IDX \
            --grid_side $K &  # <--- 传入当前循环的 grid_side 参数
    done

    wait # 等待所有 GPU 进程完成

    # 3. 运行评估并记录结果
    echo -e "\e[1;32m[Inference Complete] Evaluating Grid ${K}x${K}...\e[0m"
    
    echo "Results for Grid ${K}x${K}:" >> $LOG_FILE
    # 执行评估脚本，并将输出追加到日志文件
    python videomind/eval/eval_auto.py $pred_path --dataset $dataset | tee -a $LOG_FILE
    
    echo "------------------------------------------------------" >> $LOG_FILE
done

echo -e "\n\e[1;36mAll Ablation Experiments Finished!\e[0m"
echo -e "Check results in: \e[1;32m$LOG_FILE\e[0m"