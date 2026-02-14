#!/bin/bash
set -e

# # === 显卡占用检测模块 ===
# TARGET_GPU=5
# echo "🔍 Checking status of GPU ${TARGET_GPU}..."

# while true; do
#     # 查询指定 GPU 上的计算进程 PID
#     # -i: 指定显卡ID
#     # --query-compute-apps=pid: 只查询计算进程的PID
#     # --format=csv,noheader: 格式化输出，去掉表头
#     pids=$(nvidia-smi -i $TARGET_GPU --query-compute-apps=pid --format=csv,noheader)

#     # 判断 pids 字符串是否为空 (-z)
#     if [ -z "$pids" ]; then
#         echo "✅ GPU ${TARGET_GPU} is free! Starting tasks..."
#         break
#     else
#         # 获取当前时间
#         now=$(date "+%Y-%m-%d %H:%M:%S")
#         # 如果不为空，说明有进程在跑，打印提示并等待
#         # echo $pids | tr '\n' ' ' 用于把多行PID变成一行显示
#         echo "[$now] ⏳ GPU ${TARGET_GPU} is busy (PIDs: $(echo $pids | tr '\n' ' ')). Waiting 30s..."
#         sleep 30
#     fi
# done
# # ========================

# === 配置区域 ===
DATASET=$1
SPLIT=${2:-"test"}

# [修改点]：将输出路径改为包含 "retry" 的新文件夹
PRED_PATH="outputs_agent_retry/${DATASET}_${SPLIT}"

mkdir -p $PRED_PATH
LOG_DIR="logs_agent"
mkdir -p $LOG_DIR

# 获取 Python 路径
PYTHON_BIN=$(which python)
echo "Using Python: $PYTHON_BIN"


echo "=========================================================="
echo "🚀 Starting vLLM Servers on GPU 1-6..."
echo "=========================================================="

# === 1. 启动 Planner (文本模型) ===
# 替换为 Int4 版本，TP=1 (单卡即可跑飞起)
# 显卡：使用 GPU 1
echo "Starting Planner (Int4) on GPU 1..."
INT4_MODEL="/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/veot/model_zoo/Qwen2.5-72B-Instruct-Int4"

CUDA_VISIBLE_DEVICES=1 nohup $PYTHON_BIN -m vllm.entrypoints.openai.api_server \
    --model $INT4_MODEL \
    --served-model-name planner-72b \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.90 \
    --port 8000 \
    --trust-remote-code > $LOG_DIR/planner_server.log 2>&1 &

# === 2. 启动 Grounder (视觉模型) on GPU 3,4,5,6 ===
# 稳健方案: TP=4
echo "🚀 Starting Grounder (Qwen2-VL-72B) on GPU 3,4,5,6..."

VL_MODEL_PATH="/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/veot/model_zoo/Qwen2-VL-72B-Instruct"

CUDA_VISIBLE_DEVICES=2,3,4,5 nohup $PYTHON_BIN -m vllm.entrypoints.openai.api_server \
    --model $VL_MODEL_PATH \
    --served-model-name grounder-vl-72b \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --port 8001 \
    --trust-remote-code > $LOG_DIR/grounder_server.log 2>&1 &

# === 3. 循环等待服务就绪 (这就是你刚才缺少的步骤) ===
echo "⏳ Waiting for servers to be ready..."

# 检查 Planner (8000)
while ! nc -z localhost 8000; do
  sleep 5
  echo "Waiting for Planner (8000)..."
  # 检查进程是否意外死亡
  if ! pgrep -f "planner-72b" > /dev/null; then
     echo "❌ Planner died! Check logs: cat $LOG_DIR/planner_server.log"
     exit 1
  fi
done

# 检查 Grounder (8001)
while ! nc -z localhost 8001; do
  sleep 5
  echo "Waiting for Grounder (8001)..."
  if ! pgrep -f "grounder-vl-72b" > /dev/null; then
     echo "❌ Grounder died! Check logs: cat $LOG_DIR/grounder_server.log"
     exit 1
  fi
done

echo "✅ All Servers Ready! Starting Inference..."


# === 4. 运行 Python 客户端 ===
# 这里我们开 4 个进程并发请求 API
CHUNKS=4
CURRENT_DIR=$(pwd)
export PYTHONPATH="$CURRENT_DIR:$PYTHONPATH"

echo "Current PYTHONPATH: $PYTHONPATH"
for IDX in $(seq 0 $((CHUNKS-1))); do
    python videomind/eval/infer_agent_api_new.py \
        --dataset $DATASET \
        --split $SPLIT \
        --pred_path $PRED_PATH \
        --chunk $CHUNKS \
        --index $IDX &
done

wait # 等待所有 python 脚本跑完

echo "🎉 Evaluation Finished! Results saved to $PRED_PATH"

# === 5. 结束后杀掉服务器 (可选) ===
# pkill -f vllm