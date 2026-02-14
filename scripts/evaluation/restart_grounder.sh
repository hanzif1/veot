#!/bin/bash
set -e

# === 配置路径 ===
VL_MODEL_PATH="/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/veot/model_zoo/Qwen2-VL-72B-Instruct"
LOG_DIR="logs_agent"
mkdir -p $LOG_DIR

# 获取当前 Python 路径
PYTHON_BIN=$(which python)
echo "Using Python: $PYTHON_BIN"

echo "=========================================================="
echo "🔄 Restarting Grounder Service Only..."
echo "=========================================================="

# === 1. 精准杀进程 (只杀 8001 端口) ===
echo "🔪 Killing old Grounder process on Port 8001..."

# 方法 A: 通过端口杀 (最准)
# 如果没有 fuser 命令，这一行可能会报错，所以加了 || true
fuser -k 8001/tcp >/dev/null 2>&1 || true

# 方法 B: 通过名字杀 (双重保险)
pkill -f "grounder-vl-72b" || true

# === 2. 等待显存释放 ===
echo "🧹 Waiting 8 seconds for VRAM cleanup..."
sleep 8

# === 3. 启动 Grounder (GPU 3,4,5,6) ===
# 参数: TP=4, Port=8001, MaxLen=8192
echo "🚀 Starting Grounder (Qwen2-VL-72B) on GPU 3,4,5,6..."

CUDA_VISIBLE_DEVICES=3,4,5,6 nohup $PYTHON_BIN -m vllm.entrypoints.openai.api_server \
    --model $VL_MODEL_PATH \
    --served-model-name grounder-vl-72b \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 8192 \
    --port 8001 \
    --trust-remote-code > $LOG_DIR/grounder_server.log 2>&1 &

# === 4. 等待就绪 ===
echo "⏳ Waiting for Grounder to be ready..."

# 循环检查端口 8001
while ! nc -z localhost 8001; do
  sleep 5
  echo "Waiting for Grounder (8001)..."
  
  # 检查进程是否刚启动就挂了
  if ! pgrep -f "grounder-vl-72b" > /dev/null; then
     echo "❌ Error: Grounder process died immediately!"
     echo "👇 Check the error log:"
     tail -n 10 $LOG_DIR/grounder_server.log
     exit 1
  fi
done

echo "✅ Grounder is Ready on Port 8001!"