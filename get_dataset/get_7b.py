import os

# --- 1. 必须写在最前面：配置镜像 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 2. 关键：强制关闭极速下载，改回普通下载（为了稳定性）---
if 'HF_HUB_ENABLE_HF_TRANSFER' in os.environ:
    del os.environ['HF_HUB_ENABLE_HF_TRANSFER']

from huggingface_hub import snapshot_download, logging

# --- 3. 设置日志 ---
logging.set_verbosity_info()

print("正在连接镜像站 (hf-mirror.com)...")
print("正在下载 Qwen/Qwen2-VL-7B-Instruct 模型的所有文件...")

try:
    snapshot_download(
        repo_id="Qwen/Qwen2-VL-7B-Instruct",  # 修改 1: 改为 Qwen 的仓库 ID
        repo_type="model",
        # 修改 2: 建议存放在 model_zoo 下独立的文件夹中，避免混淆
        local_dir="/data2/codefile/yueagent/model_zoo/Qwen2-VL-7B-Instruct", 
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8                     
    )
    print("✅ Qwen2-VL-7B-Instruct 下载成功！")
except Exception as e:
    print("\n❌ 下载失败。错误详情：")
    print(e)