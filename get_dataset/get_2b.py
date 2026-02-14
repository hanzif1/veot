import os
from huggingface_hub import snapshot_download, logging

# --- 1. 必须写在最前面：配置镜像 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 2. 关键：强制关闭极速下载，改回普通下载（为了稳定性）---
if 'HF_HUB_ENABLE_HF_TRANSFER' in os.environ:
    del os.environ['HF_HUB_ENABLE_HF_TRANSFER']

# --- 3. 设置日志 ---
logging.set_verbosity_info()

print("正在连接镜像站 (hf-mirror.com)...")
print("正在下载 yeliudev/VideoMind-2B 模型的所有文件...")

try:
    snapshot_download(
        repo_id="yeliudev/VideoMind-2B",  # 修改 1: 替换为 VideoMind-2B 的仓库 ID
        repo_type="model",
        # 修改 2: 保存路径同步修改为 VideoMind-2B，保持目录整洁
        local_dir="/HOME/nsccgz_zgchen/nsccgz_zgchen_6/HDD_POOL/veot/model_zoo/VideoMind-2B", 
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=8                     
    )
    print("✅ yeliudev/VideoMind-2B 下载成功！")
except Exception as e:
    print("\n❌ 下载失败。错误详情：")
    print(e)