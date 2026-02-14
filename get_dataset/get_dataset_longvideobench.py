import os

# --- 1. 必须写在最前面：配置镜像 ---
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# --- 2. 关键：强制关闭极速下载，改回普通下载（为了稳定性）---
# 刚才的报错很可能是因为加速器连不上网导致的
if 'HF_HUB_ENABLE_HF_TRANSFER' in os.environ:
    del os.environ['HF_HUB_ENABLE_HF_TRANSFER']

from huggingface_hub import snapshot_download, logging

# --- 3. 设置日志 ---
logging.set_verbosity_info()

print("正在连接镜像站 (hf-mirror.com)...")
print("如果这一步卡住超过 1 分钟，请检查你的服务器网络。")

try:
    snapshot_download(
        repo_id="yeliudev/VideoMind-Dataset",
        repo_type="dataset",
        allow_patterns="longvideobench/*",
        local_dir="/XYAIFS00/HDD_POOL/nsccgz_zgchen/nsccgz_zgchen_6/veot/datasets",     
        local_dir_use_symlinks=False,
        resume_download=True,
        # max_workers=4  # 普通模式下不建议开太大，默认即可
    )
    print("✅ 下载成功！")
except Exception as e:
    print("\n❌ 依然失败。错误详情：")
    print(e)