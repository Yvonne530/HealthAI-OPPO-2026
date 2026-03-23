# RehabGuardian 10.0 v5
**针对 RTX 4070 Laptop 8.6GB + WSL2 Ubuntu 7.8GB RAM 优化**

## 关键变化：HDF5 流式加载

v4 的问题：`get_all_samples()` 把 2,776,079 帧全部加载为 Python 对象，
需要 6-8GB RAM → WSL2 只有 7.8GB → OOM Traceback。

v5 的解决方案：
```
.b3d 文件
  ↓ preprocess.py（一次性，约30分钟）
dataset.h5（约1.5-2GB，压缩存储）
  ↓ HDF5RehabDataset（按需读取，每 batch 只用几 MB RAM）
DataLoader → 训练
```

## 快速开始

```bash
# 0. 建议先增加 WSL2 内存（Windows PowerShell）
# 在 C:\Users\Lenovo\.wslconfig 添加：
# [wsl2]
# memory=14GB
# swap=8GB
# 然后 wsl --shutdown 重启 WSL2

# 1. 安装依赖（在 WSL2 conda rehab_vlm 环境中）
conda activate rehab_vlm
bash setup.sh

# 2. 预处理（只需运行一次，约30-60分钟）
python preprocess.py
# 输出：/mnt/d/.../processed/dataset.h5  (~1.5GB)

# 3. 检查环境
python main.py --mode check

# 4. 训练
python main.py --mode train

# 5. 推理演示
python main.py --mode infer

# 6. 导出 ONNX
python main.py --mode export
```

## 硬件资源占用

| 资源     | 实际占用           | 说明 |
|---------|-------------------|------|
| VRAM    | ~400MB（训练时）  | batch=16，模型10M参数 |
| RAM     | ~300MB            | HDF5系统缓存+索引 |
| 磁盘    | ~2GB              | HDF5压缩数据集 |
| 预处理时间 | 30-60分钟      | 只需一次 |

## 为什么用 Linux 而不是 Windows

- nimblephysics 在你的 conda `rehab_vlm` 环境（WSL2 Linux）中已安装
- Windows 端 `pip list` 显示无 nimblephysics，且 CUDA: False
- CUDA 通过 WSL2 → NVIDIA WDDM 透传，功能完整
- 数据在 `/mnt/d/`，WSL2 直接访问（9p 协议，速度约 200MB/s）