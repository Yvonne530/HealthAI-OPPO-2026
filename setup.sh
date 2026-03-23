#!/bin/bash
# setup.sh
# 在你的 WSL2 Ubuntu 24.04 + conda rehab_vlm 环境中运行
# 用法：bash setup.sh

set -e
echo "=================================================="
echo "RehabGuardian 环境配置脚本"
echo "硬件：RTX 4070 Laptop 8.6GB | WSL2 Ubuntu 24.04"
echo "=================================================="

# 1. 激活 conda 环境（如果还没激活）
# conda activate rehab_vlm

# 2. 安装缺少的依赖
echo "[1/4] 安装 Python 依赖..."
pip install h5py>=3.10.0 --quiet              # HDF5 支持（核心）
pip install psutil --quiet                     # 内存监控
pip install pyyaml --quiet                     # 配置文件
pip install tqdm --quiet                       # 进度条
pip install scipy>=1.10.0 --quiet             # 信号处理
pip install scikit-learn>=1.3.0 --quiet       # 消融实验
pip install matplotlib --quiet                 # 可视化（可选）

# mediapipe（推理时用，训练不需要）
# pip install mediapipe>=0.10.0 --quiet

# onnxruntime（导出验证用，可选）
# pip install onnxruntime --quiet

echo "[2/4] 检查 nimblephysics..."
python -c "import nimblephysics; print('nimblephysics OK')" || \
    echo "⚠️  nimblephysics 未安装，b3d 加载会失败"

echo "[3/4] 检查 PyTorch + CUDA..."
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM: {vram:.1f} GB')
"

echo "[4/4] 建议的 WSL2 内存配置..."
echo ""
echo "在 Windows PowerShell 中执行："
echo "  notepad C:\\Users\\Lenovo\\.wslconfig"
echo ""
echo "添加以下内容（给 WSL2 更多内存）："
echo "  [wsl2]"
echo "  memory=14GB"
echo "  swap=8GB"
echo "  processors=10"
echo ""
echo "保存后在 PowerShell 执行：wsl --shutdown"
echo "然后重启 WSL2"
echo ""
echo "=================================================="
echo "配置完成！接下来："
echo ""
echo "# 步骤1：预处理（约 20-40 分钟，只需运行一次）"
echo "python preprocess.py"
echo ""
echo "# 步骤2：检查环境"
echo "python main.py --mode check"
echo ""
echo "# 步骤3：开始训练"
echo "python main.py --mode train"
echo "=================================================="