#!/bin/bash

# ============================================================
# Kaggle Qwen2-VL 微调 - 一键执行脚本
# ============================================================

set -e  # 遇到错误立即停止

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}"
echo "============================================================"
echo "  Kaggle Qwen2-VL 医疗助手微调 - 全自动流程"
echo "============================================================"
echo -e "${NC}"

# 检查 GPU
echo -e "${YELLOW}[检查] 验证 GPU 环境...${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ========== 步骤 1: 环境初始化 ==========
echo -e "${GREEN}[步骤 1/3] 环境初始化${NC}"
echo "----------------------------------------"
bash /kaggle/working/setup.sh
echo ""

# ========== 步骤 2: 模型训练 ==========
echo -e "${GREEN}[步骤 2/3] 开始训练${NC}"
echo "----------------------------------------"

# 询问是否跳过训练（用于调试）
read -p "是否开始训练？(y/n，默认 y): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    bash /kaggle/working/train.sh
else
    echo "⊙ 跳过训练步骤"
fi
echo ""

# ========== 步骤 3: 导出与量化 ==========
echo -e "${GREEN}[步骤 3/3] 导出与量化${NC}"
echo "----------------------------------------"

read -p "是否导出并量化模型？(y/n，默认 y): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    bash /kaggle/working/export_and_quantize.sh
else
    echo "⊙ 跳过导出步骤"
fi
echo ""

# ========== 完成 ==========
echo -e "${GREEN}"
echo "============================================================"
echo "  ✓ 全部流程完成！"
echo "============================================================"
echo -e "${NC}"

echo "生成的文件:"
echo "  - LoRA 权重: /kaggle/working/rehab_lora/"
echo "  - GGUF 量化: /kaggle/working/rehab_q4_k_m.gguf"
echo ""

echo "磁盘使用情况:"
df -h /kaggle/working | tail -1
echo ""

echo -e "${YELLOW}提示: 在 Kaggle 右侧 Output 面板下载 GGUF 文件${NC}"