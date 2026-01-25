#!/bin/bash

# ============================================================
# Kaggle Qwen2-VL 微调 - 一键执行脚本
# ============================================================
#
# 环境变量控制（适用于 Kaggle 非交互式环境）：
#   SMOKE_TEST=1    - 冒烟测试：单卡、4 步即停，跳过导出
#   SKIP_TRAIN=1    - 跳过训练步骤
#   SKIP_EXPORT=1   - 跳过导出与量化步骤
#
# 使用示例：
#   bash run_all.sh                      # 执行全部流程
#   SMOKE_TEST=1 bash run_all.sh         # 冒烟测试（单卡 4 步，不导出）
#   SKIP_TRAIN=1 bash run_all.sh         # 跳过训练，只导出
#   SKIP_EXPORT=1 bash run_all.sh        # 只训练，不导出
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

# 显示当前配置
echo "执行配置："
echo "  SMOKE_TEST=${SMOKE_TEST:-0} (设为 1 冒烟测试：单卡 4 步，不导出)"
echo "  SKIP_TRAIN=${SKIP_TRAIN:-0} (设为 1 跳过训练)"
echo "  SKIP_EXPORT=${SKIP_EXPORT:-0} (设为 1 跳过导出)"
echo ""

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

# 冒烟测试时传递 SMOKE_TEST，train.sh 会单卡 + 4 步
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
    SMOKE_TEST="${SMOKE_TEST:-0}" bash /kaggle/working/train.sh
else
    echo "⊙ 跳过训练步骤 (SKIP_TRAIN=1)"
fi
echo ""

# ========== 步骤 3: 导出与量化 ==========
echo -e "${GREEN}[步骤 3/3] 导出与量化${NC}"
echo "----------------------------------------"

# 冒烟测试不导出；SKIP_EXPORT=1 也跳过
if [[ "${SMOKE_TEST:-0}" = "1" ]] || [[ "${SKIP_EXPORT:-0}" = "1" ]]; then
    if [[ "${SMOKE_TEST:-0}" = "1" ]]; then
        echo "⊙ 冒烟测试模式，跳过导出 (SMOKE_TEST=1)"
    else
        echo "⊙ 跳过导出步骤 (SKIP_EXPORT=1)"
    fi
else
    bash /kaggle/working/export_and_quantize.sh
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