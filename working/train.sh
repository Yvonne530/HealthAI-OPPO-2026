#!/bin/bash

# ============================================================
# Qwen2-VL 微调训练启动脚本 (T4×2 GPU) - 自动修复版
# ============================================================

set -e

SMOKE=${SMOKE_TEST:-0}

echo "=========================================="
if [ "$SMOKE" = "1" ]; then
    echo "开始冒烟测试 (单卡，4 步)"
else
    echo "开始训练 Qwen2-VL 医疗助手 (T4×2 GPU)"
fi
echo "=========================================="

# 设置工作目录
cd /kaggle/working/LLaMA-Factory

# 选择配置文件
if [ "$SMOKE" = "1" ]; then
    CONFIG="/kaggle/working/train_config_smoke.yaml"
else
    CONFIG="/kaggle/working/train_config.yaml"
fi

# --- 核心修复逻辑：强制删除无效参数 ---
if [ -f "$CONFIG" ]; then
    echo "正在自动修复配置文件中的兼容性参数..."
    # 强制删除导致报错的三个关键参数
    sed -i '/qwen2_vl_max_pixels/d' "$CONFIG"
    sed -i '/visual_inputs/d' "$CONFIG"
    sed -i '/modules_to_save/d' "$CONFIG"
    # 确保没有 val_dataset 冲突
    sed -i '/val_dataset/d' "$CONFIG"
else
    echo "✗ 错误: 配置文件不存在: $CONFIG"
    exit 1
fi
# ------------------------------------

# 检查数据文件
if [ ! -f "/kaggle/temp/train.json" ]; then
    echo "✗ 错误: 训练数据文件不存在，请先运行 setup.sh"
    exit 1
fi

# 显卡配置
if [ "$SMOKE" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    GPU_COUNT=1
    echo "GPU 配置: 冒烟模式单卡 (GPU 0)"
else
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    echo "GPU 配置检测: 发现 $GPU_COUNT 块显卡"
fi

echo "训练配置路径: $CONFIG"
echo "开始执行 llamafactory-cli..."
echo "=========================================="

# 执行训练
if [ "$SMOKE" = "1" ]; then
    llamafactory-cli train "$CONFIG"
elif [ "$GPU_COUNT" -ge 2 ]; then
    echo "检测到双卡，启动分布式训练..."
    # llamafactory-cli 会自动处理 torchrun 逻辑
    llamafactory-cli train "$CONFIG"
else
    llamafactory-cli train "$CONFIG"
fi

echo ""
echo "=========================================="
echo "✓ 训练指令执行完毕！"
echo "=========================================="