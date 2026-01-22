#!/bin/bash

# ============================================================
# Qwen2-VL 微调训练启动脚本
# ============================================================

set -e

echo "=========================================="
echo "开始训练 Qwen2-VL 医疗助手"
echo "=========================================="

# 设置工作目录
cd /kaggle/working/LLaMA-Factory

# 检查配置文件
if [ ! -f "/kaggle/working/train_config.yaml" ]; then
    echo "✗ 错误: 训练配置文件不存在"
    exit 1
fi

# 检查数据文件
if [ ! -f "/kaggle/temp/train.json" ]; then
    echo "✗ 错误: 训练数据文件不存在，请先运行 setup.sh"
    exit 1
fi

# 显示训练配置
echo ""
echo "训练配置摘要:"
echo "  - 模型: Qwen2-VL-7B-Instruct"
echo "  - Batch Size: 4 (梯度累积 x8 = 等效32)"
echo "  - LoRA Rank: 32"
echo "  - 梯度检查点: 启用 ✓"
echo "  - 输出目录: /kaggle/working/rehab_lora"
echo ""

# 启动训练
echo "开始训练..."
echo "=========================================="

llamafactory-cli train /kaggle/working/train_config.yaml

echo ""
echo "=========================================="
echo "✓ 训练完成！"
echo "=========================================="
echo ""
echo "LoRA 权重保存在: /kaggle/working/rehab_lora"
echo "下一步: 运行 bash export_and_quantize.sh 导出并量化模型"
echo ""