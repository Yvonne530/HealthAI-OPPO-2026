#!/bin/bash

# ============================================================
# Qwen2-VL 微调训练启动脚本 (T4×2 GPU)
# ============================================================

set -e

echo "=========================================="
echo "开始训练 Qwen2-VL 医疗助手 (T4×2 GPU)"
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

# 检测 GPU 数量
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo ""
echo "GPU 配置检测:"
echo "  GPU 数量: $GPU_COUNT"

if [ "$GPU_COUNT" -ge 2 ]; then
    echo "  训练模式: DDP 双卡并行"
    echo "  等效 batch size: 1 (per GPU) × $GPU_COUNT × 16 (grad accum) = $((1 * GPU_COUNT * 16))"
else
    echo "  训练模式: 单卡"
    echo "  等效 batch size: 1 × 16 (grad accum) = 16"
    echo "  ⚠ 建议使用双卡以提升速度"
fi

# 显示训练配置
echo ""
echo "训练配置摘要:"
echo "  - 模型: Qwen2-VL-7B-Instruct"
echo "  - 精度: FP16 (T4 原生支持)"
echo "  - Batch Size: 1 per GPU"
echo "  - 梯度累积: 16 步"
echo "  - LoRA Rank: 32"
echo "  - 梯度检查点: 启用 ✓"
echo "  - 输出目录: /kaggle/working/rehab_lora"
echo ""

# 启动训练
echo "开始训练..."
echo "=========================================="

# 根据 GPU 数量选择训练命令
if [ "$GPU_COUNT" -ge 2 ]; then
    # 双卡 DDP 训练
    echo "使用 DDP 双卡训练..."
    
    # 方法 1: 使用 llamafactory-cli (推荐)
    llamafactory-cli train /kaggle/working/train_config.yaml
    
    # 方法 2: 手动使用 torchrun (备选)
    # CUDA_VISIBLE_DEVICES=0,1 torchrun \
    #     --nproc_per_node=$GPU_COUNT \
    #     --master_port=29500 \
    #     -m llamafactory.cli train /kaggle/working/train_config.yaml
else
    # 单卡训练
    echo "使用单卡训练..."
    llamafactory-cli train /kaggle/working/train_config.yaml
fi

echo ""
echo "=========================================="
echo "✓ 训练完成！"
echo "=========================================="
echo ""
echo "LoRA 权重保存在: /kaggle/working/rehab_lora"
echo ""
echo "查看训练日志:"
echo "  tail -50 /kaggle/working/rehab_lora/trainer_log.jsonl"
echo ""
echo "下一步: 运行 bash export_and_quantize.sh 导出并量化模型"
echo ""