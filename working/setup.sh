#!/bin/bash

# ============================================================
# T4×2 GPU 环境初始化 - Qwen2-VL 微调环境
# ============================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "环境初始化开始 (T4×2 GPU 配置)"
echo "=========================================="

# ========== 1. 环境变量配置 ==========
echo "[1/6] 配置环境变量..."

# 重定向 Hugging Face 缓存到大容量临时目录
export HF_HOME="/kaggle/temp/hf_cache"
export HF_DATASETS_CACHE="/kaggle/temp/hf_datasets"
export TRANSFORMERS_CACHE="/kaggle/temp/transformers"

# 使用魔搭镜像加速（中国大陆用户）
export USE_MODELSCOPE_HUB=1

# 设置 Python 环境
export PYTHONPATH="/kaggle/working:$PYTHONPATH"

# T4 特定优化
export CUDA_LAUNCH_BLOCKING=0
export TORCH_DISTRIBUTED_DEBUG=OFF

# 创建缓存目录
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE

echo "✓ 环境变量配置完成"

# ========== 2. GPU 检查 ==========
echo "[2/6] 检查 GPU 配置..."

# 检测 GPU 数量
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
echo "检测到 $GPU_COUNT 个 GPU"

# 检测 GPU 型号
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "GPU 型号: $GPU_NAME"

if [[ $GPU_NAME == *"T4"* ]]; then
    echo "✓ 检测到 T4 GPU，配置已优化"
    if [ "$GPU_COUNT" -ge 2 ]; then
        echo "✓ 双卡配置，将使用 DDP 训练"
    else
        echo "⚠ 单卡 T4，训练速度会较慢"
    fi
else
    echo "⚠ 警告: 非 T4 GPU，当前配置针对 T4 优化"
fi

echo "✓ GPU 检查完成"

# ========== 3. 安装 LLaMA-Factory ==========
echo "[3/6] 安装 LLaMA-Factory..."

cd /kaggle/working

# 克隆最新版本
if [ ! -d "LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi

cd LLaMA-Factory

# 安装依赖（仅安装必需组件）
pip install -e ".[torch,metrics]" --no-cache-dir -q

# 验证安装
python -c "import llamafactory; print(f'✓ LLaMA-Factory 版本: {llamafactory.__version__}')"

echo "✓ LLaMA-Factory 安装完成"

# ========== 4. 创建数据集配置 ==========
echo "[4/6] 创建 dataset_info.json..."

cat > /kaggle/working/LLaMA-Factory/data/dataset_info.json << 'EOF'
{
  "rehab_train": {
    "file_name": "/kaggle/temp/train.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  },
  "rehab_val": {
    "file_name": "/kaggle/temp/val.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "images": "images"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt"
    }
  }
}
EOF

echo "✓ 数据集配置完成"

# ========== 5. 运行数据预处理 ==========
echo "[5/6] 运行数据预处理..."

python /kaggle/working/preprocess_data.py

echo "✓ 数据预处理完成"

# ========== 6. 创建训练配置 ==========
echo "[6/6] 创建训练配置文件..."

# ========== 6. 创建训练配置 (针对 16GB 显存优化) ==========
cat > /kaggle/working/train_config.yaml << 'EOF'
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct
template: qwen2_vl
dataset: rehab_train
val_size: 0.1
cutoff_len: 2048
max_samples: 1000000
preprocessing_num_workers: 4
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target: all
num_train_epochs: 3
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.1
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
gradient_checkpointing: true
fp16: true
bf16: false
optim: adamw_bnb_8bit
save_strategy: steps
save_steps: 200
save_total_limit: 1
output_dir: /kaggle/working/rehab_lora
overwrite_output_dir: true
qwen2_vl_max_pixels: 301056
logging_steps: 10
report_to: none
EOF
# ========== 7. 创建导出配置 ==========
cat > /kaggle/working/export_config.yaml << 'EOF'
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct
adapter_name_or_path: /kaggle/working/rehab_lora
template: qwen2_vl
finetuning_type: lora
export_dir: /kaggle/temp/full_model
export_size: 2
export_device: cpu
export_legacy_format: false
EOF

echo "✓ 配置文件已创建完成"

echo "✓ 导出配置文件已创建: export_config.yaml"

echo ""
echo "=========================================="
echo "✓ 环境初始化完成！(T4×2 GPU)"
echo "=========================================="
echo ""
echo "配置摘要:"
echo "  • GPU: T4×2 (每卡 16GB 显存)"
echo "  • 精度: FP16 (T4 原生支持)"
echo "  • Batch size: 1 per GPU"
echo "  • 梯度累积: 16 步"
echo "  • 等效 batch size: 32 (1×2×16)"
echo "  • LoRA Rank: 16 (显存优化)"
echo "  • 序列长度: 2048 (降低以节省显存)"
echo "  • 优化器: AdamW 8bit"
echo "  • 预计训练时长: 16-24 小时"
echo ""
echo "⚠️  16GB 显存优化提示:"
echo "  • 已降低 LoRA rank (32→16)"
echo "  • 已降低序列长度 (4096→2048)"
echo "  • 已启用 8bit 优化器"
echo "  • 如仍 OOM，可进一步降低 cutoff_len 至 1024"
echo ""
echo "下一步操作："
echo "  1. 运行训练: bash train.sh"
echo "  2. 导出模型: bash export_and_quantize.sh"
echo ""