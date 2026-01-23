#!/bin/bash

# ============================================================
# Kaggle A100 环境初始化 - Qwen2-VL 微调环境
# ============================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "环境初始化开始"
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

# 创建缓存目录
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TRANSFORMERS_CACHE

echo "✓ 环境变量配置完成"

# ========== 2. 安装 LLaMA-Factory ==========
echo "[2/6] 安装 LLaMA-Factory..."

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

# ========== 3. 创建数据集配置 ==========
echo "[3/6] 创建 dataset_info.json..."

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

# ========== 4. 运行数据预处理 ==========
echo "[4/6] 运行数据预处理..."

python /kaggle/working/preprocess_data.py

echo "✓ 数据预处理完成"

# ========== 5. 创建训练配置 ==========
echo "[5/6] 创建训练配置文件..."

cat > /kaggle/working/train_config.yaml << 'EOF'
### ============================================================
### Qwen2-VL-7B 医疗助手微调配置 (80GB A100 优化)
### ============================================================

### 模型配置
model_name_or_path: Qwen/Qwen2-VL-7B-Instruct
template: qwen2_vl
visual_inputs: true  # Qwen2-VL 官方标准参数

### 数据配置
dataset: rehab_train
val_dataset: rehab_val
cutoff_len: 4096
max_samples: 1000000  # 不限制样本数

### LoRA 配置
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: all  # 自动覆盖视觉+语言模块

### 训练参数
num_train_epochs: 3
learning_rate: 1.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.1

### 显存优化策略 (核心)
per_device_train_batch_size: 4          # 稳定起点
gradient_accumulation_steps: 8          # 等效 batch_size=32
gradient_checkpointing: true            # 节省 30%+ 显存（必须开启）
bf16: true                              # A100 原生支持
optim: adamw_torch

### 评估配置
per_device_eval_batch_size: 2
eval_strategy: steps
eval_steps: 50
save_strategy: steps
save_steps: 50

### 磁盘保护
output_dir: /kaggle/working/rehab_lora
save_total_limit: 1                     # 只保留最新检查点
logging_steps: 10

### 其他
overwrite_output_dir: true
ddp_timeout: 3600
report_to: none                         # 不上传到 wandb/tensorboard
EOF

echo "✓ 训练配置文件已创建: train_config.yaml"

# ========== 6. 创建导出配置 ==========
echo "[6/6] 创建导出配置文件..."

cat > /kaggle/working/export_config.yaml << 'EOF'
### ============================================================
### LoRA 权重导出配置
### ============================================================

model_name_or_path: Qwen/Qwen2-VL-7B-Instruct
adapter_name_or_path: /kaggle/working/rehab_lora
template: qwen2_vl
finetuning_type: lora

# 导出到大容量临时目录
export_dir: /kaggle/temp/full_model
export_size: 2
export_device: cpu
export_legacy_format: false
EOF

echo "✓ 导出配置文件已创建: export_config.yaml"

echo ""
echo "=========================================="
echo "✓ 环境初始化完成！"
echo "=========================================="
echo ""
echo "下一步操作："
echo "  1. 运行训练: bash train.sh"
echo "  2. 导出模型: bash export_and_quantize.sh"
echo ""