# 📦 Kaggle Qwen2-VL 微调完整文件清单

## 核心文件（必需）

### 1. Python 脚本
| 文件名 | 用途 | 说明 |
|--------|------|------|
| `preprocess_data.py` | 数据预处理 | 路径修正、格式校验、物理路径验证 |
| `check_env.py` | 环境检查 | 验证 GPU、磁盘、依赖包、数据文件 |
| `test_inference.py` | 推理测试 | 加载模型并测试生成效果 |
| `monitor_training.py` | 训练监控 | 实时显示 Loss、显存、训练进度 |

### 2. Bash 脚本
| 文件名 | 用途 | 说明 |
|--------|------|------|
| `setup.sh` | 环境初始化 | 安装 LLaMA-Factory、配置环境变量 |
| `train.sh` | 训练启动 | 启动微调训练 |
| `export_and_quantize.sh` | 导出与量化 | LoRA 合并 + GGUF 量化 |
| `run_all.sh` | 一键执行 | 全流程自动化 |
| `cleanup.sh` | 磁盘清理 | 释放存储空间 |

### 3. 配置文件
| 文件名 | 用途 | 说明 |
|--------|------|------|
| `train_config.yaml` | 训练配置 | 超参数、显存优化、LoRA 设置 |
| `export_config.yaml` | 导出配置 | LoRA 合并参数 |
| `dataset_info.json` | 数据集配置 | LLaMA-Factory 数据集定义 |
| `requirements.txt` | Python 依赖 | 所需第三方包列表 |

### 4. 文档
| 文件名 | 用途 | 说明 |
|--------|------|------|
| `README.md` | 快速开始指南 | 使用说明、配置解释、问题排查 |
| `FILE_LIST.md` | 文件清单 | 本文件，所有文件索引 |

---

## 📂 文件结构（部署后）

```
/kaggle/
├── input/
│   └── fitness-rehab-vlm/          # 数据集（Kaggle 添加）
│       ├── train.json
│       ├── val.json
│       └── images/
│           ├── squat/
│           ├── lunge/
│           └── lateral_raise/
│
├── working/                         # 持久化目录（20GB）
│   ├── preprocess_data.py
│   ├── check_env.py
│   ├── test_inference.py
│   ├── monitor_training.py
│   ├── setup.sh
│   ├── train.sh
│   ├── export_and_quantize.sh
│   ├── run_all.sh
│   ├── cleanup.sh
│   ├── train_config.yaml
│   ├── export_config.yaml
│   ├── requirements.txt
│   ├── README.md
│   ├── FILE_LIST.md
│   │
│   ├── LLaMA-Factory/              # 克隆的框架仓库
│   │   └── data/
│   │       └── dataset_info.json
│   │
│   ├── rehab_lora/                 # 训练输出
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   └── trainer_log.jsonl
│   │
│   └── rehab_q4_k_m.gguf          # 最终量化模型（下载此文件）
│
└── temp/                            # 临时目录（大容量）
    ├── train.json                   # 预处理后的数据
    ├── val.json
    ├── hf_cache/                    # Hugging Face 缓存
    ├── full_model/                  # 合并后的完整模型（临时）
    └── llama.cpp/                   # 量化工具
```

---

## 🚀 使用流程

### 阶段 1: 部署文件
```bash
# 将所有文件上传到 Kaggle Notebook
# 或在 Notebook 中创建这些文件
```

### 阶段 2: 检查环境
```bash
python check_env.py
```

### 阶段 3: 一键执行
```bash
bash run_all.sh
```

### 阶段 4: 监控训练（可选）
```bash
# 在新的 Notebook Cell 中运行
python monitor_training.py --once
```

### 阶段 5: 清理空间（可选）
```bash
bash cleanup.sh
```

### 阶段 6: 测试模型
```bash
python test_inference.py
```

---

## 📊 文件大小预估

| 文件类型 | 大小 | 位置 |
|---------|------|------|
| 脚本文件 (.py, .sh) | ~50 KB | /kaggle/working |
| 配置文件 (.yaml, .json) | ~5 KB | /kaggle/working |
| LLaMA-Factory 仓库 | ~50 MB | /kaggle/working |
| LoRA 权重 | ~200 MB | /kaggle/working |
| 完整模型（临时） | ~15 GB | /kaggle/temp |
| GGUF 量化模型 | ~4 GB | /kaggle/working |
| 缓存文件 | ~5 GB | /kaggle/temp |

**总持久化文件**: ~4.5 GB（仅需下载 GGUF）

---

## ⚙️ 关键配置说明

### 1. 环境变量（setup.sh）
```bash
export HF_HOME="/kaggle/temp/hf_cache"
export HF_DATASETS_CACHE="/kaggle/temp/hf_datasets"
export USE_MODELSCOPE_HUB=1
```

### 2. 训练参数（train_config.yaml）
```yaml
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
gradient_checkpointing: true  # 必须开启
lora_target: all              # 自动覆盖所有层
visual_inputs: true           # Qwen2-VL 标准参数
```

### 3. 导出路径（export_config.yaml）
```yaml
export_dir: /kaggle/temp/full_model  # 临时目录
```

---

## 🔧 自定义修改

### 修改训练参数
编辑 `train_config.yaml`:
```yaml
num_train_epochs: 5              # 增加训练轮数
learning_rate: 5.0e-5            # 降低学习率
lora_rank: 64                    # 增加 LoRA 秩
```

### 修改测试样本
编辑 `test_inference.py`:
```python
test_image = "你的图片路径.jpg"
test_prompt = "你的提示词"
```

### 修改监控间隔
编辑 `monitor_training.py`:
```python
monitor_training(interval=60)  # 60秒刷新一次
```

---

## 📝 文件依赖关系

```
setup.sh
  ├─ requirements.txt (Python 依赖)
  ├─ dataset_info.json (数据集配置)
  └─ preprocess_data.py (数据预处理)

train.sh
  └─ train_config.yaml (训练配置)

export_and_quantize.sh
  └─ export_config.yaml (导出配置)

run_all.sh
  ├─ setup.sh
  ├─ train.sh
  └─ export_and_quantize.sh
```

---

## ✅ 文件完整性检查

运行以下命令验证所有文件已正确部署:

```bash
# 检查脚本文件
ls -lh /kaggle/working/*.py
ls -lh /kaggle/working/*.sh

# 检查配置文件
ls -lh /kaggle/working/*.yaml
ls -lh /kaggle/working/*.json
ls -lh /kaggle/working/*.txt

# 检查文档
ls -lh /kaggle/working/*.md

# 或使用环境检查脚本
python /kaggle/working/check_env.py
```

---

## 🎯 最终产物

训练完成后，您将获得:

1. **LoRA 权重** (`/kaggle/working/rehab_lora/`)
   - 可用于继续训练或部署
   - 约 200 MB

2. **GGUF 量化模型** (`/kaggle/working/rehab_q4_k_m.gguf`)
   - 可直接用于推理
   - 约 4 GB
   - **下载此文件用于本地部署**

---

## 📞 技术支持

遇到问题时的检查顺序:

1. 运行 `python check_env.py` 检查环境
2. 查看 `/kaggle/working/rehab_lora/trainer_log.jsonl` 训练日志
3. 使用 `nvidia-smi` 检查 GPU 状态
4. 使用 `df -h` 检查磁盘空间
5. 使用 `python monitor_training.py --summary` 查看训练总结

---

**最后更新**: 2026-01-22
**版本**: v1.0
**适用环境**: Kaggle A100 (80GB)