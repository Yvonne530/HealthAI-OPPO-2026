# Kaggle Qwen2-VL 医疗助手微调 - 快速开始

## 📋 前置要求

- **Kaggle 环境**: A100 GPU (80GB)
- **数据集**: `fitness-rehab-vlm` 已上传到 Kaggle
- **算力预算**: 约 8-12 小时（含训练+导出+量化）

---

## 🚀 快速开始（3 步完成）

### 方法 1: 一键执行（推荐）

```bash
# 在 Kaggle Notebook 中执行
!bash /kaggle/working/run_all.sh
```

### 方法 2: 分步执行

```bash
# 步骤 1: 初始化环境（约 5 分钟）
!bash /kaggle/working/setup.sh

# 步骤 2: 开始训练（约 6-10 小时）
!bash /kaggle/working/train.sh

# 步骤 3: 导出并量化（约 30 分钟）
!bash /kaggle/working/export_and_quantize.sh
```

---

## 📁 文件说明

| 文件名 | 用途 | 大小 |
|--------|------|------|
| `preprocess_data.py` | 数据预处理与校验 | 硬校验图片路径 |
| `setup.sh` | 环境初始化 | 安装 LLaMA-Factory |
| `train_config.yaml` | 训练配置 | 80GB A100 优化 |
| `train.sh` | 训练启动脚本 | - |
| `export_config.yaml` | 导出配置 | 合并 LoRA 权重 |
| `export_and_quantize.sh` | 导出+量化流程 | 零崩溃方案 |
| `run_all.sh` | 一键执行脚本 | 全流程自动化 |

---

## 🔧 关键配置说明

### 1. 显存优化策略

```yaml
per_device_train_batch_size: 4          # 保守起点
gradient_accumulation_steps: 8          # 等效 batch_size=32
gradient_checkpointing: true            # 节省 30%+ 显存 ✓
bf16: true                              # A100 原生支持
```

**为什么这样配置？**
- Qwen2-VL 图像 token 消耗显存极高
- `gradient_checkpointing` 是多模态训练的**必选项**
- 等效 batch size 32 平衡了收敛速度与显存占用

### 2. 磁盘保护机制

```yaml
output_dir: /kaggle/working/rehab_lora  # 小文件存这里
save_total_limit: 1                      # 只保留最新检查点
```

```yaml
export_dir: /kaggle/temp/full_model     # 大文件临时存这里
```

**物理转场流程**:
1. LoRA 权重 (小) → `/kaggle/working/`
2. 合并模型 (15GB) → `/kaggle/temp/` (临时)
3. GGUF 量化 (4GB) → `/kaggle/working/` (持久)
4. 删除临时文件 → 释放 20GB 空间

### 3. LoRA 目标模块

```yaml
lora_target: all  # LLaMA-Factory 自动识别
```

**自动覆盖模块**:
- 视觉编码器: `vision_model.*.linear`
- 语言模型: `language_model.*.qkvo_proj`
- 跨模态融合层

---

## 📊 训练监控

### 实时查看训练日志

```bash
# 方法 1: 在 Notebook 中
!tail -f /kaggle/working/rehab_lora/trainer_log.jsonl

# 方法 2: 查看最后 50 行
!tail -50 /kaggle/working/rehab_lora/trainer_log.jsonl
```

### 关键指标

- **Loss**: 应从 2.x 降至 0.5 以下
- **Learning Rate**: 余弦衰减曲线
- **显存占用**: 约 65-75GB (含梯度检查点)

---

## ⚠️ 常见问题

### Q1: OOM（显存不足）怎么办？

**解决方案**:
```yaml
# 修改 train_config.yaml
per_device_train_batch_size: 2  # 降低到 2
gradient_accumulation_steps: 16 # 提高到 16（保持等效 batch size）
```

### Q2: 磁盘空间不足？

**检查点**:
```bash
# 查看空间占用
!du -sh /kaggle/working/*
!du -sh /kaggle/temp/*

# 清理缓存
!rm -rf /kaggle/temp/hf_cache
!rm -rf /kaggle/temp/transformers
```

### Q3: 数据校验失败？

**常见原因**:
1. 图片路径错误 → 检查 `images/` 目录结构
2. JSON 格式错误 → 验证 `conversations` 角色交替
3. 图片文件不存在 → 确认数据集完整上传

**调试命令**:
```bash
!python /kaggle/working/preprocess_data.py
```

### Q4: 模型无法量化？

**前提条件**:
- LoRA 合并成功（检查 `/kaggle/temp/full_model`）
- llama.cpp 编译成功
- 足够的临时磁盘空间（至少 25GB）

---

## 📦 最终产出

### 文件清单

```
/kaggle/working/
├── rehab_lora/              # LoRA 权重（约 200MB）
│   ├── adapter_config.json
│   └── adapter_model.safetensors
└── rehab_q4_k_m.gguf       # 量化模型（约 4GB）
```

### 下载方式

**Kaggle 界面**:
1. 右侧 "Output" 面板
2. 找到 `rehab_q4_k_m.gguf`
3. 点击下载

**命令行**:
```bash
kaggle kernels output <your-username>/<kernel-name> -p /kaggle/working
```

---

## 🧪 本地测试（使用 llama.cpp）

```bash
# 下载 llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make

# 运行推理
./llama-cli -m rehab_q4_k_m.gguf \
  --image path/to/test_image.jpg \
  -p "请检查我的深蹲动作。" \
  -n 512
```

---

## 📈 性能基准

| 指标 | 预期值 |
|------|--------|
| 训练时长 | 6-10 小时 (3 epochs) |
| 最终 Loss | < 0.5 |
| 导出时长 | 10-15 分钟 |
| 量化时长 | 15-20 分钟 |
| GGUF 体积 | 约 4.2GB |

---

## 🔗 相关资源

- [LLaMA-Factory 官方文档](https://github.com/hiyouga/LLaMA-Factory)
- [Qwen2-VL 模型卡](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [llama.cpp 量化指南](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)

---

## 📝 更新日志

**v1.0** (2026-01-22)
- ✅ 完整的数据校验流程
- ✅ 80GB A100 显存优化
- ✅ 零崩溃磁盘管理
- ✅ 自动化 GGUF 量化

---

## 💡 技术支持

遇到问题？请检查:
1. 训练日志: `/kaggle/working/rehab_lora/trainer_log.jsonl`
2. 数据预处理输出
3. GPU 显存占用: `nvidia-smi`

**祝您训练顺利！** 🚀