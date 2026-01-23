#!/bin/bash

# ============================================================
# "零崩溃" LoRA 导出 + GGUF 量化流程
# 针对 Kaggle 20GB 磁盘限制的物理转场方案
# ============================================================

set -e

echo "=========================================="
echo "LoRA 导出与 GGUF 量化流程"
echo "=========================================="

# ========== 阶段 1: LoRA 权重合并导出 ==========
echo ""
echo "[阶段 1/4] 合并 LoRA 权重到基座模型..."
echo "----------------------------------------"

cd /kaggle/working/LLaMA-Factory

# 检查 LoRA 权重是否存在
if [ ! -d "/kaggle/working/rehab_lora" ]; then
    echo "✗ 错误: LoRA 权重目录不存在"
    exit 1
fi

# 清理旧的导出目录
rm -rf /kaggle/temp/full_model

# 执行导出（合并后的完整模型会保存到 /kaggle/temp/）
echo "开始合并导出（约需 5-10 分钟）..."
llamafactory-cli export /kaggle/working/export_config.yaml

# 验证导出结果
if [ ! -d "/kaggle/temp/full_model" ]; then
    echo "✗ 错误: 模型导出失败"
    exit 1
fi

echo "✓ 模型导出完成: /kaggle/temp/full_model"
du -sh /kaggle/temp/full_model

# ========== 阶段 2: 下载并编译 llama.cpp ==========
echo ""
echo "[阶段 2/4] 准备 llama.cpp 量化工具..."
echo "----------------------------------------"

cd /kaggle/temp

# 克隆 llama.cpp（浅克隆节省时间）
if [ ! -d "llama.cpp" ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
fi

cd llama.cpp

# 编译量化工具
echo "编译 llama.cpp..."
make -j$(nproc) quantize 2>&1 | grep -E "^\[.*\]|error|warning" || true

echo "✓ llama.cpp 编译完成"

# ========== 阶段 3: 转换为 GGUF 格式 ==========
echo ""
echo "[阶段 3/4] 转换为 GGUF 格式..."
echo "----------------------------------------"

# 安装 Python 依赖
pip install -q sentencepiece protobuf

# 转换为 FP16 GGUF（中间格式）
python convert_hf_to_gguf.py \
    /kaggle/temp/full_model \
    --outfile /kaggle/temp/model_fp16.gguf \
    --outtype f16

echo "✓ FP16 GGUF 转换完成"

# ========== 阶段 4: 量化为 Q4_K_M ==========
echo ""
echo "[阶段 4/4] 量化为 Q4_K_M 格式..."
echo "----------------------------------------"

# 执行量化
./llama-quantize \
    /kaggle/temp/model_fp16.gguf \
    /kaggle/temp/rehab_q4_k_m.gguf \
    Q4_K_M

echo "✓ 量化完成"

# ========== 阶段 5: 清理与持久化 ==========
echo ""
echo "[清理阶段] 释放磁盘空间..."
echo "----------------------------------------"

# 删除大文件（释放约 20GB 空间）
echo "删除完整模型..."
rm -rf /kaggle/temp/full_model

echo "删除中间 FP16 文件..."
rm -f /kaggle/temp/model_fp16.gguf

# 移动最终 GGUF 文件到持久化目录
echo "移动最终文件到 /kaggle/working/..."
mv /kaggle/temp/rehab_q4_k_m.gguf /kaggle/working/

# ========== 完成报告 ==========
echo ""
echo "=========================================="
echo "✓ 全流程完成！"
echo "=========================================="
echo ""
echo "最终文件:"
ls -lh /kaggle/working/rehab_q4_k_m.gguf
echo ""
echo "文件路径: /kaggle/working/rehab_q4_k_m.gguf"
echo ""
echo "下载方式:"
echo "  1. Kaggle 界面右侧 Output 面板"
echo "  2. 或使用: kaggle kernels output <your-kernel> -p /kaggle/working"
echo ""
echo "=========================================="