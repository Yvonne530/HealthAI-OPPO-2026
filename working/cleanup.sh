#!/bin/bash

# ============================================================
# 磁盘空间清理脚本
# 用于释放 Kaggle 有限的存储空间
# ============================================================

set -e

echo "=========================================="
echo "Kaggle 磁盘清理工具"
echo "=========================================="

# 显示当前磁盘使用情况
echo ""
echo "清理前磁盘使用情况:"
df -h /kaggle/working /kaggle/temp | tail -2

# ========== 清理选项 ==========
echo ""
echo "请选择清理选项:"
echo "  1) 清理缓存文件（HF Cache, pip cache）"
echo "  2) 清理训练检查点（保留最新）"
echo "  3) 清理临时文件（/kaggle/temp）"
echo "  4) 清理 LLaMA-Factory 仓库"
echo "  5) 全部清理（保留最终产物）"
echo "  6) 取消"
echo ""

read -p "请输入选项 [1-6]: " choice

case $choice in
    1)
        echo ""
        echo "[清理] 缓存文件..."
        
        # 清理 Hugging Face 缓存
        if [ -d "/kaggle/temp/hf_cache" ]; then
            rm -rf /kaggle/temp/hf_cache
            echo "✓ 已清理 HF Cache"
        fi
        
        # 清理数据集缓存
        if [ -d "/kaggle/temp/hf_datasets" ]; then
            rm -rf /kaggle/temp/hf_datasets
            echo "✓ 已清理 Datasets Cache"
        fi
        
        # 清理 Transformers 缓存
        if [ -d "/kaggle/temp/transformers" ]; then
            rm -rf /kaggle/temp/transformers
            echo "✓ 已清理 Transformers Cache"
        fi
        
        # 清理 pip 缓存
        pip cache purge 2>/dev/null
        echo "✓ 已清理 pip cache"
        ;;
        
    2)
        echo ""
        echo "[清理] 训练检查点..."
        
        lora_dir="/kaggle/working/rehab_lora"
        
        if [ -d "$lora_dir" ]; then
            # 查找所有检查点目录
            checkpoints=$(find "$lora_dir" -maxdepth 1 -type d -name "checkpoint-*" | sort)
            
            if [ -n "$checkpoints" ]; then
                # 保留最新的检查点
                latest=$(echo "$checkpoints" | tail -1)
                
                # 删除其他检查点
                for ckpt in $checkpoints; do
                    if [ "$ckpt" != "$latest" ]; then
                        echo "  删除: $ckpt"
                        rm -rf "$ckpt"
                    fi
                done
                
                echo "✓ 已清理旧检查点，保留: $(basename $latest)"
            else
                echo "⊙ 没有找到检查点目录"
            fi
        else
            echo "⊙ LoRA 目录不存在"
        fi
        ;;
        
    3)
        echo ""
        echo "[清理] 临时文件..."
        
        # 列出大文件
        echo "临时目录中的大文件 (>100MB):"
        find /kaggle/temp -type f -size +100M -exec ls -lh {} \; 2>/dev/null | awk '{print $5, $9}' || echo "  无大文件"
        
        echo ""
        read -p "是否删除 /kaggle/temp 中的所有内容? (y/n): " confirm
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            rm -rf /kaggle/temp/*
            echo "✓ 已清理 /kaggle/temp"
        else
            echo "⊙ 取消清理"
        fi
        ;;
        
    4)
        echo ""
        echo "[清理] LLaMA-Factory 仓库..."
        
        if [ -d "/kaggle/working/LLaMA-Factory" ]; then
            size=$(du -sh /kaggle/working/LLaMA-Factory | cut -f1)
            echo "LLaMA-Factory 大小: $size"
            
            read -p "是否删除? (y/n): " confirm
            
            if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
                rm -rf /kaggle/working/LLaMA-Factory
                echo "✓ 已删除 LLaMA-Factory"
            else
                echo "⊙ 取消删除"
            fi
        else
            echo "⊙ LLaMA-Factory 不存在"
        fi
        ;;
        
    5)
        echo ""
        echo "[全部清理] 保留最终产物..."
        echo ""
        echo "将保留以下文件:"
        echo "  - /kaggle/working/rehab_lora/ (LoRA 权重)"
        echo "  - /kaggle/working/*.gguf (GGUF 量化模型)"
        echo "  - /kaggle/working/*.py (脚本文件)"
        echo "  - /kaggle/working/*.sh (脚本文件)"
        echo "  - /kaggle/working/*.yaml (配置文件)"
        echo ""
        echo "将删除:"
        echo "  - /kaggle/temp/* (所有临时文件)"
        echo "  - /kaggle/working/LLaMA-Factory/"
        echo "  - 缓存文件"
        echo ""
        
        read -p "确认执行? (y/n): " confirm
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            # 清理临时目录
            rm -rf /kaggle/temp/*
            echo "✓ 已清理 /kaggle/temp"
            
            # 清理 LLaMA-Factory
            if [ -d "/kaggle/working/LLaMA-Factory" ]; then
                rm -rf /kaggle/working/LLaMA-Factory
                echo "✓ 已删除 LLaMA-Factory"
            fi
            
            # 清理 pip 缓存
            pip cache purge 2>/dev/null
            echo "✓ 已清理 pip cache"
            
            echo ""
            echo "✓ 全部清理完成"
        else
            echo "⊙ 取消清理"
        fi
        ;;
        
    6)
        echo "取消清理"
        exit 0
        ;;
        
    *)
        echo "无效选项"
        exit 1
        ;;
esac

# 显示清理后磁盘使用情况
echo ""
echo "清理后磁盘使用情况:"
df -h /kaggle/working /kaggle/temp | tail -2

echo ""
echo "=========================================="
echo "清理完成"
echo "=========================================="