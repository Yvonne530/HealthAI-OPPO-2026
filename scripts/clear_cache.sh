#!/usr/bin/env bash
# scripts/clear_cache.sh
# 清除 .b3d 数据缓存，强制下次运行重新加载

set -e

DATA_ROOT="${1:-/mnt/d/Camargo2021_Formatted_No_Arm}"
CACHE_DIR="$DATA_ROOT/.cache"

echo "======================================"
echo "清除 RehabGuardian 数据缓存"
echo "======================================"

if [ -d "$CACHE_DIR" ]; then
    echo "发现缓存目录：$CACHE_DIR"
    echo "正在删除..."
    rm -rf "$CACHE_DIR"
    echo "✓ 缓存已清除"
else
    echo "未发现缓存目录"
fi

echo ""
echo "下次运行训练时将重新加载所有 .b3d 文件"
echo "完成 ✅"
