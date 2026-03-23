#!/usr/bin/env bash
# RehabGuardian 预训练 / 流水线冒烟（合成数据，约数分钟内在 CPU 可跑完）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
# 与 configs/pretrain.yaml 一致，避免个别环境下目录未就绪
mkdir -p "${ROOT}/checkpoints/pretrain" "${ROOT}/logs/pretrain"

# 可选：强制用 GPU（需本机 CUDA 可用），取消下一行注释
# export CUDA_VISIBLE_DEVICES=0

python3 main.py --mode train --config "${ROOT}/configs/pretrain.yaml"
echo "预训练脚本结束（exit $?）"
