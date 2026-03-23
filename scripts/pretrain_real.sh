#!/usr/bin/env bash
# 使用真实 .b3d 数据预训练（见 configs/pretrain_real.yaml）
#
#   bash scripts/pretrain_real.sh                  # 使用 YAML 中的 data.raw_root
#   bash scripts/pretrain_real.sh /path/to/b3d     # 覆盖为 --data-root
#
# 也可设置环境变量后在 YAML 里写 raw_root: "$REHAB_B3D_ROOT/..."
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "${ROOT}/checkpoints/pretrain_real" "${ROOT}/logs/pretrain_real"

DATA_ARGS=()
if [[ -n "${1:-}" ]]; then
  DATA_ARGS=(--data-root "$1")
fi

python3 main.py --mode train --config "${ROOT}/configs/pretrain_real.yaml" "${DATA_ARGS[@]}"
echo "真实数据预训练结束（exit $?）"
