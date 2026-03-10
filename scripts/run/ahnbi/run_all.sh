#!/bin/bash
# ahnbi/run_all.sh — ahnbi 서버 전체 CIFAR 실험 순차 실행
# d101(9) → d152(8) → d200(20) = 37 runs
#
# 실행:
#   nohup bash scripts/run/ahnbi/run_all.sh 0 > logs/ahnbi_all.log 2>&1 &
set -e

GPU_ID=${1:-0}
SCRIPT_DIR="$(dirname "$0")"

echo "=== [ahnbi] ALL Start: $(date) | GPU: $GPU_ID ==="

bash "$SCRIPT_DIR/run_cifar_d101.sh" "$GPU_ID"
bash "$SCRIPT_DIR/run_cifar_d152.sh" "$GPU_ID"
bash "$SCRIPT_DIR/run_cifar_d200.sh" "$GPU_ID"

echo "=== [ahnbi] ALL Complete: $(date) ==="
