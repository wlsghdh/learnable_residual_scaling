#!/bin/bash
# batch10_pruning_analysis.sh — Alpha-guided Pruning FLOPs/Latency 분석
# 목적: LRS alpha 기반 pruning의 practical value 정량 증명
#
# 전제: lrs_low checkpoint가 필요 (SAVE_CHECKPOINT=1로 학습 필요)
#
# 실행: bash scripts/run/batch10_pruning_analysis.sh [GPU_ID]
#
# 결과: results-json/pruning_analysis_d200_cifar100.json
# 예상 시간: ~2h (fine-tune 포함)
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=logs
mkdir -p $LOG_DIR

echo "=== Pruning Analysis Start: $(date) ==="

# ResNet-200, CIFAR-100, seed 42
python run_pruning_analysis.py \
  --depth 200 \
  --dataset cifar100 \
  --seed 42 \
  --finetune-epochs 10 \
  --finetune-lr 0.001 \
  --latency-runs 200 \
  2>&1 | tee $LOG_DIR/pruning_analysis_d200_cifar100.log

echo "=== Pruning Analysis Complete: $(date) ==="
