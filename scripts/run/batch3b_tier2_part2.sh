#!/bin/bash
# batch3b_tier2_part2.sh — Phase 2-D: Tier 2 LRS 변형 (후반)
# 예상 소요시간: ~15시간 (GPU 1장)
# 실행: bash scripts/run/batch3b_tier2_part2.sh [GPU_ID]
#
# Models: lrs_hybrida_mid, lrs_hybrida_high
# Depths: 50, 152 / Datasets: cifar10, cifar100 / Seeds: 42, 123, 456
# = 2 × 2 × 2 × 3 = 24 runs
set -e

GPU_ID=${1:-2}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=logs
mkdir -p $LOG_DIR
BATCH_LOG=$LOG_DIR/batch3b_tier2_part2.log

echo "=== Tier2 Part2 Start: $(date) | GPU: $GPU_ID ===" | tee -a $BATCH_LOG
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv | tee -a $BATCH_LOG

MODELS=("lrs_hybrida_mid" "lrs_hybrida_high")
DATASETS=("cifar10" "cifar100")
DEPTHS=(50 152)
SEEDS=(42 123 456)

for depth in "${DEPTHS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        RESULT_FILE="results-json/${model}_depth${depth}_${dataset}_seed${seed}_result.json"
        if [ -f "$RESULT_FILE" ]; then
          echo "[$(date)] SKIP (exists): $model d$depth $dataset s$seed" | tee -a $BATCH_LOG
          continue
        fi
        echo "[$(date)] Running: $model | depth=$depth | $dataset | seed=$seed" | tee -a $BATCH_LOG
        python run_experiments.py \
          --model "$model" \
          --depth "$depth" \
          --dataset "$dataset" \
          --seed "$seed" \
          >> $LOG_DIR/${model}_d${depth}_${dataset}_s${seed}.log 2>&1
        echo "[$(date)] Done:    $model | depth=$depth | $dataset | seed=$seed" | tee -a $BATCH_LOG
        sleep 60
      done
    done
  done
done

echo "=== Tier2 Part2 Complete: $(date) ===" | tee -a $BATCH_LOG
