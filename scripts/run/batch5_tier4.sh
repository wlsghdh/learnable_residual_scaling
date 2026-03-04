#!/bin/bash
# batch5_tier4.sh — Phase 2-F: Tier 4 추가 Depth (101, 200)
# 예상 소요시간: ~25시간 (GPU 1장)
# 실행: bash scripts/run/batch5_tier4.sh [GPU_ID]
#
# Models: baseline, lrs_low, lrs_hybrida_low
# Depths: 101, 200
# Datasets: cifar10, cifar100
# Seeds: 42, 123, 456
# = 3 × 2 × 2 × 3 = 36 runs
set -e

GPU_ID=${1:-2}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=logs
mkdir -p $LOG_DIR
BATCH_LOG=$LOG_DIR/batch5_tier4.log

echo "=== Tier4 Start: $(date) | GPU: $GPU_ID ===" | tee -a $BATCH_LOG
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv | tee -a $BATCH_LOG

MODELS=("baseline" "lrs_low" "lrs_hybrida_low")
DATASETS=("cifar10" "cifar100")
DEPTHS=(101 200)
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
    sleep 300
  done
done

echo "=== Tier4 Complete: $(date) ===" | tee -a $BATCH_LOG
