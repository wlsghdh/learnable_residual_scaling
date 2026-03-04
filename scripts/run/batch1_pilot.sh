#!/bin/bash
# batch1_pilot.sh — Phase 2-A: Pilot Study
# 예상 소요시간: ~12시간 (GPU 1장)
# 실행: bash scripts/run/batch1_pilot.sh [GPU_ID]
#
# 모델 6개 × 2 데이터셋 × 2 깊이 × seed=42 = 24 runs
set -e

GPU_ID=${1:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=logs
mkdir -p $LOG_DIR
BATCH_LOG=$LOG_DIR/batch1_pilot.log

echo "=== Pilot Study Start: $(date) | GPU: $GPU_ID ===" | tee -a $BATCH_LOG
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

MODELS=("plain_he" "plain_identity" "baseline" "resnet_identity_all" "hybrida" "lrs_hybrida_low")
DATASETS=("cifar10" "cifar100")
DEPTHS=(50 152)
SEED=42

for depth in "${DEPTHS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
      echo "[$(date)] Running: $model | depth=$depth | $dataset | seed=$SEED" | tee -a $BATCH_LOG
      python run_experiments.py \
        --model "$model" \
        --depth "$depth" \
        --dataset "$dataset" \
        --seed "$SEED" \
        >> $LOG_DIR/${model}_d${depth}_${dataset}_s${SEED}.log 2>&1
      echo "[$(date)] Done:    $model | depth=$depth | $dataset" | tee -a $BATCH_LOG
      sleep 60
    done
    echo "--- Batch sleep 300s after $dataset depth=$depth ---" | tee -a $BATCH_LOG
    sleep 300
  done
done

echo "=== Pilot Study Complete: $(date) ===" | tee -a $BATCH_LOG
