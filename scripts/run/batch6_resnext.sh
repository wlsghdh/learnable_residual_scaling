#!/bin/bash
# batch6_resnext.sh — Phase 2-G: ResNeXt-50 (32×4d) 아키텍처 일반화
# 예상 소요시간: ~15시간 (GPU 1장)
# 실행: bash scripts/run/batch6_resnext.sh [GPU_ID]
#
# Models: resnext_baseline, resnext_lrs_low, resnext_lrs_hybrida_low
#         resnext_rezero, resnext_skipinit, resnext_layerscale
# Datasets: cifar10, cifar100
# Seeds: 42, 123, 456
# = 6 × 2 × 3 = 36 runs
set -e

GPU_ID=${1:-2}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=logs
mkdir -p $LOG_DIR
BATCH_LOG=$LOG_DIR/batch6_resnext.log

echo "=== ResNeXt-50 Start: $(date) | GPU: $GPU_ID ===" | tee -a $BATCH_LOG
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

MODELS=("resnext_baseline" "resnext_lrs_low" "resnext_lrs_hybrida_low" "resnext_rezero" "resnext_skipinit" "resnext_layerscale")
DATASETS=("cifar10" "cifar100")
SEEDS=(42 123 456)
DEPTH=50  # ResNeXt depth placeholder (models.py에서 ResNeXt는 depth 무시하고 50 고정)

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      RESULT_FILE="results-json/${model}_depth${DEPTH}_${dataset}_seed${seed}_result.json"
      if [ -f "$RESULT_FILE" ]; then
        echo "[$(date)] SKIP (exists): $model d$DEPTH $dataset s$seed" | tee -a $BATCH_LOG
        continue
      fi
      echo "[$(date)] Running: $model | depth=$DEPTH | $dataset | seed=$seed" | tee -a $BATCH_LOG
      python run_experiments.py \
        --model "$model" \
        --depth "$DEPTH" \
        --dataset "$dataset" \
        --seed "$seed" \
        --batch-size 512 \
        --lr 0.4 \
        >> $LOG_DIR/${model}_d${DEPTH}_${dataset}_s${seed}.log 2>&1
      echo "[$(date)] Done:    $model | depth=$DEPTH | $dataset | seed=$seed" | tee -a $BATCH_LOG
      sleep 60
    done
  done
  echo "--- Batch sleep 300s after $dataset ---" | tee -a $BATCH_LOG
  sleep 300
done

echo "=== ResNeXt-50 Complete: $(date) ===" | tee -a $BATCH_LOG
