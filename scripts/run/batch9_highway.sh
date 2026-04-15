#!/bin/bash
# batch9_highway.sh — Highway Gate vs LRS 비교 실험
# 목적: "LRS는 Highway Network gate와 뭐가 다르냐" 질문 선제 대응
# 설계: 기존 run_experiments.py 사용 (동일한 training config/data/seed)
#
# Highway: y = T(x)·F(x) + (1-T(x))·x  (input-dependent gate)
# LRS:     y = α·F(x) + (1-α)·x         (input-independent scalar)
#
# 실행: bash scripts/run/batch9_highway.sh [GPU_ID]
#
# 실험 범위:
#   highway × depth{152,200} × {cifar10,cifar100} × seed{42,123,456}
#   = 1 model × 2 depths × 2 datasets × 3 seeds = 12 runs
#
# 예상 시간:
#   d152: ~40min/run, d200: ~60min/run (RTX 3090 기준)
#   총 ~10h
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=logs
mkdir -p $LOG_DIR
BATCH_LOG=$LOG_DIR/batch9_highway.log

echo "=== Highway Gate Experiment GPU$GPU_ID Start: $(date) ===" | tee -a $BATCH_LOG
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

MODEL="highway"
SEEDS=(42 123 456)
DEPTHS=(152 200)
DATASETS=("cifar100" "cifar10")

for depth in "${DEPTHS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      RESULT_FILE="results-json/${MODEL}_depth${depth}_${dataset}_seed${seed}_result.json"
      if [ -f "$RESULT_FILE" ]; then
        echo "[$(date)] SKIP (exists): $MODEL d${depth} ${dataset} s${seed}" | tee -a $BATCH_LOG
        continue
      fi

      echo "[$(date)] === Starting: $MODEL | d${depth} | ${dataset} | seed${seed} | GPU $GPU_ID ===" | tee -a $BATCH_LOG
      python run_experiments.py \
        --dataset "$dataset" \
        --depth $depth \
        --model "$MODEL" \
        --seed $seed \
        2>&1 | tee $LOG_DIR/${MODEL}_d${depth}_${dataset}_s${seed}.log
      echo "[$(date)] === Done: $MODEL d${depth} ${dataset} s${seed} ===" | tee -a $BATCH_LOG
      sleep 30
    done
  done
done

echo "=== Highway Gate Experiment GPU$GPU_ID Complete: $(date) ===" | tee -a $BATCH_LOG
