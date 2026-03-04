#!/bin/bash
# batch7_imagenet_gpu1.sh — Phase 4: ImageNet 실험 (GPU 1)
# 예상 소요시간: ~36시간 (GPU 1장, 단일 GPU + AMP)
# 실행: bash scripts/run/batch7_imagenet_gpu1.sh [GPU_ID]
#
# GPU1 순차 실행:
#   1. baseline     (~12시간)
#   2. lrs_low      (~12시간)
#   3. lrs_mid      (~12시간)
# 합계: ~36시간 < 48시간 ✅
#
# 데이터: /nfs_share/datasets/ILSVRC2012/
# 결과:   results-json/imagenet/
set -e

GPU_ID=${1:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=logs/imagenet
mkdir -p $LOG_DIR
RESULT_DIR=results-json/imagenet
mkdir -p $RESULT_DIR
BATCH_LOG=$LOG_DIR/batch7_gpu1.log

echo "=== ImageNet GPU$GPU_ID Start: $(date) ===" | tee -a $BATCH_LOG
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

DATA_PATH=/nfs_share/datasets/ILSVRC2012

MODELS=("baseline" "lrs_low" "lrs_mid")

for model in "${MODELS[@]}"; do
  RESULT_FILE="${RESULT_DIR}/${model}_imagenet_result.json"
  if [ -f "$RESULT_FILE" ]; then
    echo "[$(date)] SKIP (exists): $model imagenet" | tee -a $BATCH_LOG
    continue
  fi

  echo "[$(date)] === Starting: $model | ImageNet | GPU $GPU_ID ===" | tee -a $BATCH_LOG
  python train_imagenet.py \
    --model "$model" \
    --data "$DATA_PATH" \
    --output "$RESULT_DIR" \
    --epochs 90 \
    --batch-size 512 \
    --lr 0.2 \
    --warmup-epochs 5 \
    --workers 8 \
    --seed 42 \
    2>&1 | tee $LOG_DIR/${model}_imagenet.log
  echo "[$(date)] === Done: $model ===" | tee -a $BATCH_LOG
  sleep 60
done

echo "=== ImageNet GPU$GPU_ID Complete: $(date) ===" | tee -a $BATCH_LOG
