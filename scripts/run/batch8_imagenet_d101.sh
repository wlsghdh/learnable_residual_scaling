#!/bin/bash
# batch8_imagenet_d101.sh — ImageNet ResNet-101 실험
# 목적: depth-alpha scaling law가 ImageNet에서도 성립하는지 검증
# 예상 소요시간: ~72시간 (A100 1장 기준, ResNet-101은 ResNet-50 대비 ~2x)
#
# 실행: bash scripts/run/batch8_imagenet_d101.sh [GPU_ID]
#
# 모델:
#   1. baseline  (ResNet-101, ~24h)
#   2. lrs_low   (ResNet-101, ~24h)
#   3. rezero    (ResNet-101, ~24h)
#
# 기대 결과: lrs_low의 mean alpha가 CIFAR d101과 유사하게 ~0.18-0.20 수렴
#           → depth-alpha law의 dataset-independence 대규모 검증
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=logs/imagenet
mkdir -p $LOG_DIR
RESULT_DIR=results-json/imagenet
mkdir -p $RESULT_DIR
BATCH_LOG=$LOG_DIR/batch8_d101.log

echo "=== ImageNet ResNet-101 GPU$GPU_ID Start: $(date) ===" | tee -a $BATCH_LOG
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv 2>/dev/null | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

DATA_PATH=/nfs_share/datasets/ILSVRC2012
DEPTH=101

# ResNet-101은 메모리 사용량이 더 크므로 batch size 조정
# A100 80GB: bs=256 가능, RTX 3090 24GB: bs=128 권장
BATCH_SIZE=256
LR=0.1

MODELS=("baseline" "lrs_low" "rezero")

for model in "${MODELS[@]}"; do
  RESULT_FILE="${RESULT_DIR}/${model}_d${DEPTH}_imagenet_result.json"
  if [ -f "$RESULT_FILE" ]; then
    echo "[$(date)] SKIP (exists): $model d${DEPTH} imagenet" | tee -a $BATCH_LOG
    continue
  fi

  echo "[$(date)] === Starting: $model | ResNet-${DEPTH} | ImageNet | GPU $GPU_ID ===" | tee -a $BATCH_LOG
  python train_imagenet.py \
    --model "$model" \
    --depth $DEPTH \
    --data "$DATA_PATH" \
    --output "$RESULT_DIR" \
    --epochs 90 \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --warmup-epochs 5 \
    --workers 8 \
    --seed 42 \
    --save-checkpoint \
    2>&1 | tee $LOG_DIR/${model}_d${DEPTH}_imagenet.log
  echo "[$(date)] === Done: $model d${DEPTH} ===" | tee -a $BATCH_LOG
  sleep 60
done

echo "=== ImageNet ResNet-101 GPU$GPU_ID Complete: $(date) ===" | tee -a $BATCH_LOG
