#!/bin/bash
# lifeai/run_imagenet.sh — ImageNet Exp 9 (lifeai 서버 전용)
# A100 80GB, 단일 GPU + AMP
# 모델: baseline, lrs_low, lrs_mid, rezero (논문에 필요한 4개만)
# 예상: ~12h/model × 4 = ~48h
#
# 실행:
#   nohup bash scripts/run/lifeai/run_imagenet.sh 0 > logs/imagenet/lifeai_master.log 2>&1 &
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

cd "$(dirname "$0")/../../.."
LOG_DIR=logs/imagenet
mkdir -p $LOG_DIR
RESULT_DIR=results-json/lifeai
mkdir -p $RESULT_DIR
BATCH_LOG=$LOG_DIR/lifeai_master.log

echo "=== [lifeai] ImageNet Start: $(date) | GPU: $GPU_ID ===" | tee -a $BATCH_LOG
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee -a $BATCH_LOG
echo "" | tee -a $BATCH_LOG

DATA_PATH=/nfs_share/datasets/ILSVRC2012

# 논문에 필요한 모델만 (HybridA 제외)
MODELS=("baseline" "lrs_low" "lrs_mid" "rezero")

for model in "${MODELS[@]}"; do
  RESULT_FILE="${RESULT_DIR}/${model}_imagenet_result.json"
  if [ -f "$RESULT_FILE" ]; then
    echo "[$(date)] SKIP (exists): $model" | tee -a $BATCH_LOG
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

echo "=== [lifeai] ImageNet Complete: $(date) ===" | tee -a $BATCH_LOG
echo "결과 확인: ls -la $RESULT_DIR/" | tee -a $BATCH_LOG
