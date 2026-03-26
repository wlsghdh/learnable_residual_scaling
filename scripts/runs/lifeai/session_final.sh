#!/bin/bash
# 최종 추가 실험: Pruning checkpoint + ImageNet 추가
# GPU 온도 관리: 실험 사이 15분 쿨다운
set -e

if [ -z "$1" ]; then
  echo "Usage: bash session_final.sh <GPU_ID>"
  echo "Example: bash session_final.sh 3"
  exit 1
fi

GPU_ID=$1
cd "$(dirname "$0")/../../.."
mkdir -p logs/lifeai checkpoints results-json/imagenet

LOG=logs/lifeai/session_final.log

echo "============================================" | tee -a $LOG
echo "[$(date)] === Final Session START (GPU$GPU_ID) ===" | tee -a $LOG
echo "Phase1: CIFAR checkpoint x2, Phase2: ImageNet x2" | tee -a $LOG
echo "============================================" | tee -a $LOG

# --- Phase 1: CIFAR Pruning용 checkpoint 재학습 ---
echo "[$(date)] --- Phase 1: CIFAR checkpoint for pruning ---" | tee -a $LOG

for config in "lrs_low 152 cifar100 42" "lrs_low 200 cifar100 42"; do
  read model depth dataset seed <<< "$config"
  CKPT="checkpoints/${model}_depth${depth}_${dataset}_seed${seed}_best.pth"
  if [ -f "$CKPT" ]; then
    echo "[$(date)] SKIP (checkpoint exists): $model d${depth} $dataset s${seed}" | tee -a $LOG
    continue
  fi
  echo "[$(date)] START: $model d${depth} $dataset s${seed} (GPU$GPU_ID) [checkpoint]" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=$GPU_ID SAVE_CHECKPOINT=1 python run_experiments.py \
    --model "$model" --depth "$depth" --dataset "$dataset" --seed "$seed" \
    >> logs/lifeai/${model}_d${depth}_${dataset}_s${seed}_ckpt.log 2>&1
  echo "[$(date)] DONE: $model d${depth} $dataset s${seed}" | tee -a $LOG
  echo "[$(date)] COOLDOWN 15min..." | tee -a $LOG
  sleep 900
done

# --- Phase 2: ImageNet 추가 ---
echo "[$(date)] --- Phase 2: ImageNet additional models ---" | tee -a $LOG

for model in lrs_ha_low lrs_high; do
  RESULT="results-json/imagenet/${model}_imagenet_result.json"
  if [ -f "$RESULT" ]; then
    echo "[$(date)] SKIP (exists): $model imagenet" | tee -a $LOG
    continue
  fi
  echo "[$(date)] START: $model ImageNet (GPU$GPU_ID)" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=$GPU_ID python train_imagenet.py \
    --model "$model" \
    --data /nfs_share/datasets/ILSVRC2012 \
    --output results-json/imagenet \
    --epochs 60 \
    --batch-size 512 \
    --lr 0.2 \
    --warmup-epochs 5 \
    --workers 8 \
    --seed 42 \
    >> logs/lifeai/${model}_imagenet.log 2>&1
  echo "[$(date)] DONE: $model ImageNet" | tee -a $LOG
  echo "[$(date)] COOLDOWN 15min..." | tee -a $LOG
  sleep 900
done

echo "============================================" | tee -a $LOG
echo "[$(date)] === Final Session COMPLETE ===" | tee -a $LOG
echo "============================================" | tee -a $LOG
