#!/bin/bash
# Pruning Fine-tuning 실험
# GPU 온도 관리: 실험 사이 15분 쿨다운
set -e

if [ -z "$1" ]; then
  echo "Usage: bash session_pruning.sh <GPU_ID>"
  echo "Example: bash session_pruning.sh 3"
  exit 1
fi

GPU_ID=$1
cd "$(dirname "$0")/../../.."
mkdir -p logs/lifeai

LOG=logs/lifeai/session_pruning.log

echo "============================================" | tee -a $LOG
echo "[$(date)] === Pruning Session START (GPU$GPU_ID) ===" | tee -a $LOG
echo "============================================" | tee -a $LOG

# --- Phase 1: Fine-tuning Pruning (d200, 10 epochs) ---
echo "[$(date)] --- Phase 1: d200 pruning + fine-tune ---" | tee -a $LOG
CUDA_VISIBLE_DEVICES=$GPU_ID python run_pruning_finetune.py \
  --depth 200 --dataset cifar100 --seed 42 \
  --finetune-epochs 10 --finetune-lr 0.001 \
  >> logs/lifeai/pruning_ft_d200.log 2>&1
echo "[$(date)] DONE: d200 pruning" | tee -a $LOG
echo "[$(date)] COOLDOWN 15min..." | tee -a $LOG
sleep 900

# --- Phase 2: Fine-tuning Pruning (d152, 10 epochs) ---
echo "[$(date)] --- Phase 2: d152 pruning + fine-tune ---" | tee -a $LOG
CUDA_VISIBLE_DEVICES=$GPU_ID python run_pruning_finetune.py \
  --depth 152 --dataset cifar100 --seed 42 \
  --finetune-epochs 10 --finetune-lr 0.001 \
  >> logs/lifeai/pruning_ft_d152.log 2>&1
echo "[$(date)] DONE: d152 pruning" | tee -a $LOG
echo "[$(date)] COOLDOWN 15min..." | tee -a $LOG
sleep 900

# --- Phase 3: Fine-tuning Pruning (d200, 20 epochs, 더 긴 적응) ---
echo "[$(date)] --- Phase 3: d200 pruning + fine-tune 20ep ---" | tee -a $LOG
CUDA_VISIBLE_DEVICES=$GPU_ID python run_pruning_finetune.py \
  --depth 200 --dataset cifar100 --seed 42 \
  --finetune-epochs 20 --finetune-lr 0.005 \
  >> logs/lifeai/pruning_ft_d200_20ep.log 2>&1
echo "[$(date)] DONE: d200 pruning 20ep" | tee -a $LOG
echo "[$(date)] COOLDOWN 15min..." | tee -a $LOG
sleep 900

echo "============================================" | tee -a $LOG
echo "[$(date)] === Pruning Session COMPLETE ===" | tee -a $LOG
echo "============================================" | tee -a $LOG
