#!/bin/bash
# Session 1: d152 전부(6개) + d200 cifar10 전부(10개) = 16개, 예상 ~24h
# GPU 온도 관리: 실험 사이 10분 쿨다운
set -e

if [ -z "$1" ]; then
  echo "Usage: bash session1.sh <GPU_ID>"
  echo "Example: bash session1.sh 0"
  exit 1
fi

GPU_ID=$1
cd "$(dirname "$0")/../../.."
mkdir -p logs/lifeai

LOG=logs/lifeai/session1.log

run_cifar() {
  local model=$1 depth=$2 dataset=$3 seed=$4
  RESULT="results-json/${model}_depth${depth}_${dataset}_seed${seed}_result.json"
  if [ -f "$RESULT" ]; then
    echo "[$(date)] SKIP: $model d${depth} $dataset s${seed}" | tee -a $LOG
    return
  fi
  echo "[$(date)] START: $model d${depth} $dataset s${seed} (GPU$GPU_ID)" | tee -a $LOG
  CUDA_VISIBLE_DEVICES=$GPU_ID python run_experiments.py \
    --model "$model" --depth "$depth" --dataset "$dataset" --seed "$seed" \
    >> logs/lifeai/${model}_d${depth}_${dataset}_s${seed}.log 2>&1
  echo "[$(date)] DONE: $model d${depth} $dataset s${seed}" | tee -a $LOG

  # 쿨다운 10분
  echo "[$(date)] COOLDOWN 10min..." | tee -a $LOG
  sleep 900
}

echo "============================================" | tee -a $LOG
echo "[$(date)] === Session 1 START (GPU$GPU_ID) ===" | tee -a $LOG
echo "d152 x6 + d200 cifar10 x10 = 16 runs" | tee -a $LOG
echo "============================================" | tee -a $LOG

# --- d152 (6개, ~9h) ---
echo "[$(date)] --- Phase 1: depth152 ---" | tee -a $LOG
run_cifar lrs_mid  152 cifar100 123
run_cifar lrs_high 152 cifar100 123
run_cifar lrs_mid  152 cifar10  456
run_cifar lrs_high 152 cifar10  456
run_cifar lrs_mid  152 cifar100 456
run_cifar lrs_high 152 cifar100 456

# --- d200 cifar10 (10개, ~15h) ---
echo "[$(date)] --- Phase 2: depth200 cifar10 ---" | tee -a $LOG
run_cifar baseline 200 cifar10 123
run_cifar lrs_low  200 cifar10 123
run_cifar lrs_mid  200 cifar10 123
run_cifar lrs_high 200 cifar10 123
run_cifar rezero   200 cifar10 123
run_cifar baseline 200 cifar10 456
run_cifar lrs_low  200 cifar10 456
run_cifar lrs_mid  200 cifar10 456
run_cifar lrs_high 200 cifar10 456
run_cifar rezero   200 cifar10 456

echo "============================================" | tee -a $LOG
echo "[$(date)] === Session 1 COMPLETE ===" | tee -a $LOG
echo "============================================" | tee -a $LOG
