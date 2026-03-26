#!/bin/bash
# Identity init 추가 실험: 11 runs
# GPU 온도 관리: 실험 사이 15분 쿨다운
set -e

if [ -z "$1" ]; then
  echo "Usage: bash session_identity.sh <GPU_ID>"
  echo "Example: bash session_identity.sh 3"
  exit 1
fi

GPU_ID=$1
cd "$(dirname "$0")/../../.."
mkdir -p logs/lifeai

LOG=logs/lifeai/session_identity.log

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

  # 쿨다운 15분
  echo "[$(date)] COOLDOWN 15min..." | tee -a $LOG
  sleep 900
}

echo "============================================" | tee -a $LOG
echo "[$(date)] === Identity Init Session START (GPU$GPU_ID) ===" | tee -a $LOG
echo "hybrida x4 + lrs_hybrida_low x2 + resnet_identity_all x5 = 11 runs" | tee -a $LOG
echo "============================================" | tee -a $LOG

# --- hybrida (4 runs) ---
echo "[$(date)] --- Phase 1: hybrida ---" | tee -a $LOG
run_cifar hybrida 152 cifar100 123
run_cifar hybrida 152 cifar100 456
run_cifar hybrida 200 cifar100 123
run_cifar hybrida 200 cifar100 456

# --- lrs_hybrida_low (2 runs) ---
echo "[$(date)] --- Phase 2: lrs_hybrida_low ---" | tee -a $LOG
run_cifar lrs_hybrida_low 200 cifar100 123
run_cifar lrs_hybrida_low 200 cifar100 456

# --- resnet_identity_all (5 runs) ---
echo "[$(date)] --- Phase 3: resnet_identity_all ---" | tee -a $LOG
run_cifar resnet_identity_all 152 cifar100 123
run_cifar resnet_identity_all 152 cifar100 456
run_cifar resnet_identity_all 200 cifar100 42
run_cifar resnet_identity_all 200 cifar100 123
run_cifar resnet_identity_all 200 cifar100 456

echo "============================================" | tee -a $LOG
echo "[$(date)] === Identity Init Session COMPLETE ===" | tee -a $LOG
echo "============================================" | tee -a $LOG
