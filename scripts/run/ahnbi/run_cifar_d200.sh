#!/bin/bash
# ahnbi/run_cifar_d200.sh — Exp 1 잔여: Depth 200 (20 runs)
# RTX 3090, ~3.5h/run → ~70h (긴 실험, GPU 여러 장 가능하면 분할)
#
# 실행:
#   nohup bash scripts/run/ahnbi/run_cifar_d200.sh 0 > logs/ahnbi_d200.log 2>&1 &
set -e

GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=$GPU_ID

cd "$(dirname "$0")/../../.."
LOG_DIR=logs
mkdir -p $LOG_DIR
BATCH_LOG=$LOG_DIR/ahnbi_d200.log

echo "=== [ahnbi] Depth200 Start: $(date) | GPU: $GPU_ID ===" | tee -a $BATCH_LOG

run_one() {
  local model=$1 depth=$2 dataset=$3 seed=$4
  RESULT_FILE="results-json/${model}_depth${depth}_${dataset}_seed${seed}_result.json"
  if [ -f "$RESULT_FILE" ]; then
    echo "[$(date)] SKIP (exists): $model d$depth $dataset s$seed" | tee -a $BATCH_LOG
    return
  fi
  echo "[$(date)] Running: $model d$depth $dataset s$seed" | tee -a $BATCH_LOG
  python run_experiments.py \
    --model "$model" --depth "$depth" --dataset "$dataset" --seed "$seed" \
    >> $LOG_DIR/${model}_d${depth}_${dataset}_s${seed}.log 2>&1
  echo "[$(date)] Done:    $model d$depth $dataset s$seed" | tee -a $BATCH_LOG
  sleep 30
}

# 5 models × 2 datasets × 2 seeds = 20 runs
for model in baseline lrs_low lrs_mid lrs_high rezero; do
  for dataset in cifar10 cifar100; do
    for seed in 123 456; do
      run_one "$model" 200 "$dataset" "$seed"
    done
  done
done

echo "=== [ahnbi] Depth200 Complete: $(date) ===" | tee -a $BATCH_LOG
