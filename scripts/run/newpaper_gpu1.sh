#!/bin/bash
# newpaper_gpu1.sh — 새 논문용 실험 (GPU 1)
# depth 50/101/152 missing runs (32 runs)
set -e

GPU_ID=${1:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=logs
mkdir -p $LOG_DIR
BATCH_LOG=$LOG_DIR/newpaper_gpu1.log

echo "=== NewPaper GPU1 Start: $(date) | GPU: $GPU_ID ===" | tee -a $BATCH_LOG

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

# === Depth 50 missing (4 runs) ===
echo "--- Depth 50 ---" | tee -a $BATCH_LOG
run_one lrs_mid 50 cifar100 123
run_one lrs_mid 50 cifar100 456
run_one lrs_high 50 cifar100 123
run_one lrs_high 50 cifar100 456

# === Depth 101 missing (20 runs) ===
echo "--- Depth 101 ---" | tee -a $BATCH_LOG
for model in baseline lrs_low lrs_mid lrs_high rezero; do
  for dataset in cifar10 cifar100; do
    for seed in 123 456; do
      run_one "$model" 101 "$dataset" "$seed"
    done
  done
done

# === Depth 152 missing (8 runs) ===
echo "--- Depth 152 ---" | tee -a $BATCH_LOG
for model in lrs_mid lrs_high; do
  for dataset in cifar10 cifar100; do
    for seed in 123 456; do
      run_one "$model" 152 "$dataset" "$seed"
    done
  done
done

echo "=== NewPaper GPU1 Complete: $(date) ===" | tee -a $BATCH_LOG
