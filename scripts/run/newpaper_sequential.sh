#!/bin/bash
# newpaper_sequential.sh — GPU1 끝나면 GPU2 작업을 이어서 실행
# GPU1 스크립트 완료 대기 후 depth200 실험 시작
set -e

GPU_ID=${1:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=logs
mkdir -p $LOG_DIR
BATCH_LOG=$LOG_DIR/newpaper_gpu2.log

echo "=== NewPaper Depth200 Start: $(date) | GPU: $GPU_ID ===" | tee -a $BATCH_LOG

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

# === Depth 200 missing (20 runs) ===
echo "--- Depth 200 ---" | tee -a $BATCH_LOG
for model in baseline lrs_low lrs_mid lrs_high rezero; do
  for dataset in cifar10 cifar100; do
    for seed in 123 456; do
      run_one "$model" 200 "$dataset" "$seed"
    done
  done
done

echo "=== NewPaper Depth200 Complete: $(date) ===" | tee -a $BATCH_LOG
