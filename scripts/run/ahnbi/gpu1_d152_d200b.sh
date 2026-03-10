#!/bin/bash
# ahnbi GPU1: d152 (8 runs) → d200 후반 (10 runs) = 18 runs
# RTX 3090 24GB, 메모리 ~70% 이하 사용
# d152: batch32 ~10GB, d200: batch32 ~12GB
# 예상: d152 ~20h + d200 ~35h = ~55h (약 2.3일)
#
# 실행:
#   nohup bash scripts/run/ahnbi/gpu1_d152_d200b.sh > logs/ahnbi_gpu1.log 2>&1 &
set -e

export CUDA_VISIBLE_DEVICES=1
cd "$(dirname "$0")/../../.."

LOG_DIR=logs
SAVE_DIR=results-json/ahnbi
mkdir -p $LOG_DIR $SAVE_DIR
BATCH_LOG=$LOG_DIR/ahnbi_gpu1.log

echo "=== [ahnbi GPU1] Start: $(date) ===" | tee -a $BATCH_LOG

run_one() {
  local model=$1 depth=$2 dataset=$3 seed=$4
  RESULT_FILE="${SAVE_DIR}/${model}_depth${depth}_${dataset}_seed${seed}_result.json"
  if [ -f "$RESULT_FILE" ]; then
    echo "[$(date)] SKIP (exists): $model d$depth $dataset s$seed" | tee -a $BATCH_LOG
    return
  fi
  echo "[$(date)] Running: $model d$depth $dataset s$seed" | tee -a $BATCH_LOG
  python run_experiments.py \
    --model "$model" --depth "$depth" --dataset "$dataset" --seed "$seed" \
    --save-dir "$SAVE_DIR" \
    >> $LOG_DIR/${model}_d${depth}_${dataset}_s${seed}.log 2>&1
  echo "[$(date)] Done:    $model d$depth $dataset s$seed" | tee -a $BATCH_LOG
  sleep 30
}

# === Depth 152 (8 runs, ~20h) ===
echo "--- Depth 152 ---" | tee -a $BATCH_LOG
for model in lrs_mid lrs_high; do
  for dataset in cifar10 cifar100; do
    for seed in 123 456; do
      run_one "$model" 152 "$dataset" "$seed"
    done
  done
done

# === Depth 200 후반 (10 runs, ~35h) ===
echo "--- Depth 200 (part B: lrs_mid/lrs_high/rezero) ---" | tee -a $BATCH_LOG
for dataset in cifar10 cifar100; do
  run_one lrs_mid 200 "$dataset" 456
done
for model in lrs_high rezero; do
  for dataset in cifar10 cifar100; do
    for seed in 123 456; do
      run_one "$model" 200 "$dataset" "$seed"
    done
  done
done

echo "=== [ahnbi GPU1] Complete: $(date) ===" | tee -a $BATCH_LOG
