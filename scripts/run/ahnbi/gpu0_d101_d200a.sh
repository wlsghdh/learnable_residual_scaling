#!/bin/bash
# ahnbi GPU0: d101 (9 runs) → d200 전반 (10 runs) = 19 runs
# RTX 3090 24GB, 메모리 ~70% 이하 사용
# d101: batch64 ~6GB, d200: batch32 ~12GB
# 예상: d101 ~14h + d200 ~35h = ~49h (약 2일)
#
# 실행:
#   nohup bash scripts/run/ahnbi/gpu0_d101_d200a.sh > logs/ahnbi_gpu0.log 2>&1 &
set -e

export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/../../.."

LOG_DIR=logs
SAVE_DIR=results-json/ahnbi
mkdir -p $LOG_DIR $SAVE_DIR
BATCH_LOG=$LOG_DIR/ahnbi_gpu0.log

echo "=== [ahnbi GPU0] Start: $(date) ===" | tee -a $BATCH_LOG

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

# === Depth 101 (9 runs, ~14h) ===
echo "--- Depth 101 ---" | tee -a $BATCH_LOG
run_one lrs_mid 101 cifar100 456
for dataset in cifar10 cifar100; do
  for seed in 123 456; do
    run_one lrs_high 101 "$dataset" "$seed"
  done
done
for dataset in cifar10 cifar100; do
  for seed in 123 456; do
    run_one rezero 101 "$dataset" "$seed"
  done
done

# === Depth 200 전반 (10 runs, ~35h) ===
echo "--- Depth 200 (part A: baseline/lrs_low/lrs_mid) ---" | tee -a $BATCH_LOG
for model in baseline lrs_low; do
  for dataset in cifar10 cifar100; do
    for seed in 123 456; do
      run_one "$model" 200 "$dataset" "$seed"
    done
  done
done
for dataset in cifar10 cifar100; do
  run_one lrs_mid 200 "$dataset" 123
done

echo "=== [ahnbi GPU0] Complete: $(date) ===" | tee -a $BATCH_LOG
