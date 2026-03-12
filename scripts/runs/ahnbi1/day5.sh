#!/bin/bash
set -e
cd "$(dirname "$0")/../../.."
mkdir -p logs/ahnbi1

echo "[$(date)] === Day5: depth200 cifar10 seed123 (baseline, lrs_low) ===" | tee -a logs/ahnbi1/day5.log

run_one() {
  local gpu=$1 model=$2 depth=$3 dataset=$4 seed=$5
  RESULT="results-json/${model}_depth${depth}_${dataset}_seed${seed}_result.json"
  if [ -f "$RESULT" ]; then
    echo "[$(date)] SKIP: $model d${depth} $dataset s${seed} (GPU$gpu)" | tee -a logs/ahnbi1/day5.log
    return
  fi
  echo "[$(date)] START: $model d${depth} $dataset s${seed} (GPU$gpu)" | tee -a logs/ahnbi1/day5.log
  CUDA_VISIBLE_DEVICES=$gpu python run_experiments.py \
    --model "$model" --depth "$depth" --dataset "$dataset" --seed "$seed" \
    >> logs/ahnbi1/${model}_d${depth}_${dataset}_s${seed}.log 2>&1
  echo "[$(date)] DONE: $model d${depth} $dataset s${seed} (GPU$gpu)" | tee -a logs/ahnbi1/day5.log
}

# 2개 모델 → GPU 0, 1 병렬
run_one 0 baseline 200 cifar10 123 &
run_one 1 lrs_low  200 cifar10 123 &
wait

echo "[$(date)] === Day5 Complete ===" | tee -a logs/ahnbi1/day5.log
