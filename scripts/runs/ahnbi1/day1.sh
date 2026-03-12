#!/bin/bash
set -e
cd "$(dirname "$0")/../../.."
mkdir -p logs/ahnbi1

echo "[$(date)] === Day1: depth101 cifar10 seed123 (lrs_high, rezero) ===" | tee -a logs/ahnbi1/day1.log

run_one() {
  local gpu=$1 model=$2 depth=$3 dataset=$4 seed=$5
  RESULT="results-json/${model}_depth${depth}_${dataset}_seed${seed}_result.json"
  if [ -f "$RESULT" ]; then
    echo "[$(date)] SKIP: $model d${depth} $dataset s${seed} (GPU$gpu)" | tee -a logs/ahnbi1/day1.log
    return
  fi
  echo "[$(date)] START: $model d${depth} $dataset s${seed} (GPU$gpu)" | tee -a logs/ahnbi1/day1.log
  CUDA_VISIBLE_DEVICES=$gpu python run_experiments.py \
    --model "$model" --depth "$depth" --dataset "$dataset" --seed "$seed" \
    >> logs/ahnbi1/${model}_d${depth}_${dataset}_s${seed}.log 2>&1
  echo "[$(date)] DONE: $model d${depth} $dataset s${seed} (GPU$gpu)" | tee -a logs/ahnbi1/day1.log
}

# 2개 모델 → GPU 0, 1 병렬
run_one 0 lrs_high 101 cifar10 123 &
run_one 1 rezero   101 cifar10 123 &
wait

echo "[$(date)] === Day1 Complete ===" | tee -a logs/ahnbi1/day1.log
