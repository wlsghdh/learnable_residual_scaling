#!/bin/bash
set -e
cd "$(dirname "$0")/../../.."
mkdir -p logs/ahnbi2

echo "[$(date)] === Day2: depth101 cifar100 seed456 (lrs_mid, lrs_high, rezero) ===" | tee -a logs/ahnbi2/day2.log

run_one() {
  local gpu=$1 model=$2 depth=$3 dataset=$4 seed=$5
  RESULT="results-json/${model}_depth${depth}_${dataset}_seed${seed}_result.json"
  if [ -f "$RESULT" ]; then
    echo "[$(date)] SKIP: $model d${depth} $dataset s${seed} (GPU$gpu)" | tee -a logs/ahnbi2/day2.log
    return
  fi
  echo "[$(date)] START: $model d${depth} $dataset s${seed} (GPU$gpu)" | tee -a logs/ahnbi2/day2.log
  CUDA_VISIBLE_DEVICES=$gpu python run_experiments.py \
    --model "$model" --depth "$depth" --dataset "$dataset" --seed "$seed" \
    >> logs/ahnbi2/${model}_d${depth}_${dataset}_s${seed}.log 2>&1
  echo "[$(date)] DONE: $model d${depth} $dataset s${seed} (GPU$gpu)" | tee -a logs/ahnbi2/day2.log
}

# 3개 모델 → 2개 병렬 + 1개 순차
run_one 0 lrs_mid  101 cifar100 456 &
run_one 1 lrs_high 101 cifar100 456 &
wait

run_one 0 rezero   101 cifar100 456

echo "[$(date)] === Day2 Complete ===" | tee -a logs/ahnbi2/day2.log
