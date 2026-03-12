#!/bin/bash
set -e

if [ -z "$1" ]; then
  echo "Usage: bash day5.sh <GPU_ID>"
  echo "Example: bash day5.sh 3"
  exit 1
fi

GPU_ID=$1
export CUDA_VISIBLE_DEVICES=$GPU_ID
cd "$(dirname "$0")/../../.."
mkdir -p logs/lifeai results-json/imagenet

MODEL=lrs_mid
echo "[$(date)] === Day5: ImageNet $MODEL (GPU$GPU_ID) ===" | tee -a logs/lifeai/day5.log

RESULT="results-json/imagenet/${MODEL}_imagenet_result.json"
if [ -f "$RESULT" ]; then
  echo "[$(date)] SKIP (exists): $MODEL" | tee -a logs/lifeai/day5.log
else
  echo "[$(date)] START: $MODEL (GPU$GPU_ID)" | tee -a logs/lifeai/day5.log
  python train_imagenet.py \
    --model "$MODEL" \
    --data /nfs_share/datasets/ILSVRC2012 \
    --output results-json/imagenet \
    --epochs 90 \
    --batch-size 512 \
    --lr 0.2 \
    --warmup-epochs 5 \
    --workers 8 \
    --seed 42 \
    >> logs/lifeai/${MODEL}_imagenet.log 2>&1
  echo "[$(date)] DONE: $MODEL" | tee -a logs/lifeai/day5.log
fi

echo "[$(date)] === Day5 Complete ===" | tee -a logs/lifeai/day5.log
