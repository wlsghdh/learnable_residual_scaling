#!/bin/bash
# batch4_tier3_ablation.sh — Phase 2-E+H: Tier 3 (Fixed-α) + Ablation
# 예상 소요시간: ~32시간 (GPU 1장)
# 실행: bash scripts/run/batch4_tier3_ablation.sh [GPU_ID]
#
# [Tier 3] Fixed-α ablation
#   Models: fixed_alpha_01/03/05/07
#   Depth: 152 / Dataset: cifar100 / Seeds: 42, 123, 456
#   = 4 × 1 × 1 × 3 = 12 runs
#
# [Ablation H] Per-channel LRS
#   Models: per_channel_lrs
#   Depths: 50, 152 / Dataset: cifar100 / Seeds: 42, 123, 456
#   = 1 × 2 × 1 × 3 = 6 runs
#
# [Ablation H] 200 Epoch (epochs override)
#   Models: baseline, lrs_low, lrs_hybrida_low
#   Depth: 152 / Dataset: cifar100 / Seeds: 42, 123, 456
#   ※ epochs=200은 config.py에서 별도 세팅 필요 → 향후 구현
#
# Total Phase 2-E: 12 runs  (즉시 실행 가능)
# Total Phase 2-H per_channel: 6 runs
set -e

GPU_ID=${1:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

LOG_DIR=logs
mkdir -p $LOG_DIR
BATCH_LOG=$LOG_DIR/batch4_tier3_ablation.log

echo "=== Tier3+Ablation Start: $(date) | GPU: $GPU_ID ===" | tee -a $BATCH_LOG
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv | tee -a $BATCH_LOG

# ── Tier 3: Fixed-α ──────────────────────────────────────────
echo "" | tee -a $BATCH_LOG
echo "=== [Tier 3] Fixed-alpha ablation ===" | tee -a $BATCH_LOG

FIXED_MODELS=("fixed_alpha_01" "fixed_alpha_03" "fixed_alpha_05" "fixed_alpha_07")
SEEDS=(42 123 456)

for model in "${FIXED_MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    RESULT_FILE="results-json/${model}_depth152_cifar100_seed${seed}_result.json"
    if [ -f "$RESULT_FILE" ]; then
      echo "[$(date)] SKIP (exists): $model d152 cifar100 s$seed" | tee -a $BATCH_LOG
      continue
    fi
    echo "[$(date)] Running: $model | depth=152 | cifar100 | seed=$seed" | tee -a $BATCH_LOG
    python run_experiments.py \
      --model "$model" \
      --depth 152 \
      --dataset cifar100 \
      --seed "$seed" \
      >> $LOG_DIR/${model}_d152_cifar100_s${seed}.log 2>&1
    echo "[$(date)] Done:    $model | depth=152 | cifar100 | seed=$seed" | tee -a $BATCH_LOG
    sleep 60
  done
done

# ── Ablation H: Per-channel LRS ──────────────────────────────
echo "" | tee -a $BATCH_LOG
echo "=== [Ablation] Per-channel LRS ===" | tee -a $BATCH_LOG

for depth in 50 152; do
  for seed in "${SEEDS[@]}"; do
    RESULT_FILE="results-json/per_channel_lrs_depth${depth}_cifar100_seed${seed}_result.json"
    if [ -f "$RESULT_FILE" ]; then
      echo "[$(date)] SKIP (exists): per_channel_lrs d$depth cifar100 s$seed" | tee -a $BATCH_LOG
      continue
    fi
    echo "[$(date)] Running: per_channel_lrs | depth=$depth | cifar100 | seed=$seed" | tee -a $BATCH_LOG
    python run_experiments.py \
      --model per_channel_lrs \
      --depth "$depth" \
      --dataset cifar100 \
      --seed "$seed" \
      >> $LOG_DIR/per_channel_lrs_d${depth}_cifar100_s${seed}.log 2>&1
    echo "[$(date)] Done:    per_channel_lrs | depth=$depth | cifar100 | seed=$seed" | tee -a $BATCH_LOG
    sleep 60
  done
done

echo "=== Tier3+Ablation Complete: $(date) ===" | tee -a $BATCH_LOG
