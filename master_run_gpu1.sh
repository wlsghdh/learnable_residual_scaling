#!/bin/bash
# master_run_gpu1.sh — GPU 1 마스터 실행 스크립트
# =====================================================
# GPU 1에서 순차적으로 배치를 실행.
# 48시간 제한 체크: 47시간 초과 시 안전하게 중단.
# 이미 결과가 있는 실험은 각 배치에서 자동 skip.
#
# 실행:
#   nohup bash master_run_gpu1.sh > logs/master_gpu1.log 2>&1 &
#   echo $! > logs/master_gpu1.pid
#
# 예상 순서:
#   batch1_pilot (~12h)         — CIFAR pilot study
#   batch2a_tier1_part1 (~20h)  — Tier1 전반
#   batch4_tier3_ablation (~32h)— Tier3 + Ablation
#   batch3a_tier2_part1 (~15h)  — Tier2 전반 (GPU2 부담 분담, lrs_mid/lrs_high/hybrida)
#   batch6_wrn (~15h)           — WRN-28-10
#   batch7_imagenet_gpu1 (~36h) — ImageNet 3 runs
#
# ⚠️ 각 배치가 48시간을 넘으면 다음 배치로 넘어가지 않고 중단.
#    다시 실행하면 이미 완료된 실험은 skip되고 이어서 진행.

GPU=1
MAX_SECONDS=$((47 * 3600))   # 47시간 (1시간 여유)
START=$(date +%s)

LOG_DIR=logs
mkdir -p $LOG_DIR
MASTER_LOG=$LOG_DIR/master_gpu${GPU}.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $MASTER_LOG
}

# 48시간 체크 후 배치 실행
run_if_time_left() {
    local script=$1
    local gpu=$2
    local elapsed=$(( $(date +%s) - START ))
    local remaining=$(( MAX_SECONDS - elapsed ))
    if [ $elapsed -gt $MAX_SECONDS ]; then
        log "⚠️  47시간 초과 (elapsed=${elapsed}s) — 안전하게 중단."
        log "    다음 실행 시 $script 부터 이어서 진행됩니다."
        exit 0
    fi
    log "▶ Starting: $script (elapsed=${elapsed}s, remaining=${remaining}s)"
    bash "$script" "$gpu"
    local rc=$?
    log "✓ Done: $script (exit=$rc)"
    sleep 300
}

log "=== Master GPU $GPU Start ==="
log "   MAX_SECONDS=$MAX_SECONDS (47h)"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv | tee -a $MASTER_LOG
echo "" | tee -a $MASTER_LOG

# ── CIFAR 배치 ─────────────────────────────────────
# batch1_pilot은 watcher 실행 전에 이미 완료됨 → skip
run_if_time_left scripts/run/batch2a_tier1_part1.sh $GPU
run_if_time_left scripts/run/batch4_tier3_ablation.sh $GPU
run_if_time_left scripts/run/batch3a_tier2_part1.sh $GPU   # GPU2 부담 분담
run_if_time_left scripts/run/batch6_wrn.sh $GPU

# ── ImageNet (CIFAR 완료 후) ────────────────────────
run_if_time_left scripts/run/batch7_imagenet_gpu1.sh $GPU

log "=== Master GPU $GPU — All batches completed ==="
