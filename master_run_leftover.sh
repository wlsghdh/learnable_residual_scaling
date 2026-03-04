#!/bin/bash
# master_run_leftover.sh — 구 GPU1 담당 잔여 실험 (단일 GPU)
# =====================================================
# batch3a_tier2_part1 (잔여) + batch6_wrn + batch7_imagenet_gpu1
# skip 로직 내장 → 이미 완료된 실험은 자동으로 건너뜀.
#
# 실행:
#   bash master_run_leftover.sh [GPU_ID]
# (loop_leftover.sh 에서 자동 호출)

GPU=${1:-1}
MAX_SECONDS=$((47 * 3600))   # 47시간 (1시간 여유)
START=$(date +%s)

LOG_DIR=logs
mkdir -p $LOG_DIR
MASTER_LOG=$LOG_DIR/master_leftover.log

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $MASTER_LOG
}

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
    log "▶ Starting: $script (GPU=$gpu, elapsed=${elapsed}s, remaining=${remaining}s)"
    bash "$script" "$gpu"
    local rc=$?
    log "✓ Done: $script (exit=$rc)"
    sleep 300
}

log "=== Master Leftover GPU $GPU Start ==="
log "   MAX_SECONDS=$MAX_SECONDS (47h)"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv | tee -a $MASTER_LOG
echo "" | tee -a $MASTER_LOG

# batch3a → ahnbi GPU1에서 처리
# batch6_wrn → ahnbi GPU1에서 처리
run_if_time_left scripts/run/batch7_imagenet_gpu1.sh $GPU

log "=== Master Leftover — All batches completed ==="
