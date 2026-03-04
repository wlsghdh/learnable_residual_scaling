#!/bin/bash
# master_run_gpu2.sh — GPU 2 마스터 실행 스크립트
# =====================================================
# GPU 2에서 순차적으로 배치를 실행.
# 48시간 제한 체크: 47시간 초과 시 안전하게 중단.
# 이미 결과가 있는 실험은 각 배치에서 자동 skip.
#
# 실행:
#   nohup bash master_run_gpu2.sh > logs/master_gpu2.log 2>&1 &
#   echo $! > logs/master_gpu2.pid
#
# 예상 순서:
#   batch2b_tier1_part2 (~20h)  — Tier1 후반
#   batch3b_tier2_part2 (~15h)  — Tier2 후반 (batch3a는 GPU1이 담당)
#   batch5_tier4 (~25h)         — Tier4 추가 Depth
#   batch6_resnext (~15h)       — ResNeXt-50
#   batch7_imagenet_gpu2 (~24h) — ImageNet 2 runs
#
# ⚠️ 각 배치가 48시간을 넘으면 다음 배치로 넘어가지 않고 중단.
#    다시 실행하면 이미 완료된 실험은 skip되고 이어서 진행.

GPU=2
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
run_if_time_left scripts/run/batch2b_tier1_part2.sh $GPU
# batch3a는 GPU1(master_run_gpu1.sh)이 담당 → 여기서 제거
run_if_time_left scripts/run/batch3b_tier2_part2.sh $GPU
run_if_time_left scripts/run/batch5_tier4.sh $GPU
# batch6_resnext → ahnbi 서버에서 병렬 실행 (GPU0)

# ── ImageNet (CIFAR 완료 후) ────────────────────────
run_if_time_left scripts/run/batch7_imagenet_gpu2.sh $GPU

log "=== Master GPU $GPU — All batches completed ==="
