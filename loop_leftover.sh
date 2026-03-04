#!/bin/bash
# loop_leftover.sh — GPU2 루프 종료 대기 → 잔여 실험 자동 실행
# =====================================================
# 1. loop_master_gpu2.sh 프로세스가 완전히 종료될 때까지 대기
# 2. 종료 확인 후 master_run_leftover.sh 를 반복 실행:
#    - 최대 47시간 실행 → 20분 sleep → 재시작
#    - 완료된 실험은 skip, 새로 완료된 실험이 없으면 종료
#
# 실행:
#   nohup bash loop_leftover.sh [GPU_ID] > logs/loop_leftover.log 2>&1 &
#   echo $! > logs/loop_leftover.pid
#
# GPU_ID 기본값: 1 (GPU2 루프 완료 후 GPU2 사용 원하면 2 지정)

GPU=${1:-1}
SLEEP_MIN=20
LOG=logs/loop_leftover.log
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG
}

log "=== Loop Leftover 시작 (GPU=${GPU}) ==="
log "    현재 정책: 최대 1 GPU (GPU 0-3만 허용)"

# ── GPU2 루프가 완전히 종료될 때까지 대기 ──────────────────────
log "    loop_master_gpu2.sh 종료 대기 중..."
WAIT_COUNT=0
while pgrep -u "$(whoami)" -f "loop_master_gpu2.sh" > /dev/null 2>&1; do
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [ $((WAIT_COUNT % 12)) -eq 0 ]; then
        log "    ... 아직 GPU2 루프 실행 중 (대기 $((WAIT_COUNT * 5))분 경과)"
    fi
    sleep 300  # 5분마다 체크
done

log "    GPU2 루프 종료 확인! 잔여 실험 시작합니다."
log "    대상: batch3a_tier2_part1 (잔여) → batch6_wrn → batch7_imagenet_gpu1"
echo "" | tee -a $LOG

# ── 잔여 실험 반복 실행 ────────────────────────────────────────
ROUND=0
while true; do
    ROUND=$((ROUND + 1))
    log "--- Round ${ROUND} 시작 ---"

    BEFORE=$(find results-json -name "*_result.json" 2>/dev/null | wc -l)

    bash master_run_leftover.sh $GPU

    AFTER=$(find results-json -name "*_result.json" 2>/dev/null | wc -l)
    NEW=$((AFTER - BEFORE))
    log "--- Round ${ROUND} 완료: 새로 완료된 실험 ${NEW}개 (총 ${AFTER}개) ---"

    if [ "$NEW" -eq 0 ]; then
        log "새로 완료된 실험 없음 → 모든 잔여 실험 완료! Loop 종료."
        break
    fi

    log ">>> ${SLEEP_MIN}분 sleep 시작 (48시간 규칙 준수) <<<"
    sleep $((SLEEP_MIN * 60))
    log ">>> sleep 완료, 재시작 <<<"
done

log "=== Loop Leftover 전체 완료 ==="
