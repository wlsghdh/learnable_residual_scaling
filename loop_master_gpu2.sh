#!/bin/bash
# loop_master_gpu2.sh — GPU 2 자동 반복 실행
# =============================================
# master_run_gpu2.sh를 반복 실행:
#   1. master 실행 (최대 47시간)
#   2. 20분 sleep (48시간 규칙 준수)
#   3. 재시작 (완료된 실험은 자동 skip)
#   4. 새로 완료된 실험이 없으면 → 모두 완료, 종료
#
# 실행:
#   nohup bash loop_master_gpu2.sh > logs/loop_master_gpu2.log 2>&1 &
#   echo $! > logs/loop_master_gpu2.pid

GPU=2
SLEEP_MIN=20
LOG=logs/loop_master_gpu${GPU}.log
mkdir -p logs

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG
}

log "=== Loop Master GPU${GPU} 시작 ==="
log "    47시간 실행 → ${SLEEP_MIN}분 sleep → 반복"

ROUND=0
while true; do
    ROUND=$((ROUND + 1))
    log "--- Round ${ROUND} 시작 ---"

    BEFORE=$(find results-json -maxdepth 1 -name "*_result.json" 2>/dev/null | wc -l)
    BEFORE=$((BEFORE + $(find results-json/imagenet -name "*_result.json" 2>/dev/null | wc -l)))

    bash master_run_gpu${GPU}.sh

    AFTER=$(find results-json -maxdepth 1 -name "*_result.json" 2>/dev/null | wc -l)
    AFTER=$((AFTER + $(find results-json/imagenet -name "*_result.json" 2>/dev/null | wc -l)))

    NEW=$((AFTER - BEFORE))
    log "--- Round ${ROUND} 완료: 새로 완료된 실험 ${NEW}개 (총 ${AFTER}개) ---"

    if [ "$NEW" -eq 0 ]; then
        log "새로 완료된 실험 없음 → 모든 실험 완료! Loop 종료."
        break
    fi

    log ">>> ${SLEEP_MIN}분 sleep 시작 (48시간 규칙 준수) <<<"
    sleep $((SLEEP_MIN * 60))
    log ">>> sleep 완료, 재시작 <<<"
done

log "=== Loop Master GPU${GPU} 전체 완료 ==="
