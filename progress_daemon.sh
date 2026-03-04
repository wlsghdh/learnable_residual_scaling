#!/bin/bash
# progress_daemon.sh — train_progress.md 30분마다 자동 갱신
# ==========================================================
# 실행:
#   nohup bash progress_daemon.sh > /dev/null 2>&1 &
#   echo $! > logs/progress_daemon.pid

INTERVAL=1800  # 30분
mkdir -p logs

echo "[$(date '+%H:%M:%S')] Progress daemon 시작 (${INTERVAL}s 간격)"

while true; do
    python scripts/update_progress.py
    sleep $INTERVAL
done
