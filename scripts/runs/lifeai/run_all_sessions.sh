#!/bin/bash
# Session 1 → 2 → 3 자동 연속 실행
# 세션 사이 30분 쿨다운 (세션 내 실험 사이는 15분)

if [ -z "$1" ]; then
  echo "Usage: bash run_all_sessions.sh <GPU_ID>"
  echo "Example: bash run_all_sessions.sh 3"
  exit 1
fi

GPU_ID=$1
DIR="$(dirname "$0")"

echo "[$(date)] === Session 1 시작 ==="
bash "$DIR/session1.sh" "$GPU_ID"

echo "[$(date)] Session 1 완료. 30분 쿨다운 후 Session 2 시작..."
sleep 1800

echo "[$(date)] === Session 2 시작 ==="
bash "$DIR/session2.sh" "$GPU_ID"

echo "[$(date)] Session 2 완료. 30분 쿨다운 후 Session 3 시작..."
sleep 1800

echo "[$(date)] === Session 3 시작 ==="
bash "$DIR/session3.sh" "$GPU_ID"

echo "[$(date)] === 전체 완료! ==="
