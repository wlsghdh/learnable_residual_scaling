#!/bin/bash
# monitor.sh — 실험 진행 상황 모니터
# =====================================
# 사용: bash monitor.sh
#       bash monitor.sh --gpu     (GPU 상태만)
#       bash monitor.sh --results (결과 요약만)
#       bash monitor.sh --running (실행 중인 것만)

MODE=${1:-"all"}

SEP="─────────────────────────────────────────────────────────"

# ── GPU 상태 ─────────────────────────────────────────────
show_gpu() {
    echo ""
    echo "━━━ GPU 상태 $(date '+%Y-%m-%d %H:%M:%S') ━━━"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader | awk -F', ' '{
        printf "  GPU %s (%s): %s util, %s/%s, %s°C\n", $1, $2, $3, $4, $5, $6
    }'
}

# ── 실행 중인 실험 ────────────────────────────────────────
show_running() {
    echo ""
    echo "━━━ 실행 중인 프로세스 ━━━"
    local procs
    procs=$(ps aux | grep -E "train\.py|train_imagenet\.py|run_experiments\.py" | grep -v grep)
    if [ -z "$procs" ]; then
        echo "  (실행 중인 실험 없음)"
    else
        echo "$procs" | awk '{printf "  PID %s | GPU=%s | %s\n", $2, ENVIRON["CUDA_VISIBLE_DEVICES"], $0}' | head -10
        ps aux | grep -E "train\.py|train_imagenet\.py|run_experiments\.py" | grep -v grep | \
            awk '{print "  PID", $2, "| CPU:", $3"%", "| MEM:", $4"%", "| CMD:", $11, $12, $13, $14, $15}'
    fi

    # Master 프로세스
    echo ""
    local masters
    masters=$(ps aux | grep -E "master_run_gpu[0-9]" | grep -v grep)
    if [ -n "$masters" ]; then
        echo "  Master 스크립트:"
        echo "$masters" | awk '{print "  PID", $2, "|", $11, $12}'
    fi
}

# ── 완료된 결과 수 ────────────────────────────────────────
show_results() {
    echo ""
    echo "━━━ 완료된 결과 ━━━"

    local RDIR=results-json

    # CIFAR 결과
    local cifar10=$(find $RDIR -maxdepth 1 -name "*cifar10*_result.json" 2>/dev/null | wc -l)
    local cifar100=$(find $RDIR -maxdepth 1 -name "*cifar100*_result.json" 2>/dev/null | wc -l)
    echo "  CIFAR-10:  ${cifar10} json files"
    echo "  CIFAR-100: ${cifar100} json files"

    # 모델별 요약
    echo ""
    echo "  모델별 완료 수 (CIFAR-10 + CIFAR-100):"
    for m in baseline lrs_low lrs_mid lrs_high rezero hybrida \
              lrs_hybrida_low lrs_hybrida_mid lrs_hybrida_high \
              skipinit fixup layerscale \
              fixed_alpha_01 fixed_alpha_03 fixed_alpha_05 fixed_alpha_07 \
              per_channel_lrs \
              wrn_baseline wrn_lrs_low wrn_lrs_hybrida_low wrn_rezero wrn_skipinit wrn_layerscale \
              resnext_baseline resnext_lrs_low resnext_lrs_hybrida_low resnext_rezero resnext_skipinit resnext_layerscale; do
        local cnt
        cnt=$(find $RDIR -maxdepth 1 -name "${m}_depth*_result.json" 2>/dev/null | wc -l)
        if [ "$cnt" -gt 0 ]; then
            printf "    %-30s %2d files\n" "$m" "$cnt"
        fi
    done

    # ImageNet 결과
    echo ""
    local imgnet_dir=$RDIR/imagenet
    if [ -d "$imgnet_dir" ]; then
        local imgnet=$(find $imgnet_dir -name "*_result.json" 2>/dev/null | wc -l)
        echo "  ImageNet: ${imgnet}/5 models"
        if [ "$imgnet" -gt 0 ]; then
            for f in $imgnet_dir/*_result.json; do
                local name=$(basename $f _imagenet_result.json)
                local top1=$(python3 -c "import json; d=json.load(open('$f')); print(d['results']['best_top1'])" 2>/dev/null || echo "?")
                local top5=$(python3 -c "import json; d=json.load(open('$f')); print(d['results']['best_top5'])" 2>/dev/null || echo "?")
                printf "    %-20s Top-1: %s%%  Top-5: %s%%\n" "$name" "$top1" "$top5"
            done
        fi
    else
        echo "  ImageNet: 0/5 models"
    fi
}

# ── 로그 마지막 줄 ────────────────────────────────────────
show_logs() {
    echo ""
    echo "━━━ 최근 로그 (마지막 줄) ━━━"
    local LDIR=logs
    if [ -d "$LDIR" ]; then
        for f in $LDIR/master_gpu*.log; do
            if [ -f "$f" ]; then
                echo "  [$(basename $f)]"
                tail -3 "$f" | sed 's/^/    /'
            fi
        done
        for f in $LDIR/imagenet/batch7_gpu*.log; do
            if [ -f "$f" ]; then
                echo "  [$(basename $f)]"
                tail -2 "$f" | sed 's/^/    /'
            fi
        done
    fi
}

# ── 전체 출력 ─────────────────────────────────────────────
case $MODE in
    --gpu)     show_gpu ;;
    --results) show_results ;;
    --running) show_running ;;
    *)
        show_gpu
        show_running
        show_results
        show_logs
        echo ""
        ;;
esac
