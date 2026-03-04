"""
update_progress.py — train_progress.md 자동 생성
=================================================
30분마다 호출됨. 결과 JSON을 읽어서 진행 상황을 마크다운으로 정리.
사용: python scripts/update_progress.py
"""

import os
import json
import subprocess
from datetime import datetime

RDIR = "results-json"
OUT  = "train_progress.md"

# ── 전체 실험 계획 ───────────────────────────────────────────
# (model, depth, dataset, seed) 형태로 정의
PLAN = {}

def add(phase, model, depths, datasets, seeds):
    for d in depths:
        for ds in datasets:
            for s in seeds:
                PLAN.setdefault(phase, []).append((model, d, ds, s))

# Phase 2-A: Pilot Study (batch1)
for m in ["plain_he", "plain_identity", "baseline", "resnet_identity_all", "hybrida", "lrs_hybrida_low"]:
    add("2-A Pilot", m, [50, 152], ["cifar10", "cifar100"], [42])

# Phase 2-C Tier1 Part1 (batch2a): baseline, lrs_low, lrs_hybrida_low
for m in ["baseline", "lrs_low", "lrs_hybrida_low"]:
    add("2-C Tier1", m, [50, 152], ["cifar10", "cifar100"], [42, 123, 456])

# Phase 2-C Tier1 Part2 (batch2b): rezero, skipinit, fixup, layerscale
for m in ["rezero", "skipinit", "fixup", "layerscale"]:
    add("2-C Tier1", m, [50, 152], ["cifar10", "cifar100"], [42, 123, 456])

# Phase 2-D Tier2 Part1 (batch3a)
for m in ["lrs_mid", "lrs_high", "hybrida"]:
    add("2-D Tier2", m, [50, 152], ["cifar10", "cifar100"], [42, 123, 456])

# Phase 2-D Tier2 Part2 (batch3b)
for m in ["lrs_hybrida_mid", "lrs_hybrida_high"]:
    add("2-D Tier2", m, [50, 152], ["cifar10", "cifar100"], [42, 123, 456])

# Phase 2-E Tier3 (batch4): fixed_alpha
for m in ["fixed_alpha_01", "fixed_alpha_03", "fixed_alpha_05", "fixed_alpha_07"]:
    add("2-E Tier3", m, [152], ["cifar100"], [42, 123, 456])
# per_channel_lrs
for m in ["per_channel_lrs"]:
    add("2-E Tier3", m, [50, 152], ["cifar100"], [42, 123, 456])

# Phase 2-F Tier4 (batch5)
for m in ["baseline", "lrs_low", "lrs_hybrida_low"]:
    add("2-F Tier4", m, [101, 200], ["cifar10", "cifar100"], [42, 123, 456])

# Phase 2-G WRN (batch6_wrn)
for m in ["wrn_baseline", "wrn_lrs_low", "wrn_lrs_hybrida_low", "wrn_rezero", "wrn_skipinit", "wrn_layerscale"]:
    add("2-G WRN", m, [28], ["cifar10", "cifar100"], [42, 123, 456])

# Phase 2-G ResNeXt (batch6_resnext)
for m in ["resnext_baseline", "resnext_lrs_low", "resnext_lrs_hybrida_low", "resnext_rezero", "resnext_skipinit", "resnext_layerscale"]:
    add("2-G ResNeXt", m, [50], ["cifar10", "cifar100"], [42, 123, 456])

# Phase 4: ImageNet (batch7)
for m in ["baseline", "lrs_low", "lrs_hybrida_low", "rezero", "lrs_mid"]:
    PLAN.setdefault("4 ImageNet", []).append((m, 50, "imagenet", 42))


# ── 완료 여부 확인 ───────────────────────────────────────────
def result_exists(model, depth, dataset, seed):
    if dataset == "imagenet":
        alias = {"lrs_hybrida_low": "lrs_ha_low"}.get(model, model)
        path = os.path.join(RDIR, "imagenet", f"{alias}_imagenet_result.json")
        return os.path.exists(path)
    # seed suffix 파일 우선, 없으면 no-suffix 파일 확인
    p1 = os.path.join(RDIR, f"{model}_depth{depth}_{dataset}_seed{seed}_result.json")
    p2 = os.path.join(RDIR, f"{model}_depth{depth}_{dataset}_result.json")
    return os.path.exists(p1) or os.path.exists(p2)


def load_best_acc(model, depth, dataset, seed):
    if dataset == "imagenet":
        alias = {"lrs_hybrida_low": "lrs_ha_low"}.get(model, model)
        path = os.path.join(RDIR, "imagenet", f"{alias}_imagenet_result.json")
    else:
        p1 = os.path.join(RDIR, f"{model}_depth{depth}_{dataset}_seed{seed}_result.json")
        p2 = os.path.join(RDIR, f"{model}_depth{depth}_{dataset}_result.json")
        path = p1 if os.path.exists(p1) else p2
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
        acc = d["results"].get("best_top1") or d["results"].get("best_acc", 0) * 100
        return round(acc, 2)
    except:
        return None


# ── 현재 실행 중인 실험 ──────────────────────────────────────
def get_running():
    try:
        out = subprocess.check_output(
            ["ps", "aux"], text=True, stderr=subprocess.DEVNULL)
        running = []
        for line in out.splitlines():
            if "run_experiments.py" in line or "train_imagenet.py" in line:
                if "grep" in line:
                    continue
                parts = line.split()
                # --model, --depth, --dataset, --seed 파싱
                args = parts[10:]
                info = {}
                for i, a in enumerate(args):
                    if a in ("--model", "--depth", "--dataset", "--seed") and i+1 < len(args):
                        info[a[2:]] = args[i+1]
                if info:
                    running.append(info)
        # deduplicate
        seen = set()
        unique = []
        for r in running:
            key = (r.get("model",""), r.get("depth",""), r.get("dataset",""), r.get("seed",""))
            if key not in seen:
                seen.add(key)
                unique.append(r)
        return unique
    except:
        return []


# ── GPU 상태 ─────────────────────────────────────────────────
def get_gpu_status():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"], text=True)
        lines = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            idx, util, mem_used, mem_total = parts
            util_val = int(util.replace("%","").strip())
            flag = "🟢" if util_val > 10 else "⚪"
            lines.append(f"| GPU{idx} | {util} | {mem_used}/{mem_total} | {flag} |")
        return lines
    except:
        return ["| (nvidia-smi 실패) | | | |"]


# ── 마크다운 생성 ─────────────────────────────────────────────
def generate():
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    running = get_running()
    gpu_lines = get_gpu_status()

    lines = []
    lines.append(f"# LRS 학습 진행 상황")
    lines.append(f"> 마지막 업데이트: **{now}** (30분마다 자동 갱신)")
    lines.append("")

    # ── 전체 진행률 ──────────────────────────────────────────
    total_all = sum(len(v) for v in PLAN.values())
    done_all  = sum(1 for phase, runs in PLAN.items()
                    for (m, d, ds, s) in runs if result_exists(m, d, ds, s))
    pct = done_all / total_all * 100 if total_all else 0

    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)

    lines.append(f"## 전체 진행률")
    lines.append(f"```")
    lines.append(f"[{bar}] {done_all}/{total_all} ({pct:.1f}%)")
    lines.append(f"```")
    lines.append("")

    # ── 현재 실행 중 ─────────────────────────────────────────
    lines.append("## 현재 실행 중")
    if running:
        for r in running:
            lines.append(f"- `{r.get('model','?')}` depth={r.get('depth','?')} "
                         f"{r.get('dataset','?')} seed={r.get('seed','?')}")
    else:
        lines.append("- (실행 중인 실험 없음 — 배치 간 sleep 중일 수 있음)")
    lines.append("")

    # ── GPU 상태 ─────────────────────────────────────────────
    lines.append("## GPU 상태")
    lines.append("| GPU | 사용률 | 메모리 | 상태 |")
    lines.append("|-----|--------|--------|------|")
    for gl in gpu_lines:
        lines.append(gl)
    lines.append("")

    # ── Phase별 상세 ─────────────────────────────────────────
    lines.append("## Phase별 진행 상황")
    lines.append("")

    for phase, runs in PLAN.items():
        done  = sum(1 for m, d, ds, s in runs if result_exists(m, d, ds, s))
        total = len(runs)
        p     = done / total * 100 if total else 0
        b_len = 20
        b_filled = int(b_len * p / 100)
        b = "█" * b_filled + "░" * (b_len - b_filled)
        status = "✅" if done == total else ("🔄" if done > 0 else "⬜")
        lines.append(f"### {status} Phase {phase}  `[{b}] {done}/{total} ({p:.0f}%)`")
        lines.append("")

        # 모델별 요약
        models_in_phase = list(dict.fromkeys(m for m, d, ds, s in runs))
        for model in models_in_phase:
            model_runs = [(d, ds, s) for m, d, ds, s in runs if m == model]
            model_done = sum(1 for d, ds, s in model_runs if result_exists(model, d, ds, s))
            model_total = len(model_runs)

            # 대표 정확도 (cifar100, 가장 높은 depth, seed42)
            best_acc = None
            for d in sorted(set(d for d, ds, s in model_runs), reverse=True):
                for ds in ["cifar100", "cifar10"]:
                    acc = load_best_acc(model, d, ds, 42)
                    if acc is not None:
                        best_acc = f"{acc:.2f}% ({ds} d{d})"
                        break
                if best_acc:
                    break

            acc_str = f" → {best_acc}" if best_acc else ""
            icon = "✅" if model_done == model_total else ("🔄" if model_done > 0 else "⬜")
            lines.append(f"  {icon} `{model}` {model_done}/{model_total}{acc_str}")
        lines.append("")

    # ── 주목할 결과 ──────────────────────────────────────────
    lines.append("## 주목할 결과")
    lines.append("")
    lines.append("| 모델 | Dataset | Depth | Best Acc |")
    lines.append("|------|---------|-------|----------|")

    highlights = [
        ("baseline",        "cifar100", [50, 152, 200]),
        ("lrs_low",         "cifar100", [50, 152, 200]),
        ("lrs_hybrida_low", "cifar100", [50, 152, 200]),
        ("rezero",          "cifar100", [152]),
        ("lrs_high",        "cifar100", [200]),  # collapse
    ]
    for model, ds, depths in highlights:
        for depth in depths:
            acc = load_best_acc(model, depth, ds, 42)
            if acc is not None:
                flag = " ⚠️ (collapse!)" if acc < 20 else ""
                lines.append(f"| `{model}` | {ds} | {depth} | **{acc:.2f}%**{flag} |")
    lines.append("")

    lines.append("---")
    lines.append(f"*자동 생성: `python scripts/update_progress.py`*")

    return "\n".join(lines)


if __name__ == "__main__":
    content = generate()
    with open(OUT, "w") as f:
        f.write(content)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] train_progress.md 업데이트 완료")
