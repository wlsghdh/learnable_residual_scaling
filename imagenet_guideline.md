# ImageNet 실험 가이드 (lifeai 서버)

## 개요
- **서버**: lifeai (A100 80GB PCIe × 6)
- **GPU 사용**: 1장만 사용
- **모델**: baseline, lrs_low, lrs_mid, rezero (총 4 runs)
- **아키텍처**: ResNet-50 (16 blocks)
- **데이터**: `/nfs_share/datasets/ILSVRC2012/`
- **예상 시간**: ~12h/model × 4 = **~48h (약 2일)**

---

## 학습 설정

| 항목 | 값 | 비고 |
|------|-----|------|
| Epochs | 90 | ImageNet 표준 |
| **Batch size** | **512** | A100 80GB + AMP 최적 (~25GB VRAM 사용) |
| Learning rate | 0.2 | linear scaling: 0.1 × (512/256) |
| LR schedule | Linear warmup → Cosine annealing | |
| Warmup epochs | 5 | |
| Optimizer | SGD (momentum=0.9, wd=1e-4) | |
| AMP | ✅ 활성화 | Mixed precision |
| Workers | 8 | |
| Seed | 42 | |

> **Batch size 512가 최적인 이유**: A100 80GB에서 ResNet-50 + AMP는 ~25GB 사용.
> 1024도 가능하지만 512가 수렴 안정성과 메모리 효율의 최적 균형점.

---

## 실행 방법

### 1. 사전 확인
```bash
# GPU 상태 확인
nvidia-smi

# ImageNet 데이터 확인
ls /nfs_share/datasets/ILSVRC2012/train/ | wc -l   # 1000 (클래스 수)
ls /nfs_share/datasets/ILSVRC2012/val/ | wc -l     # 1000
```

### 2. 백그라운드 실행
```bash
cd /home/jjh0709/Learnable_Reidual_Scaling

# GPU 0 사용, 백그라운드 실행
nohup bash scripts/run/lifeai/run_imagenet.sh 0 > logs/imagenet/lifeai_master.log 2>&1 &

# PID 확인
echo $!
```

### 3. 모니터링

```bash
# 전체 진행 상황 (어떤 모델이 시작/완료됐는지)
tail -f logs/imagenet/lifeai_master.log

# 현재 모델의 상세 로그 (epoch별 accuracy)
tail -f logs/imagenet/baseline_imagenet.log      # baseline 실행 중일 때
tail -f logs/imagenet/lrs_low_imagenet.log       # lrs_low 실행 중일 때
tail -f logs/imagenet/lrs_mid_imagenet.log       # lrs_mid 실행 중일 때
tail -f logs/imagenet/rezero_imagenet.log        # rezero 실행 중일 때

# GPU 사용량 실시간
nvidia-smi -l 5

# GPU 사용량 + 프로세스 (간결)
watch -n 10 nvidia-smi

# 학습 프로세스 확인
ps aux | grep train_imagenet
```

### 4. 중간 확인
```bash
# 완료된 결과 확인
ls -la results-json/lifeai/

# 결과 JSON에서 현재 best accuracy 확인
python3 -c "
import json, glob
for f in sorted(glob.glob('results-json/lifeai/*_result.json')):
    d = json.load(open(f))
    r = d['results']
    print(f'{d[\"name\"]:25s} Top-1: {r[\"best_top1\"]:.2f}%  Top-5: {r[\"best_top5\"]:.2f}%  @ epoch {r[\"best_epoch\"]}')
"
```

### 5. 종료/중단
```bash
# 현재 실행 중인 학습 중단 (필요시)
ps aux | grep train_imagenet | grep -v grep
kill <PID>

# 재실행 (이미 완료된 모델은 자동 SKIP)
nohup bash scripts/run/lifeai/run_imagenet.sh 0 > logs/imagenet/lifeai_master.log 2>&1 &
```

---

## 실행 순서 및 예상 시간

| 순서 | 모델 | 예상 시간 | 누적 |
|------|------|----------|------|
| 1 | baseline | ~12h | 12h |
| 2 | lrs_low | ~12h | 24h |
| 3 | lrs_mid | ~12h | 36h |
| 4 | rezero | ~12h | 48h |

> 각 모델 완료 시 `results-json/lifeai/{model}_imagenet_result.json` 자동 생성
> 이미 존재하는 파일은 자동 SKIP

---

## 결과 파일 구조

```
results-json/
├── lifeai/                          ← ImageNet (이 서버)
│   ├── baseline_imagenet_result.json
│   ├── lrs_low_imagenet_result.json
│   ├── lrs_mid_imagenet_result.json
│   └── rezero_imagenet_result.json
├── ahnbi/                           ← CIFAR (다른 서버)
│   ├── lrs_mid_depth101_cifar100_seed456_result.json
│   └── ...  (37 files)
└── *.json                           ← 기존 완료 결과 (그대로 유지)
```

각 JSON 내용:
```json
{
  "name": "baseline_imagenet",
  "depth": 50,
  "dataset": "imagenet",
  "results": {
    "best_top1": 76.54,        // ← 논문 Table에 사용
    "best_top5": 93.21,        // ← 논문 Table에 사용
    "best_epoch": 87,
    "alpha_final": {           // ← Fig 10(b) per-block α에 사용
      "mean": 0.35,
      "all": [0.42, 0.38, ...]
    },
    "history": {
      "val_top1": [...],       // ← 학습 커브
      "alpha_stats": [...]     // ← α 궤적
    }
  }
}
```

---

## 완료 후 할 일

```bash
# 1. 결과 확인
python3 -c "
import json, glob
for f in sorted(glob.glob('results-json/lifeai/*_result.json')):
    d = json.load(open(f))
    r = d['results']
    af = r.get('alpha_final', {})
    alpha_str = f'  α_mean={af[\"mean\"]:.4f}' if af else ''
    print(f'{d[\"name\"]:25s} Top-1: {r[\"best_top1\"]:.2f}%  Top-5: {r[\"best_top5\"]:.2f}%{alpha_str}')
"

# 2. Git에 결과 커밋 & push
git add results-json/lifeai/
git commit -m "Add ImageNet experiment results (Exp 9)"
git push

# 3. ahnbi에서 pull
# (ahnbi 서버에서)
# git pull
```

---

## 트러블슈팅

| 문제 | 해결 |
|------|------|
| CUDA OOM | batch-size를 256으로 줄이고 lr을 0.1로 변경 |
| 데이터 로딩 느림 | `--workers 12`로 늘리기 |
| 학습 중단됨 | 재실행하면 완료된 모델은 SKIP, 이어서 진행 |
| NaN loss | `--no-amp` 옵션 추가 (AMP 비활성화) |
