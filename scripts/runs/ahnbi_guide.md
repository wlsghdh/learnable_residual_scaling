# Ahnbi 실험 가이드

## 논문 배경

**논문**: "ResNet은 스스로 깊이를 선택한다" — Learnable Residual Scaling (LRS)
**수식**: `y = α·F(x) + (1−α)·x`, `α = sigmoid(θ)`, per-block

각 모델은 α 초기값이 다르며, 학습 과정에서 α가 어디로 수렴하는지 관찰하는 것이 핵심.

| 모델      | θ 초기값 | α 초기값 | 의미                    |
|-----------|----------|----------|-------------------------|
| baseline  | -        | -        | 일반 ResNet (α 없음)    |
| lrs_low   | -2.0     | ~0.12    | residual 약하게 시작    |
| lrs_mid   | 0.0      | 0.50     | 중간에서 시작           |
| lrs_high  | 2.0      | ~0.88    | residual 강하게 시작    |
| rezero    | -        | 0.0      | α=0에서 시작 (ReZero)   |

## 실험 목적

**Exp 1 (α 수렴)**: 서로 다른 초기값에서 시작해도 α가 같은 값으로 수렴하는지 확인
- depth 50/101/152/200 × cifar10/cifar100 × seed 42/123/456
- 3-seed 평균으로 robustness 검증

---

## Ahnbi1 스크립트 내용 (seed=123)

### day1.sh — ResNet-101, CIFAR-10
```
GPU 0: lrs_high (θ=2.0)  ─┐
                           ├─ 병렬
GPU 1: rezero   (α=0)    ─┘
```
→ 결과: `lrs_high_depth101_cifar10_seed123_result.json`
→ 결과: `rezero_depth101_cifar10_seed123_result.json`

### day2.sh — ResNet-101, CIFAR-100
```
GPU 0: lrs_high ─┐
                  ├─ 병렬
GPU 1: rezero   ─┘
```
→ 결과: `lrs_high_depth101_cifar100_seed123_result.json`
→ 결과: `rezero_depth101_cifar100_seed123_result.json`

### day3.sh — ResNet-152, CIFAR-10
```
GPU 0: lrs_mid  ─┐
                  ├─ 병렬
GPU 1: lrs_high ─┘
```
→ 결과: `lrs_mid_depth152_cifar10_seed123_result.json`
→ 결과: `lrs_high_depth152_cifar10_seed123_result.json`

### day4.sh — ResNet-152, CIFAR-100
```
GPU 0: lrs_mid  ─┐
                  ├─ 병렬
GPU 1: lrs_high ─┘
```
→ 결과: `lrs_mid_depth152_cifar100_seed123_result.json`
→ 결과: `lrs_high_depth152_cifar100_seed123_result.json`

### day5.sh — ResNet-200, CIFAR-10 (1/2)
```
GPU 0: baseline ─┐
                  ├─ 병렬
GPU 1: lrs_low  ─┘
```
→ 결과: `baseline_depth200_cifar10_seed123_result.json`
→ 결과: `lrs_low_depth200_cifar10_seed123_result.json`

### day6.sh — ResNet-200, CIFAR-10 (2/2)
```
GPU 0: lrs_mid  ─┐
                  ├─ 병렬 (1차)
GPU 1: lrs_high ─┘
GPU 0: rezero   ─── 순차 (2차)
```
→ 결과: `lrs_mid_depth200_cifar10_seed123_result.json`
→ 결과: `lrs_high_depth200_cifar10_seed123_result.json`
→ 결과: `rezero_depth200_cifar10_seed123_result.json`

### day7.sh — ResNet-200, CIFAR-100 (1/2)
```
GPU 0: baseline ─┐
                  ├─ 병렬
GPU 1: lrs_low  ─┘
```
→ 결과: `baseline_depth200_cifar100_seed123_result.json`
→ 결과: `lrs_low_depth200_cifar100_seed123_result.json`

### day8.sh — ResNet-200, CIFAR-100 (2/2)
```
GPU 0: lrs_mid  ─┐
                  ├─ 병렬 (1차)
GPU 1: lrs_high ─┘
GPU 0: rezero   ─── 순차 (2차)
```
→ 결과: `lrs_mid_depth200_cifar100_seed123_result.json`
→ 결과: `lrs_high_depth200_cifar100_seed123_result.json`
→ 결과: `rezero_depth200_cifar100_seed123_result.json`

**총 18개 실험** (2×2 + 2×2 + 2×2 + 3×2 = 18)

---

## Ahnbi2 스크립트 내용 (seed=456)

### day1.sh — ResNet-101, CIFAR-10
```
GPU 0: lrs_high ─┐
                  ├─ 병렬
GPU 1: rezero   ─┘
```

### day2.sh — ResNet-101, CIFAR-100
```
GPU 0: lrs_mid  ─┐
                  ├─ 병렬 (1차)
GPU 1: lrs_high ─┘
GPU 0: rezero   ─── 순차 (2차)
```

### day3.sh — ResNet-152, CIFAR-10
```
GPU 0: lrs_mid  ─┐
                  ├─ 병렬
GPU 1: lrs_high ─┘
```

### day4.sh — ResNet-152, CIFAR-100
```
GPU 0: lrs_mid  ─┐
                  ├─ 병렬
GPU 1: lrs_high ─┘
```

### day5.sh — ResNet-200, CIFAR-10 (1/2)
```
GPU 0: baseline ─┐
                  ├─ 병렬
GPU 1: lrs_low  ─┘
```

### day6.sh — ResNet-200, CIFAR-10 (2/2)
```
GPU 0: lrs_mid  ─┐
                  ├─ 병렬 (1차)
GPU 1: lrs_high ─┘
GPU 0: rezero   ─── 순차 (2차)
```

### day7.sh — ResNet-200, CIFAR-100 (1/2)
```
GPU 0: baseline ─┐
                  ├─ 병렬
GPU 1: lrs_low  ─┘
```

### day8.sh — ResNet-200, CIFAR-100 (2/2)
```
GPU 0: lrs_mid  ─┐
                  ├─ 병렬 (1차)
GPU 1: lrs_high ─┘
GPU 0: rezero   ─── 순차 (2차)
```

**총 19개 실험** (2 + 3 + 2 + 2 + 2 + 3 + 2 + 3 = 19)

---

## 실행 방법

### 1. 사전 준비 (각 ahnbi 서버에서)
```bash
cd ~/Learnable_Reidual_Scaling
git pull origin main
chmod +x scripts/runs/ahnbi1/*.sh   # ahnbi1 서버
chmod +x scripts/runs/ahnbi2/*.sh   # ahnbi2 서버
```

### 2. Background 실행 (nohup)
```bash
# ahnbi1 서버
nohup bash scripts/runs/ahnbi1/day1.sh > logs/ahnbi1/nohup_day1.log 2>&1 &

# ahnbi2 서버
nohup bash scripts/runs/ahnbi2/day1.sh > logs/ahnbi2/nohup_day1.log 2>&1 &
```

### 3. Background 실행 (tmux)
```bash
tmux new -s train
bash scripts/runs/ahnbi1/day1.sh
# Ctrl+B, D 로 분리
# tmux attach -t train 으로 복귀
```

### 4. 진행 확인
```bash
# 전체 진행 로그
tail -f logs/ahnbi1/day1.log

# 개별 모델 로그
tail -f logs/ahnbi1/lrs_high_d101_cifar10_s123.log

# GPU 확인
nvidia-smi

# 완료 파일 수 확인
ls results-json/*seed123* 2>/dev/null | wc -l
ls results-json/*seed456* 2>/dev/null | wc -l
```

## 주의사항
- 인자 없이 실행 (GPU 0, 1 자동 사용)
- 이미 완료된 실험은 자동 SKIP (결과 JSON 존재 여부 확인)
- 3개 모델인 날(day6, day8): 2개 병렬 → 완료 후 1개 순차 실행
- 각 서버는 자기 seed 결과만 커밋 (Git 충돌 방지)
