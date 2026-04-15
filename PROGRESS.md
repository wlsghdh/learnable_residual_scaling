# LRS 실험 진행 상황 — "ResNet은 스스로 깊이를 선택한다"
> 마지막 업데이트: 2026-04-15
> 논문: Learnable Residual Scaling을 통한 유효 깊이(Effective Depth) 자동 조절 실증 분석
> 제출 목표: **BMVC 2026** (Abstract 5/22, 논문 5/29)
> 서버 분리: **lifeai** = ImageNet, **ahnbi1/2** = CIFAR (3090×1ea), **ahnbi3** = V100×4

---

## 논문 핵심 주장

> ResNet이 깊어져도 성능이 유지되는 이유는, 네트워크가 불필요한 블록을 자동으로 건너뛰어
> 유효 깊이(effective depth)를 스스로 조절하기 때문이다.
> LRS의 per-block α가 이를 명시적으로 관찰 가능하게 한다.

---

## ✅ 완료된 실험 (2026-04-15 기준)

### Exp 1: α 수렴 실험 (Core) — ✅ **120/120 완료**
> 5 models × 4 depths × 2 datasets × 3 seeds = 120 runs
> 논문 Table 4, 5, Figure 2, 3 근거

| Depth | CIFAR-10 | CIFAR-100 |
|-------|----------|-----------|
| 50  | ✅ 5×3 | ✅ 5×3 |
| 101 | ✅ 5×3 | ✅ 5×3 |
| 152 | ✅ 5×3 | ✅ 5×3 |
| 200 | ✅ 5×3 | ✅ 5×3 |

### Exp 2: Per-block α 분포 — ✅ 완료 (Figure 4, 8)
- alpha_final.all 데이터 모두 확보
- 시각화 PDF 모두 생성됨 (figures/fig4_perblock_alpha.pdf, fig8_heatmap.pdf)

### Exp 3: 유효 깊이 정량화 — ✅ 완료 (Figure 5, Table 6, 7)
- D_eff = Σα_i 메트릭 산출
- Threshold sensitivity 분석 (Table 6)
- Threshold-free D_eff (Table 7)

### Exp 4: α 학습 궤적 — ✅ 완료 (Figure 3)
- alpha_stats per epoch 모두 기록됨

### Exp 5: Fixed-α Ablation — ✅ **12/12 완료**
- fixed_alpha_01/03/05/07 × d152 × cifar100 × 3 seeds
- 논문 Table 9, Figure 7 근거

### Exp 6: LRS_High 실패 분석 — ✅ 완료 (Figure 6)
- Exp 1의 lrs_high 데이터로 분석
- d200 catastrophic failure (37.5% on CIFAR-100) 입증

### Exp 7: HybridA / Identity Init — ✅ d152/d200 cifar100 완료 (Table 8)
- HybridA: 9/12 (cifar100 완비, cifar10 일부 missing — 논문 영향 없음)
- LRS+HybridA Low: 10/12 (동일)
- ResNet Identity All: 7/12 (d152/d200 cifar100 완비)
- 논문 핵심 비교는 모두 가능

### Exp 8: 비교 방법론 — ✅ **36/36 완료**
- skipinit/fixup/layerscale × d50/d152 × 2ds × 3seeds
- 논문 Table 10, Figure 9 근거

### Exp 9: ImageNet 검증 — ✅ **6/6 완료** (ResNet-50)
| Model | Top-1 | Top-5 |
|-------|-------|-------|
| baseline    | 75.86% | 92.92% |
| lrs_high    | 75.73% | 92.74% |
| lrs_mid     | 75.62% | 92.73% |
| rezero      | 75.36% | 92.44% |
| lrs_low     | 73.75% | 91.86% |
| lrs+ha_low  | 72.32% | 91.02% |

### Exp 10: Pruning Validation (α semantic validity) — ✅ 완료
> 논문 Table 11, Figure 11 근거. **α가 진짜 블록 중요도를 반영함을 검증.**

| Threshold | Removed | No-FT | +FT 10ep |
|-----------|---------|-------|----------|
| α<0.03 | 2/66 | 80.54% | — |
| α<0.05 | 10/66 | 73.88% | 79.53% (−1.01) |
| α<0.08 | 37/66 | 3.97% | 76.22% (−4.32) |
| α<0.10 | 48/66 | 3.13% | 74.83% (−5.71) |
| α<0.15 | 58/66 | 3.63% | 72.64% (−7.90) |

### Exp 11: Gradient Flow 분석 — ✅ 완료 (Figure 0)
- gradient_norms_d152.json 산출
- Plain vs ResNet vs LRS 비교 PDF 생성됨

---

## 🎯 BMVC accept을 위한 추가 실험 (2주 안 완료)

### 🔥 Priority 1 — 강력 권장

#### A. **ImageNet ResNet-101 + LRS Low** (1 run)
**목적**: depth-α scaling law가 ImageNet 대규모에서도 성립하는지 검증
- CIFAR d101에서 mean α ≈ 0.18이었음
- ImageNet d101도 유사한 α 값을 보이면 → **dataset-independence 강력 입증**
- 현재 ImageNet은 ResNet-50(16 blocks)만 → reviewer "이게 ImageNet에서도?" 선제 대응
- **서버**: lifeai A100, **소요**: ~36h (1.5일)
- **실행**: `bash scripts/run/batch8_imagenet_d101.sh 0` (이미 구현됨)

#### B. **FLOPs / Latency 측정 추가** (1-2 runs)
**목적**: pruning이 단순 정확도 회복이 아니라 **실질적 inference cost 절감**
- 기존 pruning_finetune은 정확도만 보고 → reviewer "FLOPs는?"
- α<0.05 (15% pruning) 시 FLOPs / GPU latency 정량 측정
- **체크포인트 필요**: 현재 0개 → `lrs_low d200 cifar100 seed42` 1회 재학습 필요
- **서버**: ahnbi3 V100, **소요**: 학습 6.3h + 분석 3h ≈ 10h
- **실행**:
  1. `SAVE_CHECKPOINT=1 python run_experiments.py --depth 200 --dataset cifar100 --model lrs_low --seed 42`
  2. `bash scripts/run/batch10_pruning_analysis.sh 0` (이미 구현됨)

### ⚠️ Priority 2 — 선택적

#### C. **Highway Network 정량 비교** (12 runs)
- 구현 완료: `scripts/run/batch9_highway.sh`
- 논문이 "관찰 도구"에 집중하므로 Related Work에서 정성적 차별화로 충분할 수 있음
- 시간 부족하면 skip

---

## ❌ 논문 범위 밖 (제거됨)

| 항목 | 제거 이유 |
|------|----------|
| WRN/ResNeXt 일반화 | 논문 핵심은 depth 변화 (아키텍처 일반화는 별개) |
| ViT/Transformer | 관찰 대상은 ResNet의 effective depth |
| Per-channel LRS | 핵심 스토리 불필요 |
| plain_he/identity 추가 | Pilot study용으로 충분 |
| 200 Epoch / StepLR Ablation | 깊이 조절 스토리와 무관 |

---

## 🗓️ BMVC 5/29 마감까지 일정

| 기간 | 작업 |
|------|------|
| 4/15 - 4/17 | lrs_low d200 checkpoint 재학습 (V100) |
| 4/16 - 4/19 | ImageNet ResNet-101 학습 (lifeai A100) |
| 4/18 - 4/19 | FLOPs/Latency 분석 |
| 4/20 - 4/30 | 논문 LaTeX BMVC 템플릿 변환 + figure 업데이트 |
| 5/1 - 5/15 | 논문 재작성, Discussion 강화, Related Work 정리 |
| 5/16 - 5/21 | 최종 검토, abstract 작성 |
| **5/22** | **Abstract 제출 마감** |
| 5/23 - 5/28 | 논문 최종 다듬기 |
| **5/29** | **Full paper 제출 마감** |

---

## 📦 자산 현황

| 항목 | 개수 |
|------|------|
| Result JSON 파일 | 314+ files |
| Figure PDF/PNG | 51 files |
| ImageNet 결과 | 6/6 (ResNet-50) |
| Pruning 분석 | 4 files |
| **Checkpoint** | **0** ⚠️ (FLOPs/Latency용 1개 재학습 필요) |
| Training logs | 0 (PID 파일만 잔존) |
