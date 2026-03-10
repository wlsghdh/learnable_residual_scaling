# LRS 실험 진행 상황 — "ResNet은 스스로 깊이를 선택한다"
> 마지막 업데이트: 2026-03-10
> 논문: Learnable Residual Scaling을 통한 유효 깊이(Effective Depth) 자동 조절 실증 분석
> 서버 분리: **lifeai** = ImageNet, **ahnbi** = CIFAR

---

## 논문 핵심 주장

> ResNet이 깊어져도 성능이 유지되는 이유는, 네트워크가 불필요한 블록을 자동으로 건너뛰어
> 유효 깊이(effective depth)를 스스로 조절하기 때문이다.
> LRS의 per-block α가 이를 명시적으로 관찰 가능하게 한다.

---

## 서버별 역할

| 서버 | GPU | 역할 | 스크립트 |
|------|-----|------|---------|
| **lifeai** | A100 80GB × 6 | ImageNet (Exp 9) | `scripts/run/lifeai/` |
| **ahnbi** | RTX 3090 24GB | CIFAR 잔여 (Exp 1) | `scripts/run/ahnbi/` |

---

## 전체 실험 현황

### Exp 1: α 수렴 실험 (Core) — CIFAR
> 목적: 초기값(Low/Mid/High)과 무관하게 α가 같은 값으로 수렴
> 설계: 5 models × 4 depths × 2 datasets × 3 seeds = **120 runs**

#### Depth 50 — ✅ 완료 (30/30)
| Model | CIFAR-10 (s42/123/456) | CIFAR-100 (s42/123/456) |
|-------|------------------------|-------------------------|
| baseline | ✅ ✅ ✅ | ✅ ✅ ✅ |
| lrs_low | ✅ ✅ ✅ | ✅ ✅ ✅ |
| lrs_mid | ✅ ✅ ✅ | ✅ ✅ ✅ |
| lrs_high | ✅ ✅ ✅ | ✅ ✅ ✅ |
| rezero | ✅ ✅ ✅ | ✅ ✅ ✅ |

#### Depth 101 — 🔄 21/30 (9 남음) → ahnbi
| Model | CIFAR-10 (s42/123/456) | CIFAR-100 (s42/123/456) |
|-------|------------------------|-------------------------|
| baseline | ✅ ✅ ✅ | ✅ ✅ ✅ |
| lrs_low | ✅ ✅ ✅ | ✅ ✅ ✅ |
| lrs_mid | ✅ ✅ ✅ | ✅ ✅ ❌ |
| lrs_high | ✅ ❌ ❌ | ✅ ❌ ❌ |
| rezero | ✅ ❌ ❌ | ✅ ❌ ❌ |

#### Depth 152 — 🔄 22/30 (8 남음) → ahnbi
| Model | CIFAR-10 (s42/123/456) | CIFAR-100 (s42/123/456) |
|-------|------------------------|-------------------------|
| baseline | ✅ ✅ ✅ | ✅ ✅ ✅ |
| lrs_low | ✅ ✅ ✅ | ✅ ✅ ✅ |
| lrs_mid | ✅ ❌ ❌ | ✅ ❌ ❌ |
| lrs_high | ✅ ❌ ❌ | ✅ ❌ ❌ |
| rezero | ✅ ✅ ✅ | ✅ ✅ ✅ |

#### Depth 200 — 🔲 10/30 (20 남음) → ahnbi
| Model | CIFAR-10 (s42/123/456) | CIFAR-100 (s42/123/456) |
|-------|------------------------|-------------------------|
| baseline | ✅ ❌ ❌ | ✅ ❌ ❌ |
| lrs_low | ✅ ❌ ❌ | ✅ ❌ ❌ |
| lrs_mid | ✅ ❌ ❌ | ✅ ❌ ❌ |
| lrs_high | ✅ ❌ ❌ | ✅ ❌ ❌ |
| rezero | ✅ ❌ ❌ | ✅ ❌ ❌ |

**Exp 1 합계: 83/120 완료, 37 남음 (ahnbi에서 실행)**

---

### Exp 2: Per-block α 분포 (Core) — ✅ 분석만 필요
- 데이터: alpha_final.all (모든 모델 seed42) 있음
- 필요: 시각화 코드 (Fig 5)

### Exp 3: 유효 깊이 정량화 (Core) — 🔲 분석 코드 작성 필요
- D_eff = Σ α_i 또는 #{α_i > τ}
- 기존 데이터로 즉시 분석 가능 (Fig 6)

### Exp 4: α 학습 궤적 (Core) — ✅ 분석만 필요
- 데이터: alpha_stats per epoch 있음 (Fig 4)

### Exp 5: Fixed-α Ablation — ✅ 완료 (12/12)
- fixed_alpha_01/03/05/07 × depth152 × CIFAR-100 × seed 42/123/456
- Fig 7에서 사용

### Exp 6: LRS_High 실패 분석 — ✅ 데이터 있음
- Exp 1의 lrs_high 데이터 활용 (Fig 8)

### Exp 8: 비교 방법론 — ✅ 완료 (48/48)
- ReZero/SkipInit/Fixup/LayerScale × depth 50/152 × CIFAR-10/CIFAR-100 × 3 seeds
- Fig 9에서 사용

### Exp 9: ImageNet 검증 — 🔲 미시작 (4 runs) → lifeai
| Model | Status |
|-------|--------|
| baseline | ❌ |
| lrs_low | ❌ |
| lrs_mid | ❌ |
| rezero | ❌ |

---

## 남은 작업 요약

| 항목 | Runs | 서버 | 예상 시간 |
|------|------|------|----------|
| Exp 1 CIFAR (d101: 9, d152: 8, d200: 20) | 37 | ahnbi | ~60h |
| Exp 9 ImageNet (4 models × ResNet-50) | 4 | lifeai | ~48h |
| 분석/시각화 (Exp 2,3,4) | - | 어디든 | 수 시간 |
| **합계** | **41 runs** | | |

---

## 제거된 실험 (논문 방향과 무관)

| 항목 | 제거 이유 |
|------|----------|
| HybridA (Identity 초기화) | 초기화 전략 ≠ 깊이 자동 조절 |
| WRN/ResNeXt 일반화 | depth 변화가 핵심, 아키텍처 일반화는 별개 |
| Per-channel LRS | 핵심 스토리 불필요 |
| plain_he/identity 실험 | Pilot study, 새 논문과 무관 |
| 200 Epoch / StepLR Ablation | 깊이 조절 스토리와 무관 |
