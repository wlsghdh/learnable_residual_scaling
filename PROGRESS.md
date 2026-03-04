# LRS 실험 진행 상황
> 마지막 업데이트: 2026-02-26 (Phase 2-A 완료, multi-seed 부분 완료)

## 전체 진행률
- [x] Phase 1: 기존 코드/결과 파악
- [x] Phase 2-A: Pilot Study (연구 동기 실험) — plain_he/identity/resnet_identity_all 완료
- [ ] Phase 2-B: 비교 대상 구현 (SkipInit, Fixup, LayerScale, Fixed-α, Plain)
- [~] Phase 2-C: Tier 1 실험 (핵심 비교, 84 runs, 3-seed) — baseline/lrs_low/lrs_ha_low 3-seed 완료 (3/7 모델)
- [ ] Phase 2-D: Tier 2 실험 (LRS 변형, 60 runs, 3-seed)
- [ ] Phase 2-E: Tier 3 실험 (Fixed-α ablation, 12 runs, 3-seed)
- [ ] Phase 2-F: Tier 4 실험 (depth 101/200, 36 runs, 3-seed)
- [ ] Phase 2-G: 아키텍처 일반화 (WRN + ResNeXt, 72 runs)
- [ ] Phase 2-H: Ablation (Per-channel, Epoch, LR schedule, 27 runs)
- [ ] Phase 3: 시각화 (Fig 1~12)
- [ ] Phase 4: ImageNet 실험 (✅ 데이터 확보 완료 — 즉시 실행 가능)

---

## 실험 배치 실행 현황

| Batch | 스크립트 | GPU | 시작 | 종료 | 상태 | 비고 |
|-------|---------|-----|------|------|------|------|
| 1 | batch1_pilot.sh | 1 | | | ⬜ | Phase 2-A, ~12h |
| 2a | batch2a_tier1_part1.sh | 1 | | | ⬜ | Tier1 전반, ~20h |
| 2b | batch2b_tier1_part2.sh | 2 | | | ⬜ | Tier1 후반, ~20h |
| 3a | batch3a_tier2_part1.sh | 2 | | | ⬜ | Tier2 전반, ~15h |
| 3b | batch3b_tier2_part2.sh | 2 | | | ⬜ | Tier2 후반, ~15h |
| 4 | batch4_tier3_ablation.sh | 1 | | | ⬜ | Tier3+Ablation, ~32h |
| 5 | batch5_tier4.sh | 2 | | | ⬜ | Tier4, ~25h |
| 6a | batch6_wrn.sh | 1 | | | ⬜ | WRN-28-10, ~15h |
| 6b | batch6_resnext.sh | 2 | | | ⬜ | ResNeXt-50, ~15h |
| 7a | batch7_imagenet_gpu1.sh | 1 | | | ⬜ | ImageNet GPU1, ~36h |
| 7b | batch7_imagenet_gpu2.sh | 2 | | | ⬜ | ImageNet GPU2, ~24h |

---

## Phase 1: 기존 코드/결과 파악
- 상태: ✅ 완료

### 기존 실험 요약 (Best Accuracy, seed=42)

#### CIFAR-10
| Model | Depth 50 | Depth 101 | Depth 152 | Depth 200 |
|-------|----------|-----------|-----------|-----------|
| Baseline | 94.80% | 95.34% | 95.73% | 95.95% |
| LRS_Low | 94.68% | 95.31% | 96.04% | 96.02% |
| LRS_Mid | 94.73% | 95.21% | 95.59% | 94.79% |
| LRS_High | 94.18% | 94.54% | 92.89% | 16.85% ⚠️ |
| ReZero | 94.64% | 95.19% | 95.82% | 95.97% |
| HybridA | 94.62% | 95.50% | 95.76% | 95.95% |
| LRS+HA_Low | 94.90% | 95.51% | 96.19% | 96.01% |
| LRS+HA_Mid | 94.98% | 95.47% | 95.92% | 96.06% |
| LRS+HA_High | 94.28% | 94.81% | 94.29% | 93.09% |

#### CIFAR-100
| Model | Depth 50 | Depth 101 | Depth 152 | Depth 200 |
|-------|----------|-----------|-----------|-----------|
| Baseline | 76.86% | 79.55% | 80.08% | 80.71% |
| LRS_Low | 77.18% | 79.00% | 80.34% | 80.66% |
| LRS_Mid | 76.94% | 78.12% | 78.07% | 77.29% |
| LRS_High | 75.22% | 75.06% | 69.98% | 5.69% ⚠️ |
| ReZero | 76.54% | 78.25% | 80.00% | 80.32% |
| HybridA | 76.24% | 78.75% | 80.10% | 80.62% |
| LRS+HA_Low | 76.89% | 78.64% | 80.76% | 80.92% |
| LRS+HA_Mid | 76.68% | 78.73% | 80.05% | 79.47% |
| LRS+HA_High | 76.07% | 74.45% | 76.73% | 69.40% |

---

## Phase 2-A: Pilot Study (연구 동기 실험)
- 상태: ✅ 완료 (seed=42, depth 50 & 152)
- 총 runs: 12 (6모델 × 2데이터셋 × 2깊이 × 1seed) — 단순화 버전 완료

| # | 모델 | Skip | Init | 구현 여부 |
|---|------|------|------|---------|
| A | plain_he | ❌ | He | ✅ |
| B | plain_identity | ❌ | Identity | ✅ |
| C | baseline | ✅ | He | ✅ |
| D | resnet_identity_all | ✅ | Identity (전층) | ✅ |
| E | hybrida | ✅ | He+Identity | ✅ |
| F | lrs_ha_low | ✅ | He+Identity+LRS | ✅ |

### 결과 (seed=42)
| 모델 | CIFAR-10 D50 | CIFAR-10 D152 | CIFAR-100 D50 | CIFAR-100 D152 |
|------|------------|--------------|--------------|----------------|
| plain_he | 92.94% | 74.47% ⚠️ | 69.17% | 1.47% ⚠️ |
| plain_identity | 49.74% ⚠️ | 15.23% ⚠️ | 10.94% ⚠️ | 3.49% ⚠️ |
| baseline | 94.68% | 96.02% | 77.02% | 80.22% |
| resnet_identity_all | 53.96% ⚠️ | 59.84% ⚠️ | 12.12% ⚠️ | 12.92% ⚠️ |
| hybrida | 94.65% | 95.65% | 76.54% | 80.30% |
| lrs_ha_low | 95.00% | 95.77% | 76.43% | 81.11% |

**핵심 관찰**: skip connection 없는 plain 네트워크는 depth 증가 시 catastrophic failure. Identity init도 도움이 안 됨. HybridA + LRS가 안정적.

---

## Phase 2-B: 비교 대상 구현
- 상태: ⬜ 대기

| 모델 | 수식 | 구현 여부 |
|------|------|---------|
| SkipInit | y = x + α·F(x), α₀=0 (BN 유지) | ⬜ |
| Fixup | y = x + F(x), last BN=0 init | ⬜ |
| LayerScale | y = x + diag(λ)·F(x), λ₀=1e-4 | ⬜ |
| fixed_alpha_01 | y = 0.1·F(x)+0.9·x (고정) | ⬜ |
| fixed_alpha_03 | y = 0.3·F(x)+0.7·x (고정) | ⬜ |
| fixed_alpha_05 | y = 0.5·F(x)+0.5·x (고정) | ⬜ |
| fixed_alpha_07 | y = 0.7·F(x)+0.3·x (고정) | ⬜ |
| per_channel_lrs | y = α_c·F(x)+(1-α_c)·x, per-ch | ⬜ |

---

## Phase 2-C: Tier 1 실험 (핵심 비교, 3-seed)
- 상태: 🔄 진행 중 (3/7 모델 완료)
- Models: baseline, lrs_low, lrs_ha_low, rezero, skipinit, fixup, layerscale
- Depths: 50, 152 / Datasets: CIFAR-10, CIFAR-100 / Seeds: 42, 123, 456
- 완료: baseline ✅, lrs_low ✅, lrs_ha_low ✅
- 미완: rezero (seed42만), skipinit ❌, fixup ❌, layerscale ❌

| 모델 | CIFAR-10 D50 | CIFAR-10 D152 | CIFAR-100 D50 | CIFAR-100 D152 |
|------|------------|--------------|--------------|----------------|
| baseline | 94.53 ± 0.11 | 95.95 ± 0.05 | 77.18 ± 0.29 | 80.42 ± 0.18 |
| lrs_low | 94.83 ± 0.10 | 95.73 ± 0.08 | 77.02 ± 0.16 | 80.28 ± 0.31 |
| lrs_ha_low | 94.83 ± 0.17 | 95.86 ± 0.10 | 76.56 ± 0.18 | 80.72 ± 0.29 |
| rezero | 94.64 (s42만) | 95.82 (s42만) | 76.54 (s42만) | 80.00 (s42만) |
| skipinit | ⬜ | ⬜ | ⬜ | ⬜ |
| fixup | ⬜ | ⬜ | ⬜ | ⬜ |
| layerscale | ⬜ | ⬜ | ⬜ | ⬜ |

---

## Phase 2-D: Tier 2 실험 (LRS 변형, 3-seed)
- 상태: ⬜ 대기
- Models: lrs_mid, lrs_high, lrs_ha_mid, lrs_ha_high, hybrida
- Depths: 50, 152 / Datasets: CIFAR-10, CIFAR-100 / Seeds: 42, 123, 456
- 총 60 runs, 예상 ~30시간

---

## Phase 2-E: Tier 3 실험 (Fixed-α ablation, 3-seed)
- 상태: ⬜ 대기
- Models: fixed_alpha_01, fixed_alpha_03, fixed_alpha_05, fixed_alpha_07
- Depth: 152 / Dataset: CIFAR-100 / Seeds: 42, 123, 456
- 총 12 runs, 예상 ~12시간

---

## Phase 2-F: Tier 4 실험 (추가 Depth, 3-seed)
- 상태: ⬜ 대기
- Models: baseline, lrs_low, lrs_ha_low
- Depths: 101, 200 / Datasets: CIFAR-10, CIFAR-100 / Seeds: 42, 123, 456
- 총 36 runs, 예상 ~25시간

---

## Phase 2-G: 아키텍처 일반화 (3-seed)
- 상태: ⬜ 대기 (WRN, ResNeXt 구현 필요)

### WRN-28-10
- Models: baseline, lrs_low, lrs_ha_low, rezero, skipinit, layerscale
- Datasets: CIFAR-10, CIFAR-100 / Seeds: 42, 123, 456
- 총 36 runs, 예상 ~15시간

### ResNeXt-50 (32×4d)
- 동일 세팅
- 총 36 runs, 예상 ~15시간

---

## Phase 2-H: Ablation Studies (3-seed)
- 상태: ⬜ 대기

| 실험 | 모델 | 세팅 | Runs |
|------|------|------|------|
| Per-channel LRS | per_channel_lrs vs lrs_low vs baseline | D50,152/CIFAR-100 | 18 |
| 200 Epoch | baseline, lrs_low, lrs_ha_low | D152/CIFAR-100 | 9 |
| StepLR | baseline, lrs_low | D152/CIFAR-100 | 6 |

---

## Phase 3: 시각화 (Fig 1~12)
| # | Figure | 파일명 | 상태 | 데이터 |
|---|--------|--------|------|--------|
| 1 | LRS Block Diagram | fig01_block_diagram | ✅ | 코드 생성 |
| 2 | Pilot Study Results | fig02_pilot_study | ⬜ | Phase 2-A |
| 3 | Depth vs Accuracy | fig03_depth_accuracy | ⬜ | Tier 1 |
| 4 | Full Method Comparison | fig04_full_comparison | ⬜ | Tier 1+2 |
| 5 | Init Sensitivity | fig05_init_sensitivity | ⬜ | Tier 2 |
| 6 | Depth-α Relationship | fig06_depth_alpha | ⬜ | Tier 1 |
| 7 | Per-block α Distribution | fig07_perblock_alpha | ⬜ | Tier 1 |
| 8 | α Trajectory | fig08_alpha_trajectory | ⬜ | Tier 1 |
| 9 | Fixed-α Ablation | fig09_fixed_alpha | ⬜ | Tier 3 |
| 10 | Cross-architecture | fig10_cross_arch | ⬜ | Phase 2-G |
| 11 | LRS High Collapse Curves | fig11_high_collapse | ⬜ | Tier 2 |
| 12 | ImageNet Results | fig12_imagenet | ⬜ | Phase 4 |

---

## Phase 4: ImageNet 실험
- 상태: ⬜ 실행 대기 (데이터 확보 완료, 스크립트 준비 완료)
- 소스: `/nfs_share/datasets/ILSVRC2012/` (train: 1000 classes, val: 1000 classes ✅)
- 코드: `train_imagenet.py` — 단일 GPU + AMP, 90 epochs, batch 256
- 실행: `bash scripts/run/batch7_imagenet_gpu1.sh 1` (GPU 1)
-       `bash scripts/run/batch7_imagenet_gpu2.sh 2` (GPU 2, 동시 실행)

| # | Model | GPU | 시작 | 종료 | 상태 | Top-1 | Top-5 | α mean |
|---|-------|-----|------|------|------|-------|-------|--------|
| 1 | baseline | 1 | | | ⬜ | | | N/A |
| 2 | lrs_low | 1 | | | ⬜ | | | |
| 3 | lrs_mid | 1 | | | ⬜ | | | |
| 4 | lrs_ha_low | 2 | | | ⬜ | | | |
| 5 | rezero | 2 | | | ⬜ | | | |

---

## 전체 실험 수 요약

| Phase | Runs | 예상 시간 (1 GPU) |
|-------|------|------------------|
| 2-A Pilot Study | 24 | ~12시간 |
| 2-C Tier 1 | 84 | ~40시간 |
| 2-D Tier 2 | 60 | ~30시간 |
| 2-E Tier 3 | 12 | ~12시간 |
| 2-F Tier 4 | 36 | ~25시간 |
| 2-G WRN+ResNeXt | 72 | ~30시간 |
| 2-H Ablation | 27 | ~20시간 |
| **총** | **315 runs** | **~169시간** |
| **2장 병렬** | | **~85시간 ≈ 3.5일** |

---

## 이슈 & 메모
- alpha_stats에는 mean/std만 저장 (per-block trajectory 없음, last epoch 값만)
- LRS_High depth 200: catastrophic failure (CIFAR-10: 16.85%, CIFAR-100: 5.69%)
- 기존 실험은 모두 seed=42 단일 시드
- run_experiments.py: 결과 저장 형식에 train/test history 추가 필요
- WRN, ResNeXt는 아직 models.py에 없음
