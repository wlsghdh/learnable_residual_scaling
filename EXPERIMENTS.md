# LRS 실험 구성 설명

> 이 문서는 전체 실험이 어떻게 설계되어 있는지, Batch/Tier라는 용어가 왜 붙었는지,
> 각 실험에서 무엇을 검증하는지를 정리한 것입니다.

---

## 1. 연구 주제 요약

**Learnable Residual Scaling (LRS)**

기존 ResNet의 residual connection 수식은:

```
y = F(x) + x          ← Baseline (고정된 1:1 비율)
```

LRS는 이 비율을 블록마다 학습 가능하게 만든다:

```
y = α·F(x) + (1−α)·x    α = sigmoid(θ),  θ는 학습 파라미터
```

- α가 크면 → transform(F) 중심
- α가 작으면 → identity(x) 중심
- 블록마다 α가 다름 (per-block learnable)

**핵심 질문:** "학습 중 네트워크가 각 블록의 skip 비율을 스스로 결정하도록 하면 성능이 나아지는가?"

---

## 2. 공통 학습 설정

| 항목 | 설정 |
|------|------|
| **데이터셋** | CIFAR-10 (10클래스), CIFAR-100 (100클래스) |
| **Epochs** | 100 |
| **Optimizer** | SGD (momentum=0.9, weight_decay=1e-4) |
| **LR Schedule** | Cosine Annealing (초기 lr=0.1, warmup 5 epochs) |
| **Seeds** | 42, 123, 456 (3-seed → 평균±표준편차로 통계적 유의성 확보) |

### Depth별 배치 크기 (GPU 메모리 이유)

| Depth | Batch Size | 블록 수 | 파라미터 |
|-------|-----------|--------|---------|
| 50 | 128 | 16 | ~23M |
| 101 | 64 | 33 | ~42M |
| 152 | 32 | 50 | ~58M |
| 200 | 32 | 66 | ~64M |

### ResNet Layer 구성

| Depth | Layer 구성 [stage1,2,3,4] |
|-------|--------------------------|
| 50 | [3, 4, 6, 3] |
| 101 | [3, 4, 23, 3] |
| 152 | [3, 8, 36, 3] |
| 200 | [3, 24, 36, 3] |

---

## 3. 모델 목록

### 3-1. Pilot Study 모델 (연구 동기 확인용)

| 모델 | 수식 | Skip | 초기화 | 목적 |
|------|------|------|--------|------|
| `plain_he` | y = F(x) | ❌ | He | skip 없는 plain network |
| `plain_identity` | y = F(x) | ❌ | Identity | skip 없이 identity init |
| `resnet_identity_all` | y = F(x)+x | ✅ | Identity (전 층) | skip 있지만 잘못된 init |
| `baseline` | y = F(x)+x | ✅ | He | 표준 ResNet |
| `hybrida` | y = F(x)+x | ✅ | He+Identity* | HybridA init |
| `lrs_hybrida_low` | y=α·F(x)+(1−α)·x | ✅ | He+Identity* | LRS 제안 모델 |

> *HybridA: stage 3,4(깊은 층)의 conv는 Identity init, 나머지는 He init

**검증 목표:** skip connection이 없거나 초기화가 잘못되면 깊은 네트워크는 학습이 안 된다 → LRS의 필요성 동기 부여

---

### 3-2. 핵심 비교 모델 (Tier 1)

LRS와 직접 비교되는 기존 방법들:

| 모델 | 수식 | 핵심 아이디어 | 논문 |
|------|------|-------------|------|
| `baseline` | y = F(x)+x | 표준 ResNet | He et al., 2016 |
| `lrs_low` | y=α·F(x)+(1−α)·x, θ₀=−2.0 | **LRS** (α₀≈0.12, identity 우세 초기화) | 본 연구 |
| `lrs_hybrida_low` | 위 + HybridA init | **LRS+HybridA** (제안 모델) | 본 연구 |
| `rezero` | y = x + α·F(x), α₀=0 | α를 0에서 시작, 점점 키움 | Bachlechner et al., 2020 |
| `skipinit` | y = x + α·F(x), α₀=0 (BN 유지) | ReZero + BN 공정 비교 | De & Smith, 2020 |
| `fixup` | y = x + s·F(x), last BN weight=0 | 마지막 BN 0 초기화로 F(x)≈0 시작 | Zhang et al., 2019 |
| `layerscale` | y = x + diag(λ)·F(x), λ₀=1e-4 | per-channel 스케일링 | Touvron et al., 2021 |

**LRS vs 기존 방법의 차이:**

| | ReZero | SkipInit | Fixup | LayerScale | **LRS** |
|--|--------|----------|-------|-----------|---------|
| α 형태 | scalar | scalar | scalar | per-channel | scalar |
| 수식 | x+α·F | x+α·F | x+s·F | x+diag(λ)·F | **α·F+(1−α)·x** |
| α 범위 | (0,∞) | (0,∞) | (0,∞) | (0,∞) | **(0,1) [sigmoid]** |
| 초기 bias | identity 없음 | identity 없음 | identity 없음 | identity 없음 | **identity 우세** |

---

### 3-3. LRS 변형 (Tier 2, ablation)

LRS의 초기값(θ₀)이 결과에 얼마나 영향을 미치는가?

| 모델 | θ₀ | α₀ = sigmoid(θ₀) | 초기 성격 |
|------|-----|------------------|---------|
| `lrs_low` | −2.0 | ≈ 0.12 | identity 강하게 선호 |
| `lrs_mid` | 0.0 | = 0.50 | 중립 (F와 x 동등) |
| `lrs_high` | +2.0 | ≈ 0.88 | transform 강하게 선호 |
| `lrs_hybrida_low` | −2.0 + HybridA | ≈ 0.12 | **LRS + 좋은 init** |
| `lrs_hybrida_mid` | 0.0 + HybridA | = 0.50 | |
| `lrs_hybrida_high` | +2.0 + HybridA | ≈ 0.88 | |
| `hybrida` | (α 없음) | N/A | HybridA init만 적용 |

---

### 3-4. Fixed-α ablation (Tier 3)

"α를 학습하는 것이 진짜 중요한가? 그냥 고정해도 되지 않나?"를 검증:

| 모델 | α 값 | 학습 여부 |
|------|------|---------|
| `fixed_alpha_01` | 0.1 | ❌ 고정 |
| `fixed_alpha_03` | 0.3 | ❌ 고정 |
| `fixed_alpha_05` | 0.5 | ❌ 고정 |
| `fixed_alpha_07` | 0.7 | ❌ 고정 |
| `lrs_low` | 시작 ≈0.12 | ✅ 학습 |

**검증 목표:** 고정 α보다 학습되는 α가 더 좋으면 → "학습 가능성"이 핵심임을 증명

---

### 3-5. Ablation: Per-channel LRS

`lrs_low`는 블록당 scalar 1개. 이걸 채널마다 따로 가지면?

| 모델 | α 형태 | 파라미터 수 |
|------|--------|-----------|
| `lrs_low` | per-block scalar | 블록 수 (16~66개) |
| `per_channel_lrs` | per-block, per-channel | 블록 수 × 채널 수 |

**검증 목표:** 더 세밀한 제어가 성능을 높이는가, 아니면 scalar로 충분한가?

---

### 3-6. 아키텍처 일반화 (batch6)

LRS가 ResNet 이외의 아키텍처에도 효과적인가?

| 아키텍처 | 모델명 prefix | Depth | 구조 특징 |
|---------|-------------|-------|---------|
| **WRN-28-10** | `wrn_` | 28 (고정) | Wide ResNet, 채널 수 10× 확대 |
| **ResNeXt-50** | `resnext_` | 50 (고정) | Grouped convolution (32 groups × 4d) |

각 아키텍처마다 6개 변형: `baseline`, `lrs_low`, `lrs_hybrida_low`, `rezero`, `skipinit`, `layerscale`

---

### 3-7. ImageNet (batch7)

CIFAR에서 검증된 결과가 대규모 데이터셋에서도 유지되는가?

| 항목 | 설정 |
|------|------|
| 데이터셋 | ImageNet-1K (1000 classes, train 1.28M / val 50K) |
| 아키텍처 | ResNet-50 기반 |
| Epochs | 90 |
| Batch size | 256 |
| LR | 0.1, cosine decay, warmup 5 epochs |
| AMP | Mixed Precision (fp16) 사용 |
| 모델 | baseline, lrs_low, lrs_mid (GPU1) / lrs_ha_low, rezero (GPU2) |

---

## 4. "Batch"와 "Tier"를 구분한 이유

### Tier (중요도 계층)

실험을 중요도 순서로 계층화한 개념:

| Tier | 설명 | 비중 |
|------|------|------|
| **Tier 1** | 핵심 비교 — LRS vs 기존 방법들 (논문의 Table 1) | 가장 중요 |
| **Tier 2** | LRS 변형 — init 값 sensitivity (α₀가 중요한가?) | 중요 |
| **Tier 3** | Ablation — fixed-α, per-channel (학습 가능성이 핵심인가?) | 보조 |
| **Tier 4** | 추가 Depth — d101, d200으로 스케일링 확인 | 보조 |

> Tier가 높을수록 먼저 실험하고, 결과가 나쁘면 Tier 3~4는 줄일 수도 있음

### Batch (실행 단위)

GPU 2장에서 순차 실행할 때의 묶음 단위. 각 Batch는:
- 1개 GPU에서 연속 실행 (nohup + loop_master로 자동화)
- 47시간 초과 시 자동 중단 후 재시작 (이미 완료된 실험은 skip)
- 결과 파일이 있으면 자동으로 건너뜀 (중복 실행 방지)

```
GPU 1: batch1 → batch2a → batch4 → batch6_wrn → batch7_imagenet_gpu1
GPU 2: batch2b → batch3a → batch3b → batch5 → batch6_resnext → batch7_imagenet_gpu2
```

---

## 5. 전체 실험 목록 한눈에 보기

| Batch | Tier | 모델들 | Depth | Dataset | Seeds | Runs | 목적 |
|-------|------|--------|-------|---------|-------|------|------|
| batch1 | Pilot | plain_he, plain_identity, resnet_identity_all, baseline, hybrida, lrs_ha_low | 50, 152 | C10, C100 | 42 | 24 | 연구 동기 |
| batch2a | Tier1 전반 | baseline, lrs_low, lrs_ha_low | 50, 152 | C10, C100 | 42/123/456 | 36 | 핵심 비교 전반 |
| batch2b | Tier1 후반 | rezero, skipinit, fixup, layerscale | 50, 152 | C10, C100 | 42/123/456 | 48 | 핵심 비교 후반 |
| batch3a | Tier2 전반 | lrs_mid, lrs_high, hybrida | 50, 152 | C10, C100 | 42/123/456 | 36 | init sensitivity |
| batch3b | Tier2 후반 | lrs_ha_mid, lrs_ha_high | 50, 152 | C10, C100 | 42/123/456 | 24 | init sensitivity |
| batch4 | Tier3 | fixed_alpha_01/03/05/07, per_channel_lrs | 152 (50) | C100 | 42/123/456 | 18 | ablation |
| batch5 | Tier4 | baseline, lrs_low, lrs_ha_low | 101, 200 | C10, C100 | 42/123/456 | 36 | depth scaling |
| batch6_wrn | 일반화 | wrn_* 6종 | 28 | C10, C100 | 42/123/456 | 36 | WRN 일반화 |
| batch6_resnext | 일반화 | resnext_* 6종 | 50 | C10, C100 | 42/123/456 | 36 | ResNeXt 일반화 |
| batch7_gpu1 | ImageNet | baseline, lrs_low, lrs_mid | 50 | ImageNet | 42 | 3 | 대규모 검증 |
| batch7_gpu2 | ImageNet | lrs_ha_low, rezero | 50 | ImageNet | 42 | 2 | 대규모 검증 |
| **합계** | | | | | | **~299 runs** | |

---

## 6. 결과 파일 구조

```
results-json/
  {model}_depth{N}_{dataset}_seed{seed}_result.json
  └── results/
        ├── best_acc          # 최고 test accuracy
        ├── best_epoch        # 최고 accuracy 달성 epoch
        ├── final_acc         # 마지막 epoch accuracy
        ├── history/
        │   ├── train_acc[]   # epoch별 train accuracy
        │   ├── test_acc[]    # epoch별 test accuracy
        │   └── alpha_stats[] # epoch별 α 평균/표준편차
        └── alpha_final/
              ├── mean        # 마지막 epoch α 평균
              ├── std         # 마지막 epoch α 표준편차
              └── all[]       # 블록별 α 값 전체 리스트
  imagenet/
    {model}_imagenet_result.json
```

---

## 7. 현재 진행 상황 요약 (2026-02-26 기준)

| Batch | 상태 | 비고 |
|-------|------|------|
| batch1 (Pilot) | ✅ 완료 | |
| batch2a (Tier1 전반) | ✅ 완료 | |
| batch2b (Tier1 후반) | 🔄 진행 중 | GPU2, skipinit 중 |
| batch3a (Tier2 전반) | ⏳ 자동 대기 | batch2b 완료 후 자동 시작 |
| batch3b (Tier2 후반) | ⏳ 자동 대기 | |
| batch4 (Tier3) | 🔄 진행 중 | GPU1, fixed_alpha 중 |
| batch5 (Tier4) | ⏳ 자동 대기 | |
| batch6_wrn | ⏳ 자동 대기 | |
| batch6_resnext | ⏳ 자동 대기 | |
| batch7_imagenet | ⏳ 자동 대기 | |
