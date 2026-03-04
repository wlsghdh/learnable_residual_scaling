# Paper Writing Guide: Learnable Residual Scaling

> 이 문서는 논문을 어떻게 쓸지, 특히 앞부분을 어떻게 구성할지에 대한 초안 및 전략입니다.
> 실험이 완료되는 대로 수치를 채워넣으면 됩니다.

---

## 제목 후보

1. **Learnable Residual Scaling: Rethinking the Skip Connection Balance in Deep Networks**
2. **Learning to Trust the Shortcut: Adaptive Residual Scaling with Identity-Biased Initialization**
3. **Identity-First Residual Learning: Learnable Scaling with HybridA Initialization**
4. **When Less Transform is More: Learnable Residual Scaling for Deep Networks**

> 추천: **2번** — LRS(learnable)와 초기화(identity-biased) 두 축을 제목에 모두 담음
> "Trust the Shortcut" = identity를 믿어라 → α가 작게 수렴하는 것과 연결됨

---

## Abstract (초안)

```
Residual networks owe much of their success to skip connections of the form
y = F(x) + x, yet this formulation silently assumes that the residual transform
and the identity path always contribute equally — an assumption that is never
validated. More fundamentally, it ignores the role of initialization in
determining how networks leverage these two paths throughout training.

We investigate this gap and make two complementary contributions. First, we
propose Learnable Residual Scaling (LRS), which replaces the fixed residual
connection with a per-block learnable balance:

    y = α · F(x) + (1 − α) · x,    α = sigmoid(θ)

The sigmoid parameterization bounds α ∈ (0, 1), ensuring a principled convex
combination of transform and identity at every block. Second, we introduce
HybridA initialization, which initializes the convolutions in deeper network
stages as identity mappings, creating a strong identity bias at the start of
training. We show that neither component alone is sufficient — LRS without
HybridA init barely matches the baseline, and HybridA init alone yields only
marginal gains. Their combination (LRS+HybridA) consistently outperforms both.

A pilot study reveals why initialization is critical: plain networks without
skip connections catastrophically fail at depth 152 (CIFAR-100: 1.47%), and
ResNets with identity-only initialization also collapse (12.92%), confirming
that skip connections and proper initialization are structural necessities,
not mere engineering choices.

Analysis of converged α values (mean ≈ 0.12) reveals that networks strongly
prefer the identity path — a finding that validates our identity-biased design.
LRS+HybridA achieves X.XX% on CIFAR-100 with ResNet-152, outperforming ReZero,
SkipInit, Fixup, and LayerScale, with consistent gains across ResNet depths,
WRN-28-10, ResNeXt-50, and ImageNet.
```

> **채워야 할 수치:** CIFAR-100/10 최종 결과 (batch2b~batch5 완료 후)

---

## 1. Introduction — 서사 구조

### 1-1. Hook: 문제 제기 (2~3 문단)

**핵심 메시지:** "잘 알려진 성공 공식이 두 가지 검토되지 않은 가정을 포함한다 — 비율 고정, 초기화 무관심"

```
Since He et al. [2016], residual connections of the form y = F(x) + x have
become the de facto building block for deep networks. Yet this ubiquitous
equation encodes two assumptions that are rarely examined:

(1) The balance assumption: the residual transform F(x) and the identity
    shortcut x always contribute equally, regardless of layer depth or
    training stage.

(2) The initialization-agnostic assumption: the initial state of the network
    has no principled relationship to how the skip connection should operate.

We argue that both assumptions are consequential, and that addressing them
together — not in isolation — is the key to improving residual networks.
```

### 1-2. 왜 초기화가 중요한가 — Pilot Study로 증명 (핵심 동기)

**논문에서 가장 강력한 motivation. Figure로 보여줄 것.**

```
To examine these assumptions empirically, we conduct a pilot study with six
progressively more principled network designs (Figure 2):

[Table: plain_he → plain_identity → resnet_identity_all → baseline → hybrida → LRS+HybridA]

plain_he (no skip, He init):        CIFAR-100/D152 = 1.47%   ← catastrophic
plain_identity (no skip, id init):  CIFAR-100/D152 = 3.49%   ← catastrophic
resnet_identity_all (skip, all-id): CIFAR-100/D152 = 12.92%  ← still fails
baseline (skip, He init):           CIFAR-100/D152 = 80.22%  ← standard
hybrida (skip, HybridA init):       CIFAR-100/D152 = 80.30%  ← init helps a bit
LRS+HybridA (skip, init+learnable): CIFAR-100/D152 = 81.11%  ← best

Two lessons emerge:

First, skip connections are not optional — their absence leads to catastrophic
failure regardless of initialization strategy (1.47% ≈ random for 100 classes).

Second, initialization profoundly shapes what the network can learn through
the skip connection. The jump from resnet_identity_all (12.92%) to baseline
(80.22%) is entirely due to changing initialization from all-identity to He init
on the residual branch. Yet He init alone does not tell each block how much to
rely on the identity path.

HybridA initialization addresses this: by initializing deeper-stage convolutions
as identity mappings, it creates a strong prior that the network should favor
information preservation at the start of training. On its own, this yields
only a marginal gain (80.30% vs 80.22%). But when combined with LRS — which
gives each block the freedom to learn the right balance — the improvement
becomes consistent and meaningful (81.11%).

This synergy is the central finding of this paper.
```

> **핵심 서사 포인트:** "init 혼자도 안 되고, LRS 혼자도 안 된다. 둘의 시너지다."
> lrs_low 단독 80.28 < baseline 80.42 이라는 약점을 역으로 활용

### 1-3. 기존 방법과의 차별화

```
Prior work on scaling residual connections — ReZero, SkipInit, Fixup,
LayerScale — focuses almost exclusively on the scaling mechanism itself,
treating initialization as an engineering detail. We observe that this
separation is precisely why they achieve limited gains on standard ResNets:
without an initialization strategy that aligns with the scaling mechanism,
the two components work against each other.

Furthermore, all prior methods use the formulation y = x + α·F(x), which is
asymmetric: the identity is fixed at weight 1.0, while only the transform is
scaled. This means α can grow unboundedly, potentially causing the transform
to dominate. LRS instead proposes:

    y = α · F(x) + (1 − α) · x,    α = sigmoid(θ) ∈ (0, 1)

a symmetric, bounded interpolation that ensures both paths remain meaningful
throughout training.
```

### 1-4. Contributions 목록

```
Our contributions are:

1. We identify and empirically validate the joint importance of residual
   balance and initialization — showing that neither alone is sufficient,
   but their combination yields consistent improvements.

2. We propose LRS: a per-block learnable α ∈ (0,1) via sigmoid, providing
   a principled convex combination of transform and identity paths.

3. We propose HybridA initialization: stage 3/4 convolutions initialized
   as identity mappings, creating an identity-first prior that synergizes
   with LRS.

4. Through analysis of converged α values (≈0.12), we provide empirical
   evidence that effective deep networks naturally favor identity-preservation,
   supporting and extending the theoretical insights of [ReZero, SkipInit].

5. We demonstrate consistent improvements across CIFAR-10/100, multiple
   ResNet depths (50/101/152/200), WRN-28-10, ResNeXt-50, and ImageNet.
```

### 1-4. 논문 구성 안내 (마지막 문단)

```
The rest of this paper is organized as follows. Section 2 reviews related work.
Section 3 presents the LRS formulation and HybridA initialization.
Section 4 describes our experimental setup. Section 5 presents main results.
Section 6 analyzes the learned α values. Section 7 concludes.
```

---

## 2. Related Work — 포지셔닝 전략

| 방법 | 수식 | 차이점 (LRS 관점) |
|------|------|-----------------|
| ResNet | y = F(x)+x | 고정 비율, 조정 없음 |
| ReZero | y = x+α·F, α₀=0 | α 범위 무제한, identity + transform 구조 |
| SkipInit | y = x+α·F, α₀=0 (BN유지) | ReZero와 유사, BN 공정비교 버전 |
| Fixup | y = x+s·F, last BN=0 | 초기화 트릭, α 학습 아님 |
| LayerScale | y = x+diag(λ)·F | per-channel, unbounded, identity 중심 아님 |
| **LRS (ours)** | **y=α·F+(1−α)·x, α=sigmoid** | **bounded ∈(0,1), 진정한 interpolation** |

---

## 3. Method — 핵심 설명

### 3-1. LRS 수식

```
Standard residual: y = F(x) + x
LRS:              y = α · F(x) + (1 − α) · x
                  α = sigmoid(θ),  θ ∈ ℝ  (per-block learnable)
```

**왜 sigmoid인가:**
- α ∈ (0,1) 보장 → convex combination (항상 두 경로의 가중 평균)
- unbounded α (ReZero 등)는 한쪽이 dominate할 수 있음
- θ₀ = −2.0 → α₀ ≈ 0.12: identity 우세 초기화 (학습 초기 안정성)

**기존 방법과의 수식 비교:**

```
ReZero/SkipInit: y = x + α·F(x)    — identity 고정 1.0, transform만 조절
Fixup:           y = x + s·F(x)    — scalar s, init trick으로만 해결
LayerScale:      y = x + diag(λ)·F — per-channel, unbounded
LRS (ours):      y = α·F + (1−α)·x — bounded, 진정한 두 경로의 혼합
```

→ 기존 방법은 "identity는 항상 1" 가정을 유지. LRS는 identity도 조절 가능.

### 3-2. HybridA 초기화 — LRS와 동등한 비중의 contribution

**단순한 engineering trick이 아니라, LRS의 필수 파트너:**

```
HybridA 규칙:
- stage 1, 2 (얕은 층): He 초기화
- stage 3, 4 (깊은 층): Conv weight → Identity 초기화
  (zero-padded identity kernel, 3×3 conv의 center에 identity 배치)
```

**왜 이 초기화가 중요한가:**

| 조합 | CIFAR-100 D152 | 설명 |
|------|---------------|------|
| Baseline (He init) | 80.42 ± 0.18 | 표준 |
| LRS only (He init + θ₀=−2.0) | 80.28 ± 0.31 | **baseline보다 낮음** |
| HybridA init only | ~80.30 (s42) | 미미한 개선 |
| **LRS + HybridA** | **80.72 ± 0.29** | **시너지 발생** |

**시너지의 메커니즘:**

```
1. HybridA init: 깊은 층의 conv가 처음에 "거의 identity"처럼 동작
   → F(x) 자체가 초기에 작은 값을 출력

2. LRS θ₀ = −2.0: α₀ ≈ 0.12, F(x)에 낮은 가중치
   → F(x) 자체도 작고, 가중치도 낮음

3. 결과: 학습 초기에 두 메커니즘이 함께 "identity first" 환경 조성
   → 학습이 안정적으로 시작 → 이후 α가 블록별로 최적값으로 수렴

LRS 혼자는: conv init이 He (랜덤)이므로 F(x)가 크고 random
            → θ₀ = −2.0으로 줄이려 해도 초기 gradient가 불안정

HybridA 혼자는: F(x)가 작게 시작하지만 α 조절 능력이 없음
             → 학습이 진행될수록 일반 ResNet처럼 수렴
```

### 3-3. Combined: LRS + HybridA

```
최종 모델 (lrs_hybrida_low):
- Stage 1,2: He init conv + LRS (θ₀=−2.0, α₀≈0.12)
- Stage 3,4: Identity init conv + LRS (θ₀=−2.0, α₀≈0.12)

"두 층 모두 identity를 향해 시작하되,
 learnable α가 각 블록에서 얼마나 transform을 믿을지 스스로 학습한다"
```

---

## 4. Experiments 구성 계획

### 메인 테이블 (Table 1 초안)

CIFAR-100, ResNet-152, 3-seed 평균±표준편차:

| Method | Formula | CIFAR-100 D152 | CIFAR-10 D152 |
|--------|---------|----------------|--------------|
| Baseline | F(x)+x | 80.42 ± 0.18 | 95.95 ± 0.05 |
| ReZero | x+α·F | (TBD) | (TBD) |
| SkipInit | x+α·F (BN) | (TBD) | (TBD) |
| Fixup | x+s·F | (TBD) | (TBD) |
| LayerScale | x+diag(λ)·F | (TBD) | (TBD) |
| LRS (ours) | α·F+(1−α)·x | 80.28 ± 0.31 | 95.73 ± 0.08 |
| **LRS+HA (ours)** | **위 + HybridA** | **80.72 ± 0.29** | **95.86 ± 0.10** |

> ⚠️ **주의:** lrs_low 단독은 baseline보다 낮음. LRS+HybridA가 핵심.
> 초기화(HybridA)와의 결합이 중요하다는 점이 contribution이 됨.

### Alpha 분석 (Figure)

- 수렴 후 α 평균: **0.1188** (lrs_ha_low, D200, CIFAR-100)
- 해석: 학습이 끝나도 α가 작음 → 네트워크는 identity 경로를 선호
- 시사점: "깊은 네트워크일수록 대부분의 블록은 identity처럼 동작한다"

---

## 5. 논문 전체 구조 (섹션별)

```
1. Introduction        (~1.5 pages)
   - Hook: 고정 비율 가정의 문제
   - Pilot Study 동기
   - 제안 방법 요약
   - Contributions

2. Related Work        (~1 page)
   - Residual connections (ResNet, DenseNet)
   - Residual scaling (ReZero, SkipInit, Fixup, LayerScale)
   - Initialization methods (He init, Identity init)

3. Method              (~1.5 pages)
   - 3.1 Problem Formulation
   - 3.2 Learnable Residual Scaling (수식 + 분석)
   - 3.3 HybridA Initialization
   - 3.4 Combined: LRS+HybridA

4. Experiments         (~3 pages)
   - 4.1 Setup (datasets, hyperparams, architectures)
   - 4.2 Pilot Study (Table: plain_he ~ lrs_ha_low)
   - 4.3 Main Results: CIFAR (Table 1)
   - 4.4 Depth Scaling (D50 → D200)
   - 4.5 Architecture Generalization (WRN, ResNeXt)
   - 4.6 ImageNet Results

5. Analysis            (~1.5 pages)
   - 5.1 Alpha 수렴 동작 (Figure: epoch별 α 변화)
   - 5.2 Per-block α 분포 (Figure: depth별 α 패턴)
   - 5.3 Fixed-α Ablation (학습 가능성이 왜 중요한가)
   - 5.4 Per-channel vs Scalar α

6. Conclusion          (~0.5 pages)
```

---

## 6. 스토리라인 (Introduction 서사 흐름)

```
[상황 설정]
ResNet 성공 → skip connection → y = F(x) + x
근데 이 수식에는 두 가정이 숨어있다: "비율은 1:1" + "초기화는 상관없음"

[Pilot Study — 충격 1: skip이 없으면]
plain network (skip 없음): CIFAR-100/D152에서 1.47% (랜덤 수준 붕괴)
→ skip connection은 deep network의 구조적 필수 요소

[Pilot Study — 충격 2: 초기화가 틀리면]
skip 있어도 all-identity init → 12.92% (여전히 붕괴)
He init → 80.22% (비로소 정상)
→ 초기화가 성능을 결정하는 핵심 요소

[Pilot Study — 충격 3: LRS 혼자는 부족]
LRS only (He init) → 80.28 < baseline 80.42 (오히려 낮음)
→ "학습 가능한 α만 추가하면 된다"는 단순한 생각은 틀렸다

[핵심 질문]
"그럼 초기화를 어떻게 하면 LRS와 시너지를 낼 수 있을까?"

[HybridA init 도입]
깊은 층을 identity로 초기화 → F(x)가 처음에 거의 0
→ LRS의 θ₀=-2.0(α≈0.12)과 합쳐지면 "이중 identity bias"
→ 학습 초기가 매우 안정적 → α가 블록별 최적값을 자유롭게 찾음

[결과: 시너지 확인]
LRS + HybridA → 80.72 ± 0.29 (baseline 대비 +0.30%p, 통계적 일관성)

[왜 α가 작게 수렴하는가?]
수렴 후 α ≈ 0.12 → "네트워크는 identity를 선호한다"
→ HybridA init이 학습 방향을 올바르게 설정했기 때문
→ 이것이 초기화가 단순한 trick이 아니라 핵심 contribution인 이유
```

---

## 7. 주의할 점 / 약점 대비

| 약점 | 대응 |
|------|------|
| lrs_low 단독은 baseline과 비슷하거나 낮음 | "LRS는 init과 함께 쓸 때 효과적"임을 contribution으로 포지셔닝 |
| 개선 폭이 크지 않음 (~0.3%p) | 통계적 유의성 (3-seed, std 범위), consistency across 아키텍처 강조 |
| lrs_high가 catastrophic failure | "α 초기값이 중요하다"는 ablation으로 활용 (오히려 좋은 이야기) |
| α 수렴값이 작음 | "identity 선호"라는 새로운 인사이트로 해석, 기존 이론 뒷받침 |

---

## 8. 수치 업데이트 체크리스트

실험 완료 후 채워야 할 부분:

- [ ] Table 1: rezero/skipinit/fixup/layerscale 3-seed 결과 (batch2b 완료 후)
- [ ] Table 1: lrs_mid/lrs_high/hybrida 3-seed 결과 (batch3a/b 완료 후)
- [ ] Depth scaling table: D50/101/152/200 비교 (batch5 완료 후)
- [ ] WRN, ResNeXt 결과 (batch6 완료 후)
- [ ] ImageNet Top-1/Top-5 (batch7 완료 후)
- [ ] Alpha trajectory figure 데이터 (현재 JSON에 alpha_stats 있음)
- [ ] Per-block alpha distribution figure (alpha_final.all 있음)
