# ResNet은 스스로 깊이를 선택한다: Learnable Residual Scaling을 통한 실증 분석

## 핵심 주장 (Thesis)

> **ResNet이 깊어져도 성능이 유지/향상되는 이유는, 네트워크가 불필요한 블록을 자동으로 건너뛰어 유효 깊이(effective depth)를 스스로 조절하기 때문이다.**

LRS(Learnable Residual Scaling)는 이 가설을 **명시적으로 관찰 가능**하게 만드는 도구이다.
기존 ResNet `y = F(x) + x`에서는 F(x)→0 여부를 직접 측정하기 어렵지만,
LRS `y = α·F(x) + (1-α)·x`에서는 학습된 α 값이 각 블록의 기여도를 직접 보여준다.

---

## 기존 결과에서 이미 관찰된 핵심 증거

### 1. 깊이가 깊을수록 α가 낮아진다 (identity 선호 강화)
| Depth | Blocks | α_mean (LRS_Low, CIFAR-100) | α_mean (LRS_Mid) |
|-------|--------|----------------------------|-------------------|
| 50    | 16     | 0.342                      | 0.347             |
| 101   | 33     | 0.183                      | 0.204             |
| 152   | 50     | 0.132                      | 0.147             |
| 200   | 66     | 0.114                      | 0.125             |

→ **초기값이 달라도(Low=-2.0, Mid=0.0) 같은 방향으로 수렴** = 네트워크가 스스로 최적 깊이를 찾는다

### 2. 초기화와 무관하게 α가 수렴한다
- LRS_Low (θ₀=-2.0, α₀≈0.12) → depth200 α_mean ≈ 0.114
- LRS_Mid (θ₀=0.0, α₀=0.50) → depth200 α_mean ≈ 0.125
- LRS_High (θ₀=2.0, α₀≈0.88) → depth200 **학습 실패** (α_mean ≈ 0.50, 수렴 못함)

→ **최적 α는 존재하지만, 초기값이 너무 멀면 도달 실패** = 최적 유효 깊이가 존재한다는 증거

### 3. Per-block α 분포: 앞 블록은 활용, 뒷 블록은 거의 스킵
LRS_Low, Depth200, CIFAR-100 per-block α:
- Block 1~3: α ≈ 0.30~0.42 (활성)
- Block 10~50: α ≈ 0.05~0.10 (거의 identity)
- Block 64 (마지막 근처): α ≈ 0.954 (다시 활성)
- Block 65~66: α ≈ 0.01 (거의 스킵)

→ **네트워크가 자발적으로 "사용할 블록"과 "건너뛸 블록"을 분류**

### 4. 유효 깊이를 강제로 높이면 성능 붕괴
LRS_High Depth200: CIFAR-10 16.85%, CIFAR-100 5.69% (사실상 랜덤)
→ **모든 블록을 강제로 활성화하면 학습 불가능** = 깊이 자동 조절이 핵심

---

## 실험 계획

### Exp 1. α 수렴 실험 (Core - 이미 데이터 있음)
**목적**: 초기값과 무관하게 α가 같은 값으로 수렴함을 보인다
**설계**:
- 모델: LRS_Low (θ₀=-2.0), LRS_Mid (θ₀=0.0), LRS_High (θ₀=2.0)
- Depth: 50, 101, 152, 200
- Dataset: CIFAR-10, CIFAR-100
- Seeds: 42, 123, 456 (3-seed)
**분석**:
- α_mean의 수렴 값 비교 (Low와 Mid가 거의 같은 곳에 도달하는지)
- α_mean vs depth 그래프 → 깊이 증가에 따른 α 감소 곡선
- LRS_High의 수렴 실패 지점 분석
**Figure**: α_mean vs Depth (3개 초기값 겹쳐 그리기)
**상태**: ✅ seed42 데이터 있음, 3-seed 추가 필요

### Exp 2. Per-block α 분포 분석 (Core)
**목적**: 어떤 블록이 "사용"되고 어떤 블록이 "스킵"되는지 보인다
**설계**:
- 모델: LRS_Low
- Depth: 50, 101, 152, 200
- Dataset: CIFAR-100
- 최종 epoch의 per-block α 값 시각화
**분석**:
- Block index vs α heatmap 또는 bar chart
- 블록 위치별 패턴 분석 (초반/중반/후반, stage 경계)
- "유효 블록 수" 계산: α > threshold (예: 0.2)인 블록 수
**Figure**: Per-block α distribution (depth별 비교)
**상태**: ✅ 데이터 있음

### Exp 3. 유효 깊이(Effective Depth) 정량화 (Core - 새 분석)
**목적**: 학습된 α로부터 "유효 깊이"를 계산하고, 실제 성능과 대응시킨다
**설계**:
- 유효 깊이 정의: D_eff = Σ α_i (각 블록의 α를 "기여도"로 보고 합산)
  - 또는 D_eff = #{i : α_i > τ} (threshold 기반 활성 블록 수)
- LRS_Low, depth 50/101/152/200의 D_eff 계산
- 비교: ResNet-{D_eff}의 실제 성능과 LRS ResNet-{N}의 성능 비교
**분석**:
- "ResNet-200 + LRS"의 유효 깊이가 약 X라면,
  실제 "ResNet-X (baseline)"의 성능과 비슷해야 한다 → 유효 깊이 가설 검증
- 유효 깊이 vs 명목 깊이 그래프
**Figure**: Nominal Depth vs Effective Depth, Effective Depth vs Accuracy
**상태**: 🔲 분석 코드 작성 필요 (데이터는 있음)

### Exp 4. α 학습 궤적 분석 (Core)
**목적**: α가 학습 과정에서 어떻게 변화하는지 추적
**설계**:
- 모델: LRS_Low, LRS_Mid
- Depth: 152, 200
- Dataset: CIFAR-100
- 매 epoch α_mean 기록 (이미 alpha_stats에 있음)
**분석**:
- α_mean vs epoch 그래프
- 학습 초기에 α가 빠르게 감소하는지 vs 점진적 감소인지
- LRS_Mid가 LRS_Low의 수렴 값에 도달하는 과정
**Figure**: α trajectory over training epochs
**상태**: ✅ epoch별 alpha_stats 데이터 있음

### Exp 5. Fixed-α Ablation (중요)
**목적**: α가 "학습 가능"해야 하는지, 고정값으로도 충분한지
**설계**:
- 모델: Fixed α = 0.1, 0.3, 0.5, 0.7 (non-learnable)
- Depth: 152
- Dataset: CIFAR-100
- Seeds: 42, 123, 456
**분석**:
- Fixed α=0.1 (학습된 최적값 근처) vs LRS_Low (학습 가능) 비교
- 만약 Fixed α=0.1 ≈ LRS_Low → "최적 유효 깊이는 존재하지만, 굳이 학습할 필요 없다"
- 만약 LRS_Low > Fixed α=0.1 → "per-block 적응이 중요하다" (블록마다 다른 α가 필요)
**Figure**: Fixed-α vs Learnable α accuracy 비교
**상태**: 🔲 일부 데이터 있음, 3-seed 필요

### Exp 6. LRS_High 실패 분석 (중요)
**목적**: α가 높으면(유효 깊이를 줄일 수 없으면) 왜 실패하는지 분석
**설계**:
- 모델: LRS_High (θ₀=2.0)
- Depth: 50, 101, 152, 200
- Dataset: CIFAR-10, CIFAR-100
**분석**:
- Depth별 accuracy 하락 패턴 (50은 괜찮지만 200은 붕괴)
- α 수렴 실패: LRS_High가 최적 α에 도달 못하는 과정
- Training loss curve 비교 (발산 여부)
- gradient flow 분석 (가능하면)
**Figure**: LRS_High failure analysis (accuracy + α trajectory)
**상태**: ✅ 데이터 있음

### Exp 7. Depth Scaling 실험 (새 실험 필요)
**목적**: "유효 깊이가 자동 조절된다"는 가설의 직접 검증
**설계**:
- 깊이를 50 → 101 → 152 → 200 → 302(?) 로 늘려가며
- Baseline ResNet과 LRS ResNet의 성능 비교
- 가설: 깊이를 무작정 늘려도 LRS는 유효 깊이를 조절하므로 성능 유지
- 반면 Baseline은 일정 깊이 이후 성능 하락 가능
**분석**:
- Accuracy vs Depth 곡선 (Baseline vs LRS)
- "LRS는 과도한 깊이의 해로움을 완화한다"
**Figure**: Depth scaling curve
**상태**: 🔲 depth 302 추가 실험 필요 (선택적)

### Exp 8. 비교 방법론 (Supporting)
**목적**: 유사한 목적의 기존 방법과 비교
**설계**:
- ReZero (y = x + α·F(x), α₀=0): "깊이 조절"을 주장하는 기존 방법
- SkipInit: 유사한 접근
- LayerScale: per-channel 스케일링
- Baseline: 표준 ResNet
- Depth: 50, 152 / Dataset: CIFAR-10, CIFAR-100 / 3-seed
**분석**:
- 성능 비교 + 각 방법의 학습된 스케일 값 비교
- LRS의 α가 다른 방법의 스케일보다 더 해석 가능한지
**Figure**: Method comparison table + α/scale distribution
**상태**: 🔄 일부 데이터 있음

### Exp 9. ImageNet 검증 (Supporting)
**목적**: 대규모 데이터셋에서도 동일한 현상 확인
**설계**:
- 모델: Baseline, LRS_Low, LRS_Mid, ReZero
- Architecture: ResNet-50
- Dataset: ImageNet (ILSVRC2012)
- 90 epochs, standard augmentation
**분석**:
- Top-1/Top-5 accuracy
- 학습된 α 분포
**상태**: 🔲 실행 대기

---

## 논문 스토리라인

### 1. Introduction
- ResNet의 성공 → 깊이를 늘릴 수 있게 됨
- 하지만 "왜" skip connection이 효과적인지는 여전히 논쟁
- 우리의 주장: **ResNet은 skip connection을 통해 유효 깊이를 자동 조절**
- LRS를 통해 이를 **명시적으로 관찰**하고 **실험적으로 검증**

### 2. Method: Learnable Residual Scaling
- 수식: y = α·F(x) + (1-α)·x, α = σ(θ), per-block learnable
- α의 의미: 해당 블록의 "기여도" 또는 "활성도"
- 유효 깊이: D_eff = Σ α_i
- 기존 ResNet과의 관계: α=1이면 standard ResNet, α=0이면 identity

### 3. Experiments

#### 3.1 α는 초기값에 무관하게 수렴한다 (Exp 1)
→ 최적 유효 깊이가 task/depth에 의해 결정됨

#### 3.2 깊이가 깊을수록 α가 낮아진다 (Exp 1, 3)
→ 불필요한 블록이 많을수록 네트워크가 더 적극적으로 스킵

#### 3.3 Per-block α 분포: 선택적 블록 활용 (Exp 2)
→ 모든 블록이 균등하게 쓰이는 것이 아니라, 특정 블록이 핵심적

#### 3.4 유효 깊이 축소를 못하면 학습 실패 (Exp 6)
→ LRS_High의 catastrophic failure = 깊이 조절이 필수

#### 3.5 Per-block 적응의 필요성 (Exp 5)
→ Fixed-α vs Learnable → per-block 적응이 중요한지 확인

#### 3.6 기존 방법과의 비교 (Exp 8)
→ ReZero, SkipInit 등도 유사한 효과를 내는지

#### 3.7 대규모 검증 (Exp 9)
→ ImageNet에서도 동일 현상

### 4. Discussion
- ResNet의 본질: "깊은 네트워크"가 아니라 "적응적 깊이 네트워크"
- 유효 깊이 관점에서의 네트워크 설계 시사점
- Limitation: CNN에 국한, Transformer에서의 검증 필요

### 5. Conclusion

---

## 제거한 것들 (이전 계획 대비)

| 항목 | 제거 이유 |
|------|----------|
| HybridA (Identity 초기화) | 초기화 전략은 "깊이 자동 조절" 스토리와 별개. 오히려 논점을 흐림 |
| Per-channel LRS | 핵심 스토리 불필요. Ablation으로만 언급 가능 |
| WRN/ResNeXt 일반화 | 핵심은 depth에 따른 α 변화이므로 ResNet depth 변화가 더 중요 |
| 다양한 LR schedule 비교 | 깊이 조절 스토리와 무관 |

---

## 실험 우선순위

| 순위 | 실험 | 필요 작업 | 예상 시간 |
|------|------|----------|----------|
| ★★★ | Exp 1: α 수렴 | 3-seed 추가 (LRS_Mid, LRS_High) | ~20h |
| ★★★ | Exp 2: Per-block α | 시각화 코드 작성 | 분석만 |
| ★★★ | Exp 3: 유효 깊이 정량화 | 분석 코드 작성 | 분석만 |
| ★★★ | Exp 4: α 궤적 | epoch별 데이터 확인 + 시각화 | 분석만 |
| ★★☆ | Exp 5: Fixed-α | 3-seed 실험 | ~12h |
| ★★☆ | Exp 6: LRS_High 실패 | 시각화 + 추가 분석 | 분석만 |
| ★★☆ | Exp 8: 방법론 비교 | ReZero/SkipInit 3-seed | ~20h |
| ★☆☆ | Exp 7: Depth 302 | 새 실험 (선택적) | ~10h |
| ★☆☆ | Exp 9: ImageNet | 실험 실행 | ~36h |

---

## 기존 데이터 활용 가능 여부

| 데이터 | 상태 | 비고 |
|--------|------|------|
| LRS_Low depth 50/101/152/200 (seed42) | ✅ | α_final, alpha_stats 있음 |
| LRS_Mid depth 50/101/152/200 (seed42) | ✅ | 동일 |
| LRS_High depth 50/101/152/200 (seed42) | ✅ | 실패 케이스 포함 |
| Baseline depth 50/101/152/200 (seed42) | ✅ | 비교군 |
| LRS_Low 3-seed (depth 50, 152) | ✅ | 이미 완료 |
| Baseline 3-seed (depth 50, 152) | ✅ | 이미 완료 |
| Fixed-α (depth 152, seed 42/123/456) | ✅ 일부 | 0.1/0.3/0.5/0.7 있음 |
| ReZero (seed42) | ✅ | 1-seed만 |
| Per-block α (all models) | ✅ | alpha_final.all에 있음 |
| Epoch별 α trajectory | ✅ | alpha_stats에 mean/std per epoch |
