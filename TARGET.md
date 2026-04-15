# BMVC 2026 Submission Target — Learnable Residual Scaling

> **목표**: BMVC 2026에 accept되는 논문 작성
> **마감**: Abstract 5/22, Full paper 5/29
> **현재 LaTeX**: `overleaf_paperB_v2.tex` (Elsevier journal format → BMVC 변환 필요)

---

## 1. 논문 한 줄 요약

**Learnable Residual Scaling (LRS)을 *분석 도구*로 사용해 deep ResNet이 자신의 effective depth를 자율적으로 압축한다는 사실을 직접 관찰·정량화한다.**

성능 향상 논문이 아니라 **observation/analysis 논문**이라는 점이 핵심 positioning.

---

## 2. BMVC reviewer가 던질 4가지 질문 — 우리의 답

### Q1. "이거 새롭냐? Veit et al. 2016 (Residual networks behave like ensembles)이 이미 effective path 짧다고 했잖아"

**답**:
- Veit는 *간접 증거* (path 분석, lesion study)
- LRS는 **direct, per-block measurement**: α 값 자체가 블록 사용량을 정량적으로 보여줌
- Pruning validation (Exp 10): α<0.03 블록 제거시 정확도 0% 하락 → α가 진짜 importance를 반영
- **차별화 포인트**: "관찰 가능한 도구를 제공" + "정량적 metric (D_eff)"

### Q2. "Highway Network gate랑 뭐가 다르냐?"

**답**:
- Highway: input-dependent gate $T(x) = \sigma(W_g x + b_g)$
- LRS: input-independent scalar $\alpha = \sigma(\theta)$ per block
- **결정적 차이**: LRS의 α는 **per-block scalar**라서 블록 단위 해석이 가능. Highway gate는 입력마다 다르므로 "이 블록이 얼마나 쓰이는가"를 단일 값으로 말할 수 없음
- → Related Work에서 명시적으로 한 단락 추가

### Q3. "ImageNet에서도 같은 패턴이냐?"

**답**:
- 현재 ImageNet ResNet-50 (16 blocks)만 있음
- **추가 실험 필요**: ImageNet ResNet-101에서 mean α가 CIFAR d101과 유사 (≈0.18)하면 → dataset-independence 강력 입증
- → Priority 1 추가 실험 항목

### Q4. "분석만 하고 끝? Practical impact는?"

**답**:
- α-guided pruning: 15% block 제거시 −1% 정확도 (Exp 10)
- **추가 강화 필요**: FLOPs/Latency 정량 측정 → "실질적 inference cost 절감 도구"로 positioning
- → Priority 1 추가 실험 항목

---

## 3. 논문 구조 (BMVC 9-page format)

| Section | 분량 | 핵심 메시지 |
|---------|------|-----------|
| 1. Introduction | 1 page | "deep ResNet이 진짜 다 쓰는가?" 질문 + LRS 도입 + 3 contribution |
| 2. Related Work | 0.75 | Residual scaling (ReZero/SkipInit/Fixup/LayerScale), Highway Network 비교, Effective depth (Veit, Stochastic Depth) |
| 3. Method | 1.25 | LRS 정의, sigmoid 정당화, gradient flow 분석, 3 init variant |
| 4. Experiments | 4 | Setup → α convergence → per-block 분포 → effective depth → init sensitivity → fixed vs learned → identity init → pruning → ImageNet |
| 5. Discussion | 1.5 | Self-selection 해석, downsampling 역할, "왜 standard ResNet도 잘 되는가", practical implications |
| 6. Conclusion | 0.5 | 6 findings 재요약 + future work |

References 별도 page.

---

## 4. 핵심 Figures (12개)

| Fig | 제목 | 상태 |
|-----|------|------|
| 0 | Gradient flow at init (plain/baseline/LRS) | ✅ |
| 1 | LRS block diagram | ✅ |
| 2 | α convergence vs depth (3 init variants) | ✅ |
| 3 | α trajectory during training | ✅ |
| 4 | Per-block α (66 blocks, ResNet-200) | ✅ |
| 5 | Effective depth visualization | ✅ |
| 6 | LRS High failure analysis | ✅ |
| 7 | Fixed vs learned α | ✅ |
| 8 | α heatmap by stage | ✅ |
| 9 | Method comparison (ReZero/Fixup/LayerScale) | ✅ |
| 10 | ImageNet validation | ✅ |
| 11 | Pruning + fine-tuning | ✅ |
| **12 (NEW)** | **FLOPs vs Accuracy trade-off** | ⚠️ 추가 실험 후 |

---

## 5. 핵심 Tables (12개)

| Tab | 제목 | 상태 |
|-----|------|------|
| 1 | Method comparison (ReZero/SkipInit/Fixup/LayerScale/LRS) | ✅ |
| 2 | LRS init variants | ✅ |
| 3 | Mean converged α by depth | ✅ |
| 4 | CIFAR-10 accuracy | ✅ |
| 5 | CIFAR-100 accuracy | ✅ |
| 6 | Effective depth threshold sensitivity | ✅ |
| 7 | D_eff (threshold-free) | ✅ |
| 8 | Fixed-α ablation | ✅ |
| 9 | Identity init effect | ✅ |
| 10 | Comparison with existing methods | ✅ |
| 11 | ImageNet validation | ✅ |
| 12 | Pruning results | ✅ |
| **13 (NEW)** | **Pruning + FLOPs/Latency** | ⚠️ 추가 실험 후 |
| **14 (NEW)** | **ImageNet ResNet-101 결과** | ⚠️ 추가 실험 후 |

---

## 6. Accept을 위한 강화 포인트

### A. Story arc 강화
**현재**: "deep ResNet은 effective depth가 짧다 + LRS로 보여줌"
**개선**: "Skip connection은 gradient flow를 안정화하지만, 네트워크는 그 안정성을 *블록을 우회*하는 데 사용한다 — LRS가 이 self-selection을 직접 관찰 가능하게 한다"

### B. Quantitative claim 강화
- ✅ "5-6/66 blocks active" (binary)
- ✅ D_eff = 7.5 in 200-layer (threshold-free)
- 🔥 추가: "**X% FLOPs 절감으로 Y% 정확도만 손실**" (α-guided pruning을 efficiency tool로 재포지셔닝)
- 🔥 추가: "ImageNet d101에서도 mean α = ~0.18로 CIFAR와 일치"

### C. Discussion 강화 항목
1. **Why this matters for architecture design**: stage 수가 block 수보다 중요 → MobileNet-style 설계 정당화
2. **Connection to NAS**: heterogeneous α 분포는 NAS 결과와 일치 (downsampling block 강조)
3. **Connection to BatchNorm bias toward identity** (De & Smith 2020): LRS가 implicit하게 일어나는 일을 explicit하게 측정

### D. Limitation 솔직히 인정
- Bottleneck ResNet에 한정 (Basic block, ViT는 future work)
- LRS 자체는 약간의 optimization landscape 변화 (소량의 측정 인공물)
- ImageNet ResNet-101 까지는 검증 (152/200은 cost로 미수행)

---

## 7. Priority 1 추가 실험 (요약)

### A. ImageNet ResNet-101 + LRS Low
- **Where**: lifeai (A100)
- **When**: 4/16 시작 → 4/19 완료
- **Output**: Table 14, ImageNet d101 mean α 값
- **Script**: `bash scripts/run/batch8_imagenet_d101.sh 0`

### B. FLOPs / Latency 분석
- **Where**: ahnbi3 (V100)
- **When**: checkpoint 학습 (4/15-4/17) → 분석 (4/18)
- **Output**: Table 13, Figure 12, Discussion paragraph 강화
- **Script**:
  ```bash
  SAVE_CHECKPOINT=1 python run_experiments.py \
    --depth 200 --dataset cifar100 --model lrs_low --seed 42
  bash scripts/run/batch10_pruning_analysis.sh 0
  ```

---

## 8. 제출 체크리스트

### 4월
- [ ] lrs_low d200 cifar100 seed42 checkpoint 학습 (V100, 6.3h)
- [ ] ImageNet ResNet-101 학습 (A100, 36h)
- [ ] FLOPs/Latency 분석 (V100, 3h)
- [ ] BMVC LaTeX 템플릿 다운로드 + 변환 시작

### 5월 1주차
- [ ] Section 4 (Experiments)에 새 결과 통합
- [ ] Figure 12, Table 13, 14 생성
- [ ] Discussion 4.X, 4.Y 추가

### 5월 2주차
- [ ] Related Work에서 Highway 비교 단락 추가
- [ ] Limitations 솔직하게 작성
- [ ] Abstract 최종화

### 5월 3주차 (마감 직전)
- [ ] **5/22 Abstract 제출**
- [ ] 동료 review 받기
- [ ] Figure 해상도/캡션 점검

### 5월 4주차
- [ ] **5/29 Full paper 제출**
- [ ] Supplementary material 정리

---

## 9. Risk & Mitigation

| Risk | 대응 |
|------|------|
| ImageNet d101 학습 실패 | seed42 고정, 기존 train_imagenet.py 검증된 코드 사용 |
| Checkpoint 학습 GPU 점유 | V100 4대 중 1대만 사용, 다른 작업과 분리 |
| BMVC 페이지 초과 | Discussion을 아껴 쓰고, supplementary로 detail 분리 |
| Reviewer "novelty 약함" | Highway Network 비교 단락 + α semantic validity (pruning) 강조 |
| Reviewer "실험 부족" | 120 CIFAR + 6+1 ImageNet + pruning + FLOPs로 충분히 방어 |
