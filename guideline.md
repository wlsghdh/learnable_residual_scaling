# 논문 Figure & 실험 가이드라인
> "ResNet은 스스로 깊이를 선택한다: Learnable Residual Scaling을 통한 실증 분석"

---

## 논문 구조 ↔ Figure 매핑

| Section | Figure | 내용 | 데이터 소스 | 상태 |
|---------|--------|------|------------|------|
| §2 Method | **Fig 1** | ResNet Block vs LRS Block 구조도 | 다이어그램 (데이터 불필요) | ✅ 완료 |
| §3.1 α 수렴 | **Fig 2** | α_mean vs Depth (초기값별) | Exp 1: lrs_low/mid/high × d50/101/152/200 | 🔄 3-seed 완료 후 업데이트 |
| §3.1 α 수렴 | **Fig 3** | α_mean 수렴 비교 (Low≈Mid, High 실패) | Exp 1: 3 초기값 비교 | 🔄 동일 |
| §3.2 α 궤적 | **Fig 4** | α_mean vs Epoch (학습 과정) | Exp 4: alpha_stats per epoch | ✅ seed42 데이터 있음 |
| §3.3 Per-block α | **Fig 5** | Block index vs α (depth별 비교) | Exp 2: alpha_final.all | ✅ seed42 데이터 있음 |
| §3.4 유효 깊이 | **Fig 6** | Nominal Depth vs Effective Depth | Exp 3: D_eff = Σα_i 계산 | 🔲 분석 코드 필요 |
| §3.5 Fixed-α | **Fig 7** | Fixed α vs Learnable α 성능 비교 | Exp 5: fixed_alpha 결과 | ✅ 데이터 완료 |
| §3.6 High 실패 | **Fig 8** | LRS_High depth별 성능 + α 궤적 | Exp 6: lrs_high 결과 | ✅ seed42 데이터 있음 |
| §3.7 방법론 비교 | **Fig 9** | ReZero/SkipInit/Fixup/LayerScale 비교 | Exp 8: 비교 방법론 결과 | ✅ 3-seed 완료 |
| §3.8 ImageNet | **Fig 10** | ImageNet Top-1/Top-5 + α 분포 | Exp 9: ImageNet 결과 | 🔲 lifeai 실험 후 |

---

## Figure 상세 스펙

### Fig 1: Block Diagram (구조도)
- **형식**: 2-panel (a) Standard ResNet Block, (b) LRS Block
- **수식**: (a) y = F(x) + x, (b) y = α·F(x) + (1-α)·x, α = σ(θ)
- **스크립트**: `scripts/visualize/fig1_block_diagram.py`
- **크기**: single column (3.5" wide)

### Fig 2: α Convergence across Depths (핵심 Figure)
- **형식**: Line plot, x=Depth(50,101,152,200), y=α_mean
- **라인**: LRS_Low (blue), LRS_Mid (green), LRS_High (red)
- **에러바**: 3-seed mean ± std
- **포인트**: Low와 Mid가 수렴, High는 깊을수록 발산
- **데이터**: `results-json/{model}_depth{D}_{dataset}_seed{S}_result.json` → `results.alpha_final.mean`
- **스크립트**: `scripts/visualize/fig2_alpha_convergence.py`
- **크기**: single column

### Fig 3: Initialization Sensitivity
- **형식**: 2-panel (a) CIFAR-10, (b) CIFAR-100
- **Bar chart**: Baseline vs LRS_Low vs LRS_Mid vs LRS_High, depth별 accuracy
- **하이라이트**: LRS_High depth200 catastrophic failure
- **데이터**: `results.best_acc`
- **스크립트**: `scripts/visualize/fig3_init_sensitivity.py`
- **크기**: double column

### Fig 4: α Training Trajectory
- **형식**: Line plot, x=Epoch, y=α_mean
- **라인**: Depth 50/101/152/200 (색상 구분)
- **모델**: LRS_Low (main), LRS_Mid (supplementary)
- **데이터**: `results.history.alpha_stats[].mean`
- **스크립트**: `scripts/visualize/fig4_alpha_trajectory.py`
- **크기**: single column

### Fig 5: Per-block α Distribution (핵심 Figure)
- **형식**: 4-panel subplot (depth 50/101/152/200)
- **Bar chart**: Block index(x) vs α value(y)
- **색상**: Stage별 구분 (stage1=blue, stage2=green, stage3=orange, stage4=red)
- **모델**: LRS_Low, CIFAR-100
- **데이터**: `results.alpha_final.all` (리스트)
- **스크립트**: `scripts/visualize/fig5_perblock_alpha.py`
- **크기**: double column

### Fig 6: Effective Depth (핵심 Figure, 새로 작성 필요)
- **형식**: 2-panel
  - (a) Nominal Depth vs D_eff = Σα_i (line plot)
  - (b) D_eff vs Accuracy, Baseline accuracy 점선으로 비교
- **포인트**: ResNet-200의 D_eff ≈ 7~8 → ResNet-50 수준의 유효 깊이
- **데이터**: alpha_final.all에서 계산
- **스크립트**: `scripts/visualize/fig6_effective_depth.py` ← **새로 작성**
- **크기**: double column

### Fig 7: Fixed-α Ablation
- **형식**: Bar chart, x=α value (0.1, 0.3, 0.5, 0.7, Learnable)
- **y**: Accuracy (CIFAR-100, depth152)
- **에러바**: 3-seed
- **포인트**: Learnable > Fixed, per-block 적응 중요
- **데이터**: `results-json/fixed_alpha_{01,03,05,07}_depth152_cifar100_seed{42,123,456}_result.json`
- **스크립트**: `scripts/visualize/fig7_fixed_alpha.py`
- **크기**: single column

### Fig 8: LRS_High Failure Analysis
- **형식**: 2-panel
  - (a) Depth vs Accuracy (Baseline/Low/Mid/High)
  - (b) α_mean trajectory for LRS_High (depth별)
- **데이터**: lrs_high 결과
- **스크립트**: `scripts/visualize/fig8_high_failure.py`
- **크기**: double column

### Fig 9: Method Comparison
- **형식**: Grouped bar chart
- **Methods**: Baseline, LRS_Low, ReZero, SkipInit, Fixup, LayerScale
- **Depths**: 50, 152
- **에러바**: 3-seed
- **데이터**: 각 method의 result JSON
- **스크립트**: `scripts/visualize/fig9_method_comparison.py`
- **크기**: double column

### Fig 10: ImageNet Results
- **형식**: 2-panel
  - (a) Bar chart: Top-1/Top-5 accuracy (Baseline, LRS_Low, LRS_Mid, ReZero)
  - (b) Per-block α distribution (LRS_Low, ResNet-50, 16 blocks)
- **데이터**: `results-json/imagenet/{model}_imagenet_result.json`
- **스크립트**: `scripts/visualize/fig10_imagenet.py` ← **새로 작성 (데이터 후)**
- **크기**: double column

---

## 실험 실행 가이드

### lifeai 서버 (ImageNet)
```bash
cd /home/jjh0709/Learnable_Reidual_Scaling

# ImageNet 데이터 확인
ls /nfs_share/datasets/ILSVRC2012/{train,val}

# 실행 (GPU 0 사용)
nohup bash scripts/run/lifeai/run_imagenet.sh 0 > logs/imagenet/lifeai_master.log 2>&1 &

# 모니터링
tail -f logs/imagenet/lifeai_master.log
nvidia-smi -l 10
```

### ahnbi 서버 (CIFAR) — GPU 2장 병렬
```bash
cd /home/jjh0709/Learnable_Reidual_Scaling

# GPU0: d101(9) → d200 전반(10) = 19 runs, ~49h
nohup bash scripts/run/ahnbi/gpu0_d101_d200a.sh > logs/ahnbi_gpu0.log 2>&1 &

# GPU1: d152(8) → d200 후반(10) = 18 runs, ~55h
nohup bash scripts/run/ahnbi/gpu1_d152_d200b.sh > logs/ahnbi_gpu1.log 2>&1 &

# 모니터링
tail -f logs/ahnbi_gpu0.log   # GPU0 진행
tail -f logs/ahnbi_gpu1.log   # GPU1 진행

# 결과 확인
ls results-json/ahnbi/
```
> 메모리: d101 batch64 ~6GB, d152 batch32 ~10GB, d200 batch32 ~12GB (RTX 3090 24GB의 ~50~70%)

---

## Git 충돌 방지 규칙

| 서버 | 커밋 범위 | 절대 수정 금지 |
|------|----------|---------------|
| **lifeai** | `results-json/lifeai/` | ImageNet 결과만 |
| **ahnbi** | `results-json/ahnbi/` | CIFAR 잔여 결과만 |

> 서버별 폴더가 완전히 분리 → **충돌 불가**

### Merge 절차
```bash
# lifeai에서
git add results-json/lifeai/
git commit -m "Add ImageNet results (lifeai)"
git push

# ahnbi에서
git add results-json/ahnbi/
git commit -m "Add CIFAR 3-seed results (ahnbi)"
git push

# 상대 서버에서
git pull
```

---

## JSON 데이터 구조 참조

### CIFAR 결과
```json
{
  "results": {
    "best_acc": 0.7892,           // 0~1 범위
    "best_epoch": 95,
    "final_acc": 0.7856,
    "alpha_final": {
      "mean": 0.132,
      "std": 0.15,
      "all": [0.42, 0.38, ..., 0.01]  // per-block α
    },
    "history": {
      "test_acc": [0.1, 0.2, ...],     // epoch별
      "alpha_stats": [
        {"epoch": 1, "mean": 0.5, "std": 0.01},
        ...
      ]
    }
  }
}
```

### ImageNet 결과
```json
{
  "results": {
    "best_acc": 0.7654,           // top1 / 100
    "best_top1": 76.54,
    "best_top5": 93.21,
    "best_epoch": 87,
    "alpha_final": { ... },       // 동일 구조
    "history": {
      "val_top1": [...],
      "val_top5": [...],
      "alpha_stats": [...]
    }
  }
}
```

---

## Figure 생성 순서 (권장)

1. **즉시 가능** (seed42 데이터 있음):
   - Fig 1 (블록 다이어그램)
   - Fig 4 (α 궤적)
   - Fig 5 (per-block α)
   - Fig 6 (유효 깊이) ← 분석 코드 작성 필요

2. **CIFAR 3-seed 완료 후** (ahnbi):
   - Fig 2 (α 수렴, 에러바 포함)
   - Fig 3 (초기화 민감도)
   - Fig 7 (Fixed-α ablation)
   - Fig 8 (LRS_High 실패)
   - Fig 9 (방법론 비교)

3. **ImageNet 완료 후** (lifeai):
   - Fig 10 (ImageNet 결과)

```bash
# 전체 Figure 생성 (모든 데이터 준비 후)
python scripts/visualize/generate_all.py
```
