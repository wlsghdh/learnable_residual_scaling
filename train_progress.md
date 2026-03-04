# LRS 학습 진행 상황
> 마지막 업데이트: **2026-03-01 10:38:55** (30분마다 자동 갱신)

## 전체 진행률
```
[█████████████████████░░░░░░░░░] 214/299 (71.6%)
```

## 현재 실행 중
- `lrs_mid` depth=50 cifar100 seed=123
- `skipinit` depth=152 cifar100 seed=123
- `lrs_mid` depth=50 cifar100 seed=42

## GPU 상태
| GPU | 사용률 | 메모리 | 상태 |
|-----|--------|--------|------|
| GPU0 | 0 % | 0 MiB/81920 MiB | ⚪ |
| GPU1 | 100 % | 14658 MiB/81920 MiB | 🟢 |
| GPU2 | 57 % | 3904 MiB/81920 MiB | 🟢 |
| GPU3 | 0 % | 0 MiB/81920 MiB | ⚪ |
| GPU4 | 0 % | 0 MiB/81920 MiB | ⚪ |
| GPU5 | 0 % | 0 MiB/81920 MiB | ⚪ |
| GPU6 | 0 % | 0 MiB/81920 MiB | ⚪ |

## Phase별 진행 상황

### ✅ Phase 2-A Pilot  `[████████████████████] 24/24 (100%)`

  ✅ `plain_he` 4/4 → 1.47% (cifar100 d152)
  ✅ `plain_identity` 4/4 → 3.49% (cifar100 d152)
  ✅ `baseline` 4/4 → 80.22% (cifar100 d152)
  ✅ `resnet_identity_all` 4/4 → 12.92% (cifar100 d152)
  ✅ `hybrida` 4/4 → 80.30% (cifar100 d152)
  ✅ `lrs_hybrida_low` 4/4 → 81.11% (cifar100 d152)

### 🔄 Phase 2-C Tier1  `[██████████████████░░] 76/84 (90%)`

  ✅ `baseline` 12/12 → 80.22% (cifar100 d152)
  ✅ `lrs_low` 12/12 → 80.55% (cifar100 d152)
  ✅ `lrs_hybrida_low` 12/12 → 81.11% (cifar100 d152)
  ✅ `rezero` 12/12 → 79.72% (cifar100 d152)
  🔄 `skipinit` 10/12 → 79.72% (cifar100 d152)
  🔄 `fixup` 9/12 → 96.10% (cifar10 d152)
  🔄 `layerscale` 9/12 → 95.69% (cifar10 d152)

### ✅ Phase 2-D Tier2  `[████████████████████] 60/60 (100%)`

  ✅ `lrs_mid` 12/12 → 78.07% (cifar100 d152)
  ✅ `lrs_high` 12/12 → 69.98% (cifar100 d152)
  ✅ `hybrida` 12/12 → 80.30% (cifar100 d152)
  ✅ `lrs_hybrida_mid` 12/12 → 80.05% (cifar100 d152)
  ✅ `lrs_hybrida_high` 12/12 → 76.73% (cifar100 d152)

### ✅ Phase 2-E Tier3  `[████████████████████] 18/18 (100%)`

  ✅ `fixed_alpha_01` 3/3 → 79.69% (cifar100 d152)
  ✅ `fixed_alpha_03` 3/3 → 61.37% (cifar100 d152)
  ✅ `fixed_alpha_05` 3/3 → 39.51% (cifar100 d152)
  ✅ `fixed_alpha_07` 3/3 → 5.88% (cifar100 d152)
  ✅ `per_channel_lrs` 6/6 → 75.50% (cifar100 d152)

### ✅ Phase 2-F Tier4  `[████████████████████] 36/36 (100%)`

  ✅ `baseline` 12/12 → 80.71% (cifar100 d200)
  ✅ `lrs_low` 12/12 → 80.66% (cifar100 d200)
  ✅ `lrs_hybrida_low` 12/12 → 80.92% (cifar100 d200)

### ⬜ Phase 2-G WRN  `[░░░░░░░░░░░░░░░░░░░░] 0/36 (0%)`

  ⬜ `wrn_baseline` 0/6
  ⬜ `wrn_lrs_low` 0/6
  ⬜ `wrn_lrs_hybrida_low` 0/6
  ⬜ `wrn_rezero` 0/6
  ⬜ `wrn_skipinit` 0/6
  ⬜ `wrn_layerscale` 0/6

### ⬜ Phase 2-G ResNeXt  `[░░░░░░░░░░░░░░░░░░░░] 0/36 (0%)`

  ⬜ `resnext_baseline` 0/6
  ⬜ `resnext_lrs_low` 0/6
  ⬜ `resnext_lrs_hybrida_low` 0/6
  ⬜ `resnext_rezero` 0/6
  ⬜ `resnext_skipinit` 0/6
  ⬜ `resnext_layerscale` 0/6

### ⬜ Phase 4 ImageNet  `[░░░░░░░░░░░░░░░░░░░░] 0/5 (0%)`

  ⬜ `baseline` 0/1 → 77.02% (cifar100 d50)
  ⬜ `lrs_low` 0/1 → 77.21% (cifar100 d50)
  ⬜ `lrs_hybrida_low` 0/1 → 76.43% (cifar100 d50)
  ⬜ `rezero` 0/1 → 76.10% (cifar100 d50)
  ⬜ `lrs_mid` 0/1 → 76.28% (cifar100 d50)

## 주목할 결과

| 모델 | Dataset | Depth | Best Acc |
|------|---------|-------|----------|
| `baseline` | cifar100 | 50 | **77.02%** |
| `baseline` | cifar100 | 152 | **80.22%** |
| `baseline` | cifar100 | 200 | **80.71%** |
| `lrs_low` | cifar100 | 50 | **77.21%** |
| `lrs_low` | cifar100 | 152 | **80.55%** |
| `lrs_low` | cifar100 | 200 | **80.66%** |
| `lrs_hybrida_low` | cifar100 | 50 | **76.43%** |
| `lrs_hybrida_low` | cifar100 | 152 | **81.11%** |
| `lrs_hybrida_low` | cifar100 | 200 | **80.92%** |
| `rezero` | cifar100 | 152 | **79.72%** |
| `lrs_high` | cifar100 | 200 | **5.69%** ⚠️ (collapse!) |

---
*자동 생성: `python scripts/update_progress.py`*