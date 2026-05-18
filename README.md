# Learnable Residual Scaling (LRS)

**깊은 잔차 네트워크를 위한 학습 가능한 잔차 스케일링**

## 📄 논문 (Paper)

- **arXiv preprint (PDF):** [`paper_bmvc/arXiv_preprint_ver.pdf`](paper_bmvc/arXiv_preprint_ver.pdf)

[![Paper](https://img.shields.io/badge/Paper-PDF-red.svg)](paper_bmvc/arXiv_preprint_ver.pdf)

> 위 PDF 링크를 클릭하면 GitHub의 PDF 뷰어로 논문을 바로 열람할 수 있습니다.

## 핵심 아이디어

ResNet의 skip connection에서 착안하여, 각 블록이 **identity와 변환의 비율을 학습**하도록 합니다.

```
기존 ResNet:  y = F(x) + x           (1:1 고정)
제안 방법:    y = α·F(x) + (1-α)·x   (α 학습)
```

- `α = sigmoid(learnable_parameter)`
- `init_scale=-2.0` → `sigmoid(-2) ≈ 0.12` (identity 선호)
- `init_scale=0.0` → `sigmoid(0) = 0.5` (균형)
- `init_scale=2.0` → `sigmoid(2) ≈ 0.88` (transform 선호)

## 9가지 모델

| # | 모델 | 설명 | α 초기값 |
|---|------|------|----------|
| 1 | `baseline` | 기존 ResNet (1:1 고정) | - |
| 2 | `lrs_low` | LRS | ≈0.12 |
| 3 | `lrs_mid` | LRS | =0.5 |
| 4 | `lrs_high` | LRS | ≈0.88 |
| 5 | `rezero` | ReZero (α=0 시작) | 0 |
| 6 | `hybrida` | Identity 초기화 (layer3,4) | - |
| 7 | `lrs_hybrida_low` | **HybridA + LRS** ← 제안 | ≈0.12 |
| 8 | `lrs_hybrida_mid` | **HybridA + LRS** ← 제안 | =0.5 |
| 9 | `lrs_hybrida_high` | **HybridA + LRS** ← 제안 | ≈0.88 |

## 사용법

### 단일 실험

```bash
# CIFAR-10, ResNet-50
python run_experiments.py --dataset cifar10 --depth 50

# CIFAR-100, ResNet-152, GPU 1
CUDA_VISIBLE_DEVICES=1 python run_experiments.py --dataset cifar100 --depth 152
```

### CIFAR-10 전체 (50/101/152)

```bash
chmod +x run_cifar10_all.sh
CUDA_VISIBLE_DEVICES=0 nohup ./run_cifar10_all.sh > cifar10_all.log 2>&1 &
```

### CIFAR-100 전체 (50/101/152)

```bash
chmod +x run_cifar100_all.sh
CUDA_VISIBLE_DEVICES=1 nohup ./run_cifar100_all.sh > cifar100_all.log 2>&1 &
```

### 로그 확인

```bash
tail -f cifar10_all.log
tail -f cifar100_all.log
```

## 파일 구조

```
lrs_project/
├── config.py              # 설정
├── data.py                # 데이터 로딩
├── models.py              # 9가지 모델 정의
├── trainer.py             # 학습 로직
├── utils.py               # 유틸리티
├── run_experiments.py     # 메인 스크립트
├── run_cifar10_all.sh     # CIFAR-10 전체 실험
├── run_cifar100_all.sh    # CIFAR-100 전체 실험
├── run_all.sh             # 전체 실험
└── README.md
```

## 결과 저장 위치

```
results-json/
├── cifar10_depth50_all_results.json
├── cifar10_depth101_all_results.json
├── cifar10_depth152_all_results.json
├── cifar100_depth50_all_results.json
├── cifar100_depth101_all_results.json
├── cifar100_depth152_all_results.json
└── figures/
    ├── cifar10_depth50_comparison.png
    └── ...
```

## Requirements

```
torch>=1.10
torchvision
numpy
matplotlib
tqdm
```



## 현재 PID
3773181 cifar10 --depth 50
3778690 cifar100 --depth 50

