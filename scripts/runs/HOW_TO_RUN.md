# 실험 실행 가이드

## 폴더 구조

```
scripts/runs/
├── ahnbi1/        # CIFAR seed=123, GPU 0+1 병렬
│   ├── day1.sh ~ day8.sh
├── ahnbi2/        # CIFAR seed=456, GPU 0+1 병렬
│   ├── day1.sh ~ day8.sh
├── lifeai/        # ImageNet (1개씩, GPU 직접 지정)
│   ├── day1.sh, day3.sh, day5.sh, day7.sh
├── ahnbi_guide.md # 실험 상세 설명
└── HOW_TO_RUN.md
```

## 사전 준비

각 서버에서 git clone/pull 후:
```bash
cd ~/Learnable_Reidual_Scaling
git pull origin main
chmod +x scripts/runs/ahnbi1/*.sh   # ahnbi1 서버
chmod +x scripts/runs/ahnbi2/*.sh   # ahnbi2 서버
chmod +x scripts/runs/lifeai/*.sh   # lifeai 서버
```

## 실행 방법

### Ahnbi (인자 없이 실행 — GPU 0,1 자동 병렬)
```bash
# nohup (권장)
nohup bash scripts/runs/ahnbi1/day1.sh > logs/ahnbi1/nohup_day1.log 2>&1 &

# tmux
tmux new -s train
bash scripts/runs/ahnbi1/day1.sh
# Ctrl+B, D 로 분리 / tmux attach -t train 으로 복귀
```

### Lifeai (GPU ID 필수 지정)
```bash
# GPU 3번 사용 예시
nohup bash scripts/runs/lifeai/day7.sh 3 > logs/lifeai/nohup_day7.log 2>&1 &
```

## GPU 사용 방식

| 서버    | GPU 방식                                      |
|---------|-----------------------------------------------|
| ahnbi1  | GPU 0, 1 자동 병렬 (2개→동시, 3개→2+1)        |
| ahnbi2  | GPU 0, 1 자동 병렬 (2개→동시, 3개→2+1)        |
| lifeai  | GPU 1개, 인자로 직접 지정 (`bash day1.sh 3`)   |

## 진행 확인

```bash
tail -f logs/ahnbi1/day1.log                         # 전체 진행
tail -f logs/ahnbi1/lrs_high_d101_cifar10_s123.log   # 개별 모델
nvidia-smi                                            # GPU 확인
ls results-json/*seed123* 2>/dev/null | wc -l         # 완료 수
```

## 실험 일정표

### Ahnbi1 (seed=123)

| Day | Depth | Dataset  | Models                      | GPU 배치       |
|-----|-------|----------|-----------------------------|----------------|
| 1   | 101   | cifar10  | lrs_high, rezero            | 0+1 병렬       |
| 2   | 101   | cifar100 | lrs_high, rezero            | 0+1 병렬       |
| 3   | 152   | cifar10  | lrs_mid, lrs_high           | 0+1 병렬       |
| 4   | 152   | cifar100 | lrs_mid, lrs_high           | 0+1 병렬       |
| 5   | 200   | cifar10  | baseline, lrs_low           | 0+1 병렬       |
| 6   | 200   | cifar10  | lrs_mid, lrs_high, rezero   | 0+1 병렬 → 0   |
| 7   | 200   | cifar100 | baseline, lrs_low           | 0+1 병렬       |
| 8   | 200   | cifar100 | lrs_mid, lrs_high, rezero   | 0+1 병렬 → 0   |

### Ahnbi2 (seed=456)

| Day | Depth | Dataset  | Models                      | GPU 배치       |
|-----|-------|----------|-----------------------------|----------------|
| 1   | 101   | cifar10  | lrs_high, rezero            | 0+1 병렬       |
| 2   | 101   | cifar100 | lrs_mid, lrs_high, rezero   | 0+1 병렬 → 0   |
| 3   | 152   | cifar10  | lrs_mid, lrs_high           | 0+1 병렬       |
| 4   | 152   | cifar100 | lrs_mid, lrs_high           | 0+1 병렬       |
| 5   | 200   | cifar10  | baseline, lrs_low           | 0+1 병렬       |
| 6   | 200   | cifar10  | lrs_mid, lrs_high, rezero   | 0+1 병렬 → 0   |
| 7   | 200   | cifar100 | baseline, lrs_low           | 0+1 병렬       |
| 8   | 200   | cifar100 | lrs_mid, lrs_high, rezero   | 0+1 병렬 → 0   |

### Lifeai (ImageNet, seed=42)

| Day | Model    | GPU          |
|-----|----------|--------------|
| 1   | baseline | 직접 지정    |
| 3   | lrs_low  | 직접 지정    |
| 5   | lrs_mid  | 직접 지정    |
| 7   | rezero   | 직접 지정    |

## 주의사항

- 이미 완료된 실험은 자동 SKIP (결과 JSON 존재 여부 확인)
- lifeai는 GPU 규칙이 까다로우므로 하루 1개씩만, GPU 번호 직접 지정
- Git 충돌 방지: 각 서버는 자기 결과만 커밋
- 실험 상세 내용은 `ahnbi_guide.md` 참고
