#!/bin/bash
# run_cifar100_all.sh - CIFAR-100 전체 실험 (50/101/152층)
#
# 사용법:
#   chmod +x run_cifar100_all.sh
#   CUDA_VISIBLE_DEVICES=1 nohup ./run_cifar100_all.sh > cifar100_all.log 2>&1 &

echo "=========================================="
echo "CIFAR-100 Full Experiments"
echo "Started: $(date)"
echo "=========================================="

# ResNet-50
echo ""
echo "[1/3] CIFAR-100 ResNet-50"
echo "Started: $(date)"
python run_experiments.py --dataset cifar100 --depth 50
echo "Finished: $(date)"

# ResNet-101
echo ""
echo "[2/3] CIFAR-100 ResNet-101"
echo "Started: $(date)"
python run_experiments.py --dataset cifar100 --depth 101
echo "Finished: $(date)"

# ResNet-152
echo ""
echo "[3/3] CIFAR-100 ResNet-152"
echo "Started: $(date)"
python run_experiments.py --dataset cifar100 --depth 152
echo "Finished: $(date)"

echo ""
echo "=========================================="
echo "CIFAR-100 All Experiments Completed!"
echo "Finished: $(date)"
echo "=========================================="
