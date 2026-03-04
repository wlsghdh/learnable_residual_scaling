#!/bin/bash
# run_cifar10_all.sh - CIFAR-10 전체 실험 (50/101/152층)
#
# 사용법:
#   chmod +x run_cifar10_all.sh
#   CUDA_VISIBLE_DEVICES=0 nohup ./run_cifar10_all.sh > cifar10_all.log 2>&1 &

echo "=========================================="
echo "CIFAR-10 Full Experiments"
echo "Started: $(date)"
echo "=========================================="

# ResNet-50
echo ""
echo "[1/3] CIFAR-10 ResNet-50"
echo "Started: $(date)"
python run_experiments.py --dataset cifar10 --depth 50
echo "Finished: $(date)"

# ResNet-101
echo ""
echo "[2/3] CIFAR-10 ResNet-101"
echo "Started: $(date)"
python run_experiments.py --dataset cifar10 --depth 101
echo "Finished: $(date)"

# ResNet-152
echo ""
echo "[3/3] CIFAR-10 ResNet-152"
echo "Started: $(date)"
python run_experiments.py --dataset cifar10 --depth 152
echo "Finished: $(date)"

echo ""
echo "=========================================="
echo "CIFAR-10 All Experiments Completed!"
echo "Finished: $(date)"
echo "=========================================="
