#!/bin/bash
# run_all.sh - 전체 실험 순차 실행
#
# 사용법:
#   chmod +x run_all.sh
#   CUDA_VISIBLE_DEVICES=0 nohup ./run_all.sh > all_experiments.log 2>&1 &

echo "=========================================="
echo "LRS Full Experiments"
echo "Started: $(date)"
echo "=========================================="

# ============================================================
# CIFAR-10
# ============================================================

echo ""
echo "[CIFAR-10] ResNet-50"
python run_experiments.py --dataset cifar10 --depth 50
echo "Done: $(date)"

echo ""
echo "[CIFAR-10] ResNet-101"
python run_experiments.py --dataset cifar10 --depth 101
echo "Done: $(date)"

echo ""
echo "[CIFAR-10] ResNet-152"
python run_experiments.py --dataset cifar10 --depth 152
echo "Done: $(date)"

# ============================================================
# CIFAR-100
# ============================================================

echo ""
echo "[CIFAR-100] ResNet-50"
python run_experiments.py --dataset cifar100 --depth 50
echo "Done: $(date)"

echo ""
echo "[CIFAR-100] ResNet-101"
python run_experiments.py --dataset cifar100 --depth 101
echo "Done: $(date)"

echo ""
echo "[CIFAR-100] ResNet-152"
python run_experiments.py --dataset cifar100 --depth 152
echo "Done: $(date)"

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Finished: $(date)"
echo "=========================================="
