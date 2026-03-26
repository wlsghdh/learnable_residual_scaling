"""
Alpha-based Block Pruning 실험
학습된 LRS 모델에서 α가 낮은 블록을 제거하고 accuracy 측정
"""

import torch
import torch.nn as nn
import json
import os
import argparse

from models import create_model
from data import get_data_loaders
from config import get_training_config, set_seed


def evaluate(model, test_loader, device):
    """Test accuracy 측정"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct / total * 100


def get_alpha_values(model):
    """모델에서 per-block alpha 값 추출"""
    alphas = {}
    for name, module in model.named_modules():
        if hasattr(module, 'residual_scale'):
            alpha = torch.sigmoid(module.residual_scale).item()
            alphas[name] = alpha
    return alphas


def prune_blocks(model, threshold):
    """alpha < threshold인 블록을 identity로 만듦 (residual_scale을 -100으로 설정)"""
    pruned = 0
    total = 0
    for name, module in model.named_modules():
        if hasattr(module, 'residual_scale'):
            total += 1
            alpha = torch.sigmoid(module.residual_scale).item()
            if alpha < threshold:
                # alpha를 0으로 만들어서 y = (1-0)*x + 0*F(x) = x (identity)
                module.residual_scale.data.fill_(-100.0)
                pruned += 1
    return pruned, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 100 if args.dataset == 'cifar100' else 10
    train_config = get_training_config(args.depth)

    # 데이터 로드
    (_, test_loader), _ = get_data_loaders(
        dataset=args.dataset,
        batch_size=train_config['batch_size'],
    )

    # Checkpoint 경로
    if args.checkpoint is None:
        args.checkpoint = f'checkpoints/lrs_low_depth{args.depth}_{args.dataset}_seed{args.seed}_best.pth'

    print(f"=== Pruning Experiment ===")
    print(f"  Depth: {args.depth}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Checkpoint: {args.checkpoint}")
    print()

    # 다양한 threshold로 pruning 테스트
    thresholds = [0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30, 0.50]

    results = []

    for threshold in thresholds:
        # 매번 새로 모델 로드 (이전 pruning 영향 제거)
        model = create_model('lrs_low', depth=args.depth, num_classes=num_classes)
        state_dict = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
        model = model.to(device)

        if threshold == 0.0:
            # pruning 없음 (원본)
            alphas = get_alpha_values(model)
            alpha_list = sorted(alphas.values())
            pruned, total = 0, len(alphas)
            acc = evaluate(model, test_loader, device)
            print(f"  Original (no pruning): {acc:.2f}% | {total} blocks")
            print(f"  Alpha stats: min={min(alpha_list):.4f} max={max(alpha_list):.4f} mean={sum(alpha_list)/len(alpha_list):.4f}")
            print()
        else:
            pruned, total = prune_blocks(model, threshold)
            kept = total - pruned
            acc = evaluate(model, test_loader, device)
            pct_removed = pruned / total * 100
            print(f"  α < {threshold:.2f}: removed {pruned}/{total} ({pct_removed:.0f}%) | kept {kept} | acc={acc:.2f}%")

        results.append({
            'threshold': threshold,
            'pruned': pruned,
            'total': total,
            'kept': total - pruned,
            'pct_removed': pruned / total * 100 if total > 0 else 0,
            'accuracy': acc,
        })

    # 결과 저장
    output = {
        'depth': args.depth,
        'dataset': args.dataset,
        'seed': args.seed,
        'checkpoint': args.checkpoint,
        'results': results,
    }

    output_path = f'results-json/pruning_d{args.depth}_{args.dataset}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
