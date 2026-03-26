"""
Fine-tuning Pruning 실험
α가 낮은 블록을 제거 후, 짧은 fine-tuning으로 네트워크를 적응시킴
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import os
import argparse
import time

from models import create_model
from data import get_data_loaders
from config import get_training_config, set_seed


def evaluate(model, test_loader, device):
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


def get_block_alphas(model):
    """블록별 alpha 값 추출"""
    alphas = {}
    for name, module in model.named_modules():
        if hasattr(module, 'residual_scale'):
            alpha = torch.sigmoid(module.residual_scale).item()
            alphas[name] = alpha
    return alphas


def prune_and_freeze(model, threshold):
    """alpha < threshold인 블록을 identity로 고정"""
    pruned_blocks = []
    total = 0
    for name, module in model.named_modules():
        if hasattr(module, 'residual_scale'):
            total += 1
            alpha = torch.sigmoid(module.residual_scale).item()
            if alpha < threshold:
                module.residual_scale.data.fill_(-100.0)
                module.residual_scale.requires_grad = False
                # 해당 블록의 conv/bn도 freeze
                for param in module.parameters():
                    if param is not module.residual_scale:
                        param.requires_grad = False
                pruned_blocks.append((name, alpha))
    return pruned_blocks, total


def finetune(model, train_loader, test_loader, device, epochs=10, lr=0.001):
    """짧은 fine-tuning"""
    model.train()
    criterion = nn.CrossEntropyLoss()

    # 학습 가능한 파라미터만 optimizer에 넣기
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc = evaluate(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
        print(f"    Epoch {epoch+1}/{epochs}: acc={acc:.2f}% (best={best_acc:.2f}%)")

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-lr', type=float, default=0.001)
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_classes = 100 if args.dataset == 'cifar100' else 10
    train_config = get_training_config(args.depth)

    (train_loader, test_loader), _ = get_data_loaders(
        dataset=args.dataset,
        batch_size=train_config['batch_size'],
    )

    if args.checkpoint is None:
        args.checkpoint = f'checkpoints/lrs_low_depth{args.depth}_{args.dataset}_seed{args.seed}_best.pth'

    print(f"=== Fine-tuning Pruning Experiment ===")
    print(f"  Depth: {args.depth}, Dataset: {args.dataset}")
    print(f"  Fine-tune: {args.finetune_epochs} epochs, lr={args.finetune_lr}")
    print(f"  Checkpoint: {args.checkpoint}")
    print()

    # 원본 accuracy
    model = create_model('lrs_low', depth=args.depth, num_classes=num_classes)
    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    original_acc = evaluate(model, test_loader, device)
    original_alphas = get_block_alphas(model)
    total_blocks = len(original_alphas)
    print(f"  Original accuracy: {original_acc:.2f}% ({total_blocks} blocks)")
    print()

    thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    results = [{
        'threshold': 0.0,
        'pruned': 0,
        'total': total_blocks,
        'kept': total_blocks,
        'layers_removed': 0,
        'pct_removed': 0,
        'acc_before_ft': original_acc,
        'acc_after_ft': original_acc,
    }]

    for threshold in thresholds:
        print(f"--- Threshold: α < {threshold:.2f} ---")

        # 매번 새로 로드
        model = create_model('lrs_low', depth=args.depth, num_classes=num_classes)
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
        model = model.to(device)

        # Pruning
        pruned_blocks, total = prune_and_freeze(model, threshold)
        kept = total - len(pruned_blocks)
        layers_removed = len(pruned_blocks) * 6
        pct = len(pruned_blocks) / total * 100

        print(f"  Removed: {len(pruned_blocks)}/{total} blocks ({pct:.0f}%), {layers_removed} layers")

        # Pruning 직후 accuracy
        acc_before = evaluate(model, test_loader, device)
        print(f"  Before fine-tune: {acc_before:.2f}%")

        # Fine-tuning
        start_time = time.time()
        acc_after = finetune(model, train_loader, test_loader, device,
                            epochs=args.finetune_epochs, lr=args.finetune_lr)
        elapsed = time.time() - start_time
        print(f"  After fine-tune:  {acc_after:.2f}% (took {elapsed/60:.1f} min)")
        print(f"  Recovery: {acc_before:.2f}% → {acc_after:.2f}% (Δ from original: {acc_after - original_acc:+.2f}%)")
        print()

        results.append({
            'threshold': threshold,
            'pruned': len(pruned_blocks),
            'total': total,
            'kept': kept,
            'layers_removed': layers_removed,
            'pct_removed': pct,
            'acc_before_ft': acc_before,
            'acc_after_ft': acc_after,
            'finetune_time': elapsed,
        })

    # 저장
    output = {
        'depth': args.depth,
        'dataset': args.dataset,
        'seed': args.seed,
        'original_acc': original_acc,
        'finetune_epochs': args.finetune_epochs,
        'finetune_lr': args.finetune_lr,
        'results': results,
    }

    output_path = f'results-json/pruning_finetune_d{args.depth}_{args.dataset}.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # 최종 요약
    print("=" * 70)
    print(f"{'Threshold':>10s} {'Removed':>10s} {'Layers':>8s} {'Before FT':>10s} {'After FT':>10s} {'Δ':>8s}")
    print("-" * 70)
    for r in results:
        t = r['threshold']
        delta = r['acc_after_ft'] - original_acc
        t_str = "None" if t == 0 else f"α<{t:.2f}"
        print(f"{t_str:>10s} {r['pruned']:>4d}/{r['total']:<4d} {r['layers_removed']:>6d}  {r['acc_before_ft']:>9.2f}% {r['acc_after_ft']:>9.2f}% {delta:>+7.2f}%")

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
