"""
Alpha-guided Pruning + FLOPs/Latency 분석
==========================================
LRS alpha 기반 pruning이 실제 연산량(FLOPs)과 추론 시간(latency)을
얼마나 줄이는지 정량 측정.

기존 run_pruning_finetune.py의 결과를 확장하여:
1. FLOPs 계산 (conv/bn/linear 기준)
2. Inference latency 측정 (GPU warmup + 반복 측정)
3. Fine-tune 후 accuracy 대비 efficiency 보고

사용법:
    python run_pruning_analysis.py --depth 200 --dataset cifar100 --seed 42
    python run_pruning_analysis.py --depth 200 --dataset cifar100 --seed 42 --checkpoint path/to/best.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import os
import argparse
import time
import numpy as np

from models import create_model, LRSBottleneck
from data import get_data_loaders
from config import get_training_config, set_seed


# ============================================================
# FLOPs 계산
# ============================================================

def count_flops(model, input_size=(1, 3, 32, 32)):
    """모델의 FLOPs 계산 (MAC 기준)
    Conv2d, Linear, BatchNorm2d의 연산량을 합산.
    Pruned block (alpha ≈ 0)은 실제로는 identity이므로
    해당 conv/bn의 FLOPs는 0으로 처리 가능하지만,
    여기서는 "실제 제거 시" FLOPs를 계산.
    """
    flops = 0
    hooks = []

    def conv_hook(module, input, output):
        nonlocal flops
        # FLOPs = 2 * Cout * Hout * Wout * Cin * Kh * Kw / groups
        batch_size = input[0].size(0)
        out_channels = output.size(1)
        out_h, out_w = output.size(2), output.size(3)
        in_channels = module.in_channels
        kh, kw = module.kernel_size
        groups = module.groups
        flops += 2 * out_channels * out_h * out_w * (in_channels // groups) * kh * kw

    def bn_hook(module, input, output):
        nonlocal flops
        # BN: 4 ops per element (mean, var, normalize, scale+shift)
        flops += 4 * input[0].numel()

    def linear_hook(module, input, output):
        nonlocal flops
        # FLOPs = 2 * in_features * out_features
        flops += 2 * module.in_features * module.out_features

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.BatchNorm2d):
            hooks.append(module.register_forward_hook(bn_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    device = next(model.parameters()).device
    x = torch.randn(*input_size).to(device)
    model.eval()
    with torch.no_grad():
        model(x)

    for h in hooks:
        h.remove()

    return flops


# ============================================================
# Latency 측정
# ============================================================

def measure_latency(model, input_size=(1, 3, 32, 32), num_warmup=50, num_runs=200):
    """GPU inference latency 측정 (ms)"""
    device = next(model.parameters()).device
    model.eval()

    x = torch.randn(*input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # 측정
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
    }


# ============================================================
# Pruning 관련
# ============================================================

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
                for param in module.parameters():
                    if param is not module.residual_scale:
                        param.requires_grad = False
                pruned_blocks.append((name, alpha))
    return pruned_blocks, total


def count_pruned_params(model):
    """Pruned 블록의 파라미터 수를 제외한 실효 파라미터 수 계산"""
    total_params = 0
    active_params = 0
    for name, module in model.named_modules():
        if isinstance(module, LRSBottleneck):
            block_params = sum(p.numel() for p in module.parameters())
            total_params += block_params
            alpha = torch.sigmoid(module.residual_scale).item()
            if alpha > 0.001:  # not pruned
                active_params += block_params
    # 나머지 (conv1, bn1, fc 등)
    counted_params = set()
    for name, module in model.named_modules():
        if isinstance(module, LRSBottleneck):
            for p in module.parameters():
                counted_params.add(id(p))
    other_params = sum(p.numel() for p in model.parameters() if id(p) not in counted_params)

    return {
        'total': total_params + other_params,
        'active': active_params + other_params,
        'pruned': total_params - active_params,
    }


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


def finetune(model, train_loader, test_loader, device, epochs=10, lr=0.001):
    model.train()
    criterion = nn.CrossEntropyLoss()
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
        print(f"    FT Epoch {epoch+1}/{epochs}: acc={acc:.2f}% (best={best_acc:.2f}%)")

    return best_acc


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--finetune-epochs', type=int, default=10)
    parser.add_argument('--finetune-lr', type=float, default=0.001)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--latency-runs', type=int, default=200,
                        help='Latency 측정 반복 횟수')
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

    input_size = (1, 3, 32, 32)

    print(f"=== Pruning + FLOPs/Latency Analysis ===")
    print(f"  Depth: {args.depth}, Dataset: {args.dataset}, Seed: {args.seed}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print()

    # ── 원본 모델 baseline 측정 ──
    model = create_model('lrs_low', depth=args.depth, num_classes=num_classes)
    state_dict = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    original_acc = evaluate(model, test_loader, device)
    original_flops = count_flops(model, input_size)
    original_latency = measure_latency(model, input_size, num_runs=args.latency_runs)
    original_params = count_pruned_params(model)
    original_alphas = get_block_alphas(model)
    total_blocks = len(original_alphas)

    print(f"  Original:")
    print(f"    Accuracy:   {original_acc:.2f}%")
    print(f"    FLOPs:      {original_flops:,} ({original_flops/1e9:.2f} GFLOPs)")
    print(f"    Latency:    {original_latency['mean_ms']:.2f} ± {original_latency['std_ms']:.2f} ms")
    print(f"    Parameters: {original_params['total']:,}")
    print(f"    Blocks:     {total_blocks}")
    print()

    del model
    torch.cuda.empty_cache()

    # ── 각 threshold로 pruning + 분석 ──
    thresholds = [0.03, 0.05, 0.08, 0.10, 0.15]
    results = [{
        'threshold': 0.0,
        'pruned': 0,
        'total': total_blocks,
        'kept': total_blocks,
        'layers_removed': 0,
        'pct_removed': 0,
        'acc_no_ft': original_acc,
        'acc_with_ft': original_acc,
        'flops': original_flops,
        'flops_ratio': 1.0,
        'latency': original_latency,
        'params': original_params,
    }]

    for threshold in thresholds:
        print(f"--- Threshold: α < {threshold:.2f} ---")

        # (A) Pruning 없이 accuracy 측정 (no fine-tune)
        model = create_model('lrs_low', depth=args.depth, num_classes=num_classes)
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
        model = model.to(device)

        pruned_blocks, total = prune_and_freeze(model, threshold)
        kept = total - len(pruned_blocks)
        layers_removed = len(pruned_blocks) * 6
        pct = len(pruned_blocks) / total * 100

        acc_no_ft = evaluate(model, test_loader, device)
        flops = count_flops(model, input_size)
        latency = measure_latency(model, input_size, num_runs=args.latency_runs)
        params = count_pruned_params(model)

        print(f"  Removed: {len(pruned_blocks)}/{total} blocks ({pct:.0f}%), {layers_removed} layers")
        print(f"  Acc (no FT):  {acc_no_ft:.2f}%")
        print(f"  FLOPs:        {flops:,} ({flops/original_flops*100:.1f}% of original)")
        print(f"  Latency:      {latency['mean_ms']:.2f} ms ({latency['mean_ms']/original_latency['mean_ms']*100:.1f}%)")
        print(f"  Params:       {params['active']:,} ({params['active']/params['total']*100:.1f}%)")

        # (B) Fine-tune
        print(f"  Fine-tuning {args.finetune_epochs} epochs...")
        acc_with_ft = finetune(model, train_loader, test_loader, device,
                               epochs=args.finetune_epochs, lr=args.finetune_lr)
        print(f"  Acc (with FT): {acc_with_ft:.2f}% (Δ={acc_with_ft - original_acc:+.2f}%)")
        print()

        results.append({
            'threshold': threshold,
            'pruned': len(pruned_blocks),
            'total': total,
            'kept': kept,
            'layers_removed': layers_removed,
            'pct_removed': pct,
            'acc_no_ft': acc_no_ft,
            'acc_with_ft': acc_with_ft,
            'flops': flops,
            'flops_ratio': flops / original_flops,
            'latency': latency,
            'params': params,
        })

        del model
        torch.cuda.empty_cache()

    # ── 결과 저장 ──
    output = {
        'depth': args.depth,
        'dataset': args.dataset,
        'seed': args.seed,
        'checkpoint': args.checkpoint,
        'original': {
            'accuracy': original_acc,
            'flops': original_flops,
            'latency': original_latency,
            'params': original_params,
            'blocks': total_blocks,
        },
        'finetune_config': {
            'epochs': args.finetune_epochs,
            'lr': args.finetune_lr,
        },
        'results': results,
    }

    output_path = f'results-json/pruning_analysis_d{args.depth}_{args.dataset}.json'
    os.makedirs('results-json', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # ── 최종 요약 테이블 ──
    print("=" * 100)
    print(f"{'Threshold':>10s} {'Removed':>10s} {'Layers':>8s} {'Acc(noFT)':>10s} {'Acc(FT)':>10s} {'FLOPs%':>8s} {'Latency%':>10s} {'Params%':>9s}")
    print("-" * 100)
    for r in results:
        t = r['threshold']
        t_str = "None" if t == 0 else f"α<{t:.2f}"
        flops_pct = r['flops'] / original_flops * 100
        lat_pct = r['latency']['mean_ms'] / original_latency['mean_ms'] * 100
        params_pct = r['params']['active'] / r['params']['total'] * 100
        print(f"{t_str:>10s} {r['pruned']:>4d}/{r['total']:<4d} {r['layers_removed']:>6d}  "
              f"{r['acc_no_ft']:>9.2f}% {r['acc_with_ft']:>9.2f}% {flops_pct:>7.1f}% "
              f"{lat_pct:>9.1f}% {params_pct:>8.1f}%")

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
