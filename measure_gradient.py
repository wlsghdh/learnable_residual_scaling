"""
Gradient L2 Norm 블록별 측정 스크립트
- Plain network (skip connection 없음)
- Baseline ResNet (He init)
- LRS Low ResNet
학습 초반 1 epoch의 gradient 크기를 블록별로 기록
"""

import torch
import torch.nn as nn
import json
import argparse
import os
import math
from collections import defaultdict

from models import create_model
from data import get_data_loaders
from config import get_training_config, set_seed, CONFIG


def measure_gradient_norms(model, train_loader, device, num_batches=50):
    """각 블록의 gradient L2 norm을 측정"""
    model.train()
    criterion = nn.CrossEntropyLoss()

    # 블록별 gradient norm 저장
    block_grad_norms = defaultdict(list)

    # 블록 이름 매핑
    block_names = []
    block_params = {}

    for name, module in model.named_modules():
        # Bottleneck 블록 찾기
        if hasattr(module, 'conv1') and hasattr(module, 'conv2') and hasattr(module, 'conv3'):
            block_names.append(name)
            block_params[name] = list(module.parameters())
        # BasicBlock (plain network)
        elif hasattr(module, 'conv1') and hasattr(module, 'conv2') and not hasattr(module, 'conv3'):
            if 'layer' in name or 'blocks' in name:
                block_names.append(name)
                block_params[name] = list(module.parameters())

    if not block_names:
        # fallback: layer별로 찾기
        for name, param in model.named_parameters():
            if 'weight' in name and ('conv' in name or 'fc' in name):
                layer_name = name.rsplit('.', 1)[0]
                if layer_name not in block_params:
                    block_names.append(layer_name)
                    block_params[layer_name] = []
                block_params[layer_name].append(param)

    print(f"  Found {len(block_names)} blocks")

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= num_batches:
            break

        inputs, targets = inputs.to(device), targets.to(device)

        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # 각 블록의 gradient L2 norm 계산
        for name in block_names:
            total_norm = 0.0
            for param in block_params[name]:
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = math.sqrt(total_norm)
            block_grad_norms[name].append(total_norm)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx + 1}/{num_batches}")

    # 평균 gradient norm
    avg_norms = {}
    for name in block_names:
        if block_grad_norms[name]:
            avg_norms[name] = sum(block_grad_norms[name]) / len(block_grad_norms[name])

    return block_names, avg_norms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, default=200)
    parser.add_argument('--dataset', type=str, default='cifar100')
    parser.add_argument('--num-batches', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=str, default='results-json/gradient_norms.json')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    num_classes = 100 if args.dataset == 'cifar100' else 10
    train_config = get_training_config(args.depth)

    # 데이터 로드
    (train_loader, _), num_classes_from_data = get_data_loaders(
        dataset=args.dataset,
        batch_size=train_config['batch_size'],
    )

    # 측정할 모델들
    model_types = ['baseline', 'lrs_low', 'plain_he']
    model_labels = {
        'baseline': 'ResNet (skip connection)',
        'lrs_low': 'LRS Low (α≈0.12)',
        'plain_he': 'Plain Network (no skip)',
    }

    results = {}

    for model_type in model_types:
        print(f"\n=== {model_labels[model_type]} ===")
        try:
            model = create_model(model_type, depth=args.depth, num_classes=num_classes)
            model = model.to(device)

            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Parameters: {param_count:,}")

            block_names, avg_norms = measure_gradient_norms(
                model, train_loader, device, num_batches=args.num_batches
            )

            # 블록 인덱스로 정리
            norms_list = []
            for i, name in enumerate(block_names):
                norm = avg_norms.get(name, 0.0)
                norms_list.append(norm)
                if i < 5 or i >= len(block_names) - 3:
                    print(f"  Block {i:3d} ({name:40s}): grad_norm = {norm:.6f}")
                elif i == 5:
                    print(f"  ...")

            results[model_type] = {
                'label': model_labels[model_type],
                'block_names': block_names,
                'gradient_norms': norms_list,
                'num_blocks': len(block_names),
            }

        except Exception as e:
            print(f"  Error: {e}")
            results[model_type] = {'error': str(e)}

    # 저장
    output = {
        'depth': args.depth,
        'dataset': args.dataset,
        'seed': args.seed,
        'num_batches': args.num_batches,
        'models': results,
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
