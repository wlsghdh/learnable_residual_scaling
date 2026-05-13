"""
Re-measure FLOPs and Latency for pruning analysis.
Updates existing pruning_analysis_d200_cifar100.json with corrected values.

Requires:
- models.py with LRSBottleneck short-circuit (getattr(self, '_pruned', False))
- run_pruning_analysis.py functions for prune_and_freeze
"""

import torch
import torch.nn as nn
import json
import os
import time
import numpy as np

from models import create_model, LRSBottleneck
from config import set_seed


def count_flops(model, input_size=(1, 3, 32, 32)):
    """모델의 FLOPs 계산. Short-circuit된 블록은 conv hook이 실행되지 않으므로 자동으로 제외됨."""
    flops = 0
    hooks = []

    def conv_hook(module, input, output):
        nonlocal flops
        out_channels = output.size(1)
        out_h, out_w = output.size(2), output.size(3)
        in_channels = module.in_channels
        kh, kw = module.kernel_size
        groups = module.groups
        flops += 2 * out_channels * out_h * out_w * (in_channels // groups) * kh * kw

    def bn_hook(module, input, output):
        nonlocal flops
        flops += 4 * input[0].numel()

    def linear_hook(module, input, output):
        nonlocal flops
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


def measure_latency(model, input_size=(1, 3, 32, 32), num_warmup=300, num_runs=300):
    """GPU inference latency 측정 (ms)
    Note: num_warmup이 충분히 커야 GPU compilation/caching이 안정화됨.
    """
    device = next(model.parameters()).device
    model.eval()

    x = torch.randn(*input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x)

    if device.type == 'cuda':
        torch.cuda.synchronize()

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
            times.append((t1 - t0) * 1000)

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
    }


def global_gpu_warmup(model, input_size=(1, 3, 32, 32), num_runs=500):
    """GPU 전역 warmup — 측정 시작 전 호출"""
    device = next(model.parameters()).device
    model.eval()
    x = torch.randn(*input_size).to(device)
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()


def prune_blocks_with_shortcut(model, threshold):
    """alpha < threshold인 블록에 _pruned=True 설정 (forward에서 short-circuit)"""
    pruned_count = 0
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, LRSBottleneck):
            total += 1
            alpha = torch.sigmoid(module.residual_scale).item()
            if alpha < threshold:
                module._pruned = True
                pruned_count += 1
    return pruned_count, total


def reset_pruning(model):
    """모든 블록의 _pruned 속성 제거"""
    for module in model.modules():
        if isinstance(module, LRSBottleneck):
            if hasattr(module, '_pruned'):
                delattr(module, '_pruned')


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = 'checkpoints/lrs_low_depth200_cifar100_seed42_best.pth'
    json_path = 'results-json/pruning_analysis_d200_cifar100.json'
    input_size = (1, 3, 32, 32)
    num_runs = 200

    # 기존 결과 로드
    with open(json_path) as f:
        results = json.load(f)

    print(f"=== Re-measuring FLOPs / Latency with short-circuit ===")
    print(f"Checkpoint: {checkpoint_path}")
    print()

    # 원본 모델 측정 (no pruning)
    model = create_model('lrs_low', depth=200, num_classes=100)
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 전역 GPU warmup (compilation/caching 안정화)
    print("Performing global GPU warmup (500 runs)...")
    global_gpu_warmup(model, input_size, num_runs=500)

    original_flops = count_flops(model, input_size)
    original_latency = measure_latency(model, input_size, num_runs=num_runs)

    print(f"Original:")
    print(f"  FLOPs:   {original_flops/1e9:.3f} GFLOPs")
    print(f"  Latency: {original_latency['mean_ms']:.2f} ± {original_latency['std_ms']:.2f} ms")
    print()

    # 원본 업데이트
    results['original']['flops'] = original_flops
    results['original']['latency'] = original_latency

    # 첫 번째 결과 (threshold=0, 원본)도 업데이트
    results['results'][0]['flops'] = original_flops
    results['results'][0]['flops_ratio'] = 1.0
    results['results'][0]['latency'] = original_latency

    # 각 threshold마다 측정
    thresholds = [0.03, 0.05, 0.08, 0.10, 0.15]

    for i, threshold in enumerate(thresholds):
        reset_pruning(model)
        pruned, total = prune_blocks_with_shortcut(model, threshold)

        flops = count_flops(model, input_size)
        latency = measure_latency(model, input_size, num_runs=num_runs)

        flops_pct = flops / original_flops * 100
        lat_pct = latency['mean_ms'] / original_latency['mean_ms'] * 100

        print(f"α < {threshold:.2f}: pruned {pruned}/{total} blocks")
        print(f"  FLOPs:   {flops/1e9:.3f} GFLOPs ({flops_pct:.1f}% of original)")
        print(f"  Latency: {latency['mean_ms']:.2f} ms ({lat_pct:.1f}% of original)")
        print()

        # 기존 결과 (results[i+1]) 업데이트 — accuracy는 그대로 유지
        results['results'][i+1]['flops'] = flops
        results['results'][i+1]['flops_ratio'] = flops / original_flops
        results['results'][i+1]['latency'] = latency

    # 파일 저장
    backup_path = json_path.replace('.json', '_backup_before_fix.json')
    os.rename(json_path, backup_path)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Updated: {json_path}")
    print(f"Backup:  {backup_path}")

    # 최종 요약 테이블
    print()
    print("=" * 80)
    print(f"{'Threshold':>10s} {'Pruned':>10s} {'AccNoFT':>10s} {'AccFT':>10s} {'FLOPs%':>10s} {'Latency%':>10s}")
    print("-" * 80)
    for r in results['results']:
        t = r['threshold']
        t_str = 'None' if t == 0 else f'α<{t:.2f}'
        flops_pct = r['flops'] / original_flops * 100
        lat_pct = r['latency']['mean_ms'] / original_latency['mean_ms'] * 100
        print(f"{t_str:>10s} {r['pruned']:>3d}/{r['total']:<5d} "
              f"{r['acc_no_ft']:>9.2f}% {r['acc_with_ft']:>9.2f}% "
              f"{flops_pct:>9.1f}% {lat_pct:>9.1f}%")


if __name__ == '__main__':
    main()
