"""
fig7_high_collapse.py - LRS-High Catastrophic Failure
=======================================================
ResNet-152/200에서 LRS_High의 학습 붕괴 vs Baseline, LRS_Low
Training accuracy curve 비교
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style import set_paper_style, save_figure, get_result_path, COLORS

set_paper_style()


def load_history(model_key, depth, dataset, key='test_acc'):
    path = get_result_path(model_key, depth, dataset)
    if not os.path.exists(path):
        print(f'  WARNING: not found: {path}')
        return None
    with open(path) as f:
        data = json.load(f)
    hist = data['results']['history'].get(key)
    if hist is None:
        return None
    return [v * 100 for v in hist]  # 0~1 → %


def plot_collapse(ax, dataset, title):
    """단일 dataset에서 depth 152, 200의 비교"""

    configs = [
        # (model_key, depth, label, color, ls, lw)
        ('baseline',  152, 'Baseline ResNet-152',   COLORS['baseline'],  '-',  1.6),
        ('lrs_low',   152, 'LRS-Low ResNet-152',    COLORS['lrs_low'],   '-',  1.6),
        ('lrs_high',  152, 'LRS-High ResNet-152',   COLORS['lrs_high'],  '-',  1.6),
        ('baseline',  200, 'Baseline ResNet-200',   COLORS['baseline'],  '--', 1.6),
        ('lrs_low',   200, 'LRS-Low ResNet-200',    COLORS['lrs_low'],   '--', 1.6),
        ('lrs_high',  200, 'LRS-High ResNet-200',   COLORS['lrs_high'],  '--', 2.0),
    ]

    any_data = False
    for model_key, depth, label, color, ls, lw in configs:
        vals = load_history(model_key, depth, dataset, 'test_acc')
        if vals is None:
            continue
        any_data = True
        epochs = list(range(1, len(vals) + 1))
        ax.plot(epochs, vals,
                color=color, linestyle=ls, linewidth=lw,
                label=label, zorder=4, alpha=0.9)

    if not any_data:
        ax.text(0.5, 0.5, 'No data available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='#7f8c8d')

    # LRS-High 200 final value 주석
    fail_val = load_history('lrs_high', 200, dataset, 'test_acc')
    if fail_val is not None and max(fail_val) < 50:
        # 실패한 경우
        best = max(fail_val)
        best_ep = fail_val.index(best) + 1
        ax.annotate(
            f'LRS-High-200\nMax {best:.1f}%',
            xy=(best_ep, best),
            xytext=(best_ep + 5, best + 5),
            fontsize=8, color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.0),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#fdf2f2',
                      edgecolor='#e74c3c', lw=0.8),
            zorder=10
        )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(title, fontsize=10, fontweight='bold')

    # 정상 범위만 표시 (깊이 별 표시)
    ax.legend(loc='lower right', fontsize=7.5, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.35, linestyle='--')


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    fig.patch.set_facecolor('white')

    plot_collapse(axes[0], 'cifar10',  '(a) CIFAR-10: LRS-High Collapse')
    plot_collapse(axes[1], 'cifar100', '(b) CIFAR-100: LRS-High Collapse')

    # Shared note
    fig.text(0.5, -0.02,
             'Solid: ResNet-152  |  Dashed: ResNet-200  |  '
             'LRS-High (orange) collapses at depth 200',
             ha='center', fontsize=8.5, color='#7f8c8d')

    plt.tight_layout(pad=1.5)
    save_figure(fig, 'fig7_high_collapse')
    plt.close()
    print('Fig 7 done.')


if __name__ == '__main__':
    main()
