"""
fig2_depth_accuracy.py - Depth vs Accuracy Grouped Bar Chart
=============================================================
CIFAR-10 / CIFAR-100에서 Baseline, LRS_Low, LRS+HA_Low 비교
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from style import set_paper_style, save_figure, get_result_path, COLORS

set_paper_style()

DEPTHS = [50, 101, 152, 200]
VARIANTS = ['baseline', 'lrs_low', 'lrs_ha_low']
VARIANT_LABELS = {
    'baseline':   'Baseline',
    'lrs_low':    'LRS-Low',
    'lrs_ha_low': 'LRS+HA-Low',
}
VARIANT_COLORS = {
    'baseline':   COLORS['baseline'],
    'lrs_low':    COLORS['lrs_low'],
    'lrs_ha_low': COLORS['lrs_ha_low'],
}


def load_accuracy(model_key, depth, dataset):
    path = get_result_path(model_key, depth, dataset)
    if not os.path.exists(path):
        print(f'  WARNING: not found: {path}')
        return None
    with open(path) as f:
        data = json.load(f)
    return data['results']['best_acc'] * 100


def plot_dataset(ax, dataset, title):
    n_depths = len(DEPTHS)
    n_variants = len(VARIANTS)
    width = 0.22
    x = np.arange(n_depths)

    # 데이터 로드
    accs = {}
    for v in VARIANTS:
        accs[v] = [load_accuracy(v, d, dataset) for d in DEPTHS]

    # Y 범위 계산 (결과 근처만)
    all_vals = [v for vals in accs.values() for v in vals if v is not None]
    failed = [v for v in all_vals if v < 50]
    valid = [v for v in all_vals if v >= 50]
    y_min = min(valid) - 0.5 if valid else 0
    y_max = max(valid) + 0.3 if valid else 100

    for i, v in enumerate(VARIANTS):
        offsets = (i - (n_variants - 1) / 2) * width
        vals = accs[v]
        bars = ax.bar(x + offsets, vals, width,
                      label=VARIANT_LABELS[v],
                      color=VARIANT_COLORS[v],
                      edgecolor='white', linewidth=0.5,
                      alpha=0.9, zorder=3)

        # 수치 표시
        for bar, val in zip(bars, vals):
            if val is None:
                continue
            if val < 50:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        y_min + 0.1, f'{val:.1f}%',
                        ha='center', va='bottom', fontsize=6,
                        color='#e74c3c', rotation=90)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.05,
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=6.5,
                        color='#2c3e50', rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f'ResNet-{d}' for d in DEPTHS])
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.4, linestyle='--')
    ax.set_axisbelow(True)

    # LRS_High 붕괴 주석 (해당 depth에만)
    for i, (d, v) in enumerate(zip(DEPTHS, accs['lrs_ha_low'])):
        pass  # 이 figure는 3개 variant만 표시


def main():
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.0))
    fig.patch.set_facecolor('white')

    plot_dataset(axes[0], 'cifar10', '(a) CIFAR-10')
    plot_dataset(axes[1], 'cifar100', '(b) CIFAR-100')

    plt.tight_layout(pad=1.5, h_pad=2.0)
    save_figure(fig, 'fig2_depth_accuracy')
    plt.close()
    print('Fig 2 done.')


if __name__ == '__main__':
    main()
