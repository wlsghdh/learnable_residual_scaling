"""
fig_full_comparison.py - Full Method Comparison
================================================
Tier 1: Baseline, LRS-Low, LRS+HA-Low, ReZero, SkipInit, Fixup, LayerScale
Depth 152 / CIFAR-10 + CIFAR-100 / 3 seeds (mean ± std)

데이터 없으면 WARNING 출력 후 placeholder 생성.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style import set_paper_style, save_figure, get_result_path, COLORS, LABELS, MARKERS

set_paper_style()

# Tier 1 models in order
MODELS = [
    ('baseline',   COLORS['baseline'],  'Baseline'),
    ('lrs_low',    COLORS['lrs_low'],   r'LRS-Low'),
    ('lrs_ha_low', COLORS['lrs_ha_low'], r'LRS+HA-Low$^\dagger$'),
    ('rezero',     COLORS['rezero'],    'ReZero'),
    ('skipinit',   COLORS['skipinit'],  'SkipInit'),
    ('fixup',      COLORS['fixup'],     'Fixup'),
    ('layerscale', COLORS['layerscale'], 'LayerScale'),
]

SEEDS = [42, 123, 456]
DEPTH = 152
DATASETS = ['cifar10', 'cifar100']


def load_acc_seeds(model_key, depth, dataset, seeds):
    accs = []
    for seed in seeds:
        json_name = {
            'lrs_ha_low': 'lrs_hybrida_low',
        }.get(model_key, model_key)
        fname = f'{json_name}_depth{depth}_{dataset}_seed{seed}_result.json'
        path = os.path.join(os.path.dirname(__file__), '../../results-json', fname)
        if not os.path.exists(path):
            # seed=42 without seed suffix fallback
            fname2 = f'{json_name}_depth{depth}_{dataset}_result.json'
            path2 = os.path.join(os.path.dirname(__file__), '../../results-json', fname2)
            if seed == 42 and os.path.exists(path2):
                path = path2
            else:
                continue
        with open(path) as f:
            data = json.load(f)
        accs.append(data['results']['best_acc'] * 100)
    return accs


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5))
    fig.patch.set_facecolor('white')

    for ax, dataset, title in zip(axes, DATASETS,
                                   ['(a) CIFAR-10, ResNet-152',
                                    '(b) CIFAR-100, ResNet-152']):
        means, stds, colors, labels = [], [], [], []
        any_data = False

        for mk, color, label in MODELS:
            accs = load_acc_seeds(mk, DEPTH, dataset, SEEDS)
            if not accs:
                print(f'  WARNING: no data for {mk} depth{DEPTH} {dataset}')
                means.append(None)
                stds.append(None)
            else:
                any_data = True
                means.append(np.mean(accs))
                stds.append(np.std(accs))
            colors.append(color)
            labels.append(label)

        x = np.arange(len(MODELS))
        width = 0.60

        # Placeholder text if no data
        if not any_data:
            ax.text(0.5, 0.5,
                    f'{title}\n(data pending — run batch2/3 first)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='#7f8c8d',
                    bbox=dict(boxstyle='round', facecolor='#f8f9fa',
                              edgecolor='#bdc3c7'))
            ax.axis('off')
            continue

        # Bars
        for i, (m, s, c, lbl) in enumerate(zip(means, stds, colors, labels)):
            if m is None:
                ax.bar(i, 0, width, color='#ecf0f1', edgecolor='#bdc3c7',
                       linewidth=0.5, hatch='///', alpha=0.5, zorder=3)
                ax.text(i, 0.2, 'N/A', ha='center', va='bottom',
                        fontsize=8, color='#95a5a6')
            else:
                bar = ax.bar(i, m, width, color=c, alpha=0.88,
                             edgecolor='white', linewidth=0.5, zorder=3)
                if s is not None and s > 0:
                    ax.errorbar(i, m, yerr=s, fmt='none',
                                color='#2c3e50', capsize=5, capthick=1.5,
                                lw=1.5, zorder=5)
                # 수치 표시
                ax.text(i, m + (s if s else 0) + 0.05,
                        f'{m:.2f}', ha='center', va='bottom',
                        fontsize=7.5, color='#2c3e50', fontweight='bold')

        # Y축 범위
        valid_means = [m for m in means if m is not None]
        valid_stds = [s if s else 0 for m, s in zip(means, stds) if m is not None]
        if valid_means:
            ymin = min(valid_means) - 0.8
            ymax = max(m + s for m, s in zip(valid_means, valid_stds)) + 0.5
            ax.set_ylim(ymin, ymax)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.35, linestyle='--')
        ax.set_axisbelow(True)

    # Proposed 표시
    fig.text(0.99, 0.01, r'$\dagger$ Proposed method',
             ha='right', fontsize=8, color='#7f8c8d')

    plt.tight_layout(pad=1.5)
    save_figure(fig, 'fig_full_comparison')
    plt.close()
    print('Fig full_comparison done.')


if __name__ == '__main__':
    main()
