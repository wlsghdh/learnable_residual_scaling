"""
fig_pilot_study.py - Pilot Study: Skip Connection & Initialization
===================================================================
Motivation figure: Skip connection과 초기화 방식이 깊은 네트워크 학습에 미치는 영향
6개 모델 × 2 depth × 2 dataset

모델:
  plain_he          : Skip 없음, He init
  plain_identity    : Skip 없음, Identity init
  baseline          : Skip 있음, He init (표준 ResNet)
  resnet_id_all     : Skip 있음, Identity init (모든 레이어)
  hybrida           : Skip 있음, He+Identity 혼합
  lrs_ha_low        : Skip 있음, He+Identity+LRS (제안)
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from style import set_paper_style, save_figure, get_result_path, COLORS, LABELS, MARKERS

set_paper_style()

MODELS = [
    ('plain_he',       '#7f8c8d', 'x',  '--', 'Plain (He init)'),
    ('plain_identity', '#bdc3c7', '+',  '--', 'Plain (Identity init)'),
    ('resnet_id_all',  '#95a5a6', '*',  '-.',  'ResNet (Identity init)'),
    ('baseline',       COLORS['baseline'],  'o', '-', 'Baseline ResNet (He init)'),
    ('hybrida',        COLORS['hybrida'],   'h', '-', 'HybridA'),
    ('lrs_ha_low',     COLORS['lrs_ha_low'],'D', '-', 'LRS+HA-Low (Proposed)'),
]

DEPTHS = [50, 152]
DATASETS = ['cifar10', 'cifar100']


def load_acc(model_key, depth, dataset):
    path = get_result_path(model_key, depth, dataset)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    return data['results']['best_acc'] * 100


def main():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.5))
    fig.patch.set_facecolor('white')

    for ax, dataset, title in zip(axes,
                                   DATASETS,
                                   ['(a) CIFAR-10', '(b) CIFAR-100']):
        n_models = len(MODELS)
        n_depths = len(DEPTHS)
        width = 0.35
        x = np.arange(n_depths)

        for i, (mk, color, marker, ls, label) in enumerate(MODELS):
            vals = [load_acc(mk, d, dataset) for d in DEPTHS]
            offset = (i - (n_models - 1) / 2) * (width / n_models * 1.5)
            plot_vals = [v if v is not None else 0 for v in vals]
            bars = ax.bar(x + offset, plot_vals, width / n_models * 1.3,
                          label=label, color=color,
                          edgecolor='white', linewidth=0.5,
                          alpha=0.88, zorder=3)
            # 수치 표시 (정상 결과만)
            for bar, val in zip(bars, vals):
                if val is None:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            0.2, 'N/A', ha='center', va='bottom',
                            fontsize=6, color='#95a5a6', rotation=90)
                    continue
                if val < 20:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.1,
                            f'{val:.0f}%', ha='center', va='bottom',
                            fontsize=6, color='#e74c3c', rotation=90)
                else:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.1,
                            f'{val:.1f}', ha='center', va='bottom',
                            fontsize=6.5, color='#2c3e50')

        # Y축 범위 (정상 결과 근처)
        all_vals = [load_acc(mk, d, dataset)
                    for mk, *_ in MODELS for d in DEPTHS]
        valid = [v for v in all_vals if v is not None and v > 20]
        if valid:
            ax.set_ylim(min(valid) - 1.5, max(valid) + 1.0)

        ax.set_xticks(x)
        ax.set_xticklabels([f'ResNet-{d}' for d in DEPTHS])
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(title, fontweight='bold')
        ax.grid(axis='y', alpha=0.35, linestyle='--')
        ax.set_axisbelow(True)

    # Shared legend below
    handles = [mpatches.Patch(color=c, label=lbl, alpha=0.88)
               for _, c, _, _, lbl in MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=3,
               fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(pad=1.5, rect=[0, 0.10, 1, 1])
    save_figure(fig, 'fig_pilot_study')
    plt.close()
    print('Fig pilot_study done.')


if __name__ == '__main__':
    main()
