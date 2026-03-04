"""
fig4_depth_alpha.py - Depth vs Learned Alpha Relationship ⭐ 핵심 Figure
=========================================================================
LRS_Low와 LRS+HA_Low에서 depth 증가에 따른 mean α 변화
CIFAR-10과 CIFAR-100 결과가 겹치는 것이 핵심 포인트
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style import set_paper_style, save_figure, get_result_path, COLORS

set_paper_style()

DEPTHS = [50, 101, 152, 200]


def load_alpha(model_key, depth, dataset):
    """alpha_final 통계 로드"""
    path = get_result_path(model_key, depth, dataset)
    if not os.path.exists(path):
        print(f'  WARNING: not found: {path}')
        return None, None
    with open(path) as f:
        data = json.load(f)
    alpha = data['results'].get('alpha_final')
    if alpha is None:
        return None, None
    return alpha['mean'], alpha['std']


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    fig.patch.set_facecolor('white')

    datasets = [
        ('cifar10',  'CIFAR-10',  '●', '-'),
        ('cifar100', 'CIFAR-100', '▲', '--'),
    ]

    panel_info = [
        ('lrs_low',    '(a) LRS-Low ($\\alpha_0 \\approx 0.12$)',  COLORS['lrs_low']),
        ('lrs_ha_low', '(b) LRS+HA-Low (Proposed)',               COLORS['lrs_ha_low']),
    ]

    for ax, (model_key, panel_title, color) in zip(axes, panel_info):
        for ds, ds_label, marker_char, ls in datasets:
            means, stds = [], []
            for d in DEPTHS:
                m, s = load_alpha(model_key, d, ds)
                means.append(m)
                stds.append(s)

            valid_mask = [m is not None for m in means]
            vd = [d for d, v in zip(DEPTHS, valid_mask) if v]
            vm = [m for m, v in zip(means, valid_mask) if v]
            vs = [s for s, v in zip(stds, valid_mask) if v]

            marker = 'o' if ds == 'cifar10' else '^'
            ax.errorbar(vd, vm, yerr=vs,
                        color=color, linestyle=ls,
                        linewidth=1.8, marker=marker,
                        markersize=7, capsize=4, capthick=1.2,
                        label=ds_label, zorder=4, elinewidth=1)

        # α=0.5 수평 기준선
        ax.axhline(y=0.5, color='#95a5a6', linestyle=':', linewidth=1.2,
                   zorder=1, label='ResNet 1:1 default ($\\alpha=0.5$)')

        # α 초기값 표시
        init_alpha = 1 / (1 + np.exp(2))  # sigmoid(-2) ≈ 0.119
        ax.axhline(y=init_alpha, color='#bdc3c7', linestyle='-.', linewidth=1.0,
                   zorder=1, alpha=0.7, label=f'Init. $\\alpha_0 \\approx {init_alpha:.2f}$')

        ax.set_xlabel('ResNet Depth')
        ax.set_ylabel('Learned $\\alpha$ (mean ± std)')
        ax.set_title(panel_title, fontsize=10, fontweight='bold')
        ax.set_xticks(DEPTHS)
        ax.set_xticklabels([str(d) for d in DEPTHS])

        # Y축 범위
        all_means = [m for m in [load_alpha(model_key, d, ds)[0]
                                 for d in DEPTHS
                                 for ds in ['cifar10', 'cifar100']] if m is not None]
        if all_means:
            ylo = max(0, min(all_means) - 0.05)
            yhi = min(1.0, max(all_means) + 0.1)
            ax.set_ylim(ylo, max(yhi, 0.55))

        ax.legend(fontsize=8, framealpha=0.9, loc='upper right')
        ax.grid(alpha=0.35, linestyle='--')

    plt.tight_layout(pad=1.5, w_pad=2.0)
    save_figure(fig, 'fig4_depth_alpha')
    plt.close()
    print('Fig 4 done.')


if __name__ == '__main__':
    main()
