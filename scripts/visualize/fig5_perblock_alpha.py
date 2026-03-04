"""
fig5_perblock_alpha.py - Per-block α Distribution (Depth 152)
==============================================================
152층 (50 blocks)의 block별 학습된 α 값
CIFAR-10 실선, CIFAR-100 점선, Layer 구간 배경색, Downsample block 마커
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

# Depth 152: [3, 8, 36, 3] = 50 blocks
# Layer 경계 (누적)
DEPTH = 152
LAYERS_152 = [3, 8, 36, 3]
LAYER_BOUNDARIES = []
total = 0
for n in LAYERS_152:
    LAYER_BOUNDARIES.append((total, total + n - 1))
    total += n
# [(0,2), (3,10), (11,46), (47,49)]

# Downsample block: 각 layer의 첫 번째 block
DOWNSAMPLE_BLOCKS = [b[0] for b in LAYER_BOUNDARIES]  # [0, 3, 11, 47]

LAYER_COLORS = ['#eaf4fb', '#eafaf1', '#fdf2f8', '#fff3e0']
LAYER_NAMES = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']


def load_perblock_alpha(model_key, depth, dataset):
    path = get_result_path(model_key, depth, dataset)
    if not os.path.exists(path):
        print(f'  WARNING: not found: {path}')
        return None
    with open(path) as f:
        data = json.load(f)
    alpha = data['results'].get('alpha_final')
    if alpha is None:
        return None
    return alpha.get('all')


def main():
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.0))
    fig.patch.set_facecolor('white')

    panel_info = [
        ('lrs_low',    '(a) LRS-Low, Depth-152', COLORS['lrs_low']),
        ('lrs_ha_low', '(b) LRS+HA-Low, Depth-152 (Proposed)', COLORS['lrs_ha_low']),
    ]

    for ax, (model_key, title, color) in zip(axes, panel_info):
        n_blocks = sum(LAYERS_152)  # 50 blocks
        x = np.arange(1, n_blocks + 1)

        # Layer 배경색
        for i, (start, end) in enumerate(LAYER_BOUNDARIES):
            ax.axvspan(start + 0.5, end + 1.5,
                       alpha=0.35, color=LAYER_COLORS[i],
                       label=LAYER_NAMES[i], zorder=0)
            # Layer 이름 텍스트
            mid = (start + end) / 2 + 1
            ax.text(mid, 0.97, LAYER_NAMES[i],
                    ha='center', va='top', fontsize=8,
                    color='#7f8c8d', transform=ax.get_xaxis_transform())

        # α=0.5 기준선
        ax.axhline(y=0.5, color='#95a5a6', linestyle=':', linewidth=1.2, zorder=1)
        ax.text(n_blocks + 0.5, 0.5, '$\\alpha=0.5$',
                ha='left', va='center', fontsize=8, color='#95a5a6')

        # CIFAR-10 / CIFAR-100 데이터
        for ds, ds_label, ls, marker_char in [
            ('cifar10',  'CIFAR-10',  '-',  'o'),
            ('cifar100', 'CIFAR-100', '--', '^'),
        ]:
            alphas = load_perblock_alpha(model_key, DEPTH, ds)
            if alphas is None:
                print(f'  WARNING: no alpha data for {model_key} depth{DEPTH} {ds}')
                continue

            # 데이터 길이 확인
            if len(alphas) != n_blocks:
                print(f'  WARNING: expected {n_blocks} blocks, got {len(alphas)}')
                x_plot = np.arange(1, len(alphas) + 1)
            else:
                x_plot = x

            ax.plot(x_plot, alphas, color=color, linestyle=ls,
                    linewidth=1.3, marker=marker_char,
                    markersize=3, label=ds_label, zorder=4, alpha=0.85)

        # Downsample blocks 마커 (▼)
        for block_idx in DOWNSAMPLE_BLOCKS:
            # block_idx는 0-based, x는 1-based
            ax.annotate('▼', xy=(block_idx + 1, -0.025),
                        xycoords=('data', 'axes fraction'),
                        ha='center', va='top', fontsize=7,
                        color='#7f8c8d')

        ax.set_xlabel('Block Index')
        ax.set_ylabel('Learned $\\alpha$ value')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(0.5, n_blocks + 1.5)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--', zorder=0)

        # Downsample legend
        ax.text(0.02, -0.08, '▼ = downsample block',
                transform=ax.transAxes, fontsize=7.5, color='#7f8c8d')

    plt.tight_layout(pad=1.5, h_pad=2.5)
    save_figure(fig, 'fig5_perblock_alpha')
    plt.close()
    print('Fig 5 done.')


if __name__ == '__main__':
    main()
