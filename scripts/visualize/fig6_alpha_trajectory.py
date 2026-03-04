"""
fig6_alpha_trajectory.py - Alpha Trajectory During Training
============================================================
Epoch에 따른 mean α 변화 (LRS_Low, depth 50/101/152/200)
좌: CIFAR-10, 우: CIFAR-100
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
DEPTH_STYLES = {
    50:  {'ls': '-',  'lw': 1.8},
    101: {'ls': '--', 'lw': 1.8},
    152: {'ls': '-.',  'lw': 1.8},
    200: {'ls': ':',  'lw': 2.0},
}
DEPTH_COLORS = {
    50:  '#e74c3c',
    101: '#e67e22',
    152: '#3498db',
    200: '#8e44ad',
}


def load_alpha_trajectory(model_key, depth, dataset):
    """epoch별 alpha_stats (mean, std) 로드"""
    path = get_result_path(model_key, depth, dataset)
    if not os.path.exists(path):
        print(f'  WARNING: not found: {path}')
        return None, None
    with open(path) as f:
        data = json.load(f)
    alpha_stats = data['results']['history'].get('alpha_stats')
    if not alpha_stats:
        return None, None
    epochs = [s['epoch'] for s in alpha_stats]
    means = [s['mean'] for s in alpha_stats]
    return epochs, means


def plot_trajectory(ax, model_key, dataset, title):
    has_data = False
    for depth in DEPTHS:
        epochs, means = load_alpha_trajectory(model_key, depth, dataset)
        if epochs is None:
            print(f'  WARNING: no alpha trajectory for {model_key} depth{depth} {dataset}')
            continue
        has_data = True
        cfg = DEPTH_STYLES[depth]
        ax.plot(epochs, means,
                color=DEPTH_COLORS[depth],
                linestyle=cfg['ls'], linewidth=cfg['lw'],
                label=f'ResNet-{depth}',
                zorder=4, alpha=0.9)

    if not has_data:
        ax.text(0.5, 0.5, 'No alpha trajectory data available',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='#7f8c8d',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#bdc3c7'))
        ax.set_title(title, fontsize=10, fontweight='bold')
        return

    # α=0.5 기준선
    ax.axhline(y=0.5, color='#95a5a6', linestyle=':', linewidth=1.0, zorder=1)

    # Sigmoid(-2) 초기값
    init_alpha = 1 / (1 + np.exp(2))
    ax.axhline(y=init_alpha, color='#bdc3c7', linestyle='-.', linewidth=0.9,
               zorder=1, alpha=0.7, label=f'Init $\\alpha_0$')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean $\\alpha$')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.35, linestyle='--')
    ax.set_ylim(bottom=0)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    fig.patch.set_facecolor('white')

    plot_trajectory(axes[0], 'lrs_low', 'cifar10',
                    '(a) LRS-Low: CIFAR-10')
    plot_trajectory(axes[1], 'lrs_low', 'cifar100',
                    '(b) LRS-Low: CIFAR-100')

    plt.tight_layout(pad=1.5, w_pad=2.0)
    save_figure(fig, 'fig6_alpha_trajectory')
    plt.close()
    print('Fig 6 done.')


if __name__ == '__main__':
    main()
