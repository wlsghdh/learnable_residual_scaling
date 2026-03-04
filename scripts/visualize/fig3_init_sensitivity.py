"""
fig3_init_sensitivity.py - Initialization Sensitivity
======================================================
CIFAR-100에서 깊이별 LRS Low/Mid/High + Baseline 비교
LRS High 200층의 catastrophic failure 강조
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, os.path.dirname(__file__))
from style import set_paper_style, save_figure, get_result_path, COLORS

set_paper_style()

DEPTHS = [50, 101, 152, 200]
DATASET = 'cifar100'

MODELS = {
    'baseline':  {'color': COLORS['baseline'],  'label': 'Baseline',              'ls': '--', 'lw': 1.8, 'marker': 'o'},
    'lrs_low':   {'color': COLORS['lrs_low'],   'label': 'LRS-Low ($\\alpha_0\\approx 0.12$)',  'ls': '-',  'lw': 1.8, 'marker': 's'},
    'lrs_mid':   {'color': COLORS['lrs_mid'],   'label': 'LRS-Mid ($\\alpha_0=0.5$)',           'ls': '-',  'lw': 1.8, 'marker': '^'},
    'lrs_high':  {'color': COLORS['lrs_high'],  'label': 'LRS-High ($\\alpha_0\\approx 0.88$)', 'ls': '-',  'lw': 1.8, 'marker': 'v'},
}


def load_accuracy(model_key, depth, dataset):
    path = get_result_path(model_key, depth, dataset)
    if not os.path.exists(path):
        print(f'  WARNING: not found: {path}')
        return None
    with open(path) as f:
        data = json.load(f)
    return data['results']['best_acc'] * 100


def main():
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.patch.set_facecolor('white')

    accs = {}
    for mk in MODELS:
        accs[mk] = [load_accuracy(mk, d, DATASET) for d in DEPTHS]

    # Baseline 점선
    base_vals = accs['baseline']
    ax.plot(DEPTHS, base_vals,
            color=COLORS['baseline'], linestyle='--',
            linewidth=1.8, marker='o', markersize=6,
            label='Baseline', zorder=3)

    # LRS variants
    for mk in ['lrs_low', 'lrs_mid', 'lrs_high']:
        cfg = MODELS[mk]
        vals = accs[mk]
        # 정상 구간 (값이 50% 이상인 경우만 연결)
        valid_depths = [d for d, v in zip(DEPTHS, vals) if v is not None and v >= 50]
        valid_vals = [v for v in vals if v is not None and v >= 50]

        ax.plot(valid_depths, valid_vals,
                color=cfg['color'], linestyle=cfg['ls'],
                linewidth=cfg['lw'], marker=cfg['marker'],
                markersize=6, label=cfg['label'], zorder=4)

    # LRS High 200층 실패 강조
    fail_val_high = load_accuracy('lrs_high', 200, DATASET)
    if fail_val_high is not None and fail_val_high < 50:
        # 실패 포인트 빨간 X
        ax.scatter([200], [fail_val_high], marker='X', s=150,
                   color='#e74c3c', zorder=10, clip_on=False)
        ax.annotate(
            f'LRS-High\nDepth-200\n{fail_val_high:.1f}% (Collapse!)',
            xy=(200, fail_val_high),
            xytext=(175, 72),
            fontsize=8, color='#e74c3c',
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdf2f2',
                      edgecolor='#e74c3c', lw=0.8),
            zorder=11
        )
        # 점선으로 잇기
        last_high_val = load_accuracy('lrs_high', 152, DATASET)
        if last_high_val is not None:
            ax.plot([152, 200], [last_high_val, fail_val_high],
                    color=COLORS['lrs_high'], linestyle=':', lw=1.2,
                    marker='v', markersize=6, zorder=3)

    # LRS High 152층도 표시
    val_152 = load_accuracy('lrs_high', 152, DATASET)
    if val_152 is not None:
        pass  # 이미 위에서 처리됨

    ax.set_xlabel('ResNet Depth')
    ax.set_ylabel('Best Test Accuracy (%)')
    ax.set_title('Initialization Sensitivity (CIFAR-100)', fontweight='bold')
    ax.set_xticks(DEPTHS)
    ax.set_xticklabels([f'ResNet-{d}' for d in DEPTHS])

    # Y축 범위 - 정상 결과 근처만
    all_valid = [v for vals in accs.values() for v in vals if v is not None and v >= 50]
    if all_valid:
        ax.set_ylim(min(all_valid) - 1.5, max(all_valid) + 0.8)

    ax.legend(loc='upper left', fontsize=8.5, framealpha=0.9)
    ax.grid(alpha=0.35, linestyle='--')

    plt.tight_layout()
    save_figure(fig, 'fig3_init_sensitivity')
    plt.close()
    print('Fig 3 done.')


if __name__ == '__main__':
    main()
