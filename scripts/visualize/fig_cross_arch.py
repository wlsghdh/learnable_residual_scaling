"""
fig_cross_arch.py - Cross-Architecture Generalization
======================================================
WRN-28-10 & ResNeXt-50 에서 LRS 방법 적용 결과
6개 variant × 2 arch × 2 dataset / 3 seeds (mean ± std)

데이터 없으면 WARNING + placeholder.
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(__file__))
from style import set_paper_style, save_figure, COLORS

set_paper_style()

SEEDS = [42, 123, 456]
DATASETS = ['cifar10', 'cifar100']

# WRN uses depth=28, ResNeXt uses depth=50
ARCH_INFO = [
    ('wrn',     28,  'WRN-28-10',    ['wrn_baseline', 'wrn_lrs_low', 'wrn_lrs_hybrida_low',
                                       'wrn_rezero', 'wrn_skipinit', 'wrn_layerscale']),
    ('resnext', 50,  'ResNeXt-50',   ['resnext_baseline', 'resnext_lrs_low', 'resnext_lrs_hybrida_low',
                                       'resnext_rezero', 'resnext_skipinit', 'resnext_layerscale']),
]

VARIANT_COLORS = {
    'baseline':         COLORS['baseline'],
    'lrs_low':          COLORS['lrs_low'],
    'lrs_hybrida_low':  COLORS['lrs_ha_low'],
    'rezero':           COLORS['rezero'],
    'skipinit':         COLORS['skipinit'],
    'layerscale':       COLORS['layerscale'],
}

VARIANT_LABELS = {
    'baseline':         'Baseline',
    'lrs_low':          'LRS-Low',
    'lrs_hybrida_low':  r'LRS+HA-Low$^\dagger$',
    'rezero':           'ReZero',
    'skipinit':         'SkipInit',
    'layerscale':       'LayerScale',
}

RDIR = os.path.join(os.path.dirname(__file__), '../../results-json')


def get_variant_key(model_name):
    """wrn_lrs_low → lrs_low"""
    for prefix in ('wrn_', 'resnext_'):
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    return model_name


def load_accs(model_name, depth, dataset, seeds):
    accs = []
    for seed in seeds:
        fname = f'{model_name}_depth{depth}_{dataset}_seed{seed}_result.json'
        path = os.path.join(RDIR, fname)
        if not os.path.exists(path):
            fname2 = f'{model_name}_depth{depth}_{dataset}_result.json'
            path2 = os.path.join(RDIR, fname2)
            if seed == 42 and os.path.exists(path2):
                path = path2
            else:
                continue
        with open(path) as f:
            data = json.load(f)
        accs.append(data['results']['best_acc'] * 100)
    return accs


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
    fig.patch.set_facecolor('white')
    fig.suptitle('Cross-Architecture Generalization of LRS', fontsize=13,
                 fontweight='bold', y=1.01)

    for row, (arch_key, depth, arch_name, model_names) in enumerate(ARCH_INFO):
        for col, dataset in enumerate(DATASETS):
            ax = axes[row][col]
            title = f'({chr(97 + row*2 + col)}) {arch_name} — {dataset.upper()}'

            means, stds, colors, labels = [], [], [], []
            any_data = False

            for mn in model_names:
                vk = get_variant_key(mn)
                accs = load_accs(mn, depth, dataset, SEEDS)
                colors.append(VARIANT_COLORS.get(vk, '#95a5a6'))
                labels.append(VARIANT_LABELS.get(vk, vk))
                if accs:
                    any_data = True
                    means.append(np.mean(accs))
                    stds.append(np.std(accs))
                else:
                    print(f'  WARNING: no data for {mn} d{depth} {dataset}')
                    means.append(None)
                    stds.append(None)

            if not any_data:
                ax.text(0.5, 0.5,
                        f'{title}\n(data pending — run batch6 first)',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=9.5, color='#7f8c8d',
                        bbox=dict(boxstyle='round', facecolor='#f8f9fa',
                                  edgecolor='#bdc3c7'))
                ax.axis('off')
                continue

            x = np.arange(len(model_names))
            width = 0.60

            for i, (m, s, c, lbl) in enumerate(zip(means, stds, colors, labels)):
                if m is None:
                    ax.bar(i, 0, width, color='#ecf0f1', edgecolor='#bdc3c7',
                           hatch='///', alpha=0.5, zorder=3)
                    ax.text(i, 0.2, 'N/A', ha='center', va='bottom',
                            fontsize=7.5, color='#95a5a6')
                else:
                    ax.bar(i, m, width, color=c, alpha=0.88,
                           edgecolor='white', linewidth=0.5, zorder=3)
                    if s > 0:
                        ax.errorbar(i, m, yerr=s, fmt='none',
                                    color='#2c3e50', capsize=3.5, capthick=1.0,
                                    lw=1.2, zorder=5)
                    ax.text(i, m + (s or 0) + 0.05, f'{m:.2f}',
                            ha='center', va='bottom', fontsize=7,
                            color='#2c3e50')

            valid = [m for m in means if m is not None]
            valid_stds = [s for m, s in zip(means, stds) if m is not None]
            if valid:
                ax.set_ylim(min(valid) - 0.8,
                            max(m + (s or 0) for m, s in zip(valid, valid_stds)) + 0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha='right')
            ax.set_ylabel('Test Accuracy (%)')
            ax.set_title(title, fontweight='bold', fontsize=10)
            ax.grid(axis='y', alpha=0.35, linestyle='--')
            ax.set_axisbelow(True)

    fig.text(0.99, 0.01, r'$\dagger$ Proposed method',
             ha='right', fontsize=8, color='#7f8c8d')
    plt.tight_layout(pad=1.5)
    save_figure(fig, 'fig_cross_arch')
    plt.close()
    print('Fig cross_arch done.')


if __name__ == '__main__':
    main()
