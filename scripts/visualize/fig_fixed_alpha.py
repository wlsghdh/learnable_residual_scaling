"""
fig_fixed_alpha.py - Fixed-α Ablation Study
=============================================
고정 α vs 학습 α (LRS-Low) vs Baseline 비교
Depth 152 / CIFAR-100 / 3 seeds (mean ± std)

Fixed α values: 0.1, 0.3, 0.5, 0.7
핵심 메시지: 적절한 초기값도 중요하지만,
            학습 가능한 α (LRS)가 깊이에 적응적으로 변함
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
DEPTH = 152
DATASET = 'cifar100'

# Fixed alpha models
FIXED_MODELS = [
    ('fixed_alpha_01', 0.1, '#f9ebea'),
    ('fixed_alpha_03', 0.3, '#f1948a'),
    ('fixed_alpha_05', 0.5, '#e74c3c'),
    ('fixed_alpha_07', 0.7, '#922b21'),
]

# Reference models
REF_MODELS = [
    ('baseline',   COLORS['baseline'],   'Baseline\n(He init)'),
    ('lrs_low',    COLORS['lrs_low'],    r'LRS-Low$^\dagger$' '\n' r'($\alpha$ learnable)'),
    ('lrs_ha_low', COLORS['lrs_ha_low'], r'LRS+HA-Low$^\dagger$' '\n' r'($\alpha$ learnable)'),
]


def load_accs(model_key, depth, dataset, seeds):
    accs = []
    json_key = {
        'lrs_ha_low': 'lrs_hybrida_low',
    }.get(model_key, model_key)
    rdir = os.path.join(os.path.dirname(__file__), '../../results-json')
    for seed in seeds:
        fname = f'{json_key}_depth{depth}_{dataset}_seed{seed}_result.json'
        path = os.path.join(rdir, fname)
        if not os.path.exists(path):
            fname2 = f'{json_key}_depth{depth}_{dataset}_result.json'
            path2 = os.path.join(rdir, fname2)
            if seed == 42 and os.path.exists(path2):
                path = path2
            else:
                continue
        with open(path) as f:
            data = json.load(f)
        accs.append(data['results']['best_acc'] * 100)
    return accs


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.2))
    fig.patch.set_facecolor('white')

    # ── (a) Bar chart: Fixed α vs Learnable ────────────────
    ax = axes[0]

    all_models = ([(mk, color, f'Fixed\n$\\alpha={av}$')
                   for mk, av, color in FIXED_MODELS] +
                  [(mk, color, lbl) for mk, color, lbl in REF_MODELS])

    means, stds, colors, labels = [], [], [], []
    any_data = False
    for mk, color, label in all_models:
        accs = load_accs(mk, DEPTH, DATASET, SEEDS)
        colors.append(color)
        labels.append(label)
        if accs:
            any_data = True
            means.append(np.mean(accs))
            stds.append(np.std(accs))
        else:
            print(f'  WARNING: no data for {mk}')
            means.append(None)
            stds.append(None)

    x = np.arange(len(all_models))
    width = 0.65

    if not any_data:
        ax.text(0.5, 0.5,
                '(a) Fixed-α Ablation\n(data pending — run batch4 first)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=10, color='#7f8c8d',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa',
                          edgecolor='#bdc3c7'))
        ax.axis('off')
    else:
        for i, (m, s, c, lbl) in enumerate(zip(means, stds, colors, labels)):
            if m is None:
                ax.bar(i, 0, width, color='#ecf0f1', edgecolor='#bdc3c7',
                       hatch='///', alpha=0.5, zorder=3)
                ax.text(i, 0.3, 'N/A', ha='center', va='bottom',
                        fontsize=8, color='#95a5a6')
            else:
                ax.bar(i, m, width, color=c, alpha=0.88,
                       edgecolor='white', linewidth=0.5, zorder=3)
                if s > 0:
                    ax.errorbar(i, m, yerr=s, fmt='none',
                                color='#2c3e50', capsize=4, capthick=1.2,
                                lw=1.3, zorder=5)
                ax.text(i, m + s + 0.05, f'{m:.2f}',
                        ha='center', va='bottom', fontsize=7.5,
                        color='#2c3e50')

        # 구분선 (fixed vs learnable)
        ax.axvline(x=len(FIXED_MODELS) - 0.5, color='#7f8c8d',
                   linestyle='--', lw=1.2, alpha=0.7)
        ax.text(len(FIXED_MODELS) / 2 - 0.5, ax.get_ylim()[0] + 0.1,
                'Fixed α', ha='center', fontsize=8.5, color='#7f8c8d')
        ax.text(len(FIXED_MODELS) + len(REF_MODELS) / 2 - 0.5,
                ax.get_ylim()[0] + 0.1,
                'Learnable α', ha='center', fontsize=8.5, color='#2c3e50')

        valid = [m for m in means if m is not None]
        if valid:
            ax.set_ylim(min(valid) - 0.8,
                        max(m + (s or 0) for m, s in zip(means, stds)
                            if m is not None) + 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha='right')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('(a) Fixed vs Learnable α\n(CIFAR-100, ResNet-152)',
                     fontweight='bold', fontsize=10)
        ax.grid(axis='y', alpha=0.35, linestyle='--')
        ax.set_axisbelow(True)

    # ── (b) Line plot: Fixed α accuracy vs α value ──────────
    ax2 = axes[1]
    fixed_alphas = [0.1, 0.3, 0.5, 0.7]
    fixed_means = [means[i] if means[i] is not None else np.nan
                   for i in range(len(FIXED_MODELS))]
    fixed_stds = [stds[i] if stds[i] is not None else 0.0
                  for i in range(len(FIXED_MODELS))]

    # Learnable α reference lines
    lrs_accs = {
        'lrs_low':    load_accs('lrs_low', DEPTH, DATASET, SEEDS),
        'lrs_ha_low': load_accs('lrs_ha_low', DEPTH, DATASET, SEEDS),
    }
    lrs_means = {k: np.mean(v) if v else None for k, v in lrs_accs.items()}

    if any(not np.isnan(m) for m in fixed_means):
        ax2.errorbar(fixed_alphas, fixed_means, yerr=fixed_stds,
                     color='#e74c3c', marker='o', linewidth=1.8,
                     markersize=7, capsize=4, capthick=1.2, lw=1.5,
                     label=r'Fixed $\alpha$ (constant)', zorder=4)
        ax2.fill_between(fixed_alphas,
                          [m - s for m, s in zip(fixed_means, fixed_stds)],
                          [m + s for m, s in zip(fixed_means, fixed_stds)],
                          alpha=0.15, color='#e74c3c')

    # Learnable LRS-Low reference
    for mk, color, label in [('lrs_low', COLORS['lrs_low'], 'LRS-Low (learnable)'),
                               ('lrs_ha_low', COLORS['lrs_ha_low'], 'LRS+HA-Low (learnable)')]:
        mean = lrs_means.get(mk)
        if mean is not None:
            ax2.axhline(y=mean, color=color, linestyle='--',
                        linewidth=1.5, label=label, zorder=3)
            ax2.text(0.72, mean + 0.02, f'{mean:.2f}%',
                     color=color, fontsize=8, ha='right')

    ax2.set_xlabel(r'Fixed $\alpha$ value')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('(b) Accuracy vs. Fixed α value\n(CIFAR-100, ResNet-152)',
                  fontweight='bold', fontsize=10)
    ax2.set_xticks(fixed_alphas)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(alpha=0.35, linestyle='--')

    if not any(not np.isnan(m) for m in fixed_means):
        ax2.text(0.5, 0.5, '(data pending)', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=10, color='#7f8c8d',
                 bbox=dict(boxstyle='round', facecolor='#f8f9fa',
                           edgecolor='#bdc3c7'))

    fig.text(0.99, 0.01, r'$\dagger$ Proposed', ha='right',
             fontsize=8, color='#7f8c8d')
    plt.tight_layout(pad=1.5)
    save_figure(fig, 'fig_fixed_alpha')
    plt.close()
    print('Fig fixed_alpha done.')


if __name__ == '__main__':
    main()
