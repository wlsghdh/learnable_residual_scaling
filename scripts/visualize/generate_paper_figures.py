#!/usr/bin/env python3
"""
Generate ALL paper figures for the LRS (Learnable Residual Scaling) paper.
Produces publication-quality PDFs for a computer vision conference (BMVC/CVPR style).

Saves figures to both:
  - figures/
  - results-json/figures/paper/

Usage:
    python scripts/visualize/generate_paper_figures.py
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path

# ==============================================================================
# Global Configuration
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / 'results-json'
IMAGENET_DIR = RESULTS_DIR / 'imagenet'
FIG_DIR_1 = BASE_DIR / 'figures'
FIG_DIR_2 = RESULTS_DIR / 'figures' / 'paper'

# Create output directories
FIG_DIR_1.mkdir(parents=True, exist_ok=True)
FIG_DIR_2.mkdir(parents=True, exist_ok=True)

# Color palette
COLORS = {
    'baseline':   '#616161',
    'lrs_low':    '#1976D2',
    'lrs_mid':    '#388E3C',
    'lrs_high':   '#D32F2F',
    'rezero':     '#F57C00',
    'plain_he':   '#9C27B0',
    'fixup':      '#00796B',
    'skipinit':   '#5D4037',
    'layerscale': '#C2185B',
    'lrs_ha_low': '#7B1FA2',
}

# Display names for legend
DISPLAY_NAMES = {
    'baseline':   'Baseline (ResNet)',
    'lrs_low':    'LRS Low',
    'lrs_mid':    'LRS Mid',
    'lrs_high':   'LRS High',
    'rezero':     'ReZero',
    'plain_he':   'Plain Network',
    'fixup':      'Fixup',
    'skipinit':   'SkipInit',
    'layerscale': 'LayerScale',
    'lrs_ha_low': 'LRS-HA Low',
}

# Block counts per depth
BLOCK_COUNTS = {50: 16, 101: 33, 152: 50, 200: 66}

# Stage boundaries (block indices, 0-indexed)
STAGE_BOUNDARIES = {
    50:  {'stage1': (0, 2),   'stage2': (3, 6),   'stage3': (7, 12),  'stage4': (13, 15)},
    101: {'stage1': (0, 2),   'stage2': (3, 6),   'stage3': (7, 29),  'stage4': (30, 32)},
    152: {'stage1': (0, 2),   'stage2': (3, 10),  'stage3': (11, 46), 'stage4': (47, 49)},
    200: {'stage1': (0, 2),   'stage2': (3, 26),  'stage3': (27, 62), 'stage4': (63, 65)},
}

# Downsampling blocks (first block of stage 2, 3, 4)
DOWNSAMPLE_BLOCKS = {
    50:  [3, 7, 13],
    101: [3, 7, 30],
    152: [3, 11, 47],
    200: [3, 27, 63],
}

DEPTHS = [50, 101, 152, 200]
SEEDS = [42, 123, 456]

# ==============================================================================
# Matplotlib Style Setup
# ==============================================================================

def setup_style():
    """Set up matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 9,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8.5,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.8,
        'lines.markersize': 5,
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'mathtext.fontset': 'cm',
    })

# ==============================================================================
# Data Loading Helpers
# ==============================================================================

def load_result(model, depth, dataset, seed=42):
    """
    Load a single result JSON file.
    Handles dual naming convention:
      - seed42 without suffix: {model}_depth{N}_{dataset}_result.json
      - seed42 with suffix:    {model}_depth{N}_{dataset}_seed42_result.json
      - other seeds:           {model}_depth{N}_{dataset}_seed{S}_result.json
    Returns dict or None if file not found.
    """
    # Try seed-suffixed name first
    path_seed = RESULTS_DIR / f'{model}_depth{depth}_{dataset}_seed{seed}_result.json'
    if path_seed.exists():
        with open(path_seed) as f:
            return json.load(f)
    # For seed 42, also try without suffix
    if seed == 42:
        path_noseed = RESULTS_DIR / f'{model}_depth{depth}_{dataset}_result.json'
        if path_noseed.exists():
            with open(path_noseed) as f:
                return json.load(f)
    return None


def load_imagenet_result(model):
    """Load an ImageNet result JSON."""
    path = IMAGENET_DIR / f'{model}_imagenet_result.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def get_best_acc(data):
    """Extract best accuracy (as percentage) from result dict."""
    if data is None:
        return None
    r = data.get('results', data)
    acc = r.get('best_acc')
    if acc is not None:
        return acc * 100 if acc <= 1.0 else acc
    return None


def get_alpha_final(data):
    """Extract alpha_final dict from result."""
    if data is None:
        return None
    r = data.get('results', data)
    return r.get('alpha_final')


def get_alpha_stats(data):
    """Extract alpha_stats history list from result."""
    if data is None:
        return None
    r = data.get('results', data)
    h = r.get('history', {})
    return h.get('alpha_stats')


def avg_results(model, depth, dataset):
    """
    Compute 3-seed average accuracy and std.
    Returns (mean_acc, std_acc, list_of_accs) or (None, None, []).
    """
    accs = []
    for seed in SEEDS:
        data = load_result(model, depth, dataset, seed)
        acc = get_best_acc(data)
        if acc is not None:
            accs.append(acc)
    if len(accs) == 0:
        return None, None, []
    return np.mean(accs), np.std(accs), accs


def avg_alpha_mean(model, depth, dataset):
    """
    Compute 3-seed average of alpha_final.mean.
    Returns (mean, std, list_of_values).
    """
    vals = []
    for seed in SEEDS:
        data = load_result(model, depth, dataset, seed)
        af = get_alpha_final(data)
        if af is not None and 'mean' in af:
            vals.append(af['mean'])
    if len(vals) == 0:
        return None, None, []
    return np.mean(vals), np.std(vals), vals


def save_figure(fig, filename):
    """Save figure to both output directories."""
    path1 = FIG_DIR_1 / filename
    path2 = FIG_DIR_2 / filename
    fig.savefig(path1, format='pdf', dpi=300, bbox_inches='tight')
    fig.savefig(path2, format='pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path1}")
    print(f"  Saved: {path2}")


# ==============================================================================
# Figure 0: Gradient Flow
# ==============================================================================

def fig0_gradient_flow():
    """Gradient flow comparison: plain_he vs baseline vs lrs_low."""
    print("\n[Fig 0] Gradient Flow")

    path = RESULTS_DIR / 'gradient_norms_d200.json'
    if not path.exists():
        print("  SKIPPED: gradient_norms_d200.json not found")
        return

    with open(path) as f:
        data = json.load(f)

    models_data = data['models']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # (a) All 3 models, log scale
    for model_key in ['plain_he', 'baseline', 'lrs_low']:
        if model_key not in models_data:
            continue
        md = models_data[model_key]
        norms = md['gradient_norms']
        blocks = np.arange(len(norms))
        ax1.semilogy(blocks, norms, color=COLORS[model_key],
                     label=DISPLAY_NAMES[model_key], linewidth=1.8)

    ax1.set_xlabel('Block Index')
    ax1.set_ylabel('Gradient L2 Norm (log scale)')
    ax1.set_title('(a) Gradient Flow Comparison')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.set_xlim(0, 65)

    # (b) ResNet vs LRS only, linear scale
    for model_key in ['baseline', 'lrs_low']:
        if model_key not in models_data:
            continue
        md = models_data[model_key]
        norms = md['gradient_norms']
        blocks = np.arange(len(norms))
        ax2.plot(blocks, norms, color=COLORS[model_key],
                 label=DISPLAY_NAMES[model_key], linewidth=1.8)

    ax2.set_xlabel('Block Index')
    ax2.set_ylabel('Gradient L2 Norm')
    ax2.set_title('(b) ResNet vs LRS (Linear Scale)')
    ax2.legend(loc='best', framealpha=0.9)
    ax2.set_xlim(0, 65)

    fig.tight_layout(w_pad=3)
    save_figure(fig, 'fig0_gradient_flow.pdf')


# ==============================================================================
# Figure 2: Alpha Mean vs Depth
# ==============================================================================

def fig2_alpha_depth():
    """Alpha mean vs depth for CIFAR-10 and CIFAR-100."""
    print("\n[Fig 2] Alpha Mean vs Depth")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    datasets = ['cifar10', 'cifar100']
    dataset_labels = ['CIFAR-10', 'CIFAR-100']
    models = ['lrs_low', 'lrs_mid', 'lrs_high']
    markers = ['o', 's', '^']

    for ax, dataset, ds_label in zip(axes, datasets, dataset_labels):
        for model, marker in zip(models, markers):
            means = []
            stds = []
            valid_depths = []
            for depth in DEPTHS:
                m, s, vals = avg_alpha_mean(model, depth, dataset)
                if m is not None:
                    means.append(m)
                    stds.append(s)
                    valid_depths.append(depth)

            if valid_depths:
                means = np.array(means)
                stds = np.array(stds)
                ax.plot(valid_depths, means, color=COLORS[model],
                        label=DISPLAY_NAMES[model], marker=marker, linewidth=1.8)
                ax.fill_between(valid_depths, means - stds, means + stds,
                                color=COLORS[model], alpha=0.2)

        ax.set_xlabel('Depth')
        ax.set_ylabel(r'$\bar{\alpha}$ (Mean)')
        ax.set_title(ds_label)
        ax.set_xticks(DEPTHS)
        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim(bottom=0)

    fig.tight_layout(w_pad=3)
    save_figure(fig, 'fig2_alpha_depth.pdf')


# ==============================================================================
# Figure 3: Alpha Trajectory
# ==============================================================================

def fig3_alpha_trajectory():
    """Alpha mean trajectory over training epochs."""
    print("\n[Fig 3] Alpha Trajectory")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    configs = [
        (50, 'cifar10',  'Depth 50, CIFAR-10'),
        (50, 'cifar100', 'Depth 50, CIFAR-100'),
        (200, 'cifar10', 'Depth 200, CIFAR-10'),
        (200, 'cifar100','Depth 200, CIFAR-100'),
    ]
    models = ['lrs_low', 'lrs_mid', 'lrs_high']

    for ax, (depth, dataset, title) in zip(axes.flat, configs):
        for model in models:
            data = load_result(model, depth, dataset, seed=42)
            stats = get_alpha_stats(data)
            if stats is None:
                continue
            epochs = [s['epoch'] for s in stats]
            means = [s['mean'] for s in stats]
            ax.plot(epochs, means, color=COLORS[model],
                    label=DISPLAY_NAMES[model], linewidth=1.8)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(r'$\bar{\alpha}$')
        ax.set_title(title)
        ax.legend(loc='best', framealpha=0.9, fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, 100)

    fig.tight_layout(h_pad=2.5, w_pad=3)
    save_figure(fig, 'fig3_alpha_trajectory.pdf')


# ==============================================================================
# Figure 4: Per-block Alpha
# ==============================================================================

def fig4_perblock_alpha():
    """Per-block alpha values for different depths (lrs_low, CIFAR-100)."""
    print("\n[Fig 4] Per-block Alpha")

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=False)

    for ax, depth in zip(axes, DEPTHS):
        data = load_result('lrs_low', depth, 'cifar100', seed=42)
        af = get_alpha_final(data)
        if af is None or 'all' not in af:
            ax.set_title(f'Depth {depth} (no data)')
            continue

        alphas = af['all']
        n_blocks = len(alphas)
        block_indices = np.arange(n_blocks)

        # Color bars: darker for downsampling blocks
        bar_colors = [COLORS['lrs_low']] * n_blocks
        ds_blocks = DOWNSAMPLE_BLOCKS.get(depth, [])
        for idx in ds_blocks:
            if idx < n_blocks:
                bar_colors[idx] = '#0D47A1'  # darker blue for downsample

        ax.bar(block_indices, alphas, color=bar_colors, width=0.8, edgecolor='none')

        # Stage boundary lines
        boundaries = STAGE_BOUNDARIES.get(depth, {})
        for stage_name, (start, end) in boundaries.items():
            if start > 0:
                ax.axvline(x=start - 0.5, color='gray', linestyle='--',
                           linewidth=0.8, alpha=0.7)

        ax.set_ylabel(r'$\alpha$')
        ax.set_title(f'Depth {depth} ({n_blocks} blocks)')
        ax.set_ylim(0, 1.0)
        ax.set_xlim(-0.5, n_blocks - 0.5)

        if depth == DEPTHS[-1]:
            ax.set_xlabel('Block Index')

    fig.suptitle('Per-block Learned $\\alpha$ (LRS Low, CIFAR-100)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout(h_pad=1.5)
    save_figure(fig, 'fig4_perblock_alpha.pdf')


# ==============================================================================
# Figure 5: Effective Depth
# ==============================================================================

def fig5_effective_depth():
    """Effective depth (sum of alphas) vs nominal depth."""
    print("\n[Fig 5] Effective Depth")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(len(DEPTHS))
    width = 0.35

    nominal_vals = []
    deff_means = []
    deff_stds = []
    pct_labels = []

    for depth in DEPTHS:
        n_blocks = BLOCK_COUNTS[depth]
        nominal_vals.append(n_blocks)

        # 3-seed average of sum(alpha)
        sums = []
        for seed in SEEDS:
            data = load_result('lrs_low', depth, 'cifar100', seed)
            af = get_alpha_final(data)
            if af is not None and 'all' in af:
                sums.append(np.sum(af['all']))

        if sums:
            deff_means.append(np.mean(sums))
            deff_stds.append(np.std(sums))
            pct_labels.append(f'{np.mean(sums)/n_blocks*100:.1f}%')
        else:
            deff_means.append(0)
            deff_stds.append(0)
            pct_labels.append('N/A')

    bars1 = ax.bar(x - width/2, nominal_vals, width, label='Nominal Depth (# blocks)',
                   color='#BDBDBD', edgecolor='#757575', linewidth=0.5)
    bars2 = ax.bar(x + width/2, deff_means, width, yerr=deff_stds,
                   label=r'Effective Depth ($D_{\mathrm{eff}} = \Sigma\alpha$)',
                   color=COLORS['lrs_low'], edgecolor='#0D47A1', linewidth=0.5,
                   capsize=3)

    # Add percentage labels on top of D_eff bars
    for i, (bar, pct) in enumerate(zip(bars2, pct_labels)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + deff_stds[i] + 0.3,
                pct, ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=COLORS['lrs_low'])

    ax.set_xlabel('Network Depth')
    ax.set_ylabel('Number of Blocks')
    ax.set_title('Effective Depth vs Nominal Depth (LRS Low, CIFAR-100)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'ResNet-{d}' for d in DEPTHS])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, max(nominal_vals) * 1.15)

    fig.tight_layout()
    save_figure(fig, 'fig5_effective_depth.pdf')


# ==============================================================================
# Figure 6: Init Failure (lrs_high collapse at deep networks)
# ==============================================================================

def fig6_init_failure():
    """Show lrs_high accuracy collapse and alpha trajectory at d200."""
    print("\n[Fig 6] Init Failure")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Accuracy vs depth for all models, CIFAR-100
    models = ['baseline', 'lrs_low', 'lrs_mid', 'lrs_high', 'rezero']
    markers = ['o', 's', '^', 'D', 'v']

    for model, marker in zip(models, markers):
        acc_means = []
        acc_stds = []
        valid_depths = []
        for depth in DEPTHS:
            m, s, vals = avg_results(model, depth, 'cifar100')
            if m is not None:
                acc_means.append(m)
                acc_stds.append(s)
                valid_depths.append(depth)

        if valid_depths:
            acc_means = np.array(acc_means)
            acc_stds = np.array(acc_stds)
            ax1.plot(valid_depths, acc_means, color=COLORS[model],
                     label=DISPLAY_NAMES[model], marker=marker, linewidth=1.8)
            ax1.fill_between(valid_depths, acc_means - acc_stds, acc_means + acc_stds,
                             color=COLORS[model], alpha=0.15)

    ax1.set_xlabel('Depth')
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('(a) Accuracy vs Depth (CIFAR-100)')
    ax1.set_xticks(DEPTHS)
    ax1.legend(loc='lower left', framealpha=0.9, fontsize=8)

    # (b) Alpha trajectory at d200 showing lrs_high stuck
    for model in ['lrs_low', 'lrs_mid', 'lrs_high']:
        data = load_result(model, 200, 'cifar100', seed=42)
        stats = get_alpha_stats(data)
        if stats is None:
            continue
        epochs = [s['epoch'] for s in stats]
        means = [s['mean'] for s in stats]
        ax2.plot(epochs, means, color=COLORS[model],
                 label=DISPLAY_NAMES[model], linewidth=1.8)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(r'$\bar{\alpha}$')
    ax2.set_title(r'(b) $\alpha$ Trajectory at Depth 200')
    ax2.legend(loc='best', framealpha=0.9, fontsize=8)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(bottom=0)

    fig.tight_layout(w_pad=3)
    save_figure(fig, 'fig6_init_failure.pdf')


# ==============================================================================
# Figure 7: Fixed Alpha Comparison
# ==============================================================================

def fig7_fixed_alpha():
    """Compare learned vs fixed alpha values at depth 152, CIFAR-100."""
    print("\n[Fig 7] Fixed Alpha")

    fig, ax = plt.subplots(figsize=(8, 4.5))

    configs = [
        ('Baseline', 'baseline', None, COLORS['baseline']),
        ('LRS Low\n(Learned)', 'lrs_low', None, COLORS['lrs_low']),
        ('Fixed 0.1', 'fixed_alpha_01', None, '#4CAF50'),
        ('Fixed 0.3', 'fixed_alpha_03', None, '#FFC107'),
        ('Fixed 0.5', 'fixed_alpha_05', None, '#FF9800'),
        ('Fixed 0.7', 'fixed_alpha_07', None, '#F44336'),
    ]

    labels = []
    means = []
    stds = []
    colors = []

    for label, model, _, color in configs:
        accs = []
        for seed in SEEDS:
            if model == 'baseline':
                data = load_result('baseline', 152, 'cifar100', seed)
            elif model == 'lrs_low':
                data = load_result('lrs_low', 152, 'cifar100', seed)
            else:
                # fixed_alpha files
                path = RESULTS_DIR / f'{model}_depth152_cifar100_seed{seed}_result.json'
                if path.exists():
                    with open(path) as f:
                        data = json.load(f)
                else:
                    data = None

            acc = get_best_acc(data)
            if acc is not None:
                accs.append(acc)

        if accs:
            labels.append(label)
            means.append(np.mean(accs))
            stds.append(np.std(accs))
            colors.append(color)

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='white',
                  linewidth=0.5, capsize=4, width=0.6)

    # Add accuracy labels on bars
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{m:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Learned vs Fixed $\\alpha$ (Depth 152, CIFAR-100)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)

    # Set y-axis to show differences clearly
    if means:
        ymin = min(means) - max(stds) - 2
        ymax = max(means) + max(stds) + 2
        ax.set_ylim(max(0, ymin), ymax)

    fig.tight_layout()
    save_figure(fig, 'fig7_fixed_alpha.pdf')


# ==============================================================================
# Figure 8: Heatmap of Alpha Values
# ==============================================================================

def fig8_heatmap():
    """Heatmap of per-block alpha across depths."""
    print("\n[Fig 8] Alpha Heatmap")

    fig, ax = plt.subplots(figsize=(10, 4))

    # Build a matrix: rows = depths, columns = normalized position
    n_cols = 200  # resolution for normalized position
    heatmap_data = np.full((len(DEPTHS), n_cols), np.nan)

    for i, depth in enumerate(DEPTHS):
        data = load_result('lrs_low', depth, 'cifar100', seed=42)
        af = get_alpha_final(data)
        if af is None or 'all' not in af:
            continue
        alphas = af['all']
        n_blocks = len(alphas)
        for j, alpha_val in enumerate(alphas):
            # Map block index to normalized position
            col_start = int(j / n_blocks * n_cols)
            col_end = int((j + 1) / n_blocks * n_cols)
            heatmap_data[i, col_start:col_end] = alpha_val

    im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd',
                   extent=[0, 1, len(DEPTHS) - 0.5, -0.5],
                   interpolation='nearest', vmin=0, vmax=0.8)

    ax.set_yticks(range(len(DEPTHS)))
    ax.set_yticklabels([f'd{d}' for d in DEPTHS])
    ax.set_xlabel('Normalized Block Position')
    ax.set_ylabel('Depth')
    ax.set_title(r'Per-block $\alpha$ Heatmap (LRS Low, CIFAR-100)')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(r'$\alpha$ value', fontsize=10)

    fig.tight_layout()
    save_figure(fig, 'fig8_heatmap.pdf')


# ==============================================================================
# Figure 9: Method Comparison
# ==============================================================================

def fig9_comparison():
    """Grouped bar chart comparing methods on CIFAR-10 and CIFAR-100 at depth 152."""
    print("\n[Fig 9] Method Comparison")

    fig, ax = plt.subplots(figsize=(10, 4.5))

    methods = ['baseline', 'lrs_low', 'rezero', 'skipinit', 'fixup', 'layerscale']
    method_labels = ['Baseline', 'LRS Low', 'ReZero', 'SkipInit', 'Fixup', 'LayerScale']
    method_colors = [COLORS[m] for m in methods]

    datasets = ['cifar10', 'cifar100']
    dataset_labels = ['CIFAR-10', 'CIFAR-100']

    n_methods = len(methods)
    n_groups = len(datasets)
    bar_width = 0.12
    group_width = n_methods * bar_width

    for g, (dataset, ds_label) in enumerate(zip(datasets, dataset_labels)):
        for m_idx, (method, mlabel) in enumerate(zip(methods, method_labels)):
            depth = 152
            mean_acc, std_acc, accs = avg_results(method, depth, dataset)

            x_pos = g * (group_width + 0.3) + m_idx * bar_width

            if mean_acc is not None:
                ax.bar(x_pos, mean_acc, bar_width, yerr=std_acc,
                       color=method_colors[m_idx], edgecolor='white',
                       linewidth=0.5, capsize=2,
                       label=mlabel if g == 0 else '')
            else:
                ax.bar(x_pos, 0, bar_width, color='lightgray',
                       label=mlabel if g == 0 else '')

    # X-axis labels for groups
    group_centers = []
    for g in range(n_groups):
        center = g * (group_width + 0.3) + (n_methods - 1) * bar_width / 2
        group_centers.append(center)

    ax.set_xticks(group_centers)
    ax.set_xticklabels(dataset_labels, fontsize=11)
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Method Comparison (Depth 152, 3-seed Average)')
    ax.legend(loc='upper right', ncol=2, framealpha=0.9, fontsize=8)

    # Set y-axis for visibility
    ax.set_ylim(bottom=70)

    fig.tight_layout()
    save_figure(fig, 'fig9_comparison.pdf')


# ==============================================================================
# Figure 10: ImageNet Results
# ==============================================================================

def fig10_imagenet():
    """ImageNet results: accuracy bar chart and per-block alpha."""
    print("\n[Fig 10] ImageNet")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # (a) Top-1 accuracy bar chart
    models = ['baseline', 'lrs_low', 'lrs_mid', 'lrs_high', 'rezero', 'lrs_ha_low']
    model_labels = ['Baseline', 'LRS Low', 'LRS Mid', 'LRS High', 'ReZero', 'LRS-HA\nLow']
    model_colors = [COLORS.get(m, '#999999') for m in models]

    accs = []
    valid_labels = []
    valid_colors = []

    for model, label, color in zip(models, model_labels, model_colors):
        data = load_imagenet_result(model)
        acc = get_best_acc(data)
        if acc is not None:
            accs.append(acc)
            valid_labels.append(label)
            valid_colors.append(color)

    if accs:
        x = np.arange(len(accs))
        bars = ax1.bar(x, accs, color=valid_colors, edgecolor='white',
                       linewidth=0.5, width=0.6)

        # Add accuracy labels
        for bar, acc_val in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{acc_val:.1f}', ha='center', va='bottom', fontsize=8,
                     fontweight='bold')

        ax1.set_xticks(x)
        ax1.set_xticklabels(valid_labels, fontsize=8)
        ax1.set_ylabel('Top-1 Accuracy (%)')
        ax1.set_title('(a) ImageNet Top-1 Accuracy')
        if accs:
            ax1.set_ylim(min(accs) - 3, max(accs) + 3)

    # (b) Per-block alpha for lrs_low on ImageNet
    data = load_imagenet_result('lrs_low')
    af = get_alpha_final(data) if data else None

    if af is not None and 'all' in af:
        alphas = af['all']
        blocks = np.arange(len(alphas))
        ax2.bar(blocks, alphas, color=COLORS['lrs_low'], width=0.7, edgecolor='none')
        ax2.set_xlabel('Block Index')
        ax2.set_ylabel(r'$\alpha$')
        ax2.set_title('(b) Per-block $\\alpha$ (LRS Low, ImageNet)')
        ax2.set_ylim(0, max(alphas) * 1.2 if max(alphas) > 0 else 1.0)
        ax2.set_xlim(-0.5, len(alphas) - 0.5)
    else:
        ax2.set_title('(b) Per-block alpha (no data)')

    fig.tight_layout(w_pad=3)
    save_figure(fig, 'fig10_imagenet.pdf')


# ==============================================================================
# Figure 11: Pruning Results
# ==============================================================================

def fig11_pruning():
    """Pruning results: without and with fine-tuning."""
    print("\n[Fig 11] Pruning Results")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # (a) Without fine-tuning
    path_prune = RESULTS_DIR / 'pruning_d200_cifar100.json'
    if path_prune.exists():
        with open(path_prune) as f:
            prune_data = json.load(f)

        results = prune_data['results']
        pct_removed = [r['pct_removed'] for r in results]
        accuracy = [r['accuracy'] for r in results]

        ax1.plot(pct_removed, accuracy, color=COLORS['lrs_low'], marker='o',
                 linewidth=1.8, markersize=6)
        ax1.axhline(y=accuracy[0], color='gray', linestyle='--', linewidth=1.0,
                     alpha=0.7, label=f'Original ({accuracy[0]:.1f}%)')
        ax1.set_xlabel('Blocks Removed (%)')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('(a) Pruning Without Fine-tuning')
        ax1.legend(loc='lower left', framealpha=0.9)
        ax1.set_xlim(-2, 102)
        ax1.set_ylim(0, max(accuracy) + 5)
    else:
        ax1.set_title('(a) Pruning (no data)')

    # (b) With fine-tuning
    path_ft = RESULTS_DIR / 'pruning_finetune_d200_cifar100.json'
    if path_ft.exists():
        with open(path_ft) as f:
            ft_data = json.load(f)

        original_acc = ft_data.get('original_acc', 0)
        results = ft_data['results']
        pct_removed = [r['pct_removed'] for r in results]
        acc_before = [r['acc_before_ft'] for r in results]
        acc_after = [r['acc_after_ft'] for r in results]

        ax2.plot(pct_removed, acc_before, color=COLORS['lrs_low'], marker='o',
                 linewidth=1.5, markersize=5, linestyle='--', label='Before FT',
                 alpha=0.7)
        ax2.plot(pct_removed, acc_after, color=COLORS['lrs_low'], marker='s',
                 linewidth=1.8, markersize=5, linestyle='-', label='After FT')
        ax2.axhline(y=original_acc, color='gray', linestyle='--', linewidth=1.0,
                     alpha=0.7, label=f'Original ({original_acc:.1f}%)')

        ax2.set_xlabel('Blocks Removed (%)')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('(b) Pruning With Fine-tuning')
        ax2.legend(loc='lower left', framealpha=0.9)
        ax2.set_xlim(-2, 95)
        ax2.set_ylim(0, max(max(acc_after), original_acc) + 5)
    else:
        ax2.set_title('(b) Pruning FT (no data)')

    fig.tight_layout(w_pad=3)
    save_figure(fig, 'fig11_pruning.pdf')


# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 60)
    print("LRS Paper Figure Generator")
    print(f"Base directory: {BASE_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Output dirs: {FIG_DIR_1}, {FIG_DIR_2}")
    print("=" * 60)

    setup_style()

    figure_funcs = [
        fig0_gradient_flow,
        fig2_alpha_depth,
        fig3_alpha_trajectory,
        fig4_perblock_alpha,
        fig5_effective_depth,
        fig6_init_failure,
        fig7_fixed_alpha,
        fig8_heatmap,
        fig9_comparison,
        fig10_imagenet,
        fig11_pruning,
    ]

    success = 0
    failed = 0

    for func in figure_funcs:
        try:
            func()
            success += 1
        except Exception as e:
            print(f"  ERROR in {func.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Done! {success} figures generated, {failed} failed.")
    print(f"Figures saved to:")
    print(f"  {FIG_DIR_1}")
    print(f"  {FIG_DIR_2}")
    print("=" * 60)


if __name__ == '__main__':
    main()
