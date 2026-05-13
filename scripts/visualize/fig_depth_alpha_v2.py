"""
Figure: depth-alpha scaling law (CIFAR + ImageNet combined)
Updated to include ImageNet d50/d101 (new ImageNet stem experiments).
"""
import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Style — clean, BMVC-friendly
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.linewidth': 0.8,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42,  # editable text in PDF
    'ps.fonttype': 42,
})

ROOT = '/home/jjh0709/Learnable_Residual_Scaling'

def collect_cifar_alpha(model='lrs_low', dataset='cifar100'):
    """Collect mean alpha per depth across seeds."""
    depths = [50, 101, 152, 200]
    data = {d: [] for d in depths}
    for d in depths:
        # try seed42 (no suffix), then 123, 456
        candidates = [
            f'results-json/{model}_depth{d}_{dataset}_result.json',
            f'results-json/{model}_depth{d}_{dataset}_seed42_result.json',
            f'results-json/{model}_depth{d}_{dataset}_seed123_result.json',
            f'results-json/{model}_depth{d}_{dataset}_seed456_result.json',
        ]
        for path in candidates:
            full = os.path.join(ROOT, path)
            if os.path.exists(full):
                try:
                    with open(full) as f:
                        r = json.load(f)
                    af = r.get('results', {}).get('alpha_final', {})
                    if af and 'mean' in af:
                        data[d].append(af['mean'])
                except:
                    pass
    return data

def collect_imagenet():
    """Collect ImageNet alpha values."""
    results = {}
    for d, fname in [(50, 'lrs_low_imagenet_result.json'),
                     (101, 'lrs_low_d101_imagenet_result.json')]:
        path = os.path.join(ROOT, 'results-json/imagenet_v2', fname)
        if os.path.exists(path):
            with open(path) as f:
                r = json.load(f)
            af = r.get('results', {}).get('alpha_final', {})
            if af:
                results[d] = af['mean']
    return results

def main():
    c10 = collect_cifar_alpha('lrs_low', 'cifar10')
    c100 = collect_cifar_alpha('lrs_low', 'cifar100')
    imgnet = collect_imagenet()

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # CIFAR-10
    depths_c10 = sorted([d for d in c10 if c10[d]])
    means_c10 = [np.mean(c10[d]) for d in depths_c10]
    stds_c10 = [np.std(c10[d]) for d in depths_c10]
    ax.errorbar(depths_c10, means_c10, yerr=stds_c10,
                marker='o', markersize=7, linewidth=1.5,
                capsize=3, color='#1f77b4',
                label='CIFAR-10', zorder=3)

    # CIFAR-100
    depths_c100 = sorted([d for d in c100 if c100[d]])
    means_c100 = [np.mean(c100[d]) for d in depths_c100]
    stds_c100 = [np.std(c100[d]) for d in depths_c100]
    ax.errorbar(depths_c100, means_c100, yerr=stds_c100,
                marker='s', markersize=7, linewidth=1.5,
                capsize=3, color='#ff7f0e',
                label='CIFAR-100', zorder=3)

    # ImageNet (single seed, no error bar)
    if imgnet:
        depths_img = sorted(imgnet.keys())
        vals_img = [imgnet[d] for d in depths_img]
        ax.plot(depths_img, vals_img,
                marker='D', markersize=11, linewidth=1.8,
                color='#2ca02c',
                markeredgecolor='black', markeredgewidth=0.5,
                label='ImageNet (1 seed)', zorder=5)

    # Annotation: ImageNet d50 and d101 close to CIFAR pattern
    if 50 in imgnet:
        ax.annotate(f'{imgnet[50]:.3f}', xy=(50, imgnet[50]),
                    xytext=(56, imgnet[50]-0.030),
                    fontsize=9, color='black', fontweight='bold')
    if 101 in imgnet:
        ax.annotate(f'{imgnet[101]:.3f}', xy=(101, imgnet[101]),
                    xytext=(108, imgnet[101]-0.030),
                    fontsize=9, color='black', fontweight='bold')

    # Thesis annotation — bottom-left for visibility
    ax.text(0.04, 0.06,
            r'$\bar{\alpha}$: $0.34 \to 0.12$ as $L: 50 \to 200$' + '\n' +
            r'$\approx$ 65% reduction',
            transform=ax.transAxes, fontsize=9.5,
            color='black', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff5e6',
                      edgecolor='#ff7f0e', linewidth=0.8))

    ax.set_xlabel('Nominal depth $L$', fontsize=12)
    ax.set_ylabel(r'Converged mean $\bar{\alpha}$', fontsize=12)
    ax.set_xticks([50, 101, 152, 200])
    ax.set_xticklabels([50, 101, 152, 200])
    ax.set_ylim(0.05, 0.40)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', frameon=True, fontsize=9.5,
              edgecolor='gray', framealpha=0.95)

    plt.tight_layout()
    out_dir = os.path.join(ROOT, 'paper_bmvc/figures')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'fig_depth_alpha_v2.pdf'),
                bbox_inches='tight', pad_inches=0.05)
    plt.savefig(os.path.join(out_dir, 'fig_depth_alpha_v2.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=200)

    # Console summary
    print('=== CIFAR-10 ===')
    for d in depths_c10:
        print(f'  d{d}: mean alpha = {np.mean(c10[d]):.4f} (n={len(c10[d])})')
    print('=== CIFAR-100 ===')
    for d in depths_c100:
        print(f'  d{d}: mean alpha = {np.mean(c100[d]):.4f} (n={len(c100[d])})')
    print('=== ImageNet ===')
    for d, v in sorted(imgnet.items()):
        print(f'  d{d}: mean alpha = {v:.4f}')
    print()
    print(f'Saved: {out_dir}/fig_depth_alpha_v2.pdf')

if __name__ == '__main__':
    main()
