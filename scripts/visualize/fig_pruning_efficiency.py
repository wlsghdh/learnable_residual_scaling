"""
Figure: alpha-guided pruning efficiency
- Left: Accuracy vs % blocks pruned
- Right: FLOPs / Latency reduction with accuracy preserved
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

ROOT = '/home/jjh0709/Learnable_Residual_Scaling'

def main():
    with open(os.path.join(ROOT, 'results-json/pruning_analysis_d200_cifar100.json')) as f:
        d = json.load(f)

    orig_flops = d['original']['flops']
    orig_lat = d['original']['latency']['mean_ms']
    orig_acc = d['original']['accuracy']

    rows = d['results']
    pct = [r['pruned'] / r['total'] * 100 for r in rows]
    acc_noft = [r['acc_no_ft'] for r in rows]
    acc_ft = [r['acc_with_ft'] for r in rows]
    flops_pct = [r['flops'] / orig_flops * 100 for r in rows]
    lat_pct = [r['latency']['mean_ms'] / orig_lat * 100 for r in rows]
    thresh = [r['threshold'] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))

    # Left: Accuracy vs % blocks pruned
    ax = axes[0]

    # Sweet spot 강조 (a<0.08, 56% pruned)
    ax.axvspan(50, 60, alpha=0.12, color='gold', zorder=0)

    ax.plot(pct, acc_noft, marker='x', markersize=9, linewidth=1.2,
            color='#d62728', linestyle='--', label='Without fine-tuning', alpha=0.85)
    ax.plot(pct, acc_ft, marker='o', markersize=8, linewidth=1.5,
            color='#2ca02c', label='With 10-epoch fine-tuning')
    ax.axhline(orig_acc, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    ax.text(80, orig_acc + 1.5, f'Original: {orig_acc:.2f}%',
            fontsize=8.5, color='gray', ha='right')

    # Annotations (staggered offsets)
    ann_offsets = {0.03: (8, -10), 0.05: (10, -15), 0.08: (-8, 10), 0.10: (5, -20)}
    for i, (p, a, t) in enumerate(zip(pct, acc_ft, thresh)):
        if t in ann_offsets:
            dx, dy = ann_offsets[t]
            ax.annotate(rf'$\alpha<{t:.2f}$',
                        xy=(p, a), xytext=(p+dx, a+dy),
                        fontsize=8.5, color='#2ca02c',
                        arrowprops=dict(arrowstyle='->', color='gray',
                                        lw=0.6, alpha=0.7))

    # Sweet spot label
    ax.text(55, 30, 'Sweet spot\n' + r'$\alpha<0.08$' + '\n(−4.6%, 56% removed)',
            ha='center', fontsize=8.5, color='#b8860b', fontweight='bold')

    ax.set_xlabel('Blocks pruned (\%)', fontsize=11)
    ax.set_ylabel('Top-1 accuracy (\%)', fontsize=11)
    ax.set_xlim(-3, 95)
    ax.set_ylim(0, 90)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower left', frameon=True, fontsize=9.5,
              edgecolor='gray', framealpha=0.95)
    ax.set_title('(a) Accuracy retention via fine-tuning',
                 fontsize=11, pad=8)

    # Right: FLOPs / Latency reduction (skip threshold=0)
    ax = axes[1]
    rows_pruned = [r for r in rows if r['threshold'] > 0]
    flops_r = [100 - r['flops'] / orig_flops * 100 for r in rows_pruned]
    lat_r = [100 - r['latency']['mean_ms'] / orig_lat * 100 for r in rows_pruned]
    acc_drop = [orig_acc - r['acc_with_ft'] for r in rows_pruned]
    labels = [rf'$\alpha<{r["threshold"]:.2f}$' for r in rows_pruned]

    x = np.arange(len(rows_pruned))
    width = 0.35
    b1 = ax.bar(x - width/2, flops_r, width,
                color='#1f77b4', label='FLOPs reduction', edgecolor='black', linewidth=0.4)
    b2 = ax.bar(x + width/2, lat_r, width,
                color='#ff7f0e', label='Latency reduction', edgecolor='black', linewidth=0.4)

    # Bar value labels
    for bar, val in zip(b1, flops_r):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f'{val:.0f}', ha='center', fontsize=8, color='#1f77b4')
    for bar, val in zip(b2, lat_r):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1.5,
                f'{val:.0f}', ha='center', fontsize=8, color='#ff7f0e')

    # Accuracy drop secondary annotation (purple to avoid red confusion)
    ax2 = ax.twinx()
    ax2.plot(x, acc_drop, marker='o', markersize=7, linewidth=1.5,
             color='#7b3294', label='Accuracy drop')
    ax2.set_ylabel('Accuracy drop (%)', fontsize=11, color='#7b3294')
    ax2.tick_params(axis='y', labelcolor='#7b3294')
    ax2.spines['top'].set_visible(False)
    ax2.set_ylim(0, max(acc_drop) * 1.3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel('Reduction (%)', fontsize=11)
    ax.set_xlabel('Pruning threshold', fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              frameon=True, fontsize=9, edgecolor='gray', framealpha=0.95)
    ax.set_title('(b) FLOPs and latency reduction',
                 fontsize=11, pad=8)

    plt.tight_layout()
    out_dir = os.path.join(ROOT, 'paper_bmvc/figures')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'fig_pruning_efficiency.pdf'),
                bbox_inches='tight', pad_inches=0.05)
    plt.savefig(os.path.join(out_dir, 'fig_pruning_efficiency.png'),
                bbox_inches='tight', pad_inches=0.05, dpi=200)

    print(f'Saved: {out_dir}/fig_pruning_efficiency.pdf')
    print()
    print('Summary:')
    for r in rows_pruned:
        t = r['threshold']
        print(f"  alpha<{t:.2f}: {r['pruned']}/{r['total']} pruned, "
              f"acc={r['acc_with_ft']:.2f}%, "
              f"FLOPs={r['flops']/orig_flops*100:.1f}%, "
              f"Latency={r['latency']['mean_ms']/orig_lat*100:.1f}%")

if __name__ == '__main__':
    main()
