"""
fig8_imagenet_results.py - ImageNet Experiment Results
=======================================================
ImageNet 실험 완료 후 실행.
결과 JSON 경로: results/imagenet/{model}_imagenet_result.json
"""

import json
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from style import set_paper_style, save_figure, COLORS

set_paper_style()

DEFAULT_DIR = os.path.join(os.path.dirname(__file__), '../../results-json/imagenet')

VARIANTS = ['baseline', 'lrs_low', 'lrs_ha_low', 'rezero', 'lrs_mid']
VARIANT_LABELS = {
    'baseline':   'Baseline\nResNet-50',
    'lrs_low':    'LRS-Low\n($\\alpha_0\\approx0.12$)',
    'lrs_ha_low': 'LRS+HA-Low\n(Proposed)',
    'rezero':     'ReZero',
    'lrs_mid':    'LRS-Mid\n($\\alpha_0=0.5$)',
}
VARIANT_COLORS = {
    'baseline':   COLORS['baseline'],
    'lrs_low':    COLORS['lrs_low'],
    'lrs_ha_low': COLORS['lrs_ha_low'],
    'rezero':     COLORS['rezero'],
    'lrs_mid':    COLORS['lrs_mid'],
}


def load_imagenet(result_dir, model_key):
    """ImageNet 결과 로드. 없으면 None 반환"""
    fname = os.path.join(result_dir, f'{model_key}_imagenet_result.json')
    if not os.path.exists(fname):
        print(f'  WARNING: not found: {fname}')
        return None
    with open(fname) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', default=DEFAULT_DIR,
                        help='ImageNet 결과 JSON 디렉토리')
    args = parser.parse_args()

    results = {}
    for v in VARIANTS:
        results[v] = load_imagenet(args.result_dir, v)

    # 데이터 있는지 확인
    any_data = any(r is not None for r in results.values())
    if not any_data:
        print('WARNING: No ImageNet results found.')
        print('         ImageNet 실험 완료 후 다시 실행하세요.')
        print(f'         예상 경로: {args.result_dir}')

        # Placeholder figure 생성
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5,
                'ImageNet Results\n(To be updated after ImageNet experiments)',
                ha='center', va='center', transform=ax.transAxes,
                fontsize=12, color='#7f8c8d',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa',
                          edgecolor='#bdc3c7'))
        ax.axis('off')
        save_figure(fig, 'fig8_imagenet_results')
        plt.close()
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5))
    fig.patch.set_facecolor('white')

    x = np.arange(len(VARIANTS))
    width = 0.55

    for ax, metric_key, title in [
        (axes[0], 'best_top1', '(a) Top-1 Accuracy'),
        (axes[1], 'best_top5', '(b) Top-5 Accuracy'),
    ]:
        vals = []
        colors = []
        for v in VARIANTS:
            r = results[v]
            if r is None:
                vals.append(0)
            else:
                vals.append(r['results'].get(metric_key,
                            r['results']['best_acc'] * 100))
            colors.append(VARIANT_COLORS[v])

        bars = ax.bar(x, vals, width, color=colors,
                      edgecolor='white', linewidth=0.5,
                      alpha=0.9, zorder=3)

        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        val + 0.1, f'{val:.2f}%',
                        ha='center', va='bottom', fontsize=8.5)

        ax.set_xticks(x)
        ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANTS], fontsize=8.5)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title, fontweight='bold')
        valid_vals = [v for v in vals if v > 0]
        if valid_vals:
            ax.set_ylim(min(valid_vals) - 1.0, max(valid_vals) + 1.5)
        ax.grid(axis='y', alpha=0.35, linestyle='--')
        ax.set_axisbelow(True)

    plt.tight_layout(pad=1.5)
    save_figure(fig, 'fig8_imagenet_results')
    plt.close()
    print('Fig 8 done.')


if __name__ == '__main__':
    main()
