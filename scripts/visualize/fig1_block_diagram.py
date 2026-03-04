"""
fig1_block_diagram.py - ResNet Block vs LRS Block 비교 다이어그램
================================================================
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from style import set_paper_style, save_figure

set_paper_style()


def draw_block(ax, x_offset, title, is_lrs=False):
    """하나의 Residual Block 다이어그램 그리기"""
    ax.set_xlim(0, 4)
    ax.set_ylim(-0.5, 9.5)
    ax.axis('off')

    # 색상 정의
    box_color = '#ecf0f1'
    border_color = '#7f8c8d'
    skip_color = '#2c3e50'
    alpha_color = '#e74c3c'
    arrow_kw = dict(arrowstyle='->', color='#2c3e50', lw=1.5,
                    connectionstyle='arc3,rad=0.0')

    def box(ax, cx, cy, w, h, label, color=box_color, fontsize=9):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                               boxstyle='round,pad=0.05',
                               facecolor=color, edgecolor=border_color,
                               linewidth=1.2)
        ax.add_patch(rect)
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold')

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))

    cx = 2.0  # center x

    # 제목
    ax.text(cx, 9.2, title, ha='center', va='center',
            fontsize=12, fontweight='bold', color='#2c3e50')

    # Input x
    ax.text(cx, 8.6, '$x$', ha='center', va='center', fontsize=11)
    arrow(ax, cx, 8.3, cx, 7.8)

    # Conv-BN-ReLU blocks
    box(ax, cx, 7.4, 2.2, 0.6, 'Conv-BN-ReLU')
    arrow(ax, cx, 7.1, cx, 6.6)
    box(ax, cx, 6.2, 2.2, 0.6, 'Conv-BN-ReLU')
    arrow(ax, cx, 5.9, cx, 5.4)
    box(ax, cx, 5.0, 2.2, 0.6, 'Conv-BN')
    arrow(ax, cx, 4.7, cx, 4.1)

    # F(x) label
    ax.text(cx + 0.15, 4.4, '$F(x)$', ha='left', va='center',
            fontsize=9, color='#7f8c8d', style='italic')

    # Skip connection
    ax.annotate('', xy=(3.3, 3.8), xytext=(3.3, 8.6),
                arrowprops=dict(arrowstyle='->', color=skip_color, lw=1.5))
    ax.plot([cx, 3.3], [8.6, 8.6], color=skip_color, lw=1.5)
    ax.plot([3.3, 3.3], [8.6, 3.8], color=skip_color, lw=1.5)

    # Skip label
    ax.text(3.55, 6.2, '$x$', ha='center', va='center',
            fontsize=10, color=skip_color)

    if not is_lrs:
        # Add 서클 (summation)
        circle = plt.Circle((cx, 3.7), 0.28, color='white',
                             ec=border_color, lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(cx, 3.7, '$+$', ha='center', va='center', fontsize=13,
                color='#2c3e50', zorder=4)
        ax.plot([3.3, cx + 0.28], [3.8, 3.7], color=skip_color, lw=1.5)

        # y = F(x) + x label
        ax.text(cx - 1.1, 3.7, '$y = F(x) + x$', ha='center', va='center',
                fontsize=8.5, color='#7f8c8d',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f8f9fa',
                          edgecolor='none'))
    else:
        # Alpha scaling box
        box(ax, cx - 0.6, 3.7, 0.8, 0.5, '$\\alpha$', color='#fadbd8',
            fontsize=10)
        box(ax, 3.0, 3.7, 0.8, 0.5, '$1\\!-\\!\\alpha$', color='#d5f5e3',
            fontsize=9)

        ax.plot([cx, cx - 0.6], [4.1, 3.95], color=alpha_color, lw=1.5)
        ax.plot([3.3, 3.0], [3.8, 3.95], color='#27ae60', lw=1.5)

        # summation circle
        circle = plt.Circle((cx, 2.95), 0.28, color='white',
                             ec=border_color, lw=1.5, zorder=3)
        ax.add_patch(circle)
        ax.text(cx, 2.95, '$+$', ha='center', va='center', fontsize=13,
                color='#2c3e50', zorder=4)

        arrow(ax, cx - 0.6, 3.45, cx - 0.28, 2.95)
        arrow(ax, 3.0, 3.45, cx + 0.28, 2.95)

        # sigmoid(θ) annotation
        ax.text(cx, 4.55, r'$\alpha = \sigma(\theta)$', ha='center',
                va='center', fontsize=9, color=alpha_color,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fdf2f2',
                          edgecolor=alpha_color, lw=0.8))

        # formula label
        ax.text(cx - 1.4, 2.95,
                '$y = \\alpha F(x)$\n$+ (1\\!-\\!\\alpha) x$',
                ha='center', va='center', fontsize=8, color='#7f8c8d',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#f8f9fa',
                          edgecolor='none'))

    # ReLU
    y_relu = 2.2 if not is_lrs else 1.5
    arrow(ax, cx, 3.42 if not is_lrs else 2.67, cx, y_relu + 0.3)
    box(ax, cx, y_relu, 1.6, 0.5, 'ReLU')
    arrow(ax, cx, y_relu - 0.25, cx, y_relu - 0.7)

    # Output y
    ax.text(cx, y_relu - 0.85, '$y$', ha='center', va='center', fontsize=11)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 6.5))
    fig.patch.set_facecolor('white')

    draw_block(axes[0], 0, '(a) Standard ResNet Block', is_lrs=False)
    draw_block(axes[1], 0, '(b) LRS Block (Proposed)', is_lrs=True)

    plt.tight_layout(pad=1.5)
    save_figure(fig, 'fig1_block_diagram')
    plt.close()
    print('Fig 1 done.')


if __name__ == '__main__':
    main()
