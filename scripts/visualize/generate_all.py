"""
generate_all.py - 모든 논문용 Figure 일괄 생성
================================================
사용법:
    cd ~/Learnable_Reidual_Scaling
    python scripts/visualize/generate_all.py

결과 저장: figures/ (PDF + PNG 300dpi)
"""

import os
import sys
import importlib
import traceback

sys.path.insert(0, os.path.dirname(__file__))

FIGURES = [
    # 현재 데이터로 즉시 생성 가능
    ('fig1_block_diagram',    'Fig 1: Block Diagram'),
    ('fig_pilot_study',       'Fig  : Pilot Study (Skip & Init)'),
    ('fig2_depth_accuracy',   'Fig 2: Depth vs Accuracy'),
    ('fig3_init_sensitivity', 'Fig 3: Initialization Sensitivity'),
    ('fig4_depth_alpha',      'Fig 4: Depth-α Relationship ⭐'),
    ('fig5_perblock_alpha',   'Fig 5: Per-block α Distribution'),
    ('fig6_alpha_trajectory', 'Fig 6: α Trajectory During Training'),
    ('fig7_high_collapse',    'Fig 7: LRS-High Collapse'),
    # 추가 데이터 필요 (batch2/3 후)
    ('fig_full_comparison',   'Fig  : Full Method Comparison (needs batch2/3)'),
    # 추가 데이터 필요 (batch4 후)
    ('fig_fixed_alpha',       'Fig  : Fixed-α Ablation (needs batch4)'),
    # 추가 데이터 필요 (batch6 후)
    ('fig_cross_arch',        'Fig  : Cross-Architecture (needs batch6)'),
    # ImageNet 완료 후
    ('fig8_imagenet_results', 'Fig 8: ImageNet Results (needs batch7)'),
]


def main():
    print('=' * 60)
    print('  LRS Paper Figure Generation')
    print('=' * 60)

    success = []
    failed = []

    for module_name, desc in FIGURES:
        print(f'\n[{module_name}] {desc}')
        try:
            mod = importlib.import_module(module_name)
            mod.main()
            success.append(module_name)
        except Exception as e:
            print(f'  ERROR: {e}')
            traceback.print_exc()
            failed.append((module_name, str(e)))

    print('\n' + '=' * 60)
    print(f'  Done: {len(success)}/{len(FIGURES)} figures generated')
    if failed:
        print(f'  Failed:')
        for name, err in failed:
            print(f'    - {name}: {err}')
    print('=' * 60)
    print(f'\n  Output directory: figures/')


if __name__ == '__main__':
    main()
