"""
style.py - 논문용 공통 스타일 및 색상 정의
============================================
"""

import matplotlib
import matplotlib.pyplot as plt
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '../../results-json')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '../../figures')


def set_paper_style():
    matplotlib.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'legend.framealpha': 0.8,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


COLORS = {
    # Core models
    'baseline':         '#2c3e50',
    'lrs_low':          '#e74c3c',
    'lrs_mid':          '#3498db',
    'lrs_high':         '#e67e22',
    'lrs_ha_low':       '#c0392b',
    'lrs_ha_mid':       '#2980b9',
    'lrs_ha_high':      '#d35400',
    'rezero':           '#27ae60',
    'hybrida':          '#8e44ad',
    # Pilot study
    'plain_he':         '#7f8c8d',
    'plain_identity':   '#bdc3c7',
    'resnet_id_all':    '#95a5a6',
    # Comparison baselines
    'skipinit':         '#16a085',
    'fixup':            '#1abc9c',
    'layerscale':       '#2ecc71',
    # Fixed-alpha ablation (gradient: light → dark red)
    'fixed_01':         '#fadbd8',
    'fixed_03':         '#f1948a',
    'fixed_05':         '#e74c3c',
    'fixed_07':         '#922b21',
}

MARKERS = {
    'baseline':         'o',
    'lrs_low':          's',
    'lrs_mid':          '^',
    'lrs_high':         'v',
    'lrs_ha_low':       'D',
    'lrs_ha_mid':       'P',
    'lrs_ha_high':      'X',
    'rezero':           'p',
    'hybrida':          'h',
    'plain_he':         'x',
    'plain_identity':   '+',
    'resnet_id_all':    '*',
    'skipinit':         '<',
    'fixup':            '>',
    'layerscale':       'H',
}

LABELS = {
    'baseline':         'Baseline',
    'lrs_low':          r'LRS-Low ($\alpha_0\approx0.12$)',
    'lrs_mid':          r'LRS-Mid ($\alpha_0=0.5$)',
    'lrs_high':         r'LRS-High ($\alpha_0\approx0.88$)',
    'lrs_ha_low':       'LRS+HA-Low (Proposed)',
    'lrs_ha_mid':       'LRS+HA-Mid (Proposed)',
    'lrs_ha_high':      'LRS+HA-High',
    'rezero':           'ReZero',
    'hybrida':          'HybridA',
    'plain_he':         'Plain (He init)',
    'plain_identity':   'Plain (Identity init)',
    'resnet_id_all':    'ResNet (Identity init)',
    'skipinit':         'SkipInit',
    'fixup':            'Fixup',
    'layerscale':       'LayerScale',
    'fixed_alpha_01':   r'Fixed $\alpha=0.1$',
    'fixed_alpha_03':   r'Fixed $\alpha=0.3$',
    'fixed_alpha_05':   r'Fixed $\alpha=0.5$',
    'fixed_alpha_07':   r'Fixed $\alpha=0.7$',
}

# JSON 파일명에서 model_type → JSON key 매핑
JSON_NAMES = {
    'baseline':             'baseline',
    'lrs_low':              'lrs_low',
    'lrs_mid':              'lrs_mid',
    'lrs_high':             'lrs_high',
    'rezero':               'rezero',
    'hybrida':              'hybrida',
    'lrs_ha_low':           'lrs_hybrida_low',
    'lrs_ha_mid':           'lrs_hybrida_mid',
    'lrs_ha_high':          'lrs_hybrida_high',
    'plain_he':             'plain_he',
    'plain_identity':       'plain_identity',
    'resnet_id_all':        'resnet_identity_all',
    'skipinit':             'skipinit',
    'fixup':                'fixup',
    'layerscale':           'layerscale',
    'fixed_alpha_01':       'fixed_alpha_01',
    'fixed_alpha_03':       'fixed_alpha_03',
    'fixed_alpha_05':       'fixed_alpha_05',
    'fixed_alpha_07':       'fixed_alpha_07',
    'per_channel_lrs':      'per_channel_lrs',
}


def get_result_path(model_key, depth, dataset):
    """결과 JSON 파일 경로 반환"""
    json_name = JSON_NAMES.get(model_key, model_key)
    fname = f'{json_name}_depth{depth}_{dataset}_result.json'
    return os.path.join(RESULTS_DIR, fname)


def save_figure(fig, name):
    """PDF + PNG 300dpi 저장"""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    pdf_path = os.path.join(FIGURES_DIR, f'{name}.pdf')
    png_path = os.path.join(FIGURES_DIR, f'{name}.png')
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f'  Saved: {pdf_path}')
    print(f'  Saved: {png_path}')
