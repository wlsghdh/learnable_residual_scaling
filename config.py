"""
config.py - 실험 설정 및 하이퍼파라미터
======================================
"""

import torch
import random
import numpy as np

# ============================================================
# 기본 설정
# ============================================================

CONFIG = {
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 4,
    'pin_memory': True,
    
    # 데이터
    'data_dir': './data',
    'save_dir': './results-json',
    
    # 학습 설정 (층수별)
    'training': {
        50: {
            'epochs': 100,
            'batch_size': 128,
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'warmup_epochs': 5,
        },
        101: {
            'epochs': 100,
            'batch_size': 64,
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'warmup_epochs': 5,
        },
        152: {
            'epochs': 100,
            'batch_size': 32,
            'lr': 0.1,
            'momentum': 0.9,
            'weight_decay': 1e-4,
            'warmup_epochs': 5,
        },
        200: {  
        'epochs': 100,
        'batch_size': 32,  
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'warmup_epochs': 5,
        }
        
    }
}

# ResNet 구조 설정
RESNET_CONFIGS = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3],
}


def set_seed(seed=None):
    """재현성을 위한 시드 설정"""
    if seed is None:
        seed = CONFIG['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_training_config(depth):
    """층수에 맞는 학습 설정 반환"""
    if depth not in CONFIG['training']:
        return CONFIG['training'][50]
    return CONFIG['training'][depth]
