"""
run_experiments.py - 메인 실험 실행 스크립트
==========================================

Learnable Residual Scaling (LRS) 실험

사용법:
    # CIFAR-10, ResNet-50
    python run_experiments.py --dataset cifar10 --depth 50
    
    # CIFAR-100, ResNet-101
    python run_experiments.py --dataset cifar100 --depth 101
    
    # CIFAR-100, ResNet-152, GPU 1
    CUDA_VISIBLE_DEVICES=1 python run_experiments.py --dataset cifar100 --depth 152
    
    # 백그라운드 실행
    CUDA_VISIBLE_DEVICES=0 nohup python run_experiments.py --dataset cifar10 --depth 50 > exp.log 2>&1 &

7가지 모델 비교:
    1. Baseline        - 기존 ResNet (1:1 고정)
    2. LRS_Low         - Learnable Residual Scaling (α≈0.12 시작)
    3. LRS_Mid         - Learnable Residual Scaling (α=0.5 시작)
    4. ReZero          - ReZero (α=0 시작)
    5. HybridA         - Identity 초기화만 (layer3,4)
    6. LRS_HybridA_Low - HybridA + LRS (α≈0.12) ← 최종 제안
    7. LRS_HybridA_Mid - HybridA + LRS (α=0.5) ← 최종 제안
"""

import argparse
import time
from datetime import datetime

import torch

from config import CONFIG, set_seed, get_training_config, RESNET_CONFIGS
from models import create_model, count_parameters, MODEL_TYPES, MODEL_DESCRIPTIONS
from data import get_data_loaders
from trainer import run_single_experiment
from utils import (
    setup_directories, save_results, plot_comparison,
    plot_accuracy_bar, print_summary_table, create_done_file
)


def print_header(dataset, depth):
    """실험 시작 헤더"""
    print("\n" + "="*70)
    print(f"Learnable Residual Scaling (LRS) Experiments")
    print(f"Dataset: {dataset.upper()}, Depth: {depth}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {CONFIG['device']}")
    
    config = get_training_config(depth)
    print(f"\nTraining Config:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  LR: {config['lr']} (Cosine Annealing)")
    print(f"  Warmup: {config['warmup_epochs']} epochs")
    
    print(f"\nResNet-{depth} Structure: {RESNET_CONFIGS[depth]}")
    print(f"\nModels to test: {len(MODEL_TYPES)}")
    for mt in MODEL_TYPES:
        print(f"  - {mt}: {MODEL_DESCRIPTIONS[mt]}")
    print("="*70 + "\n")


def run_all_experiments(dataset, depth, device=None):
    """모든 실험 실행"""
    
    print_header(dataset, depth)
    setup_directories()
    
    # 학습 설정
    config = get_training_config(depth)
    batch_size = config['batch_size']
    
    # 데이터 로더
    (train_loader, test_loader), num_classes = get_data_loaders(dataset, batch_size)
    
    experiments = []
    total_start_time = time.time()
    
    for idx, model_type in enumerate(MODEL_TYPES, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(MODEL_TYPES)}] {model_type}")
        print('='*60)
        
        # 시드 설정
        set_seed()
        
        # 모델 생성
        model = create_model(model_type, depth, num_classes)
        total_params, trainable_params = count_parameters(model)
        print(f"Model: {model.model_name}")
        print(f"Parameters: {trainable_params:,}")
        
        # 실험 이름
        exp_name = f"{model_type}_depth{depth}_{dataset}"
        
        # 실험 실행
        result = run_single_experiment(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            depth=depth,
            name=exp_name,
            device=device
        )
        
        experiments.append(result)
        
        # 중간 저장
        save_results(result, f'{exp_name}_result.json')
        
        # GPU 메모리 정리
        del model
        torch.cuda.empty_cache()
    
    total_elapsed_time = time.time() - total_start_time
    
    # ================================================================
    # 결과 정리
    # ================================================================
    
    exp_tag = f'{dataset}_depth{depth}'
    
    final_results = {
        'experiment_name': f'LRS_{dataset.upper()}_ResNet{depth}',
        'dataset': dataset,
        'depth': depth,
        'num_classes': num_classes,
        'config': config,
        'total_time_seconds': total_elapsed_time,
        'total_time_minutes': total_elapsed_time / 60,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': experiments
    }
    
    save_results(final_results, f'{exp_tag}_all_results.json')
    
    # 그래프
    plot_comparison(experiments, f'{exp_tag}_comparison.png')
    plot_accuracy_bar(experiments, f'{exp_tag}_accuracy_bar.png')
    
    # 요약
    print_summary_table(experiments)
    
    # 완료 파일
    create_done_file(exp_tag, experiments, total_elapsed_time)
    
    print(f"\n✅ All experiments completed!")
    print(f"Total time: {total_elapsed_time/60:.1f} min ({total_elapsed_time/3600:.1f} hours)")
    
    return experiments


def main():
    parser = argparse.ArgumentParser(description='Learnable Residual Scaling Experiments')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset (cifar10 or cifar100)')
    parser.add_argument('--depth', type=int, default=50,
                        choices=[50, 101, 152,200],
                        help='ResNet depth (50, 101,152, or 200)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto)')
    args = parser.parse_args()
    
    device = args.device or CONFIG['device']
    
    run_all_experiments(
        dataset=args.dataset,
        depth=args.depth,
        device=device
    )


if __name__ == '__main__':
    main()
