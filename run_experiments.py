"""
run_experiments.py - 메인 실험 실행 스크립트
==========================================

Learnable Residual Scaling (LRS) 실험

사용법:
    # 단일 모델, 특정 seed
    python run_experiments.py --dataset cifar10 --depth 50 --model baseline --seed 42

    # 전체 모델 (기존 방식)
    python run_experiments.py --dataset cifar100 --depth 152

    # GPU 지정
    CUDA_VISIBLE_DEVICES=1 python run_experiments.py --dataset cifar100 --depth 152 --model lrs_low --seed 123

    # 백그라운드
    CUDA_VISIBLE_DEVICES=1 nohup python run_experiments.py --dataset cifar10 --depth 50 \\
        --model baseline --seed 42 > logs/baseline_d50_c10_s42.log 2>&1 &
"""

import argparse
import time
import os
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


def print_header(dataset, depth, model_type=None, seed=42):
    """실험 시작 헤더"""
    print("\n" + "=" * 70)
    print(f"Learnable Residual Scaling (LRS) Experiments")
    print(f"Dataset: {dataset.upper()}, Depth: {depth}, Seed: {seed}")
    if model_type:
        print(f"Model: {model_type}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Device: {CONFIG['device']}")

    config = get_training_config(depth)
    print(f"\nTraining Config:")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  LR: {config['lr']} (Cosine Annealing + Warmup {config['warmup_epochs']}ep)")
    print("=" * 70 + "\n")


def get_result_filename(model_type, depth, dataset, seed):
    """결과 파일명 생성
    - seed=42이고 기존 호환이 필요한 경우: {model}_depth{N}_{dataset}_result.json
    - 그 외: {model}_depth{N}_{dataset}_seed{seed}_result.json
    """
    return f"{model_type}_depth{depth}_{dataset}_seed{seed}_result.json"


def run_single_model(dataset, depth, model_type, seed=42, device=None):
    """단일 모델 실험 실행"""
    print_header(dataset, depth, model_type, seed)
    setup_directories()

    config = get_training_config(depth)
    batch_size = config['batch_size']

    # 시드 설정
    set_seed(seed)

    # 데이터 로더
    (train_loader, test_loader), num_classes = get_data_loaders(dataset, batch_size)

    # 모델 생성
    model = create_model(model_type, depth, num_classes)
    total_params, trainable_params = count_parameters(model)
    print(f"Model: {model.model_name}")
    print(f"Parameters: {trainable_params:,}")

    exp_name = f"{model_type}_depth{depth}_{dataset}_seed{seed}"

    # 실험 실행
    result = run_single_experiment(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        depth=depth,
        name=exp_name,
        device=device
    )

    # 결과 저장 (seed 포함 파일명)
    filename = get_result_filename(model_type, depth, dataset, seed)
    save_results(result, filename)

    print(f"\n✅ Done: {exp_name}")
    print(f"   Best Acc: {result['results']['best_acc']:.4f} "
          f"(epoch {result['results']['best_epoch']})")
    print(f"   Saved: results-json/{filename}")

    # GPU 메모리 정리
    del model
    torch.cuda.empty_cache()

    return result


def run_all_experiments(dataset, depth, seed=42, device=None):
    """모든 모델 순차 실행 (기존 방식)"""
    print_header(dataset, depth, seed=seed)
    setup_directories()

    config = get_training_config(depth)
    batch_size = config['batch_size']

    (train_loader, test_loader), num_classes = get_data_loaders(dataset, batch_size)

    # ResNet 기반 모델만 실행 (WRN/ResNeXt 제외)
    resnet_models = [m for m in MODEL_TYPES
                     if not m.startswith('wrn_') and not m.startswith('resnext_')]

    experiments = []
    total_start_time = time.time()

    for idx, model_type in enumerate(resnet_models, 1):
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(resnet_models)}] {model_type}")
        print('=' * 60)

        set_seed(seed)

        model = create_model(model_type, depth, num_classes)
        total_params, trainable_params = count_parameters(model)
        print(f"Model: {model.model_name}")
        print(f"Parameters: {trainable_params:,}")

        exp_name = f"{model_type}_depth{depth}_{dataset}_seed{seed}"

        result = run_single_experiment(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            depth=depth,
            name=exp_name,
            device=device
        )

        experiments.append(result)
        save_results(result, get_result_filename(model_type, depth, dataset, seed))

        del model
        torch.cuda.empty_cache()

    total_elapsed_time = time.time() - total_start_time

    exp_tag = f'{dataset}_depth{depth}_seed{seed}'

    final_results = {
        'experiment_name': f'LRS_{dataset.upper()}_ResNet{depth}_seed{seed}',
        'dataset': dataset,
        'depth': depth,
        'seed': seed,
        'num_classes': num_classes,
        'config': config,
        'total_time_seconds': total_elapsed_time,
        'total_time_minutes': total_elapsed_time / 60,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'experiments': experiments
    }

    save_results(final_results, f'{exp_tag}_all_results.json')
    plot_comparison(experiments, f'{exp_tag}_comparison.png')
    plot_accuracy_bar(experiments, f'{exp_tag}_accuracy_bar.png')
    print_summary_table(experiments)
    create_done_file(exp_tag, experiments, total_elapsed_time)

    print(f"\n✅ All experiments completed!")
    print(f"Total time: {total_elapsed_time / 60:.1f} min ({total_elapsed_time / 3600:.1f} hours)")

    return experiments


def main():
    parser = argparse.ArgumentParser(description='Learnable Residual Scaling Experiments')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--depth', type=int, default=50,
                        choices=[28, 50, 101, 152, 200])
    parser.add_argument('--model', type=str, default=None,
                        help='특정 모델만 실행 (default: 전체). 예: baseline, lrs_low')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 (default: 42)')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None,
                        help='결과 저장 경로 override (default: config.py의 save_dir)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='배치 크기 override (default: config.py 설정 사용)')
    parser.add_argument('--lr', type=float, default=None,
                        help='학습률 override (default: config.py 설정 사용)')
    args = parser.parse_args()

    # save_dir override
    if args.save_dir is not None:
        CONFIG['save_dir'] = args.save_dir

    # batch_size / lr override — config.py 기준값을 덮어씀
    if args.batch_size is not None or args.lr is not None:
        base_key = args.depth if args.depth in CONFIG['training'] else 50
        config_copy = dict(CONFIG['training'][base_key])
        if args.batch_size is not None:
            config_copy['batch_size'] = args.batch_size
        if args.lr is not None:
            config_copy['lr'] = args.lr
        CONFIG['training'][args.depth] = config_copy

    device = args.device or CONFIG['device']

    if args.model:
        # 단일 모델 실행
        mt = args.model.lower()
        # 별칭 처리
        aliases = {'lrs_ha_low': 'lrs_hybrida_low',
                   'lrs_ha_mid': 'lrs_hybrida_mid',
                   'lrs_ha_high': 'lrs_hybrida_high'}
        mt = aliases.get(mt, mt)
        if mt not in MODEL_TYPES:
            print(f"Error: unknown model '{args.model}'")
            print(f"Available: {MODEL_TYPES}")
            return
        run_single_model(args.dataset, args.depth, mt, args.seed, device)
    else:
        # 전체 실행
        run_all_experiments(args.dataset, args.depth, args.seed, device)


if __name__ == '__main__':
    main()
