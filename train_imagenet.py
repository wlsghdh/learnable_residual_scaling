"""
train_imagenet.py - ImageNet ResNet-50 학습 (단일 GPU + AMP)
=============================================================
사용법:
    CUDA_VISIBLE_DEVICES=1 python train_imagenet.py --model baseline
    CUDA_VISIBLE_DEVICES=2 python train_imagenet.py --model lrs_low
    CUDA_VISIBLE_DEVICES=1 python train_imagenet.py --model lrs_ha_low
    CUDA_VISIBLE_DEVICES=2 python train_imagenet.py --model rezero
    CUDA_VISIBLE_DEVICES=1 python train_imagenet.py --model lrs_mid

결과 저장: results-json/imagenet/{model}_imagenet_result.json
"""

import os
import sys
import json
import time
import random
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(__file__))
from models import create_model


# ============================================================
# 설정
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=['baseline', 'lrs_low', 'lrs_mid', 'lrs_high',
                                 'rezero', 'hybrida',
                                 'lrs_ha_low', 'lrs_ha_mid', 'lrs_ha_high',
                                 'highway'],
                        help='모델 종류')
    parser.add_argument('--depth', type=int, default=50,
                        choices=[50, 101, 152, 200],
                        help='ResNet depth (default: 50)')
    parser.add_argument('--data', type=str,
                        default='/nfs_share/datasets/ILSVRC2012',
                        help='ImageNet 데이터 경로')
    parser.add_argument('--output', type=str,
                        default='results-json/imagenet',
                        help='결과 저장 경로')
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=256,
                        help='batch size (single GPU)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-amp', action='store_true', default=False,
                        help='AMP 비활성화')
    parser.add_argument('--save-checkpoint', action='store_true', default=False,
                        help='최고 모델 체크포인트 저장')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 데이터 로더
# ============================================================

def build_dataloaders(data_dir, batch_size, num_workers):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, 'val'), transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


# ============================================================
# Accuracy 계산
# ============================================================

def accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


# ============================================================
# 학습 / 검증
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    total_loss, total_top1, total_top5, n = 0., 0., 0., 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            output = model(images)
            loss = criterion(output, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        top1, top5 = accuracy(output, targets, topk=(1, 5))
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_top1 += top1 * bs
        total_top5 += top5 * bs
        n += bs

    return total_loss / n, total_top1 / n, total_top5 / n


def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss, total_top1, total_top5, n = 0., 0., 0., 0

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                output = model(images)
                loss = criterion(output, targets)
            top1, top5 = accuracy(output, targets, topk=(1, 5))
            bs = images.size(0)
            total_loss += loss.item() * bs
            total_top1 += top1 * bs
            total_top5 += top5 * bs
            n += bs

    return total_loss / n, total_top1 / n, total_top5 / n


def get_alpha_stats(model):
    """학습된 α 통계 반환 (LRS/ReZero/Highway 모델)"""
    alphas = []
    for module in model.modules():
        if hasattr(module, 'residual_scale'):
            alphas.append(torch.sigmoid(module.residual_scale).item())
        elif hasattr(module, 'alpha') and isinstance(module.alpha, nn.Parameter):
            if module.alpha.numel() == 1:
                alphas.append(module.alpha.item())
        elif hasattr(module, 'gate_fc') and hasattr(module, 'conv1'):
            # Highway gate: sigmoid(bias) as default gate value
            alphas.append(torch.sigmoid(module.gate_fc.bias).item())
    if not alphas:
        return None
    return {
        'mean': float(np.mean(alphas)),
        'std': float(np.std(alphas)),
        'min': float(np.min(alphas)),
        'max': float(np.max(alphas)),
        'all': [float(a) for a in alphas],
    }


# ============================================================
# 메인
# ============================================================

def main():
    args = parse_args()
    use_amp = not args.no_amp

    # model alias 처리
    model_alias = {
        'lrs_ha_low': 'lrs_hybrida_low',
        'lrs_ha_mid': 'lrs_hybrida_mid',
        'lrs_ha_high': 'lrs_hybrida_high',
    }
    model_type = model_alias.get(args.model, args.model)

    # 결과 파일 경로
    os.makedirs(args.output, exist_ok=True)
    depth_suffix = f'_d{args.depth}' if args.depth != 50 else ''
    out_path = os.path.join(args.output, f'{args.model}{depth_suffix}_imagenet_result.json')
    if os.path.exists(out_path):
        print(f'[SKIP] Result already exists: {out_path}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.seed)

    print(f'=== ImageNet Training (Single GPU) ===')
    print(f'  Model: {model_type}')
    print(f'  Depth: {args.depth}')
    print(f'  Device: {device}')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Epochs: {args.epochs}')
    print(f'  Batch size: {args.batch_size}')
    print(f'  Data: {args.data}')
    print(f'  AMP: {use_amp}')
    print(f'  Seed: {args.seed}')
    print(f'  Output: {out_path}')
    print()

    # 모델
    model = create_model(model_type, depth=args.depth, num_classes=1000)
    model = model.to(device)
    print(f'  Model name: {model.model_name}')
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {n_params:,}')
    print()

    # 데이터
    train_loader, val_loader = build_dataloaders(
        args.data, args.batch_size, args.workers)
    print(f'  Train batches: {len(train_loader)}')
    print(f'  Val batches:   {len(val_loader)}')
    print()

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    # Scheduler: linear warmup → cosine annealing
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        t = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * t))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = GradScaler(enabled=use_amp)

    # 결과 저장 구조
    history = {
        'train_loss': [], 'train_top1': [], 'train_top5': [],
        'val_loss': [], 'val_top1': [], 'val_top5': [],
        'lr': [], 'alpha_stats': [],
    }

    best_top1 = 0.0
    best_top5 = 0.0
    best_epoch = 0
    start_time = time.time()

    ckpt_dir = os.path.join(args.output, 'checkpoints')
    if args.save_checkpoint:
        os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t_loss, t_top1, t_top5 = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, use_amp)

        v_loss, v_top1, v_top5 = validate(
            model, val_loader, criterion, device, use_amp)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # α 통계
        alpha_stats = get_alpha_stats(model)

        history['train_loss'].append(round(t_loss, 6))
        history['train_top1'].append(round(t_top1, 4))
        history['train_top5'].append(round(t_top5, 4))
        history['val_loss'].append(round(v_loss, 6))
        history['val_top1'].append(round(v_top1, 4))
        history['val_top5'].append(round(v_top5, 4))
        history['lr'].append(current_lr)

        if alpha_stats:
            history['alpha_stats'].append({
                'epoch': epoch,
                'mean': alpha_stats['mean'],
                'std': alpha_stats['std'],
            })

        if v_top1 > best_top1:
            best_top1 = v_top1
            best_top5 = v_top5
            best_epoch = epoch
            if args.save_checkpoint:
                ckpt_path = os.path.join(ckpt_dir, f'{args.model}_best.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_top1': best_top1,
                }, ckpt_path)

        elapsed = time.time() - start_time
        eta = elapsed / epoch * (args.epochs - epoch)
        alpha_str = (f'  α={alpha_stats["mean"]:.4f}' if alpha_stats else '')
        print(f'Ep[{epoch:3d}/{args.epochs}] '
              f'Train: {t_top1:.2f}% | Val: {v_top1:.2f}% (Top5: {v_top5:.2f}%) '
              f'| Best: {best_top1:.2f}%@{best_epoch} '
              f'| LR: {current_lr:.5f}'
              f'{alpha_str} '
              f'| ETA: {datetime.timedelta(seconds=int(eta))}',
              flush=True)

    # 결과 저장
    elapsed_total = time.time() - start_time
    alpha_final = get_alpha_stats(model)

    result = {
        'name': f'{args.model}_d{args.depth}_imagenet',
        'model_name': getattr(model, 'model_name', model_type),
        'depth': args.depth,
        'dataset': 'imagenet',
        'results': {
            'best_acc': round(best_top1 / 100, 6),
            'best_top1': round(best_top1, 4),
            'best_top5': round(best_top5, 4),
            'best_epoch': best_epoch,
            'final_top1': round(history['val_top1'][-1], 4),
            'final_top5': round(history['val_top5'][-1], 4),
            'history': history,
            'elapsed_time': round(elapsed_total, 2),
            'alpha_final': alpha_final,
        },
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'warmup_epochs': args.warmup_epochs,
            'amp': use_amp,
            'seed': args.seed,
        },
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\n=== DONE ===')
    print(f'Best Top-1: {best_top1:.2f}%  Top-5: {best_top5:.2f}%  @ epoch {best_epoch}')
    print(f'Elapsed: {datetime.timedelta(seconds=int(elapsed_total))}')
    print(f'Result saved: {out_path}')


if __name__ == '__main__':
    main()
