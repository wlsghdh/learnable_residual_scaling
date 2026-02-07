"""
data.py - 데이터 로딩 및 전처리 (CIFAR-10/100)
=============================================
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import CONFIG


# ============================================================
# CIFAR-10
# ============================================================

def get_cifar10_transforms():
    """CIFAR-10 전처리"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])
    
    return train_transform, test_transform


def get_cifar10_loaders(batch_size=128):
    """CIFAR-10 데이터로더"""
    train_transform, test_transform = get_cifar10_transforms()
    
    train_dataset = datasets.CIFAR10(
        root=CONFIG['data_dir'],
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root=CONFIG['data_dir'],
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    print(f"CIFAR-10 loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    
    return train_loader, test_loader


# ============================================================
# CIFAR-100
# ============================================================

def get_cifar100_transforms():
    """CIFAR-100 전처리"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        ),
    ])
    
    return train_transform, test_transform


def get_cifar100_loaders(batch_size=128):
    """CIFAR-100 데이터로더"""
    train_transform, test_transform = get_cifar100_transforms()
    
    train_dataset = datasets.CIFAR100(
        root=CONFIG['data_dir'],
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=CONFIG['data_dir'],
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    print(f"CIFAR-100 loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    
    return train_loader, test_loader


# ============================================================
# 통합 함수
# ============================================================

def get_data_loaders(dataset='cifar10', batch_size=128):
    """데이터셋에 따라 적절한 로더 반환"""
    if dataset.lower() == 'cifar10':
        return get_cifar10_loaders(batch_size), 10
    elif dataset.lower() == 'cifar100':
        return get_cifar100_loaders(batch_size), 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'cifar10' or 'cifar100'")
