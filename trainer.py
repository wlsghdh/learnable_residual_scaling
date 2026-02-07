"""
trainer.py - 학습 및 평가 로직
==============================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import numpy as np
from tqdm import tqdm
import time

from config import CONFIG, get_training_config
from models import count_parameters


class Trainer:
    """모델 학습 클래스"""
    
    def __init__(self, model, train_loader, test_loader, depth, device=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device or CONFIG['device']
        
        # 층수에 맞는 설정
        self.config = get_training_config(depth)
        self.epochs = self.config['epochs']
        self.warmup_epochs = self.config['warmup_epochs']
        
        # 모델을 디바이스로 이동
        self.model = self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.config['lr'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay'],
            nesterov=True
        )
        
        # Scheduler
        self._setup_scheduler()
        
        # 기록
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'lr': [],
            'alpha_stats': []
        }
        self.best_acc = 0.0
        self.best_epoch = 0
    
    def _setup_scheduler(self):
        """Warmup + Cosine Annealing"""
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_epochs
        )
        
        main_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs - self.warmup_epochs,
            eta_min=1e-6
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.warmup_epochs]
        )
    
    def train_epoch(self):
        """한 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / total, correct / total
    
    @torch.no_grad()
    def evaluate(self):
        """평가"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / total, correct / total
    
    def train(self, experiment_name=""):
        """전체 학습"""
        total_params, trainable_params = count_parameters(self.model)
        
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"Model: {getattr(self.model, 'model_name', 'Unknown')}")
        print(f"Epochs: {self.epochs}, Warmup: {self.warmup_epochs}")
        print(f"Parameters: {trainable_params:,}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        pbar = tqdm(range(1, self.epochs + 1), desc=experiment_name[:30])
        
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch()
            test_loss, test_acc = self.evaluate()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()
            
            # 기록
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['lr'].append(current_lr)
            
            # α 통계
            alpha_stats = self.model.get_alpha_stats() if hasattr(self.model, 'get_alpha_stats') else None
            if alpha_stats:
                self.history['alpha_stats'].append({
                    'epoch': epoch,
                    'mean': alpha_stats['mean'],
                    'std': alpha_stats['std']
                })
            
            # Best 체크
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_epoch = epoch
            
            pbar.set_postfix({
                'train': f'{train_acc:.4f}',
                'test': f'{test_acc:.4f}',
                'best': f'{self.best_acc:.4f}'
            })
        
        elapsed_time = time.time() - start_time
        
        print(f"\n  Best Acc: {self.best_acc:.4f} (epoch {self.best_epoch})")
        print(f"  Time: {elapsed_time/60:.1f} min")
        
        if alpha_stats:
            print(f"  Final α: mean={alpha_stats['mean']:.4f}")
        
        return {
            'best_acc': self.best_acc,
            'best_epoch': self.best_epoch,
            'final_acc': test_acc,
            'history': self.history,
            'elapsed_time': elapsed_time,
            'alpha_final': alpha_stats
        }


def run_single_experiment(model, train_loader, test_loader, depth, name, device=None):
    """단일 실험 실행"""
    trainer = Trainer(model, train_loader, test_loader, depth, device)
    result = trainer.train(experiment_name=name)
    
    return {
        'name': name,
        'model_name': getattr(model, 'model_name', 'Unknown'),
        'depth': depth,
        'results': result,
        'config': trainer.config
    }
