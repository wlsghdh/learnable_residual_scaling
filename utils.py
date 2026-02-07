"""
utils.py - 유틸리티 함수
========================
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from config import CONFIG


def setup_directories():
    """필요한 디렉토리 생성"""
    Path(CONFIG['data_dir']).mkdir(exist_ok=True)
    Path(CONFIG['save_dir']).mkdir(exist_ok=True)
    Path(CONFIG['save_dir'] + '/figures').mkdir(exist_ok=True)


def save_results(results, filename):
    """결과를 JSON으로 저장"""
    save_path = Path(CONFIG['save_dir']) / filename
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(i) for i in obj]
        return obj
    
    results_converted = convert(results)
    
    with open(save_path, 'w') as f:
        json.dump(results_converted, f, indent=2)
    
    print(f"Results saved: {save_path}")


def plot_comparison(experiments, save_name='comparison.png'):
    """실험 결과 비교 그래프"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for idx, exp in enumerate(experiments):
        name = exp['name']
        history = exp['results']['history']
        color = colors[idx]
        
        axes[0, 0].plot(history['test_acc'], label=name, color=color, linewidth=2)
        axes[0, 1].plot(history['train_loss'], label=name, color=color, linewidth=2)
        axes[1, 0].plot(history['lr'], label=name, color=color, linewidth=2)
        
        if history['alpha_stats']:
            alpha_means = [s['mean'] for s in history['alpha_stats']]
            axes[1, 1].plot(alpha_means, label=name, color=color, linewidth=2)
    
    axes[0, 0].set_title('Test Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('Train Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('LR')
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].set_title('Alpha (α) Mean')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('α mean')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = Path(CONFIG['save_dir']) / 'figures' / save_name
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {save_path}")


def plot_accuracy_bar(experiments, save_name='accuracy_bar.png'):
    """정확도 막대 그래프"""
    names = [exp['name'].split('_depth')[0] for exp in experiments]
    accs = [exp['results']['best_acc'] * 100 for exp in experiments]
    
    colors = ['#6b7280' if 'Baseline' in n else '#10b981' for n in names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(names)), accs, color=colors)
    
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Best Accuracy (%)')
    ax.set_title('Model Comparison')
    ax.set_ylim(min(accs) - 3, max(accs) + 2)
    
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.3, f'{acc:.2f}%',
                ha='center', fontsize=9)
    
    plt.tight_layout()
    save_path = Path(CONFIG['save_dir']) / 'figures' / save_name
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {save_path}")


def print_summary_table(experiments):
    """결과 요약 테이블"""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Model':<35} {'Best Acc':<12} {'Final Acc':<12} {'Time (min)':<10}")
    print("-"*70)
    
    for exp in sorted(experiments, key=lambda x: x['results']['best_acc'], reverse=True):
        name = exp['name'].split('_depth')[0]
        best_acc = exp['results']['best_acc'] * 100
        final_acc = exp['results']['final_acc'] * 100
        time_min = exp['results']['elapsed_time'] / 60
        print(f"{name:<35} {best_acc:<12.2f} {final_acc:<12.2f} {time_min:<10.1f}")
    
    print("="*70)
    
    best_exp = max(experiments, key=lambda x: x['results']['best_acc'])
    print(f"\n🏆 Best: {best_exp['name']} ({best_exp['results']['best_acc']*100:.2f}%)")


def create_done_file(experiment_name, experiments, elapsed_time):
    """완료 파일 생성"""
    filepath = Path(CONFIG['save_dir']) / f'{experiment_name}_done.md'
    
    with open(filepath, 'w') as f:
        f.write(f"# ✅ {experiment_name} Completed\n\n")
        f.write(f"- **Finished**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- **Total Time**: {elapsed_time/60:.1f} min\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Model | Best Acc | Final Acc |\n")
        f.write(f"|-------|----------|----------|\n")
        
        for exp in sorted(experiments, key=lambda x: x['results']['best_acc'], reverse=True):
            name = exp['name'].split('_depth')[0]
            best = exp['results']['best_acc'] * 100
            final = exp['results']['final_acc'] * 100
            f.write(f"| {name} | {best:.2f}% | {final:.2f}% |\n")
    
    print(f"Done file: {filepath}")
