"""
models.py - Learnable Residual Scaling (LRS) 모델 정의
=====================================================

7가지 핵심 모델:
1. Baseline        - 기존 ResNet (1:1 고정)
2. LRS_Low         - Learnable Residual Scaling (α≈0.12 시작)
3. LRS_Mid         - Learnable Residual Scaling (α=0.5 시작)
4. ReZero          - ReZero 방식 (α=0 시작)
5. HybridA         - Identity 초기화 (layer3,4)
6. LRS_HybridA_Low - HybridA + LRS (α≈0.12 시작) ← 최종 제안
7. LRS_HybridA_Mid - HybridA + LRS (α=0.5 시작) ← 최종 제안
"""

import torch
import torch.nn as nn
import numpy as np
from config import RESNET_CONFIGS


# ============================================================
# 초기화 함수
# ============================================================

def he_init(m):
    """He 초기화"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def identity_init_conv(weight):
    """Conv layer를 Identity로 초기화"""
    nn.init.zeros_(weight)
    out_c, in_c, kh, kw = weight.shape
    ch, cw = kh // 2, kw // 2
    min_c = min(out_c, in_c)
    for i in range(min_c):
        weight.data[i, i, ch, cw] = 1.0


def identity_init(m):
    """모듈을 Identity로 초기화"""
    if isinstance(m, nn.Conv2d):
        identity_init_conv(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# ============================================================
# Bottleneck 블록
# ============================================================

class Bottleneck(nn.Module):
    """기본 Bottleneck (Baseline)"""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = out + identity  # 1:1 고정
        out = self.relu(out)
        return out


class LRSBottleneck(nn.Module):
    """Learnable Residual Scaling Bottleneck
    
    y = α * F(x) + (1 - α) * x
    α = sigmoid(scale)
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, init_scale=-2.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
        # Learnable residual scale
        # init_scale=-2.0 → sigmoid(-2) ≈ 0.12 (identity 선호)
        # init_scale=0.0  → sigmoid(0) = 0.5 (균형)
        self.residual_scale = nn.Parameter(torch.tensor(init_scale))
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Learnable weighted combination
        alpha = torch.sigmoid(self.residual_scale)
        out = alpha * out + (1 - alpha) * identity
        out = self.relu(out)
        return out


class ReZeroBottleneck(nn.Module):
    """ReZero Bottleneck
    
    y = x + α * F(x), α starts at 0
    """
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        
        # ReZero: α starts at 0
        self.alpha = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = identity + self.alpha * out
        out = self.relu(out)
        return out


# ============================================================
# ResNet 모델
# ============================================================

class ResNet(nn.Module):
    """ResNet Base Class"""
    
    def __init__(self, block, layers, num_classes=10, init_scale=-2.0):
        super().__init__()
        self.in_channels = 64
        self.block = block
        self.init_scale = init_scale
        
        # CIFAR용 첫 번째 conv (3x3, stride=1)
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # 초기화
        self._init_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.block.expansion,
                          1, stride, bias=False),
                nn.BatchNorm2d(out_channels * self.block.expansion)
            )
        
        layers = []
        
        # 블록 생성 (LRS 블록이면 init_scale 전달)
        if self.block == LRSBottleneck:
            layers.append(self.block(self.in_channels, out_channels, stride,
                                     downsample, init_scale=self.init_scale))
        else:
            layers.append(self.block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels * self.block.expansion
        
        for _ in range(1, num_blocks):
            if self.block == LRSBottleneck:
                layers.append(self.block(self.in_channels, out_channels,
                                        init_scale=self.init_scale))
            else:
                layers.append(self.block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """He 초기화"""
        for m in self.modules():
            he_init(m)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def get_alpha_stats(self):
        """학습된 α 통계 반환"""
        alphas = []
        for module in self.modules():
            if hasattr(module, 'residual_scale'):
                alphas.append(torch.sigmoid(module.residual_scale).item())
            elif hasattr(module, 'alpha'):
                alphas.append(module.alpha.item())
        
        if alphas:
            return {
                'mean': np.mean(alphas),
                'std': np.std(alphas),
                'min': np.min(alphas),
                'max': np.max(alphas),
                'all': alphas
            }
        return None


class ResNetHybridA(ResNet):
    """HybridA: layer3, layer4를 Identity로 초기화"""
    
    def _init_weights(self):
        """Hybrid 초기화: layer1,2=He, layer3,4=Identity"""
        for name, m in self.named_modules():
            if 'layer3' in name or 'layer4' in name:
                identity_init(m)
            else:
                he_init(m)


# ============================================================
# 모델 생성 함수
# ============================================================

def create_model(model_type, depth=50, num_classes=10):
    """
    모델 생성 함수
    
    Args:
        model_type: 모델 종류
            - 'baseline': 기존 ResNet
            - 'lrs_low': LRS (α≈0.12 시작)
            - 'lrs_mid': LRS (α=0.5 시작)
            - 'rezero': ReZero
            - 'hybrida': HybridA 초기화
            - 'lrs_hybrida_low': HybridA + LRS (α≈0.12)
            - 'lrs_hybrida_mid': HybridA + LRS (α=0.5)
        depth: ResNet 깊이 (50, 101, 152)
        num_classes: 클래스 수 (CIFAR-10=10, CIFAR-100=100)
    
    Returns:
        model: PyTorch 모델
    """
    layers = RESNET_CONFIGS[depth]
    
    model_type = model_type.lower()
    
    if model_type == 'baseline':
        model = ResNet(Bottleneck, layers, num_classes)
        model.model_name = f'Baseline_ResNet{depth}'
    
    elif model_type == 'lrs_low':
        model = ResNet(LRSBottleneck, layers, num_classes, init_scale=-2.0)
        model.model_name = f'LRS_Low_ResNet{depth}'
    
    elif model_type == 'lrs_mid':
        model = ResNet(LRSBottleneck, layers, num_classes, init_scale=0.0)
        model.model_name = f'LRS_Mid_ResNet{depth}'
    
    elif model_type == 'rezero':
        model = ResNet(ReZeroBottleneck, layers, num_classes)
        model.model_name = f'ReZero_ResNet{depth}'
    
    elif model_type == 'hybrida':
        model = ResNetHybridA(Bottleneck, layers, num_classes)
        model.model_name = f'HybridA_ResNet{depth}'
    
    elif model_type == 'lrs_hybrida_low':
        model = ResNetHybridA(LRSBottleneck, layers, num_classes, init_scale=-2.0)
        model.model_name = f'LRS_HybridA_Low_ResNet{depth}'
    
    elif model_type == 'lrs_hybrida_mid':
        model = ResNetHybridA(LRSBottleneck, layers, num_classes, init_scale=0.0)
        model.model_name = f'LRS_HybridA_Mid_ResNet{depth}'
    
    elif model_type == 'lrs_high':
        model = ResNet(LRSBottleneck, layers, num_classes, init_scale=2.0)
        model.model_name = f'LRS_High_ResNet{depth}'
    
    elif model_type == 'lrs_hybrida_high':
        model = ResNetHybridA(LRSBottleneck, layers, num_classes, init_scale=2.0)
        model.model_name = f'LRS_HybridA_High_ResNet{depth}'
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def count_parameters(model):
    """파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# 모델 목록
# ============================================================

MODEL_TYPES = [
    'baseline',
    'lrs_low',
    'lrs_mid',
    'lrs_high',
    'rezero',
    'hybrida',
    'lrs_hybrida_low',
    'lrs_hybrida_mid',
    'lrs_hybrida_high',
]

MODEL_DESCRIPTIONS = {
    'baseline': 'Baseline ResNet (1:1 fixed)',
    'lrs_low': 'LRS (α≈0.12 start, identity-favored)',
    'lrs_mid': 'LRS (α=0.5 start, balanced)',
    'lrs_high': 'LRS (α≈0.88 start, transform-favored)',
    'rezero': 'ReZero (α=0 start)',
    'hybrida': 'HybridA Init (layer3,4=Identity)',
    'lrs_hybrida_low': 'LRS + HybridA (α≈0.12) [Proposed]',
    'lrs_hybrida_mid': 'LRS + HybridA (α=0.5) [Proposed]',
    'lrs_hybrida_high': 'LRS + HybridA (α≈0.88) [Proposed]',
}
