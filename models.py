"""
models.py - Learnable Residual Scaling (LRS) + 비교 모델 정의
=============================================================

[ResNet 기반 — 20가지]
  baseline, lrs_low, lrs_mid, lrs_high
  rezero, skipinit, fixup, layerscale
  hybrida, lrs_hybrida_low, lrs_hybrida_mid, lrs_hybrida_high
  fixed_alpha_01/03/05/07, per_channel_lrs
  plain_he, plain_identity, resnet_identity_all

[WRN-28-10 기반 — 6가지]
  wrn_baseline, wrn_lrs_low, wrn_lrs_hybrida_low
  wrn_rezero, wrn_skipinit, wrn_layerscale

[ResNeXt-50 (32×4d) 기반 — 6가지]
  resnext_baseline, resnext_lrs_low, resnext_lrs_hybrida_low
  resnext_rezero, resnext_skipinit, resnext_layerscale
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
# ResNet Bottleneck 블록들
# ============================================================

class Bottleneck(nn.Module):
    """기본 Bottleneck (Baseline)
    y = F(x) + x
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
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
        out = out + identity
        out = self.relu(out)
        return out


class LRSBottleneck(nn.Module):
    """Learnable Residual Scaling Bottleneck
    y = α·F(x) + (1−α)·x,  α = sigmoid(θ),  per-block learnable
    Pruning: set module._pruned = True to skip F(x) computation entirely
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 init_scale=-2.0, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, x):
        # Short-circuit for pruned blocks: skip F(x) computation entirely
        if getattr(self, '_pruned', False):
            identity = self.downsample(x) if self.downsample is not None else x
            return self.relu(identity)

        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        alpha = torch.sigmoid(self.residual_scale)
        out = alpha * out + (1 - alpha) * identity
        out = self.relu(out)
        return out


class ReZeroBottleneck(nn.Module):
    """ReZero Bottleneck (Bachlechner et al., 2020)
    y = x + α·F(x),  α₀ = 0
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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


class SkipInitBottleneck(nn.Module):
    """SkipInit Bottleneck (De & Smith, 2020)
    y = x + α·F(x),  α₀ = 0  (BN 유지, fair comparison)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
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


class FixupBottleneck(nn.Module):
    """Fixup Bottleneck (Zhang et al., 2019)
    y = x + fixup_scale·F(x),  bn3.weight=0 at init → F(x)≈0 at start
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.fixup_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = identity + self.fixup_scale * out
        out = self.relu(out)
        return out


class LayerScaleBottleneck(nn.Module):
    """LayerScale CNN Bottleneck (Touvron et al., 2021)
    y = x + diag(λ)·F(x),  λ per-channel,  λ₀=1e-4
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.layer_scale = nn.Parameter(
            torch.full((out_channels * self.expansion,), 1e-4)
        )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = identity + self.layer_scale.view(1, -1, 1, 1) * out
        out = self.relu(out)
        return out


class FixedAlphaBottleneck(nn.Module):
    """Fixed Alpha Bottleneck (non-learnable ablation)
    y = α·F(x) + (1−α)·x,  α 고정 (nn.Parameter 아님)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 alpha=0.5, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.fixed_alpha = alpha  # plain float, not nn.Parameter

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.fixed_alpha * out + (1 - self.fixed_alpha) * identity
        out = self.relu(out)
        return out


class PerChannelLRSBottleneck(nn.Module):
    """Per-Channel LRS Bottleneck
    y = α_c·F(x) + (1−α_c)·x,  α_c per-channel learnable
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 init_scale=-2.0, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        out_ch = out_channels * self.expansion
        self.residual_scale = nn.Parameter(torch.full((out_ch,), float(init_scale)))

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        alpha = torch.sigmoid(self.residual_scale).view(1, -1, 1, 1)
        out = alpha * out + (1 - alpha) * identity
        out = self.relu(out)
        return out


class HighwayBottleneck(nn.Module):
    """Highway-style Bottleneck (Srivastava et al., 2015)
    y = T(x)·F(x) + (1−T(x))·x,  T(x) = sigmoid(W_g·GAP(x) + b_g)
    Input-dependent gate vs LRS's input-independent scalar.
    GAP(x)를 써서 spatial dimension을 처리하고,
    gate는 per-block scalar로 출력하여 LRS와 공정 비교.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 init_bias=-2.0, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        # Highway gate: input-dependent scalar
        # GAP → Linear → Sigmoid で per-block scalar gate を生成
        gate_in_ch = in_channels
        self.gate_fc = nn.Linear(gate_in_ch, 1)
        nn.init.zeros_(self.gate_fc.weight)
        nn.init.constant_(self.gate_fc.bias, init_bias)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)

        # Input-dependent gate: GAP → FC → sigmoid
        gate_input = x.mean(dim=[2, 3])  # (B, C)
        alpha = torch.sigmoid(self.gate_fc(gate_input))  # (B, 1)
        alpha = alpha.view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        out = alpha * out + (1 - alpha) * identity
        out = self.relu(out)
        return out


class PlainBottleneck(nn.Module):
    """Plain Bottleneck (skip connection 없음)
    y = F(x)
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.relu(out)
        return out


# ============================================================
# ResNet 모델 클래스
# ============================================================

class ResNet(nn.Module):
    """ResNet Base Class
    - CIFAR (num_classes<=200): 3×3 stride=1 stem, no maxpool
    - ImageNet (num_classes==1000): 7×7 stride=2 + maxpool stem (He et al. 2016)
    """

    def __init__(self, block, layers, num_classes=10, init_kwargs=None):
        super().__init__()
        self.in_channels = 64
        self.block = block
        self.init_kwargs = init_kwargs or {}
        self.imagenet_stem = (num_classes == 1000)

        if self.imagenet_stem:
            # Standard ImageNet stem: 7x7 stride=2 + maxpool stride=2 → 56x56
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # CIFAR stem: 3x3 stride=1, no maxpool
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64,  layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * self.block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * self.block.expansion,
                          1, stride, bias=False),
                nn.BatchNorm2d(out_channels * self.block.expansion)
            )

        layers = [self.block(self.in_channels, out_channels, stride, downsample,
                             **self.init_kwargs)]
        self.in_channels = out_channels * self.block.expansion

        for _ in range(1, num_blocks):
            layers.append(self.block(self.in_channels, out_channels,
                                     **self.init_kwargs))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            he_init(m)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.imagenet_stem:
            x = self.maxpool(x)
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
            if hasattr(module, 'residual_scale') and isinstance(module.residual_scale, nn.Parameter):
                s = module.residual_scale
                if s.dim() == 0:
                    alphas.append(torch.sigmoid(s).item())
                else:
                    alphas.extend(torch.sigmoid(s).tolist())
            elif hasattr(module, 'alpha') and isinstance(module.alpha, nn.Parameter):
                alphas.append(module.alpha.item())
            elif hasattr(module, 'fixup_scale') and isinstance(module.fixup_scale, nn.Parameter):
                alphas.append(module.fixup_scale.item())
            elif hasattr(module, 'layer_scale') and isinstance(module.layer_scale, nn.Parameter):
                alphas.extend(module.layer_scale.tolist())
            elif hasattr(module, 'gate_fc') and isinstance(module, HighwayBottleneck):
                # Highway: sigmoid(bias)를 default gate value로 보고
                alphas.append(torch.sigmoid(module.gate_fc.bias).item())

        if alphas:
            return {
                'mean': float(np.mean(alphas)),
                'std': float(np.std(alphas)),
                'min': float(np.min(alphas)),
                'max': float(np.max(alphas)),
                'all': alphas
            }
        return None


class ResNetHybridA(ResNet):
    """HybridA: layer3, layer4를 Identity로 초기화"""

    def _init_weights(self):
        for name, m in self.named_modules():
            if 'layer3' in name or 'layer4' in name:
                identity_init(m)
            else:
                he_init(m)


class ResNetIdentityAll(ResNet):
    """Identity All: 모든 층을 Identity로 초기화"""

    def _init_weights(self):
        for m in self.modules():
            identity_init(m)


class ResNetFixup(ResNet):
    """Fixup ResNet: bn3.weight를 0으로 초기화 (F(x)≈0 at start)"""

    def _init_weights(self):
        for m in self.modules():
            he_init(m)
        # Fixup: 각 블록의 마지막 BN weight를 0으로
        for module in self.modules():
            if isinstance(module, FixupBottleneck):
                nn.init.zeros_(module.bn3.weight)


class ResNetHighway(ResNet):
    """Highway ResNet: he_init 후 gate_fc bias를 복원"""

    def _init_weights(self):
        for m in self.modules():
            he_init(m)
        # he_init이 gate_fc.bias를 0으로 리셋하므로,
        # init_kwargs의 init_bias로 복원
        init_bias = self.init_kwargs.get('init_bias', -2.0)
        for module in self.modules():
            if isinstance(module, HighwayBottleneck):
                nn.init.zeros_(module.gate_fc.weight)
                nn.init.constant_(module.gate_fc.bias, init_bias)


class ResNetPlain(nn.Module):
    """Plain Network (skip connection 없음, CIFAR용)
    Pilot Study: plain_he, plain_identity
    """

    def __init__(self, layers, num_classes=10, use_identity_init=False):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64,  layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * PlainBottleneck.expansion, num_classes)

        if use_identity_init:
            self.apply(identity_init)
        else:
            self.apply(he_init)

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = [PlainBottleneck(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels * PlainBottleneck.expansion
        for _ in range(1, num_blocks):
            layers.append(PlainBottleneck(self.in_channels, out_channels))
        return nn.Sequential(*layers)

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
        return None


# ============================================================
# Wide ResNet (WRN-28-10)
# ============================================================

class WideBlock(nn.Module):
    """Wide ResNet Block (pre-activation: BN-ReLU-Conv)
    variant: baseline | lrs | rezero | skipinit | layerscale
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 dropout=0.0, variant='baseline', init_scale=-2.0, **kwargs):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.variant = variant

        if variant == 'lrs':
            self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))
        elif variant in ('rezero', 'skipinit'):
            self.alpha = nn.Parameter(torch.zeros(1))
        elif variant == 'layerscale':
            self.layer_scale = nn.Parameter(torch.full((out_channels,), 1e-4))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(self.relu(self.bn1(x)))
        if self.dropout:
            out = self.dropout(out)
        out = self.conv2(self.relu(self.bn2(out)))

        if self.variant == 'lrs':
            alpha = torch.sigmoid(self.residual_scale)
            out = alpha * out + (1 - alpha) * identity
        elif self.variant in ('rezero', 'skipinit'):
            out = identity + self.alpha * out
        elif self.variant == 'layerscale':
            out = identity + self.layer_scale.view(1, -1, 1, 1) * out
        else:
            out = out + identity
        return out


class WideResNet(nn.Module):
    """Wide ResNet (WRN-depth-widen_factor)
    Default: WRN-28-10
    """

    def __init__(self, depth=28, widen_factor=10, num_classes=10,
                 dropout=0.0, variant='baseline', init_scale=-2.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth must be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor
        widths = [16, 16 * k, 32 * k, 64 * k]

        self.in_channels = widths[0]
        self.variant = variant

        self.conv1 = nn.Conv2d(3, widths[0], 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(widths[1], n, stride=1, dropout=dropout,
                                       variant=variant, init_scale=init_scale)
        self.layer2 = self._make_layer(widths[2], n, stride=2, dropout=dropout,
                                       variant=variant, init_scale=init_scale)
        self.layer3 = self._make_layer(widths[3], n, stride=2, dropout=dropout,
                                       variant=variant, init_scale=init_scale)
        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3], num_classes)

        self._init_weights(variant)

    def _make_layer(self, out_channels, num_blocks, stride, dropout, variant, init_scale):
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            downsample = None
            if s != 1 or self.in_channels != out_channels:
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channels, 1, s, bias=False)
                )
            layers.append(WideBlock(self.in_channels, out_channels, s, downsample,
                                    dropout, variant, init_scale))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def _init_weights(self, variant):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

        # HybridA: layer2, layer3을 Identity로
        if variant == 'lrs_hybrida':
            for name, m in self.named_modules():
                if 'layer2' in name or 'layer3' in name:
                    identity_init(m)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_alpha_stats(self):
        alphas = []
        for module in self.modules():
            if hasattr(module, 'residual_scale') and isinstance(module.residual_scale, nn.Parameter):
                s = module.residual_scale
                alphas.append(torch.sigmoid(s).item() if s.dim() == 0 else
                               torch.sigmoid(s).mean().item())
            elif hasattr(module, 'alpha') and isinstance(module.alpha, nn.Parameter):
                alphas.append(module.alpha.item())
            elif hasattr(module, 'layer_scale') and isinstance(module.layer_scale, nn.Parameter):
                alphas.extend(module.layer_scale.tolist())
        if alphas:
            return {'mean': float(np.mean(alphas)), 'std': float(np.std(alphas)),
                    'min': float(np.min(alphas)), 'max': float(np.max(alphas)), 'all': alphas}
        return None


# ============================================================
# ResNeXt-50 (32×4d)
# ============================================================

class ResNeXtBottleneck(nn.Module):
    """ResNeXt Bottleneck (32×4d)
    Grouped convolution in the 3×3 stage
    variant: baseline | lrs | rezero | skipinit | layerscale
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 groups=32, width_per_group=4,
                 variant='baseline', init_scale=-2.0, **kwargs):
        super().__init__()
        width = int(out_channels * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride, 1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.variant = variant

        if variant == 'lrs':
            self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))
        elif variant in ('rezero', 'skipinit'):
            self.alpha = nn.Parameter(torch.zeros(1))
        elif variant == 'layerscale':
            self.layer_scale = nn.Parameter(
                torch.full((out_channels * self.expansion,), 1e-4)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)

        if self.variant == 'lrs':
            alpha = torch.sigmoid(self.residual_scale)
            out = alpha * out + (1 - alpha) * identity
        elif self.variant in ('rezero', 'skipinit'):
            out = identity + self.alpha * out
        elif self.variant == 'layerscale':
            out = identity + self.layer_scale.view(1, -1, 1, 1) * out
        else:
            out = out + identity
        out = self.relu(out)
        return out


class ResNeXtNet(nn.Module):
    """ResNeXt for CIFAR (3×3 first conv, no maxpool)"""

    def __init__(self, layers, num_classes=10, groups=32, width_per_group=4,
                 variant='baseline', init_scale=-2.0):
        super().__init__()
        self.in_channels = 64
        self.groups = groups
        self.width_per_group = width_per_group
        self.variant = variant
        self.init_scale = init_scale

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64,  layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResNeXtBottleneck.expansion, num_classes)

        self._init_weights(variant)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * ResNeXtBottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * ResNeXtBottleneck.expansion,
                          1, stride, bias=False),
                nn.BatchNorm2d(out_channels * ResNeXtBottleneck.expansion)
            )

        layers = [ResNeXtBottleneck(
            self.in_channels, out_channels, stride, downsample,
            self.groups, self.width_per_group, self.variant, self.init_scale
        )]
        self.in_channels = out_channels * ResNeXtBottleneck.expansion

        for _ in range(1, num_blocks):
            layers.append(ResNeXtBottleneck(
                self.in_channels, out_channels, 1, None,
                self.groups, self.width_per_group, self.variant, self.init_scale
            ))
        return nn.Sequential(*layers)

    def _init_weights(self, variant):
        for m in self.modules():
            he_init(m)
        if variant == 'lrs_hybrida':
            for name, m in self.named_modules():
                if 'layer3' in name or 'layer4' in name:
                    identity_init(m)

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
        alphas = []
        for module in self.modules():
            if hasattr(module, 'residual_scale') and isinstance(module.residual_scale, nn.Parameter):
                s = module.residual_scale
                alphas.append(torch.sigmoid(s).item())
            elif hasattr(module, 'alpha') and isinstance(module.alpha, nn.Parameter):
                alphas.append(module.alpha.item())
            elif hasattr(module, 'layer_scale') and isinstance(module.layer_scale, nn.Parameter):
                alphas.extend(module.layer_scale.tolist())
        if alphas:
            return {'mean': float(np.mean(alphas)), 'std': float(np.std(alphas)),
                    'min': float(np.min(alphas)), 'max': float(np.max(alphas)), 'all': alphas}
        return None


# ============================================================
# 모델 생성 함수
# ============================================================

def create_model(model_type, depth=50, num_classes=10):
    """
    모델 생성 함수

    Args:
        model_type (str): 모델 종류 (MODEL_TYPES 참조)
        depth (int): ResNet depth (50, 101, 152, 200)
                     WRN/ResNeXt는 depth 무시 (고정 구조)
        num_classes (int): 클래스 수

    Returns:
        model: PyTorch 모델 (model.model_name 속성 포함)
    """
    mt = model_type.lower()
    layers = RESNET_CONFIGS.get(depth, RESNET_CONFIGS[50])

    # Pilot Study: Plain networks
    if mt == 'plain_he':
        model = ResNetPlain(layers, num_classes, use_identity_init=False)
        model.model_name = f'PlainHe_ResNet{depth}'

    elif mt == 'plain_identity':
        model = ResNetPlain(layers, num_classes, use_identity_init=True)
        model.model_name = f'PlainIdentity_ResNet{depth}'

    elif mt == 'resnet_identity_all':
        model = ResNetIdentityAll(Bottleneck, layers, num_classes)
        model.model_name = f'IdentityAll_ResNet{depth}'

    # Baseline
    elif mt == 'baseline':
        model = ResNet(Bottleneck, layers, num_classes)
        model.model_name = f'Baseline_ResNet{depth}'

    # LRS variants
    elif mt == 'lrs_low':
        model = ResNet(LRSBottleneck, layers, num_classes,
                       init_kwargs={'init_scale': -2.0})
        model.model_name = f'LRS_Low_ResNet{depth}'

    elif mt == 'lrs_mid':
        model = ResNet(LRSBottleneck, layers, num_classes,
                       init_kwargs={'init_scale': 0.0})
        model.model_name = f'LRS_Mid_ResNet{depth}'

    elif mt == 'lrs_high':
        model = ResNet(LRSBottleneck, layers, num_classes,
                       init_kwargs={'init_scale': 2.0})
        model.model_name = f'LRS_High_ResNet{depth}'

    # ReZero / SkipInit
    elif mt == 'rezero':
        model = ResNet(ReZeroBottleneck, layers, num_classes)
        model.model_name = f'ReZero_ResNet{depth}'

    elif mt == 'skipinit':
        model = ResNet(SkipInitBottleneck, layers, num_classes)
        model.model_name = f'SkipInit_ResNet{depth}'

    # Fixup
    elif mt == 'fixup':
        model = ResNetFixup(FixupBottleneck, layers, num_classes)
        model.model_name = f'Fixup_ResNet{depth}'

    # LayerScale
    elif mt == 'layerscale':
        model = ResNet(LayerScaleBottleneck, layers, num_classes)
        model.model_name = f'LayerScale_ResNet{depth}'

    # HybridA variants
    elif mt == 'hybrida':
        model = ResNetHybridA(Bottleneck, layers, num_classes)
        model.model_name = f'HybridA_ResNet{depth}'

    elif mt in ('lrs_hybrida_low', 'lrs_ha_low'):
        model = ResNetHybridA(LRSBottleneck, layers, num_classes,
                              init_kwargs={'init_scale': -2.0})
        model.model_name = f'LRS_HybridA_Low_ResNet{depth}'

    elif mt in ('lrs_hybrida_mid', 'lrs_ha_mid'):
        model = ResNetHybridA(LRSBottleneck, layers, num_classes,
                              init_kwargs={'init_scale': 0.0})
        model.model_name = f'LRS_HybridA_Mid_ResNet{depth}'

    elif mt in ('lrs_hybrida_high', 'lrs_ha_high'):
        model = ResNetHybridA(LRSBottleneck, layers, num_classes,
                              init_kwargs={'init_scale': 2.0})
        model.model_name = f'LRS_HybridA_High_ResNet{depth}'

    # Fixed-α ablation
    elif mt == 'fixed_alpha_01':
        model = ResNet(FixedAlphaBottleneck, layers, num_classes,
                       init_kwargs={'alpha': 0.1})
        model.model_name = f'FixedAlpha0.1_ResNet{depth}'

    elif mt == 'fixed_alpha_03':
        model = ResNet(FixedAlphaBottleneck, layers, num_classes,
                       init_kwargs={'alpha': 0.3})
        model.model_name = f'FixedAlpha0.3_ResNet{depth}'

    elif mt == 'fixed_alpha_05':
        model = ResNet(FixedAlphaBottleneck, layers, num_classes,
                       init_kwargs={'alpha': 0.5})
        model.model_name = f'FixedAlpha0.5_ResNet{depth}'

    elif mt == 'fixed_alpha_07':
        model = ResNet(FixedAlphaBottleneck, layers, num_classes,
                       init_kwargs={'alpha': 0.7})
        model.model_name = f'FixedAlpha0.7_ResNet{depth}'

    # Per-channel LRS
    elif mt == 'per_channel_lrs':
        model = ResNet(PerChannelLRSBottleneck, layers, num_classes,
                       init_kwargs={'init_scale': -2.0})
        model.model_name = f'PerChannelLRS_ResNet{depth}'

    # Highway gate (input-dependent)
    elif mt == 'highway':
        model = ResNetHighway(HighwayBottleneck, layers, num_classes,
                              init_kwargs={'init_bias': -2.0})
        model.model_name = f'Highway_ResNet{depth}'

    # ── WRN-28-10 기반 ────────────────────────────────────────
    elif mt == 'wrn_baseline':
        model = WideResNet(28, 10, num_classes, variant='baseline')
        model.model_name = 'WRN28x10_Baseline'

    elif mt == 'wrn_lrs_low':
        model = WideResNet(28, 10, num_classes, variant='lrs', init_scale=-2.0)
        model.model_name = 'WRN28x10_LRS_Low'

    elif mt == 'wrn_lrs_hybrida_low':
        model = WideResNet(28, 10, num_classes, variant='lrs_hybrida', init_scale=-2.0)
        model.model_name = 'WRN28x10_LRS_HybridA_Low'

    elif mt == 'wrn_rezero':
        model = WideResNet(28, 10, num_classes, variant='rezero')
        model.model_name = 'WRN28x10_ReZero'

    elif mt == 'wrn_skipinit':
        model = WideResNet(28, 10, num_classes, variant='skipinit')
        model.model_name = 'WRN28x10_SkipInit'

    elif mt == 'wrn_layerscale':
        model = WideResNet(28, 10, num_classes, variant='layerscale')
        model.model_name = 'WRN28x10_LayerScale'

    # ── ResNeXt-50 (32×4d) 기반 ──────────────────────────────
    elif mt == 'resnext_baseline':
        model = ResNeXtNet(layers, num_classes, variant='baseline')
        model.model_name = f'ResNeXt50_Baseline'

    elif mt == 'resnext_lrs_low':
        model = ResNeXtNet(layers, num_classes, variant='lrs', init_scale=-2.0)
        model.model_name = f'ResNeXt50_LRS_Low'

    elif mt == 'resnext_lrs_hybrida_low':
        model = ResNeXtNet(layers, num_classes, variant='lrs_hybrida', init_scale=-2.0)
        model.model_name = f'ResNeXt50_LRS_HybridA_Low'

    elif mt == 'resnext_rezero':
        model = ResNeXtNet(layers, num_classes, variant='rezero')
        model.model_name = f'ResNeXt50_ReZero'

    elif mt == 'resnext_skipinit':
        model = ResNeXtNet(layers, num_classes, variant='skipinit')
        model.model_name = f'ResNeXt50_SkipInit'

    elif mt == 'resnext_layerscale':
        model = ResNeXtNet(layers, num_classes, variant='layerscale')
        model.model_name = f'ResNeXt50_LayerScale'

    else:
        raise ValueError(f"Unknown model type: {model_type}\nAvailable: {MODEL_TYPES}")

    return model


def count_parameters(model):
    """파라미터 수 계산"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ============================================================
# 모델 목록 및 설명
# ============================================================

MODEL_TYPES = [
    # Pilot Study
    'plain_he', 'plain_identity', 'resnet_identity_all',
    # 기존 9개
    'baseline',
    'lrs_low', 'lrs_mid', 'lrs_high',
    'rezero',
    'hybrida',
    'lrs_hybrida_low', 'lrs_hybrida_mid', 'lrs_hybrida_high',
    # Phase 2-B 추가
    'skipinit', 'fixup', 'layerscale',
    # Fixed-α ablation
    'fixed_alpha_01', 'fixed_alpha_03', 'fixed_alpha_05', 'fixed_alpha_07',
    # Per-channel ablation
    'per_channel_lrs',
    # Highway gate (input-dependent comparison)
    'highway',
    # WRN-28-10
    'wrn_baseline', 'wrn_lrs_low', 'wrn_lrs_hybrida_low',
    'wrn_rezero', 'wrn_skipinit', 'wrn_layerscale',
    # ResNeXt-50
    'resnext_baseline', 'resnext_lrs_low', 'resnext_lrs_hybrida_low',
    'resnext_rezero', 'resnext_skipinit', 'resnext_layerscale',
]

MODEL_DESCRIPTIONS = {
    'plain_he':             'Plain Network, He init (no skip)',
    'plain_identity':       'Plain Network, Identity init (no skip)',
    'resnet_identity_all':  'ResNet, Identity init (all layers)',
    'baseline':             'Baseline ResNet, y=F(x)+x',
    'lrs_low':              'LRS α≈0.12 (identity-favored)',
    'lrs_mid':              'LRS α=0.5 (balanced)',
    'lrs_high':             'LRS α≈0.88 (transform-favored)',
    'rezero':               'ReZero y=x+α·F(x), α₀=0',
    'skipinit':             'SkipInit y=x+α·F(x), α₀=0 (BN retained)',
    'fixup':                'Fixup, last BN=0 init',
    'layerscale':           'LayerScale y=x+diag(λ)·F(x), λ₀=1e-4',
    'hybrida':              'HybridA init (layer3,4=Identity)',
    'lrs_hybrida_low':      'LRS+HybridA α≈0.12 [Proposed]',
    'lrs_hybrida_mid':      'LRS+HybridA α=0.5 [Proposed]',
    'lrs_hybrida_high':     'LRS+HybridA α≈0.88',
    'fixed_alpha_01':       'Fixed α=0.1 (non-learnable ablation)',
    'fixed_alpha_03':       'Fixed α=0.3 (non-learnable ablation)',
    'fixed_alpha_05':       'Fixed α=0.5 (non-learnable ablation)',
    'fixed_alpha_07':       'Fixed α=0.7 (non-learnable ablation)',
    'per_channel_lrs':      'Per-channel LRS α_c',
    'highway':              'Highway gate y=T(x)·F(x)+(1-T(x))·x, input-dependent',
    'wrn_baseline':         'WRN-28-10 Baseline',
    'wrn_lrs_low':          'WRN-28-10 LRS Low',
    'wrn_lrs_hybrida_low':  'WRN-28-10 LRS+HybridA Low',
    'wrn_rezero':           'WRN-28-10 ReZero',
    'wrn_skipinit':         'WRN-28-10 SkipInit',
    'wrn_layerscale':       'WRN-28-10 LayerScale',
    'resnext_baseline':     'ResNeXt-50 (32×4d) Baseline',
    'resnext_lrs_low':      'ResNeXt-50 LRS Low',
    'resnext_lrs_hybrida_low': 'ResNeXt-50 LRS+HybridA Low',
    'resnext_rezero':       'ResNeXt-50 ReZero',
    'resnext_skipinit':     'ResNeXt-50 SkipInit',
    'resnext_layerscale':   'ResNeXt-50 LayerScale',
}

# 별칭 (기존 스크립트 호환)
MODEL_TYPES_ALIASES = {
    'lrs_ha_low':  'lrs_hybrida_low',
    'lrs_ha_mid':  'lrs_hybrida_mid',
    'lrs_ha_high': 'lrs_hybrida_high',
}
