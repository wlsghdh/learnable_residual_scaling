# LRS 논문용 시각화 스크립트

## 전체 생성
```bash
cd ~/Learnable_Reidual_Scaling
python scripts/visualize/generate_all.py
```

## 개별 생성
```bash
python scripts/visualize/fig1_block_diagram.py
python scripts/visualize/fig2_depth_accuracy.py
python scripts/visualize/fig3_init_sensitivity.py
python scripts/visualize/fig4_depth_alpha.py
python scripts/visualize/fig5_perblock_alpha.py
python scripts/visualize/fig6_alpha_trajectory.py
python scripts/visualize/fig7_high_collapse.py
python scripts/visualize/fig8_imagenet_results.py  # ImageNet 완료 후
```

## Figure 목록

| File | Description | Data Source |
|------|-------------|-------------|
| fig1_block_diagram | ResNet vs LRS Block 구조 비교 | 없음 (직접 그림) |
| fig2_depth_accuracy | Depth별 Accuracy Bar Chart | results-json/*.json |
| fig3_init_sensitivity | LRS Low/Mid/High 초기화 민감도 | results-json/*.json |
| fig4_depth_alpha | Depth-α 관계 (핵심 Figure) | results-json/*.json |
| fig5_perblock_alpha | Block별 α 분포 (Depth 152) | results-json/*.json |
| fig6_alpha_trajectory | 학습 중 α 궤적 | results-json/*.json |
| fig7_high_collapse | LRS-High 붕괴 시각화 | results-json/*.json |
| fig8_imagenet_results | ImageNet 실험 결과 | results/imagenet/*.json |

## 출력 위치
`figures/` 디렉토리에 `.pdf` + `.png` (300dpi) 저장

## 의존성
```
matplotlib >= 3.5
numpy >= 1.21
```
