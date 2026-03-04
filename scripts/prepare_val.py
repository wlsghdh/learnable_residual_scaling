"""
prepare_val.py - ImageNet validation set 정리
=============================================
val 이미지를 클래스별 폴더로 구성: val/{synset}/ILSVRC2012_val_XXXXXXXX.JPEG

사용법:
    python scripts/prepare_val.py

전제조건:
    ~/data/imagenet/val/       <- 이미 JPEG 파일들이 flat하게 존재
    ~/data/imagenet/devkit/ILSVRC2012_devkit_t12/data/meta.mat
    ~/data/imagenet/devkit/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt
"""

import os
import shutil
import struct
import scipy.io
import numpy as np

VAL_DIR = os.path.expanduser('~/data/imagenet/val')
DEVKIT_DIR = os.path.expanduser(
    '~/data/imagenet/devkit/ILSVRC2012_devkit_t12/data')

GT_FILE = os.path.join(DEVKIT_DIR, 'ILSVRC2012_validation_ground_truth.txt')
META_FILE = os.path.join(DEVKIT_DIR, 'meta.mat')


def get_synsets():
    """meta.mat에서 synset 이름(wnid) 읽기
    Returns: dict {1-indexed class_id -> 'n01440764'}
    shape: synsets (1860, 1), each entry has ILSVRC2012_ID, WNID fields
    """
    meta = scipy.io.loadmat(META_FILE)
    synsets = meta['synsets']  # shape: (1860, 1)
    id_to_wnid = {}
    for i in range(synsets.shape[0]):
        entry = synsets[i, 0]
        ilsvrc_id = int(entry['ILSVRC2012_ID'][0][0])
        wnid = str(entry['WNID'][0])
        id_to_wnid[ilsvrc_id] = wnid
    return id_to_wnid


def main():
    print('Reading synset map from meta.mat ...')
    id_to_wnid = get_synsets()
    print(f'  Found {len(id_to_wnid)} synsets')

    print('Reading ground truth ...')
    with open(GT_FILE) as f:
        gt_labels = [int(line.strip()) for line in f]
    print(f'  {len(gt_labels)} val images')

    # val 이미지 목록 (flat)
    val_images = sorted([
        f for f in os.listdir(VAL_DIR)
        if f.endswith('.JPEG') and not os.path.isdir(os.path.join(VAL_DIR, f))
    ])
    print(f'  Found {len(val_images)} JPEG files in val/')

    if len(val_images) != len(gt_labels):
        print(f'WARNING: mismatch! images={len(val_images)}, labels={len(gt_labels)}')

    # synset 폴더 생성
    wnids = set(id_to_wnid[l] for l in gt_labels if l in id_to_wnid)
    for wnid in wnids:
        os.makedirs(os.path.join(VAL_DIR, wnid), exist_ok=True)

    print('Moving val images to class folders ...')
    moved = 0
    for img_name, label in zip(val_images, gt_labels):
        wnid = id_to_wnid.get(label)
        if wnid is None:
            print(f'  WARNING: no synset for label {label}')
            continue
        src = os.path.join(VAL_DIR, img_name)
        dst = os.path.join(VAL_DIR, wnid, img_name)
        shutil.move(src, dst)
        moved += 1
        if moved % 5000 == 0:
            print(f'  Moved {moved}/{len(val_images)} ...')

    print(f'Done. Moved {moved} images into {len(wnids)} class folders.')

    # 검증
    n_classes = len([d for d in os.listdir(VAL_DIR)
                     if os.path.isdir(os.path.join(VAL_DIR, d))])
    print(f'Val directory: {n_classes} class subdirectories')


if __name__ == '__main__':
    main()
