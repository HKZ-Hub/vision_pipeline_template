# transforms.py

import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import CFG


def get_train_transforms():
    """
    学習用のデータ拡張。
    - 左右反転 / 縦反転 / 回転
    - 色味のゆらぎ（ColorJitter）
    - ImageNet の平均・分散で正規化
    - CFG.image_size にリサイズ
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.75,
            ),

            # 必要ならここに Blur や ShiftScaleRotate などを足してもよい
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
            #                    rotate_limit=10, border_mode=0, p=0.5),

            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet
                std=[0.229, 0.224, 0.225],
            ),
            A.Resize(CFG.image_size, CFG.image_size),
            ToTensorV2(),
        ]
    )


def get_valid_transforms():
    """
    検証 / テスト用の変換。
    - 形状とスケールだけを揃え、ランダム性は入れない
    """
    return A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            A.Resize(CFG.image_size, CFG.image_size),
            ToTensorV2(),
        ]
    )
