# dataset.py

import os
from typing import Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import CFG


class BiomassDataset(Dataset):
    """
    画像 + 数値ターゲット用 Dataset（CSIRO Image2Biomass をベースにした汎用版）

    - CFG.use_two_streams = True のとき:
        画像を左右 2 枚 (left / right) に分割して返す two-stream モード
    - False のとき:
        画像全体を 1 枚だけ返す one-stream モード

    train モード:
        return (img_left, img_right, train_targets, all_targets)
        ※ one-stream の場合は img_right は None を返す設計でもよいが、
          エンジン側を合わせやすいように left/right を必ず返す。

    test モード:
        return (img_left, img_right, image_path)
    """

    def __init__(
        self,
        df,
        image_dir: str,
        transforms: Optional[object] = None,
        is_train: bool = True,
    ):
        """
        Args:
            df          : pandas.DataFrame
                          必須列: 'image_path'
                          train 時は CFG.train_target_cols / CFG.all_target_cols も含むこと
            image_dir   : 画像ファイルが置かれているディレクトリ
            transforms  : Albumentations の Compose など（image=img を受けて dict を返すもの）
            is_train    : 学習/検証用かどうか（True: ターゲットも返す）
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transforms = transforms
        self.is_train = is_train

        self.image_paths = self.df["image_path"].values

        if self.is_train:
            # 3 ターゲット (Total / GDM / Green)
            self.train_targets = self.df[CFG.train_target_cols].values.astype(
                "float32"
            )
            # 5 ターゲット (Green / Dead / Clover / GDM / Total)
            self.all_targets = self.df[CFG.all_target_cols].values.astype("float32")

    def __len__(self):
        return len(self.df)

    def _load_image(self, img_path_suffix: str) -> np.ndarray:
        """
        画像を読み込んで RGB の numpy 配列 (H, W, C) を返す。
        """
        filename = os.path.basename(img_path_suffix)
        full_path = os.path.join(self.image_dir, filename)

        image = cv2.imread(full_path)
        if image is None:
            raise FileNotFoundError(f"画像を読み込めませんでした: {full_path}")

        # BGR -> RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _split_two_streams(self, image: np.ndarray):
        """
        画像を左右 2 枚に分割して返す。
        """
        h, w, _ = image.shape
        mid = w // 2
        img_left = image[:, :mid]
        img_right = image[:, mid:]
        return img_left, img_right

    def _apply_transforms(self, img: np.ndarray) -> torch.Tensor:
        """
        Albumentations を適用して Tensor に変換。
        transforms が None の場合は numpy -> tensor だけ行う。
        """
        if self.transforms is not None:
            augmented = self.transforms(image=img)
            img = augmented["image"]

            # Albumentations + ToTensorV2 を使っていれば既に tensor になっているはず
            if isinstance(img, torch.Tensor):
                return img

        # transforms が None の場合のフォールバック
        img = img.astype("float32") / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(img)

    def __getitem__(self, idx: int):
        # 1. 画像読み込み
        img_path_suffix = self.image_paths[idx]
        image = self._load_image(img_path_suffix)

        # 2. 1ストリーム / 2ストリーム
        if CFG.use_two_streams:
            img_left, img_right = self._split_two_streams(image)
        else:
            img_left = image
            img_right = image  # エンジン側のコードを簡単にするため同じものを入れておく

        # 3. Augmentation + Tensor 化
        img_left_tensor = self._apply_transforms(img_left)
        img_right_tensor = self._apply_transforms(img_right)

        if self.is_train:
            # ターゲットも tensor に変換
            train_target_vals = self.train_targets[idx]
            all_target_vals = self.all_targets[idx]

            train_targets_tensor = torch.tensor(train_target_vals, dtype=torch.float32)
            all_targets_tensor = torch.tensor(all_target_vals, dtype=torch.float32)

            return img_left_tensor, img_right_tensor, train_targets_tensor, all_targets_tensor
        else:
            # 推論時は画像と元の image_path を返す
            return img_left_tensor, img_right_tensor, img_path_suffix
