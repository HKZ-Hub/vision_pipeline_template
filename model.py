# model.py

import torch
import torch.nn as nn
import timm

from config import CFG


class BiomassModel(nn.Module):
    """
    二流路（two-stream）モデル + マルチタスク回帰ヘッド

    - 1つのバックボーン（例: ConvNeXt, ViT）を共有して使用
    - 左画像 (img_left)、右画像 (img_right) の両方をバックボーンに通して特徴量を得る
    - 2つの特徴ベクトルを結合して 1 本のベクトルにする
    - 結合ベクトルを 3 つの専用ヘッドに入力し、それぞれ
        1. Dry_Total_g
        2. GDM_g
        3. Dry_Green_g
      を回帰出力する

    出力はタプル (out_total, out_gdm, out_green) を返す。
    """

    def __init__(
        self,
        model_name: str = None,
        pretrained: bool = None,
        n_targets: int = 3,
    ):
        super().__init__()

        if model_name is None:
            model_name = CFG.model_name
        if pretrained is None:
            pretrained = CFG.pretrained

        # timm でバックボーンを作成
        # num_classes=0, global_pool='avg' にすることで
        # 「特徴ベクトルを返すエンコーダ」として利用する
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # バックボーンが出力する特徴量の次元数
        self.n_features = self.backbone.num_features

        # 2ストリームなので、結合後の次元は 2 倍
        self.n_combined_features = self.n_features * 2

        hidden_dim = self.n_combined_features // 2

        # --- 3つの専用ヘッド（すべて同じ構造） ---
        self.head_total = nn.Sequential(
            nn.Linear(self.n_combined_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

        self.head_gdm = nn.Sequential(
            nn.Linear(self.n_combined_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

        self.head_green = nn.Sequential(
            nn.Linear(self.n_combined_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, img_left: torch.Tensor, img_right: torch.Tensor):
        """
        Args:
            img_left  : [B, 3, H, W]
            img_right : [B, 3, H, W]

        Returns:
            out_total : [B, 1]
            out_gdm   : [B, 1]
            out_green : [B, 1]
        """
        # 左右それぞれから特徴ベクトルを抽出
        feat_left = self.backbone(img_left)    # [B, C]
        feat_right = self.backbone(img_right)  # [B, C]

        # 結合
        combined = torch.cat([feat_left, feat_right], dim=1)  # [B, 2C]

        # 各ヘッドに通す
        out_total = self.head_total(combined)
        out_gdm = self.head_gdm(combined)
        out_green = self.head_green(combined)

        return out_total, out_gdm, out_green


def create_model_from_config() -> BiomassModel:
    """
    CFG から設定を読み込んでモデルを作成するヘルパー。

    例:
        from model import create_model_from_config
        model = create_model_from_config().to(CFG.device)
    """
    model = BiomassModel(
        model_name=CFG.model_name,
        pretrained=CFG.pretrained,
        n_targets=len(CFG.train_target_cols),
    )
    return model
