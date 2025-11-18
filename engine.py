# engine.py

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from sklearn.metrics import r2_score

from config import CFG


def calculate_competition_score(preds_dict_3, targets_5):
    """
    コンペ仕様の「重み付き R^2 スコア」を計算する関数。

    Args:
        preds_dict_3 (dict):
            {
                "total": np.ndarray,  # 予測 Dry_Total_g
                "gdm":   np.ndarray,  # 予測 GDM_g
                "green": np.ndarray,  # 予測 Dry_Green_g
            }
            shape はいずれも [N,] を想定（flatten されていればOK）

        targets_5 (np.ndarray):
            正解値 [N, 5]
            列の順番は CFG.all_target_cols
            ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]

    Returns:
        float: 加重平均された R^2 スコア
    """
    pred_total = preds_dict_3["total"]
    pred_gdm   = preds_dict_3["gdm"]
    pred_green = preds_dict_3["green"]

    # 物理的整合性を保つための再構成
    # clover = gdm - green, dead = total - gdm
    pred_clover = np.maximum(0, pred_gdm - pred_green)
    pred_dead   = np.maximum(0, pred_total - pred_gdm)

    # 予測の 5 列をコンペ仕様の順番で並べる
    # ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]
    y_preds = np.stack(
        [pred_green, pred_dead, pred_clover, pred_gdm, pred_total],
        axis=1,
    )

    y_true = targets_5  # 形状 [N, 5]

    r2_scores = r2_score(y_true, y_preds, multioutput="raw_values")

    weighted_r2 = 0.0
    for r2, w in zip(r2_scores, CFG.r2_weights):
        weighted_r2 += r2 * w

    return float(weighted_r2)


def train_one_epoch(model: nn.Module,
                    loader,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    """
    1エポック分の学習ループ。

    Args:
        model     : 学習対象モデル
        loader    : train DataLoader
        criterion : 損失関数（例: WeightedBiomassLoss）
        optimizer : Optimizer（Adam 等）
        device    : CPU / GPU

    Returns:
        float: そのエポックの平均 train loss
    """
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)

    for (img_left, img_right, train_targets, _all_targets_ignored) in pbar:
        img_left  = img_left.to(device)
        img_right = img_right.to(device)
        targets   = train_targets.to(device)

        optimizer.zero_grad()

        # モデルの forward
        preds = model(img_left, img_right)

        # 損失計算
        loss = criterion(preds, targets)

        # 逆伝播 & 更新
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        epoch_loss += loss_val
        pbar.set_postfix(loss=f"{loss_val:.4f}")

    avg_loss = epoch_loss / len(loader)
    return avg_loss


def validate_one_epoch(model: nn.Module,
                       loader,
                       criterion: nn.Module,
                       device: torch.device):
    """
    1エポック分の検証ループ。

    - validation loss
    - コンペ用の weighted R^2

    の両方を計算して返す。

    Returns:
        (avg_valid_loss: float, competition_score: float)
    """
    model.eval()
    epoch_loss = 0.0

    all_preds_total = []
    all_preds_gdm   = []
    all_preds_green = []
    all_targets_5   = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validating", leave=False)

        for (img_left, img_right, train_targets, all_targets) in pbar:
            img_left  = img_left.to(device)
            img_right = img_right.to(device)
            train_targets = train_targets.to(device)

            # forward
            pred_total, pred_gdm, pred_green = model(img_left, img_right)

            # 損失計算（train と同じく 3タスクぶん）
            loss = criterion(
                (pred_total, pred_gdm, pred_green),
                train_targets,
            )
            loss_val = loss.item()
            epoch_loss += loss_val

            # CPU に戻して numpy 化
            all_preds_total.append(pred_total.cpu().numpy())
            all_preds_gdm.append(pred_gdm.cpu().numpy())
            all_preds_green.append(pred_green.cpu().numpy())
            all_targets_5.append(all_targets.numpy())

            pbar.set_postfix(loss=f"{loss_val:.4f}")

    # バッチごとの配列を結合
    preds_dict_np = {
        "total": np.concatenate(all_preds_total).flatten(),
        "gdm":   np.concatenate(all_preds_gdm).flatten(),
        "green": np.concatenate(all_preds_green).flatten(),
    }
    targets_np_5 = np.concatenate(all_targets_5)  # [N, 5]

    # コンペスコア計算
    competition_score = calculate_competition_score(preds_dict_np, targets_np_5)

    avg_loss = epoch_loss / len(loader)

    return avg_loss, competition_score
