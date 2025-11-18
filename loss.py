# loss.py

import torch.nn as nn

from config import CFG


class WeightedBiomassLoss(nn.Module):
    """
    3タスク回帰用の重み付き損失関数。

    モデル出力:
        (out_total, out_gdm, out_green)  各 [B, 1]

    ターゲット:
        targets: [B, 3]
        カラム順は CFG.train_target_cols を想定
        例:
            0: Dry_Total_g
            1: GDM_g
            2: Dry_Green_g

    CFG.loss_weights で各タスクの重みを指定する:
        {
            "total_loss": 0.5,
            "gdm_loss":   0.2,
            "green_loss": 0.1
        }
    """

    def __init__(self, loss_weights_dict=None):
        super().__init__()

        # デフォルトは CFG.loss_weights を利用
        if loss_weights_dict is None:
            loss_weights_dict = CFG.loss_weights

        self.weights = loss_weights_dict

        # SmoothL1Loss は回帰タスクでよく使われる Huber 的な損失
        self.criterion = nn.SmoothL1Loss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions:
                (pred_total, pred_gdm, pred_green)
                いずれも shape [B, 1]

            targets:
                shape [B, 3]
                [total, gdm, green] に対応する値が入っている想定
        """
        pred_total, pred_gdm, pred_green = predictions

        # ターゲットをそれぞれ取り出す
        # targets[:, 0] -> total, [:, 1] -> gdm, [:, 2] -> green
        true_total = targets[:, 0].unsqueeze(-1)  # [B, 1]
        true_gdm   = targets[:, 1].unsqueeze(-1)
        true_green = targets[:, 2].unsqueeze(-1)

        # 個々のタスクの損失
        loss_total = self.criterion(pred_total, true_total)
        loss_gdm   = self.criterion(pred_gdm, true_gdm)
        loss_green = self.criterion(pred_green, true_green)

        # 重み付きで合計
        total_loss = (
            self.weights["total_loss"] * loss_total
            + self.weights["gdm_loss"] * loss_gdm
            + self.weights["green_loss"] * loss_green
        )

        return total_loss


def create_criterion():
    """
    CFG をもとに損失関数インスタンスを作るヘルパー。

    例:
        from loss import create_criterion
        criterion = create_criterion().to(CFG.device)
    """
    return WeightedBiomassLoss(loss_weights_dict=CFG.loss_weights)
