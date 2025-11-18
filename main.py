# main.py

import os
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn
import torch.optim as optim

from config import CFG
from dataset import BiomassDataset
from transforms import get_train_transforms, get_valid_transforms
from model import create_model_from_config
from loss import create_criterion
from engine import train_one_epoch, validate_one_epoch
from utils import seed_everything, ensure_dir, save_model


# ----------------------------------------
# 1. メイン関数
# ----------------------------------------
def run_training(fold: int):

    print(f"\n{'=' * 50}")
    print(f"🚀 Start Training Fold {fold}")
    print(f"{'=' * 50}")

    # ---------------------------
    #  データ分割
    # ---------------------------
    df = pd.read_csv(CFG.train_csv)

    # long → wide ピボット
    df_wide = df.pivot(index='image_path', columns='target_name', values='target')
    df_wide = df_wide.reset_index()
    df_wide.columns.name = None

    # Stratified KFold のためのビン作成
    num_bins = 10
    df_wide['total_bin'] = pd.cut(df_wide['Dry_Total_g'], bins=num_bins, labels=False)

    df_wide['fold'] = -1
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.random_state)

    for i, (_, valid_index) in enumerate(skf.split(df_wide, df_wide['total_bin'])):
        df_wide.loc[valid_index, 'fold'] = i

    train_df = df_wide[df_wide['fold'] != fold].reset_index(drop=True)
    valid_df = df_wide[df_wide['fold'] == fold].reset_index(drop=True)

    print(f"Train samples: {len(train_df)} | Valid samples: {len(valid_df)}")

    # ---------------------------
    # Dataset / DataLoader
    # ---------------------------
    train_dataset = BiomassDataset(
        df=train_df,
        image_dir=CFG.train_image_dir,
        transforms=get_train_transforms(),
        is_train=True,
    )
    valid_dataset = BiomassDataset(
        df=valid_df,
        image_dir=CFG.train_image_dir,
        transforms=get_valid_transforms(),
        is_train=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size * 2,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # ---------------------------
    # モデル / 損失 / Optimizer
    # ---------------------------
    device = CFG.device
    model = create_model_from_config().to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    criterion = create_criterion().to(device)

    # ---------------------------
    #  Two-stage Training
    # ---------------------------

    best_score = -999

    # ========== Stage 1: Freeze backbone ==========
    print("\n--- Stage 1: Training Heads (Backbone Frozen) ---")

    for param in model.module.backbone.parameters() if hasattr(model, "module") else model.backbone.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=CFG.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)

    for epoch in range(1, CFG.freeze_epochs + 1):
        print(f"\nEpoch {epoch}/{CFG.epochs} (Stage 1)")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, score = validate_one_epoch(model, valid_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | R2: {score:.4f}")

        scheduler.step(valid_loss)

        # モデル保存
        if score > best_score:
            best_score = score
            save_model(model, f"./outputs/best_fold_{fold}.pth")

    # ========== Stage 2: Unfreeze & Fine-tune ==========
    print("\n--- Stage 2: Fine-tuning All Layers ---")

    for param in model.module.backbone.parameters() if hasattr(model, "module") else model.backbone.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=CFG.finetune_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=3)

    for epoch in range(CFG.freeze_epochs + 1, CFG.epochs + 1):
        print(f"\nEpoch {epoch}/{CFG.epochs} (Stage 2)")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss, score = validate_one_epoch(model, valid_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | R2: {score:.4f}")

        scheduler.step(valid_loss)

        if score > best_score:
            best_score = score
            save_model(model, f"./outputs/best_fold_{fold}.pth")

    print(f"\n🎉 Fold {fold} Finished. Best R2 = {best_score:.4f}\n")

    # メモリ解放
    del model, train_loader, valid_loader, train_dataset, valid_dataset
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------
# 2. 実行パート
# ----------------------------------------
if __name__ == "__main__":

    seed_everything()

    ensure_dir("./outputs")

    for fold in range(CFG.n_folds):
        run_training(fold)
