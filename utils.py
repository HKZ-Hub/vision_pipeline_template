# utils.py

import os
import random
import numpy as np
import torch

from config import CFG


# -----------------------------
# 乱数シード固定（再現性の確保）
# -----------------------------
def seed_everything(seed: int = None):
    """
    乱数シードを固定して再現性を高める。
    CPU / GPU / cudnn / numpy / random に対応。
    """
    if seed is None:
        seed = CFG.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 再現性モード（完全固定）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Random seed set to {seed}")


# -----------------------------
# ディレクトリ作成（存在していてもOK）
# -----------------------------
def ensure_dir(path: str):
    """
    path のディレクトリを作成する（存在していてもエラーにしない）。
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[INFO] Created directory: {path}")


# -----------------------------
# モデル保存（DataParallel 両対応）
# -----------------------------
def save_model(model, path: str):
    """
    DataParallel / 非DataParallel どちらでも保存できるようにする。
    """
    ensure_dir(os.path.dirname(path))

    if hasattr(model, "module"):  # DataParallel
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)

    print(f"[INFO] Model saved to {path}")


# -----------------------------
# 学習率（LR）を取得
# -----------------------------
def get_lr(optimizer):
    """
    Optimizer の現在の学習率を取得する。
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# -----------------------------
# GPU メモリ情報（任意）
# -----------------------------
def gpu_memory_used():
    """
    現在の GPU メモリ使用量を MB 単位で返す。
    """
    if not torch.cuda.is_available():
        return 0.0
    mem = torch.cuda.memory_allocated() / 1024 ** 2
    return round(mem, 2)


def log_gpu_memory(prefix="GPU"):
    """
    GPU 使用メモリをログとして出力。
    """
    if torch.cuda.is_available():
        mem = gpu_memory_used()
        print(f"[INFO] {prefix} memory used: {mem} MB")


# -----------------------------
# 例：tqdm の表示を軽くする
# -----------------------------
from contextlib import contextmanager
import sys

@contextmanager
def suppress_stdout():
    """
    with suppress_stdout():
        timm_model = timm.create_model(...)
    のように使うと、timm の verbose ログを消せる。
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
