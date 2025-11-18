import torch
import os


class CFG:

    # -----------------------------
    # 基本設定
    # -----------------------------
    project_name = "vision_pipeline_template"

    # GPU / CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2
    seed = 42

    # -----------------------------
    # データセット関連
    # -----------------------------
    # 例：/kaggle/input/xxx などに置き換えるだけで使える
    base_path = "./data"      
    train_csv = os.path.join(base_path, "train.csv")
    test_csv  = os.path.join(base_path, "test.csv")

    # 画像
    train_image_dir = os.path.join(base_path, "train")
    test_image_dir  = os.path.join(base_path, "test")

    # 二流路モデル（two-stream）を使うか？
    use_two_streams = True

    # -----------------------------
    # ターゲット（マルチタスクに対応）
    # -----------------------------
    # 学習で使用するターゲット
    train_target_cols = [
        "_",
        "_",
        "_"
    ]

    # 全ての出力（5つの出力を作る時など）
    all_target_cols = [
        "_",
        "_",
        "_",
        "_",
        "_"
    ]

    # 大会の重み付き R2 用
    r2_weights = [0.1, 0.1, 0.1, 0.2, 0.5]

    # -----------------------------
    # モデル / 学習パラメータ
    # -----------------------------
    model_name = "convnext_tiny"
    pretrained = True

    image_size = 768
    batch_size = 4
    epochs = 20

    # 2段階学習（凍結→ファインチューニング）
    freeze_epochs = 5
    learning_rate = 1e-4
    finetune_lr   = 1e-5

    # -----------------------------
    # 損失関数の重み
    # -----------------------------
    loss_weights = {
        "total_loss": 0.5,
        "gdm_loss": 0.2,
        "green_loss": 0.1
    }

    # -----------------------------
    # クロスバリデーション設定
    # -----------------------------
    n_folds = 5
    stratify_target = "_"  # どの列で層化するか
    random_state = 42

    # -----------------------------
    # 保存
    # -----------------------------
    output_dir = "./outputs"
    model_save_path = os.path.join(output_dir, "model.pth")


# 必要なら utils で呼び出す
def print_config():
    print("===== CONFIG =====")
    for key, value in CFG.__dict__.items():
        if not key.startswith("__"):
            print(f"{key}: {value}")
