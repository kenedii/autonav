import argparse
import os
import shutil
import sys
import time
from datetime import timedelta

import numpy as np

TORCH_IMPORT_ERROR = None
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models
except Exception as exc:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    models = None
    TORCH_IMPORT_ERROR = exc

    class Dataset(object):
        pass

BaseModule = nn.Module if nn is not None else object

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

try:
    from .dataset_loader import (
        load_dataset,
        log_dataset_metadata,
        validate_dataset_metadata,
    )
except ImportError:
    from dataset_loader import (
        load_dataset,
        log_dataset_metadata,
        validate_dataset_metadata,
    )


MODEL_ARCHITECTURE = "resnet101"
DATASET_PATH = "combined_augmented_dataset.csv"
BATCH_SIZE = 64
NUM_EPOCHS = 64
DEVICE = torch.device("cuda" if torch and torch.cuda.is_available() else "cpu") if torch else "cpu"
VRAM_ALLOCATION = 0.6
SAVE_DIR = "checkpoints"


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Train a steering model from a flattened combined CSV.")
    parser.add_argument("--dataset-path", default=DATASET_PATH)
    parser.add_argument("--model-architecture", default=MODEL_ARCHITECTURE, choices=["resnet18", "resnet34", "resnet50", "resnet101"])
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--allow-mixed", action="store_true", help="Allow mixed rgb_source / preprocess_profile values.")
    parser.add_argument("--rgb-source", default=None, help="Filter to a single rgb_source before training.")
    return parser.parse_args()


def _require_torch_stack():
    if torch is None or nn is None or optim is None or DataLoader is None or models is None:
        raise RuntimeError(
            "PyTorch and torchvision are required for training. "
            f"Original import error: {TORCH_IMPORT_ERROR!r}"
        )


def _load_training_dependencies():
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    return {
        "plt": plt,
        "pd": pd,
        "mean_absolute_error": mean_absolute_error,
        "r2_score": r2_score,
        "train_test_split": train_test_split,
        "tqdm": tqdm,
    }


class CustomDataset(Dataset):
    def __init__(self, df, pixel_columns, img_height=120, img_width=160):
        self.df = df.reset_index(drop=True)
        self.pixel_columns = pixel_columns
        self.img_height = img_height
        self.img_width = img_width
        self.pixels_per_image = img_height * img_width

        rgb_data = self.df[self.pixel_columns].values.astype(np.float32) / 255.0
        images = []
        for row in rgb_data:
            img = row.reshape(self.pixels_per_image, 3).reshape(self.img_height, self.img_width, 3)
            img = np.transpose(img, (2, 0, 1))
            images.append(img)
        self.images = np.array(images, dtype=np.float32)
        self.targets = self.df[["steer_norm"]].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        _require_torch_stack()
        return torch.tensor(self.images[idx]), torch.tensor(self.targets[idx])


class ControlModel(BaseModule):
    def __init__(self, architecture):
        _require_torch_stack()
        super().__init__()
        architecture = architecture.lower()
        if architecture == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            input_size = 512
        elif architecture == "resnet34":
            backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            input_size = 512
        elif architecture == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            input_size = 2048
        else:
            backbone = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
            input_size = 2048

        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def train_model(args):
    _require_torch_stack()
    deps = _load_training_dependencies()
    plt = deps["plt"]
    pd = deps["pd"]
    mean_absolute_error = deps["mean_absolute_error"]
    r2_score = deps["r2_score"]
    train_test_split = deps["train_test_split"]
    tqdm = deps["tqdm"]

    os.makedirs(SAVE_DIR, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(VRAM_ALLOCATION, 0)

    original_stdout = sys.stdout
    temp_log_path = "temp_training_log.txt"
    log_file = open(temp_log_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)

    try:
        print("=" * 65)
        print("           JETRACER BEHAVIORAL CLONING TRAINING")
        print("=" * 65)
        print(f"Dataset             : {args.dataset_path}")
        print(f"Device              : {DEVICE}")
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"GPU                 : {props.name}")
            print(f"GPU Memory          : {props.total_memory / 1e9:.1f} GB")
        print(f"Batch Size          : {args.batch_size}")
        print(f"Epochs              : {args.epochs}")
        print(f"Targets             : steer_norm")
        print(f"Output Activation   : Tanh() -> [-1, 1]")
        print("=" * 65 + "\n")

        print(f"Loading dataset: {args.dataset_path}")
        df, pixel_columns = load_dataset(args.dataset_path, rgb_source_filter=args.rgb_source)
        source_counts, profile_counts = log_dataset_metadata(df)
        validate_dataset_metadata(source_counts, profile_counts, allow_mixed=args.allow_mixed)
        print(f"Total samples       : {len(df):,}")

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        train_dataset = CustomDataset(train_df, pixel_columns)
        test_dataset = CustomDataset(test_df, pixel_columns)

        num_workers = 0 if os.name == "nt" else 4
        pin_memory = torch.cuda.is_available()
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

        print(f"Train batches       : {len(train_loader)}")
        print(f"Test batches        : {len(test_loader)}\n")

        model = ControlModel(args.model_architecture).to(DEVICE)
        arch_path = os.path.join(SAVE_DIR, "model_architecture.txt")
        with open(arch_path, "w", encoding="utf-8") as f:
            f.write(f"ControlModel - {args.model_architecture} backbone + Tanh head\n")
            f.write("=" * 60 + "\n")
            f.write(str(model))
        print(f"Model architecture saved -> {arch_path}\n")

        def weighted_mse_loss(pred, target):
            return torch.mean((pred - target) ** 2)

        criterion = weighted_mse_loss
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

        history = {"epoch": [], "train_loss": [], "val_mae": [], "val_r2": [], "epoch_time": []}
        best_val_mae = float("inf")
        total_start_time = time.time()

        print("Starting training...\n")

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            model.train()
            train_loss = 0.0
            for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs} [Train]", leave=False):
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / max(1, len(train_loader))

            model.eval()
            preds_list, targets_list = [], []
            with torch.no_grad():
                for imgs, targets in tqdm(test_loader, desc=f"Epoch {epoch:02d}/{args.epochs} [Val]", leave=False):
                    imgs = imgs.to(DEVICE)
                    outputs = model(imgs)
                    preds_list.append(outputs.cpu().numpy())
                    targets_list.append(targets.numpy())

            preds = np.concatenate(preds_list)
            targets = np.concatenate(targets_list)
            val_mae = mean_absolute_error(targets, preds)
            val_r2 = r2_score(targets, preds, multioutput="uniform_average")
            steer_mae = mean_absolute_error(targets[:, 0], preds[:, 0])
            epoch_time = time.time() - epoch_start

            history["epoch"].append(epoch)
            history["train_loss"].append(avg_train_loss)
            history["val_mae"].append(val_mae)
            history["val_r2"].append(val_r2)
            history["epoch_time"].append(epoch_time)

            print(f"\nEpoch {epoch:02d} | Time: {epoch_time:.1f}s")
            print(f"   Train Loss : {avg_train_loss:.5f}")
            print(f"   Val MAE    : {val_mae:.4f}  (Steer: {steer_mae:.4f})")
            print(f"   Val R²     : {val_r2:.4f}")

            scheduler.step(val_mae)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": val_mae,
                },
                os.path.join(SAVE_DIR, "latest_model.pth"),
            )

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
                print(f"   New best model! Val MAE = {best_val_mae:.4f}")

        total_time = time.time() - total_start_time
        print("\n" + "=" * 65)
        print("TRAINING COMPLETED!")
        print("=" * 65)
        print(f"Total training time : {str(timedelta(seconds=int(total_time)))}")
        print(f"Best Val MAE        : {best_val_mae:.4f}")
        print(f"Final Val MAE       : {val_mae:.4f}")
        print(f"Final Val R²        : {val_r2:.4f}")
        print(f"Final Steer MAE     : {steer_mae:.4f}")

        pd.DataFrame(history).to_csv("training_history_normalized.csv", index=False)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(history["epoch"], history["train_loss"], "o-")
        plt.title("Train Loss")
        plt.grid(True)
        plt.subplot(1, 3, 2)
        plt.plot(history["epoch"], history["val_mae"], "o-", color="orange")
        plt.title("Validation MAE")
        plt.grid(True)
        plt.subplot(1, 3, 3)
        plt.plot(history["epoch"], history["val_r2"], "o-", color="green")
        plt.title("Validation R²")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("training_curves_normalized.png", dpi=200)
        plt.close()

        final_log_path = os.path.join(SAVE_DIR, "training_log.txt")
        if os.path.exists(final_log_path):
            os.remove(final_log_path)
        shutil.move(temp_log_path, final_log_path)
        print(f"\nAll files saved in './{SAVE_DIR}/':")
        print("   • best_model.pth")
        print("   • latest_model.pth")
        print("   • model_architecture.txt")
        print("   • training_log.txt")
        print("   • training_history_normalized.csv")
        print("   • training_curves_normalized.png")
        print(f"\nTraining log saved -> {final_log_path}")
        print("Ready for inference on JetRacer!")
    finally:
        sys.stdout = original_stdout
        log_file.close()


def main():
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
