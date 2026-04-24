#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "autonav-mpl"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "docs" / "final_artifacts" / "model_eval"
RUNS_ROOT = REPO_ROOT / "jetracer" / "train" / "runs_rgb_depth"
TRAIN_ROOT = REPO_ROOT / "jetracer" / "train"
MODEL_PATH = REPO_ROOT / "checkpoints" / "AutoNav-v2" / "AutoNav-v2-34" / "AutoNav-v2-34.pth"
ARCHITECTURE = "resnet34"
INPUT_SHAPE = (3, 120, 160)
BATCH_SIZE = 64
SEED = 42
LEFT_THRESHOLD = -0.15
RIGHT_THRESHOLD = 0.15

sys.path.insert(0, str(REPO_ROOT / "fleet" / "fleet_management_app" / "client_api"))
from models import (  # noqa: E402
    build_control_model,
    infer_control_architecture,
    infer_control_output_dim,
    unwrap_checkpoint_state_dict,
)


@dataclass
class Sample:
    image_path: Path
    steer: float
    throttle: float
    run_name: str


class RgbDataset(Dataset):
    def __init__(self, samples: list[Sample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not load image: {sample.image_path}")
        image = cv2.resize(image, (INPUT_SHAPE[2], INPUT_SHAPE[1]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        target = np.array([sample.steer, sample.throttle], dtype=np.float32)
        return torch.from_numpy(image), torch.from_numpy(target)


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_blank(value) -> bool:
    return value is None or str(value).strip() == ""


def parse_float(value):
    if is_blank(value):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def resolve_recorded_path(relative_value: str | None, run_dir: Path) -> Path | None:
    if is_blank(relative_value):
        return None
    value = str(relative_value).strip()
    candidate = Path(value)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    possible_paths = [
        REPO_ROOT / value,
        TRAIN_ROOT / value,
        run_dir / value,
        run_dir.parent / value,
        run_dir.parent.parent / value,
    ]
    for path in possible_paths:
        if path.exists():
            return path
    return None


def load_samples() -> tuple[list[Sample], list[str], int]:
    csv_paths = sorted(path for path in RUNS_ROOT.glob("run_*/dataset.csv"))
    samples: list[Sample] = []
    limitations: list[str] = []
    dropped_rows = 0

    for csv_path in csv_paths:
        run_dir = csv_path.parent
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rgb_rel = row.get("rgb_path")
                steer = parse_float(row.get("steer_norm"))
                throttle = parse_float(row.get("throttle_norm"))
                if is_blank(rgb_rel) or steer is None or throttle is None:
                    dropped_rows += 1
                    continue

                resolved = resolve_recorded_path(rgb_rel, run_dir)
                if resolved is None:
                    dropped_rows += 1
                    continue

                samples.append(Sample(image_path=resolved, steer=steer, throttle=throttle, run_name=run_dir.name))

    if not samples:
        limitations.append("No evaluation samples with resolvable rgb_path + steer_norm + throttle_norm were found.")
    return samples, limitations, dropped_rows


def split_indices(n_samples: int, seed: int = SEED):
    rng = np.random.RandomState(seed)
    order = rng.permutation(n_samples)

    temp_count = int(math.ceil(n_samples * 0.3))
    train_count = n_samples - temp_count
    train_idx = order[:train_count]
    temp_idx = order[train_count:]

    rng_temp = np.random.RandomState(seed)
    temp_order = rng_temp.permutation(len(temp_idx))
    temp_shuffled = temp_idx[temp_order]

    test_count = int(math.ceil(len(temp_shuffled) * 0.5))
    test_idx = temp_shuffled[:test_count]
    val_idx = temp_shuffled[test_count:]
    return train_idx, val_idx, test_idx


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = unwrap_checkpoint_state_dict(checkpoint)
    inferred_arch = infer_control_architecture(state_dict)
    output_dim = infer_control_output_dim(state_dict)

    model = build_control_model(architecture=inferred_arch or ARCHITECTURE, num_outputs=output_dim, layout="sequential")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, inferred_arch, output_dim


def evaluate_model(model, samples: list[Sample], batch_size: int = BATCH_SIZE):
    loader = DataLoader(RgbDataset(samples), batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    tgts = []
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            preds.append(outputs.cpu().numpy())
            tgts.append(targets.cpu().numpy())
    predictions = np.concatenate(preds, axis=0) if preds else np.empty((0, 2), dtype=np.float32)
    targets = np.concatenate(tgts, axis=0) if tgts else np.empty((0, 2), dtype=np.float32)
    return predictions, targets


def mae(true_values, pred_values) -> float:
    return float(np.mean(np.abs(true_values - pred_values)))


def classify_steering(value: float) -> int:
    if value < LEFT_THRESHOLD:
        return 0
    if value > RIGHT_THRESHOLD:
        return 2
    return 1


def confusion_matrix(true_cls: np.ndarray, pred_cls: np.ndarray) -> np.ndarray:
    matrix = np.zeros((3, 3), dtype=int)
    for truth, pred in zip(true_cls, pred_cls):
        matrix[int(truth), int(pred)] += 1
    return matrix


def create_scatter_plot(true_values, pred_values, title: str, output_path: Path, axis_label: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.scatter(true_values, pred_values, alpha=0.35, s=12, color="#2E86AB")
    combined = np.concatenate([true_values, pred_values]) if len(true_values) else np.array([-1.0, 1.0])
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(f"True {axis_label}")
    ax.set_ylabel(f"Predicted {axis_label}")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_error_histogram(errors, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.hist(errors, bins=30, color="#A23B72", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Samples")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_confusion_plot(cm: np.ndarray, output_path: Path) -> None:
    labels = ["Left", "Center", "Right"]
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Steering pseudo-class confusion matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_metrics_csv(metric_rows: list[dict[str, str]]) -> None:
    path = OUTPUT_DIR / "model_eval_metrics.csv"
    fieldnames = ["category", "metric", "value", "confidence", "notes"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metric_rows)


def write_text_file(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ensure_output_dir()

    eval_command = "python3 docs/final_artifacts/model_eval/generate_model_eval.py"
    samples, sample_limitations, dropped_rows = load_samples()

    if not samples:
        report = "\n".join(
            [
                "# AutoNav-v2-34 evaluation report",
                "",
                f"- Model path: `{MODEL_PATH.relative_to(REPO_ROOT)}`",
                "- Architecture: `resnet34`",
                "- Status: could not evaluate",
                "- Reason: no usable evaluation samples were found.",
                "",
                "## Limitations",
                *[f"- {item}" for item in sample_limitations],
                "",
                "## Minimum requirement to complete evaluation",
                "- Provide a dataset CSV with `rgb_path`, `steer_norm`, and `throttle_norm` values plus accessible image files.",
            ]
        )
        write_text_file(OUTPUT_DIR / "model_eval_report.md", report)
        print(f"Evaluation command: {eval_command}")
        print("Could not evaluate: no usable samples found.")
        return

    model, inferred_arch, output_dim = load_model()

    train_idx, val_idx, test_idx = split_indices(len(samples), seed=SEED)
    test_samples = [samples[i] for i in test_idx]
    predictions, targets = evaluate_model(model, test_samples)

    steer_true = targets[:, 0]
    throttle_true = targets[:, 1]
    steer_pred = predictions[:, 0]
    throttle_pred = predictions[:, 1]

    combined_mae = mae(targets, predictions)
    steer_mae = mae(steer_true, steer_pred)
    throttle_mae = mae(throttle_true, throttle_pred)

    cls_true = np.array([classify_steering(v) for v in steer_true], dtype=np.int64)
    cls_pred = np.array([classify_steering(v) for v in steer_pred], dtype=np.int64)
    cm = confusion_matrix(cls_true, cls_pred)
    pseudo_acc = float(np.mean(cls_true == cls_pred))

    per_class_acc = {}
    for idx, label in enumerate(["left", "center", "right"]):
        row_total = int(cm[idx].sum())
        per_class_acc[label] = float(cm[idx, idx] / row_total) if row_total else float("nan")

    create_scatter_plot(steer_true, steer_pred, "Steering: predicted vs true", OUTPUT_DIR / "steering_pred_vs_true.png", "steering")
    create_scatter_plot(throttle_true, throttle_pred, "Throttle: predicted vs true", OUTPUT_DIR / "throttle_pred_vs_true.png", "throttle")
    create_error_histogram(steer_pred - steer_true, "Steering prediction error histogram", OUTPUT_DIR / "steering_error_histogram.png")
    create_confusion_plot(cm, OUTPUT_DIR / "confusion_left_center_right.png")

    limitations = [
        "This evaluation uses the archived raw run CSVs under `jetracer/train/runs_rgb_depth/`, because the original `combined_augmented_dataset.csv` / `combined_cleaned_dataset.csv` artifacts are not checked in.",
        "The recreated 70/15/15 split matches the project proportions but is not guaranteed to be identical to the original training/test partition used when the checkpoint was first produced.",
        "Evaluation compares raw checkpoint outputs to normalized `steer_norm` / `throttle_norm` labels. It does not apply runtime-only steering inversion or throttle scaling hooks.",
        "Original training curves were not available, so overfitting/underfitting cannot be concluded directly from this checkpoint-only evaluation.",
        "The archived dataset is center-heavy, so pseudo-accuracy may overstate edge-case performance on sharp recoveries.",
    ]
    if dropped_rows:
        limitations.append(f"{dropped_rows} rows were excluded because they lacked `rgb_path`, `steer_norm`, `throttle_norm`, or a resolvable image path.")
    limitations.extend(sample_limitations)

    report_lines = [
        "# AutoNav-v2-34 evaluation report",
        "",
        f"- Evaluation command: `{eval_command}`",
        f"- Model path: `{MODEL_PATH.relative_to(REPO_ROOT)}`",
        f"- Architecture: `{inferred_arch or ARCHITECTURE}`",
        f"- Input shape: `{INPUT_SHAPE[0]} x {INPUT_SHAPE[1]} x {INPUT_SHAPE[2]}`",
        "- Output targets: normalized `steer_norm` and `throttle_norm`",
        "- Dataset used: archived raw run CSVs under `jetracer/train/runs_rgb_depth/run_*/dataset.csv` (RGB-only / experiment-3 style evaluation)",
        f"- Usable labeled RGB rows found: `{len(samples)}`",
        f"- Evaluation samples used: `{len(test_samples)}`",
        f"- Train / val / test split used for recovered evaluation: `{len(train_idx)} / {len(val_idx)} / {len(test_idx)}`",
        "",
        "## Metrics",
        f"- Steering MAE: `{steer_mae:.4f}`",
        f"- Throttle MAE: `{throttle_mae:.4f}`",
        f"- Combined MAE: `{combined_mae:.4f}`",
        f"- Steering pseudo-accuracy: `{pseudo_acc * 100:.2f}%`",
        f"- Left bin accuracy: `{per_class_acc['left'] * 100:.2f}%`",
        f"- Center bin accuracy: `{per_class_acc['center'] * 100:.2f}%`",
        f"- Right bin accuracy: `{per_class_acc['right'] * 100:.2f}%`",
        "",
        "## Overfitting / underfitting evidence",
        "- Original training and validation curves were not available in this repo snapshot.",
        "- This means overfitting/underfitting cannot be concluded directly from the checkpoint alone.",
        "- What we can say: the recovered evaluation uses a held-out split from the archived raw runs, and the dataset remains center-heavy.",
        "",
        "## Limitations",
    ]
    report_lines.extend(f"- {item}" for item in limitations)
    write_text_file(OUTPUT_DIR / "model_eval_report.md", "\n".join(report_lines))

    architecture_lines = [
        "# Model architecture summary",
        "",
        f"- Backbone: `{inferred_arch or ARCHITECTURE}` from torchvision, truncated before the classification head.",
        "- Feature extraction: convolutional backbone outputs are passed through `AdaptiveAvgPool2d((1, 1))` and flattened.",
        "- Regression head: `Linear(feature_dim -> 256)`, `ReLU`, `Dropout(0.4)`, `Linear(256 -> 128)`, `ReLU`, `Dropout(0.3)`, `Linear(128 -> 2)`, `Tanh`.",
        "- Output dimensions: 2 values representing normalized steering and normalized throttle.",
        "",
        "## Why Tanh is used",
        "- `Tanh` bounds the raw control outputs to `[-1, 1]`, which matches the normalized control-label convention used across the repo.",
        "- Bounded outputs reduce the chance of unreasonably large regression values during inference.",
        "",
        "## Why normalized steering / throttle are used",
        "- Steering and throttle are learned on a common normalized scale, which stabilizes regression training compared with raw PWM units.",
        "- Normalized outputs are easier to compare across datasets and can later be mapped back into hardware control space inside runtime code.",
    ]
    write_text_file(OUTPUT_DIR / "model_architecture_summary.md", "\n".join(architecture_lines))

    slide_lines = [
        "# Slide-ready model bullets",
        "",
        f"- Final live lane-follow checkpoint: `AutoNav-v2-34` at `{MODEL_PATH.relative_to(REPO_ROOT)}`.",
        f"- Architecture recovered from the checkpoint: `{inferred_arch or ARCHITECTURE}` with a 2-output regression head for steering and throttle.",
        f"- Recovered evaluation used `{len(samples)}` usable RGB-labeled rows from archived run CSVs, with a recreated `70 / 15 / 15` split (`{len(test_samples)}` held-out test samples).",
        f"- Held-out recovered metrics: steering MAE `{steer_mae:.4f}`, throttle MAE `{throttle_mae:.4f}`, steering pseudo-accuracy `{pseudo_acc * 100:.2f}%`.",
        f"- Bin-level steering accuracy: left `{per_class_acc['left'] * 100:.2f}%`, center `{per_class_acc['center'] * 100:.2f}%`, right `{per_class_acc['right'] * 100:.2f}%`.",
        "- This is a recovered evaluation on archived raw runs, not the original combined training CSV artifact used when the checkpoint was first created.",
        "- The README-reported `94.20%` pseudo-accuracy should be treated as historical unless regenerated from the original training artifact.",
    ]
    write_text_file(OUTPUT_DIR / "slide_ready_model_bullets.md", "\n".join(slide_lines))

    speaker_lines = [
        "# Speaker notes: model and evaluation",
        "",
        "For the final live lane-follow model, we are using the AutoNav-v2-34 checkpoint, which is a ResNet34-based regression model with two outputs: normalized steering and normalized throttle.",
        "",
        "Because the original combined training CSVs are not checked into this repo snapshot, this evaluation is a recovered evaluation on the archived raw run data. I used the RGB-only path that matches experiment 3 and recreated the same 70/15/15 split proportions from the training code.",
        "",
        f"On that recovered held-out split, the checkpoint reached a steering MAE of {steer_mae:.4f}, a throttle MAE of {throttle_mae:.4f}, and a steering pseudo-accuracy of {pseudo_acc * 100:.2f} percent.",
        "",
        "One thing to explain honestly is that pseudo-accuracy is based on left, center, and right steering bins. It is useful for presentation, but it can look better than the true edge-case control quality because the dataset is still center-heavy.",
        "",
        "Also, this repo snapshot does not include the original training curves, so we should avoid making strong claims about overfitting or underfitting. The safest wording is that we recovered a held-out evaluation from the archived raw data and used that as evidence for the final checkpoint.",
    ]
    write_text_file(OUTPUT_DIR / "speaker_notes_model.md", "\n".join(speaker_lines))

    metric_rows = [
        {"category": "model", "metric": "model_path", "value": str(MODEL_PATH.relative_to(REPO_ROOT)), "confidence": "high", "notes": "local checkpoint path"},
        {"category": "model", "metric": "architecture", "value": inferred_arch or ARCHITECTURE, "confidence": "high", "notes": "inferred from checkpoint state dict"},
        {"category": "model", "metric": "output_dim", "value": str(output_dim), "confidence": "high", "notes": "inferred from checkpoint head"},
        {"category": "data", "metric": "usable_labeled_rgb_rows", "value": str(len(samples)), "confidence": "high", "notes": "rows with rgb_path + steer_norm + throttle_norm + loadable image"},
        {"category": "data", "metric": "evaluation_samples", "value": str(len(test_samples)), "confidence": "high", "notes": "held-out split size"},
        {"category": "split", "metric": "train_val_test", "value": f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}", "confidence": "medium", "notes": "recreated 70/15/15 split proportions"},
        {"category": "eval", "metric": "steering_mae", "value": f"{steer_mae:.6f}", "confidence": "medium", "notes": "recovered evaluation on archived raw runs"},
        {"category": "eval", "metric": "throttle_mae", "value": f"{throttle_mae:.6f}", "confidence": "medium", "notes": "recovered evaluation on archived raw runs"},
        {"category": "eval", "metric": "combined_mae", "value": f"{combined_mae:.6f}", "confidence": "medium", "notes": "mean absolute error across both outputs"},
        {"category": "eval", "metric": "steering_pseudo_accuracy_pct", "value": f"{pseudo_acc * 100:.6f}", "confidence": "medium", "notes": "left/center/right bins with thresholds -0.15/+0.15"},
        {"category": "eval", "metric": "left_bin_accuracy_pct", "value": f"{per_class_acc['left'] * 100:.6f}", "confidence": "medium", "notes": "per-class recall"},
        {"category": "eval", "metric": "center_bin_accuracy_pct", "value": f"{per_class_acc['center'] * 100:.6f}", "confidence": "medium", "notes": "per-class recall"},
        {"category": "eval", "metric": "right_bin_accuracy_pct", "value": f"{per_class_acc['right'] * 100:.6f}", "confidence": "medium", "notes": "per-class recall"},
        {"category": "eval", "metric": "readme_94_20_supported", "value": "no", "confidence": "high", "notes": "original combined CSV / original metrics artifact missing from repo snapshot"},
    ]
    write_metrics_csv(metric_rows)

    created = sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file())
    print(f"Evaluation command: {eval_command}")
    print(f"Model path: {MODEL_PATH.relative_to(REPO_ROOT)}")
    print(f"Architecture: {inferred_arch or ARCHITECTURE}")
    print(f"Usable labeled RGB rows: {len(samples)}")
    print(f"Evaluation samples: {len(test_samples)}")
    print(f"Steering MAE: {steer_mae:.6f}")
    print(f"Throttle MAE: {throttle_mae:.6f}")
    print(f"Combined MAE: {combined_mae:.6f}")
    print(f"Steering pseudo-accuracy: {pseudo_acc * 100:.6f}%")
    print(f"Left/Center/Right bin accuracy: {per_class_acc['left'] * 100:.6f}% / {per_class_acc['center'] * 100:.6f}% / {per_class_acc['right'] * 100:.6f}%")
    print("Created files:")
    for name in created:
        print(f"- {OUTPUT_DIR / name}")


if __name__ == "__main__":
    main()
