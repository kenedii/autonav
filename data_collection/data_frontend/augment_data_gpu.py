#!/usr/bin/env python3
"""GPU-accelerated augmentation script using PyTorch + Kornia."""

import os
import math
import random
import re
from typing import Tuple

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn.functional as F
import kornia.augmentation as K
import kornia.filters as KF

# ---------------------------
# USER-CONFIGURABLE PARAMETERS
# ---------------------------
INPUT_CSV = "data/combined_dataset.csv"
OUTPUT_CSV = "combined_augmented_dataset.csv"
TEST_OUTPUT_DIR = "test_combination_pngs"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

IMG_W = 160
IMG_H = 120
PIXELS = IMG_W * IMG_H
PIXEL_PATTERN = re.compile(r"^[RGB](\d+)$")

# Set how much of free GPU memory (fraction between 0 and 1) this script may use.
# If CUDA not available, this is ignored.
VRAM_FRACTION = 0.65

# Soft cap on computed batch size (in case of wildly large GPU)
MAX_BATCH = 512

# Reproducible-ish (still uses random.uniform in places)
SEED = None  # set to an int for reproducibility, e.g. 42

# ---------------------------
# Utility & Depth handling
# ---------------------------
if SEED is not None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def fix_depth_value(d):
    if pd.isna(d):
        return d
    try:
        d = float(d)
    except Exception:
        return d
    if d > 50:
        return d
    if 1.0 < d < 10.0:
        return d * 100.0
    if 0.05 < d < 2.0:
        return d * 1000.0
    return d

# ---------------------------
# Device & batch sizing
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = device.type == "cuda"

def compute_batch_size(vram_fraction: float, img_bytes_float32: int, max_batch: int = MAX_BATCH) -> int:
    if not use_cuda:
        # Reasonable CPU batch (avoid huge memory usage)
        return min(32, max_batch)
    try:
        torch.cuda.synchronize(device)
    except Exception:
        pass
    # Try to fetch free memory info
    try:
        free, total = torch.cuda.mem_get_info(device)  # bytes
        target_bytes = int(free * vram_fraction)
    except Exception:
        # Fallback: approximate using total memory
        try:
            total = torch.cuda.get_device_properties(device).total_memory
            target_bytes = int(total * vram_fraction * 0.6)  # be conservative
        except Exception:
            target_bytes = int(1024**3 * 1.0)  # 1GB fallback
    batch = max(1, int(target_bytes // img_bytes_float32))
    return max(1, min(batch, max_batch))

# Per-image on-GPU memory estimate: (C x H x W) x 4 bytes (float32) plus small overhead
per_image_bytes = IMG_H * IMG_W * 3 * 4
BATCH_SIZE = compute_batch_size(VRAM_FRACTION, per_image_bytes)

print(f"Using device: {device}. Computed batch size: {BATCH_SIZE}")

# ---------------------------
# Augmentation building blocks
# ---------------------------

# We will create functions that take batched tensors & per-image strengths / probability masks
# Input format: images Tensor[B,3,H,W], float in [0,1]

def apply_color_jitter_batch(imgs: torch.Tensor, strengths: torch.Tensor) -> torch.Tensor:
    # strengths: [B] in 0.25..0.75
    # Use Kornia ColorJitter but varying parameters per-item: we will loop by sub-batches if needed.
    out = imgs
    B = imgs.shape[0]
    # We will build random parameters per-image using python and apply per-image with kornia transform
    augmented = []
    for i in range(B):
        s = strengths[i].item()
        # map s to jitter params similar to your earlier ranges
        brightness = 0.4 * s
        contrast = 0.4 * s
        saturation = 0.4 * s
        hue = 10.0 * s / 180.0  # hue as fraction in kornia (kornia expects float in [-0.5, 0.5])
        aug = K.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        # expects shape (B, C, H, W) but we'll pass single
        augmented.append(aug(out[i:i+1]))
    return torch.cat(augmented, dim=0)

def apply_blur_or_sharpen_batch(imgs: torch.Tensor, strengths: torch.Tensor) -> torch.Tensor:
    # apply gaussian blur with sigma = 0.5 + 1.5 * s or a sharpening kernel
    B = imgs.shape[0]
    out = torch.empty_like(imgs)
    for i in range(B):
        s = strengths[i].item()
        if random.random() < 0.5:
            sigma = 0.5 + 1.5 * s
            # ksize choose 5
            kernel_size = (5, 5)
            out[i:i+1] = KF.gaussian_blur2d(imgs[i:i+1], kernel_size=kernel_size, sigma=(sigma, sigma))
        else:
            # sharpening via unsharp-like kernel: add scaled laplacian
            kernel = torch.tensor([[0.0, -1.0, 0.0],
                                   [-1.0, 5.0 + 2.0 * s, -1.0],
                                   [0.0, -1.0, 0.0]], device=imgs.device, dtype=imgs.dtype)
            kernel = kernel.view(1, 1, 3, 3)
            # apply same kernel to each channel via group conv
            channels = imgs.shape[1]
            kernel = kernel.repeat(channels, 1, 1, 1)
            padding = 1
            out[i:i+1] = F.conv2d(imgs[i:i+1], kernel, padding=padding, groups=channels)
    return out

def apply_noise_batch(imgs: torch.Tensor, strengths: torch.Tensor) -> torch.Tensor:
    B, C, H, W = imgs.shape
    out = imgs.clone()
    for i in range(B):
        s = strengths[i].item()
        # num pixels fraction between 0.02 and 0.10 times strength (as before)
        frac = random.uniform(0.02, 0.10) * s
        num = int(frac * H * W)
        if num <= 0:
            continue
        ys = np.random.randint(0, H, size=num)
        xs = np.random.randint(0, W, size=num)
        # build noise map
        noise = np.random.normal(0, 20.0 * s / 255.0, size=(num, C)).astype(np.float32)
        for j in range(num):
            out[i, :, ys[j], xs[j]] = torch.clamp(out[i, :, ys[j], xs[j]] + torch.from_numpy(noise[j]).to(imgs.device), 0.0, 1.0)
    return out

def apply_random_shadow_batch(imgs: torch.Tensor, strengths: torch.Tensor) -> torch.Tensor:
    # create a linear shadow mask between two random x positions and blur it
    B, C, H, W = imgs.shape
    out = imgs.clone()
    for i in range(B):
        s = strengths[i].item()
        x1 = random.randint(0, W)
        x2 = random.randint(0, W)
        # create mask on CPU then transfer
        mask = np.zeros((H, W), dtype=np.float32)
        cv2.line(mask, (x1, 0), (x2, H-1), 1.0, thickness=W)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        mask = torch.from_numpy(mask).to(imgs.device).unsqueeze(0)  # [1,H,W]
        shadow_intensity = random.uniform(0.2, 0.6) * s
        shaded = out[i] * (1 - shadow_intensity * mask)
        out[i] = torch.clamp(shaded, 0.0, 1.0)
    return out

def apply_color_temperature_batch(imgs: torch.Tensor, strengths: torch.Tensor) -> torch.Tensor:
    B = imgs.shape[0]
    out = imgs.clone()
    for i in range(B):
        s = strengths[i].item()
        shift = int(random.uniform(-30, 30) * s)
        # imgs in [0,1] float; convert shift in 0..1 space (shift/255)
        shift_f = shift / 255.0
        b = out[i:i+1, 0:1, :, :]
        g = out[i:i+1, 1:2, :, :]
        r = out[i:i+1, 2:3, :, :]
        r = torch.clamp(r + shift_f, 0.0, 1.0)
        b = torch.clamp(b - shift_f, 0.0, 1.0)
        out[i:i+1] = torch.cat([b, g, r], dim=1)
    return out

def apply_motion_blur_batch(imgs, strengths):
    """
    imgs: tensor [B, C, H, W], float32 0..1
    strengths: list/array of per-image blur strengths (0..1)
    """

    B = imgs.shape[0]
    out = imgs.clone()

    for i in range(B):
        s = float(strengths[i])

        # no blur
        if s <= 0.01:
            continue

        # Map strength 0–1 → kernel size 3–9 (odd only)
        k = int(3 + s * 6)   # 3 → 9
        if k % 2 == 0:
            k += 1          # ensure odd

        k = max(3, k)       # Kornia requires >=3

        angle = 90.0
        direction = 0.0     # 0 = symmetric blur

        # Kornia expects kernel_size=int, NOT tuple
        try:
            blurred = KF.motion_blur(
                imgs[i:i+1],
                kernel_size=k,
                angle=angle,
                direction=direction
            )
            out[i:i+1] = blurred
        except Exception as e:
            print(f"Motion blur failed on image {i}: {e}")
            # fail-safe: skip instead of crashing
            continue

    return out

def apply_random_occlusion_batch(imgs: torch.Tensor, strengths: torch.Tensor) -> torch.Tensor:
    B, C, H, W = imgs.shape
    out = imgs.clone()
    for i in range(B):
        s = strengths[i].item()
        occ_fraction = 0.03 + 0.05 * s
        occ_area = int(H * W * occ_fraction)
        max_w = max(5, int(W * 0.3 * s + 10))
        max_h = max(5, int(H * 0.3 * s + 10))
        w = random.randint(5, min(max_w, W))
        h = random.randint(5, min(max_h, H))
        x = random.randint(0, W - w)
        y = random.randint(0, H - h)
        color_val = random.randint(0, int(60 * s))
        color_f = color_val / 255.0
        out[i, :, y:y + h, x:x + w] = color_f
    return out

def augment_flip_batch(imgs: torch.Tensor, steers: np.ndarray):
    # flip horizontally and invert steering; we return both flipped imgs and flipped steers
    flipped = torch.flip(imgs, dims=[3])  # flip width axis
    flipped_steers = -steers
    return flipped, flipped_steers

# ---------------------------
# Full combination augmentation (vectorized-ish)
# ---------------------------

def full_combination_batch(imgs: torch.Tensor, steer_norms: np.ndarray, depth_vals: np.ndarray, samples_to_save: list) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    imgs: Tensor[B,3,H,W] in [0,1]
    steer_norms: np.ndarray [B]
    depth_vals: np.ndarray [B]
    returns augmented imgs, new steers, new depths
    """
    B = imgs.shape[0]
    # strengths per-image
    strengths = torch.tensor(np.random.uniform(0.25, 0.75, size=B), dtype=torch.float32, device=imgs.device)

    out = imgs.clone()

    # Apply augmentations in sequence
    out = apply_color_jitter_batch(out, strengths)
    out = apply_blur_or_sharpen_batch(out, strengths)
    out = apply_noise_batch(out, strengths)
    out = apply_random_shadow_batch(out, strengths)
    out = apply_color_temperature_batch(out, strengths)
    out = apply_motion_blur_batch(out, strengths)
    out = apply_random_occlusion_batch(out, strengths)

    # Flip (we will always flip as per original 'Flip full strength (not harmful)')
    # Convert steer_norms to numpy, produce flipped steers
    flipped_imgs = torch.flip(out, dims=[3])
    new_steers = -steer_norms + np.array([random.uniform(-0.02, 0.02) * float(s.item()) for s in strengths])
    # depth noise: +/- 5 * strength
    new_depths = depth_vals + np.array([random.uniform(-5.0 * float(s.item()), 5.0 * float(s.item())) for s in strengths])

    # Save up to 3 samples (on CPU numpy images) -- samples_to_save is a list we append to
    for i in range(B):
        if len(samples_to_save) >= 3:
            break
        img_cpu = (flipped_imgs[i].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        samples_to_save.append(img_cpu)

    return flipped_imgs, new_steers, new_depths

# ---------------------------
# CSV Processing
# ---------------------------

df = pd.read_csv(INPUT_CSV)
# fix depth
df["depth_front"] = df["depth_front"].apply(fix_depth_value)

print("=== DEPTH CLEANUP COMPLETE ===")
print(df["depth_front"].describe())

cols_to_corr = ["depth_front", "steer_us", "throttle_us", "steer_norm", "throttle_norm"]
if set(cols_to_corr).issubset(df.columns):
    print("\n=== CORRELATION WITH DEPTH ===")
    print(df[cols_to_corr].corr())
else:
    print("Warning: some correlation columns not present in CSV; skipping correlation print.")

# Determine image columns by name so metadata tails are preserved unchanged
pixel_columns = []
for column in df.columns:
    if PIXEL_PATTERN.match(str(column)):
        pixel_columns.append(column)
pixel_columns.sort(key=lambda column: (
    int(PIXEL_PATTERN.match(str(column)).group(1)),
    0 if str(column).startswith("R") else 1 if str(column).startswith("G") else 2,
))
num_pixels_in_csv = len(pixel_columns)
expected_pixels = IMG_W * IMG_H * 3
if num_pixels_in_csv != expected_pixels:
    print(f"Warning: expected {expected_pixels} flattened pixels but CSV has {num_pixels_in_csv}. Proceeding with min length.")
    # We'll handle by truncating/padding if needed.

# Convert flattened images to numpy array for faster slicing
flat_images = df.loc[:, pixel_columns].to_numpy(dtype=np.uint8)  # shape [N, num_pixels]
N = flat_images.shape[0]
print(f"Found {N} rows, flattened image columns: {flat_images.shape[1]}")

rows_out = []
samples_saved = 0
sample_imgs = []

# Precompute base rows as pandas Series to copy quickly
# We'll iterate in batches of BATCH_SIZE
idx = 0

tq = tqdm(total=N, desc="Processing rows", ncols=80)
while idx < N:
    batch_idxs = list(range(idx, min(N, idx + BATCH_SIZE)))
    B = len(batch_idxs)
    # Load batch flattened images, handle possible mismatch in flattened length
    batch_flat = flat_images[batch_idxs, :expected_pixels]  # maybe truncation if CSV had extra
    if batch_flat.shape[1] < expected_pixels:
        # pad with zeros
        pad_cols = expected_pixels - batch_flat.shape[1]
        batch_flat = np.concatenate([batch_flat, np.zeros((B, pad_cols), dtype=np.uint8)], axis=1)

    # convert to float tensor in [0,1], shape [B, H, W, 3]
    batch_imgs_np = batch_flat.reshape(B, IMG_H, IMG_W, 3)
    batch_imgs_t = torch.from_numpy(batch_imgs_np.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)

    # metadata arrays
    steer_norms = df.iloc[batch_idxs]["steer_norm"].to_numpy(dtype=float)
    depth_vals = df.iloc[batch_idxs]["depth_front"].to_numpy(dtype=float)
    base_rows = df.iloc[batch_idxs]

    # For each item in batch, append base row (unchanged)
    for i_local, i_global in enumerate(batch_idxs):
        rows_out.append(base_rows.iloc[i_local].copy())

    # INDIVIDUAL AUGMENTS per your probabilities:
    # 25% color, 25% blur/sharpen, 25% noise, 50% flip, and FULL COMBINATION x4.
    # We'll vectorize where possible; otherwise process per-item similar to earlier.

    # 25% COLOR (per-row)
    mask_color = np.random.rand(B) < 0.25
    if mask_color.any():
        strengths = torch.tensor(np.random.uniform(0.25, 0.75, size=mask_color.sum()), dtype=torch.float32, device=device)
        # selected imgs
        sel_idx = np.where(mask_color)[0]
        sel_imgs = batch_imgs_t[sel_idx]
        auged = apply_color_jitter_batch(sel_imgs, strengths)
        for k, ii in enumerate(sel_idx):
            new_row = base_rows.iloc[ii].copy()
            img_uint8 = (auged[k].detach().cpu().numpy().transpose(1,2,0) * 255.0).astype(np.uint8).reshape(-1)
            new_row.loc[pixel_columns] = img_uint8[:len(pixel_columns)]
            rows_out.append(new_row)

    # 25% BLUR/SHARPEN
    mask_blur = np.random.rand(B) < 0.25
    if mask_blur.any():
        strengths = torch.tensor(np.random.uniform(0.25, 0.75, size=mask_blur.sum()), dtype=torch.float32, device=device)
        sel_idx = np.where(mask_blur)[0]
        sel_imgs = batch_imgs_t[sel_idx]
        auged = apply_blur_or_sharpen_batch(sel_imgs, strengths)
        for k, ii in enumerate(sel_idx):
            new_row = base_rows.iloc[ii].copy()
            img_uint8 = (auged[k].detach().cpu().numpy().transpose(1,2,0) * 255.0).astype(np.uint8).reshape(-1)
            new_row.loc[pixel_columns] = img_uint8[:len(pixel_columns)]
            rows_out.append(new_row)

    # 25% NOISE
    mask_noise = np.random.rand(B) < 0.25
    if mask_noise.any():
        strengths = torch.tensor(np.random.uniform(0.25, 0.75, size=mask_noise.sum()), dtype=torch.float32, device=device)
        sel_idx = np.where(mask_noise)[0]
        sel_imgs = batch_imgs_t[sel_idx]
        auged = apply_noise_batch(sel_imgs, strengths)
        for k, ii in enumerate(sel_idx):
            new_row = base_rows.iloc[ii].copy()
            img_uint8 = (auged[k].detach().cpu().numpy().transpose(1,2,0) * 255.0).astype(np.uint8).reshape(-1)
            new_row.loc[pixel_columns] = img_uint8[:len(pixel_columns)]
            rows_out.append(new_row)

    # 50% FLIP
    mask_flip = np.random.rand(B) < 0.50
    if mask_flip.any():
        sel_idx = np.where(mask_flip)[0]
        sel_imgs = batch_imgs_t[sel_idx]
        flipped_imgs = torch.flip(sel_imgs, dims=[3])
        for k, ii in enumerate(sel_idx):
            new_row = base_rows.iloc[ii].copy()
            new_row["steer_norm"] = -float(new_row["steer_norm"])
            img_uint8 = (flipped_imgs[k].detach().cpu().numpy().transpose(1,2,0) * 255.0).astype(np.uint8).reshape(-1)
            new_row.loc[pixel_columns] = img_uint8[:len(pixel_columns)]
            rows_out.append(new_row)

    # FULL COMBINATION x4 (per original code)
    # We'll iterate 4 times, applying full_combination_batch to entire batch (vectorized-ish)
    for _rep in range(4):
        samples_local = []  # will be appended with numpy images
        cimgs_t, s2_arr, d2_arr = full_combination_batch(batch_imgs_t.clone(), steer_norms.copy(), depth_vals.copy(), samples_local)
        # cimgs_t: Tensor[B,3,H,W] on device
        for i_local in range(B):
            new_row = base_rows.iloc[i_local].copy()
            new_row["steer_norm"] = float(s2_arr[i_local])
            new_row["depth_front"] = float(d2_arr[i_local])
            img_uint8 = (cimgs_t[i_local].detach().cpu().numpy().transpose(1,2,0) * 255.0).astype(np.uint8).reshape(-1)
            new_row.loc[pixel_columns] = img_uint8[:len(pixel_columns)]
            rows_out.append(new_row)
        # Save up to 3 sample PNGs
        for sample_img in samples_local:
            if samples_saved >= 3:
                break
            png_path = os.path.join(TEST_OUTPUT_DIR, f"combo_sample_{samples_saved+1}.png")
            cv2.imwrite(png_path, cv2.cvtColor(sample_img, cv2.COLOR_RGB2BGR))
            samples_saved += 1

    idx += B
    tq.update(B)

tq.close()

# Build output DataFrame
out_df = pd.DataFrame(rows_out, columns=df.columns)
out_df.to_csv(OUTPUT_CSV, index=False)

print("\nAugmentation complete!")
print("Original rows:", len(df))
print("Augmented rows:", len(out_df))
print("Saved combination PNG samples to:", TEST_OUTPUT_DIR)
