import os
import cv2
import pandas as pd
import numpy as np
import random

# Augmentation script that applies photographic augmentations to RGB only,
# and geometric augmentations to all relevant images (RGB, IR, Depth).

# Dynamically find the cleaned CSV from the runs_rgb_depth folder structure
# We assume this runs from train/data_frontend or train/ directory
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'runs_rgb_depth', 'runs_rgb_depth')
INPUT_CSV = os.path.join(BASE_DIR, 'combined_cleaned_dataset.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'augmented_runs')
OUTPUT_CSV = os.path.join(BASE_DIR, 'combined_augmented_dataset.csv')

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------
def ensure_full_path(rel_path):
    if pd.isna(rel_path) or not str(rel_path): 
        return None
    p = str(rel_path)
    
    # Original paths look like `runs_rgb_depth/run_2026.../rgb_0000.png`
    # We want to match it with our train directory: train/runs_rgb_depth/runs_rgb_depth/run...
    # So if it starts with runs_rgb_depth/ we prefix it with another runs_rgb_depth/ 
    # to form the actual path inside train/runs_rgb_depth/runs_rgb_depth
    if p.startswith('runs_rgb_depth/'):
        p = os.path.join('runs_rgb_depth', p)

    full_p = os.path.join(os.path.dirname(os.path.dirname(__file__)), p)
    return full_p

def augment_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + random.uniform(-10, 10)) % 180
    hsv[..., 1] *= random.uniform(0.7, 1.3)
    hsv[..., 2] *= random.uniform(0.7, 1.3)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def augment_blur_or_sharpen(img):
    if random.random() < 0.5:
        return cv2.GaussianBlur(img, (5, 5), 1.2)
    else:
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)

def augment_noise(img):
    noisy = img.copy()
    h, w, c = img.shape
    num_pixels = int(h * w * random.uniform(0.05, 0.15))
    coords_y = np.random.randint(0, h, size=num_pixels)
    coords_x = np.random.randint(0, w, size=num_pixels)
    
    noises = np.random.normal(0, 25, size=(num_pixels, c))
    noisy[coords_y, coords_x] = np.clip(noisy[coords_y, coords_x].astype(float) + noises, 0, 255)
    return noisy

def augment_random_shadow(img):
    h, w, _ = img.shape
    x1, y1 = random.randint(0, w), 0
    x2, y2 = random.randint(0, w), h
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.line(mask, (x1, y1), (x2, y2), 1.0, thickness=w)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    shadow_intensity = random.uniform(0.4, 0.8)
    shaded = img.astype(np.float32)
    shaded[:, :, :] *= (1 - shadow_intensity * mask[:, :, None])
    return np.clip(shaded, 0, 255).astype(np.uint8)

# Geometric Augmentations
def augment_motion_blur(img):
    k = random.choice([3, 5, 7])
    kernel = np.zeros((k, k))
    if random.random() < 0.5:
        kernel[int((k-1)/2), :] = np.ones(k)
    else:
        kernel[:, int((k-1)/2)] = np.ones(k)
    kernel /= k
    return cv2.filter2D(img, -1, kernel)

def augment_flip(img):
    return cv2.flip(img, 1)

def apply_photo_aug(img):
    funcs = [augment_color, augment_blur_or_sharpen, augment_noise, augment_random_shadow]
    random.shuffle(funcs)
    for f in funcs[:2]: 
        img = f(img)
    return img

def apply_geo_aug_all(img_dict, flip=False, motion_blur=False):
    out = {}
    for k, v in img_dict.items():
        if v is None:
            out[k] = None
            continue
        res = v.copy()
        if motion_blur:
            res = augment_motion_blur(res)
        if flip:
            res = augment_flip(res)
        out[k] = res
    return out

# -----------------------------------------------------------
# MAIN SCRIPT
# -----------------------------------------------------------
def main():
    print(f"Reading dataset from {INPUT_CSV}")
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found! Did you run clean_dataset.py?")
        return
    
    df = pd.read_csv(INPUT_CSV)
    
    rows = []
    pixel_data = [] # To store the new flattened pixels separately 
    total_rows = len(df)
    
    img_cols = ['rgb_path', 'cam1_path', 'ir_path', 'depth_path']
    for c in img_cols:
        if c not in df.columns:
            df[c] = None

    print(f"Starting augmentation of {total_rows} rows...")
    for idx, row in df.iterrows():
        base_row = row.copy()
        steer_norm = float(row["steer_norm"])
        
        # Read the images
        imgs = {}
        for c in img_cols:
            p = ensure_full_path(row[c])
            if p and os.path.exists(p):
                if 'depth' in c:
                    imgs[c] = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                else:
                    imgs[c] = cv2.imread(p, cv2.IMREAD_COLOR)
            else:
                imgs[c] = None

        # 1. Store original row
        # (Exclude original pixel columns starting with px_...)
        non_px_cols = [c for c in df.columns if not str(c).startswith('px_')]
        base_row_filtered = base_row[non_px_cols].copy()
        
        rows.append(base_row_filtered)
        if imgs.get('rgb_path') is not None:
             pixel_data.append(cv2.cvtColor(imgs['rgb_path'], cv2.COLOR_BGR2RGB).flatten())
        else:
             pixel_data.append(np.zeros(57600, dtype=np.uint8))
        
        base_name = f"aug_{idx}"
        
        # Variation 1: Photographic ONLY (applies to RGB)
        # Probability 50%
        if random.random() < 0.5 and imgs['rgb_path'] is not None:
            new_row = base_row_filtered.copy()
            for k, v in imgs.items():
                if v is not None:
                    res = apply_photo_aug(v.copy()) if k in ['rgb_path', 'cam1_path'] else v.copy()
                    out_name = f"{base_name}_photo_{k}.png"
                    out_p = os.path.join(OUTPUT_DIR, out_name)
                    cv2.imwrite(out_p, res)
                    new_row[k] = f"runs_rgb_depth/runs_rgb_depth/augmented_runs/{out_name}"
                    
                    if k == 'rgb_path':
                        pixel_data.append(cv2.cvtColor(res, cv2.COLOR_BGR2RGB).flatten())
            rows.append(new_row)
            
        # Variation 2: Geometric + Photographic (applies to ALL, photographic to RGB)
        # Probability 50%
        if random.random() < 0.5 and imgs['rgb_path'] is not None:
            new_row = base_row_filtered.copy()
            new_row['steer_norm'] = -steer_norm  # Horizontal Flip logic applied
            new_row['steer_us'] = 1500 - (row['steer_us'] - 1500)
            
            geo_dict = apply_geo_aug_all(imgs, flip=True, motion_blur=(random.random() < 0.3))
            
            for k, v in geo_dict.items():
                if v is not None:
                    res = apply_photo_aug(v.copy()) if k in ['rgb_path', 'cam1_path'] else v.copy()
                    out_name = f"{base_name}_geo_{k}.png"
                    out_p = os.path.join(OUTPUT_DIR, out_name)
                    cv2.imwrite(out_p, res)
                    new_row[k] = f"runs_rgb_depth/runs_rgb_depth/augmented_runs/{out_name}"
                    
                    if k == 'rgb_path':
                        pixel_data.append(cv2.cvtColor(res, cv2.COLOR_BGR2RGB).flatten())
                    
            rows.append(new_row)

        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{total_rows}")

    print("Converting generated augmentations to dataframe...")
    out_df = pd.DataFrame(rows)
    
    print("Formatting pixel data...")
    num_pixels = len(pixel_data[0]) if pixel_data else 57600
    pixel_columns = [f"px_{i}" for i in range(num_pixels)]
    pixel_df = pd.DataFrame(pixel_data, columns=pixel_columns)
    
    out_df = out_df.reset_index(drop=True)
    pixel_df = pixel_df.reset_index(drop=True)
    out_df = pd.concat([out_df, pixel_df], axis=1)

    print("Saving to CSV... (This will take a moment given file size)")
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Augmentation complete! Output: {OUTPUT_CSV}")
    print(f"Original len: {total_rows} | New len: {len(out_df)}")


if __name__ == '__main__':
    main()
