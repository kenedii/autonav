import os
import random
import re

import cv2
import numpy as np
import pandas as pd

INPUT_CSV = "combined_dataset.csv"
OUTPUT_CSV = "combined_augmented_dataset.csv"
TEST_OUTPUT_DIR = "test_combination_pngs"
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

IMG_W = 160
IMG_H = 120
PIXELS = IMG_W * IMG_H
PIXEL_PATTERN = re.compile(r"^[RGB](\d+)$")


def get_pixel_columns(df):
    pixel_columns = []
    for column in df.columns:
        if PIXEL_PATTERN.match(str(column)):
            pixel_columns.append(column)
    pixel_columns.sort(key=lambda column: (
        int(PIXEL_PATTERN.match(str(column)).group(1)),
        0 if str(column).startswith("R") else 1 if str(column).startswith("G") else 2,
    ))
    return pixel_columns


def flat_to_image(pixel_values):
    rgb = np.asarray(pixel_values, dtype=np.uint8)
    return rgb.reshape((IMG_H, IMG_W, 3))


def image_to_flat(img):
    return img.reshape((-1,))


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
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def augment_noise(img):
    noisy = img.copy()
    num_pixels = int(PIXELS * random.uniform(0.05, 0.15))
    coords = np.random.randint(0, PIXELS, size=num_pixels)
    for c in coords:
        y = c // IMG_W
        x = c % IMG_W
        noise = np.random.normal(0, 25, size=3)
        noisy[y, x] = np.clip(noisy[y, x].astype(float) + noise, 0, 255)
    return noisy


def augment_flip(img, steer_norm):
    return cv2.flip(img, 1), -steer_norm


def augment_random_shadow(img):
    x1, y1 = random.randint(0, IMG_W), 0
    x2, y2 = random.randint(0, IMG_W), IMG_H
    mask = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    cv2.line(mask, (x1, y1), (x2, y2), 1.0, thickness=IMG_W)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    shadow_intensity = random.uniform(0.4, 0.8)
    shaded = img.astype(np.float32)
    shaded[:, :, :] *= (1 - shadow_intensity * mask[:, :, None])
    return np.clip(shaded, 0, 255).astype(np.uint8)


def augment_color_temperature(img):
    shift = random.randint(-30, 30)
    b, g, r = cv2.split(img.astype(np.int16))
    r = np.clip(r + shift, 0, 255)
    b = np.clip(b - shift, 0, 255)
    return cv2.merge([b.astype(np.uint8), g.astype(np.uint8), r.astype(np.uint8)])


def depth_noise(depth):
    return depth + random.uniform(-5.0, 5.0)


def augment_motion_blur(img):
    k = random.choice([3, 5, 7, 9])
    kernel = np.zeros((k, k))
    if random.random() < 0.5:
        kernel[int((k - 1) / 2), :] = np.ones(k)
    else:
        kernel[:, int((k - 1) / 2)] = np.ones(k)
    kernel /= k
    return cv2.filter2D(img, -1, kernel)


def augment_steering_jitter(steer_norm):
    return steer_norm + random.uniform(-0.03, 0.03)


def augment_random_occlusion(img):
    occ_area = int(PIXELS * 0.06)
    max_w = int(IMG_W * 0.4)
    max_h = int(IMG_H * 0.4)
    w = random.randint(5, max_w)
    h = max(5, occ_area // max(w, 1))
    h = min(h, max_h)
    x = random.randint(0, IMG_W - w)
    y = random.randint(0, IMG_H - h)
    occluded = img.copy()
    color = random.randint(0, 50)
    occluded[y:y + h, x:x + w] = (color, color, color)
    return occluded


def full_combination(img, steer_norm, depth_val):
    img = augment_color(img)
    img = augment_blur_or_sharpen(img)
    img = augment_noise(img)
    img = augment_random_shadow(img)
    img = augment_color_temperature(img)
    img = augment_motion_blur(img)
    img = augment_random_occlusion(img)
    img, steer_norm = augment_flip(img, steer_norm)
    steer_norm = augment_steering_jitter(steer_norm)
    depth_val = depth_noise(depth_val)
    return img, steer_norm, depth_val


def fix_depth_value(d):
    if pd.isna(d):
        return d
    if d > 50:
        return d
    if 1.0 < d < 10.0:
        return d * 100.0
    if 0.05 < d < 2.0:
        return d * 1000.0
    return d


def main():
    df = pd.read_csv(INPUT_CSV)
    pixel_columns = get_pixel_columns(df)
    if not pixel_columns:
        raise RuntimeError("No pixel columns found in input CSV.")

    if "depth_front" in df.columns:
        df["depth_front"] = df["depth_front"].apply(fix_depth_value)

    print("=== DEPTH CLEANUP COMPLETE ===")
    if "depth_front" in df.columns:
        print(df["depth_front"].describe())

    cols_to_corr = ["depth_front", "steer_us", "throttle_us", "steer_norm", "throttle_norm"]
    if set(cols_to_corr).issubset(df.columns):
        print("\n=== CORRELATION WITH DEPTH ===")
        print(df[cols_to_corr].corr())

    rows = []
    combo_png_saved = 0

    for _, row in df.iterrows():
        base_row = row.copy()
        steer_norm = row["steer_norm"]
        depth_val = row["depth_front"] if "depth_front" in row else 0.0
        img = flat_to_image(row[pixel_columns].values)

        rows.append(base_row)

        if random.random() < 0.25:
            new = base_row.copy()
            new.loc[pixel_columns] = image_to_flat(augment_color(img))
            rows.append(new)

        if random.random() < 0.25:
            new = base_row.copy()
            new.loc[pixel_columns] = image_to_flat(augment_blur_or_sharpen(img))
            rows.append(new)

        if random.random() < 0.25:
            new = base_row.copy()
            new.loc[pixel_columns] = image_to_flat(augment_noise(img))
            rows.append(new)

        if random.random() < 0.25:
            new = base_row.copy()
            new.loc[pixel_columns] = image_to_flat(augment_random_shadow(img))
            rows.append(new)

        if random.random() < 0.25:
            new = base_row.copy()
            new.loc[pixel_columns] = image_to_flat(augment_color_temperature(img))
            rows.append(new)

        if random.random() < 0.25:
            new = base_row.copy()
            new.loc[pixel_columns] = image_to_flat(augment_motion_blur(img))
            rows.append(new)

        if random.random() < 0.25:
            new = base_row.copy()
            new.loc[pixel_columns] = image_to_flat(augment_random_occlusion(img))
            rows.append(new)

        if random.random() < 0.50:
            fimg, fsteer = augment_flip(img, steer_norm)
            new = base_row.copy()
            new.loc[pixel_columns] = image_to_flat(fimg)
            new["steer_norm"] = fsteer
            rows.append(new)

        for _ in range(4):
            fimg, fsteer, fdepth = full_combination(img, steer_norm, depth_val)
            new = base_row.copy()
            new.loc[pixel_columns] = image_to_flat(fimg)
            new["steer_norm"] = fsteer
            new["depth_front"] = fdepth
            rows.append(new)

        if combo_png_saved < 3:
            combo_path = os.path.join(TEST_OUTPUT_DIR, f"combo_{combo_png_saved:02d}.png")
            cv2.imwrite(combo_path, cv2.cvtColor(full_combination(img, steer_norm, depth_val)[0], cv2.COLOR_RGB2BGR))
            combo_png_saved += 1

    df_out = pd.DataFrame(rows, columns=df.columns)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"Augmented dataset saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
