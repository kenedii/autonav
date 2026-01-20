# tools/data_management.py
import os
import json
import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
import csv
import zipfile
import shutil

# Session state with unique prefix to avoid conflicts
if "dm_idx" not in st.session_state:
    st.session_state.dm_idx = 0
if "dm_current_folder" not in st.session_state:
    st.session_state.dm_current_folder = None

THUMB_WINDOW = 10

# ────────────────────────────────
# Smart image path resolver
# ────────────────────────────────
def resolve_image_path(root_folder: str, raw_path: str) -> str:
    if not raw_path or str(raw_path).strip() in ["", "nan", "None"]:
        return ""

    raw = str(raw_path).strip().replace("\\", "/")
    prefixes = ["runs_rgb_depth/", "runs_rgb_depth/run_", "rgb_depth/", "rgb/", "../rgb/"]
    cleaned = raw
    for p in prefixes:
        if cleaned.startswith(p):
            cleaned = cleaned[len(p):]

    bases = [
        root_folder,
        str(Path(root_folder).parent),
        str(Path(root_folder).parent.parent),
        str(Path(root_folder).parent / "rgb"),
        str(Path(root_folder).parent / "runs_rgb_depth"),
    ]

    candidates = []
    for base in bases:
        if os.path.isdir(base):
            p = os.path.normpath(os.path.join(base, cleaned))
            candidates.extend([p, p.replace(".png", ".jpg"), p.replace(".jpg", ".png")])

    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0] if candidates else os.path.join(root_folder, raw)


# ────────────────────────────────
# Create combined pixel-flattened CSV
# ────────────────────────────────
def create_combined_csv(root_dir: str):
    output_csv = os.path.join(root_dir, "combined_dataset.csv")

    if os.path.exists(output_csv):
        st.warning("Overwriting existing combined_dataset.csv...")

    with st.spinner("Scanning runs and finding sample image..."):
        subdirs = [d for d in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('run_')]
        if not subdirs:
            st.error("No run_* folders found.")
            return

        # Find one valid image to get image size
        sample_image_path = None
        for run_dir in subdirs:
            run_path = os.path.join(root_dir, run_dir)
            csv_path = os.path.join(run_path, "dataset.csv")
            if not os.path.exists(csv_path):
                continue
            df_temp = pd.read_csv(csv_path)
            img_col = next((c for c in df_temp.columns if any(x in c.lower() for x in ["rgb", "image", "path"])), None)
            if not img_col:
                continue
            for val in df_temp[img_col]:
                if pd.notna(val):
                    full_path = resolve_image_path(run_path, val)
                    if os.path.exists(full_path):
                        sample_image_path = full_path
                        break
            if sample_image_path:
                break

        if not sample_image_path:
            st.error("Could not find any valid image in the runs.")
            return

        img = Image.open(sample_image_path).convert('RGB')
        width, height = img.size
        num_pixels = width * height

    # Header
    header = ['timestamp', 'steer_us', 'throttle_us', 'steer_norm', 'throttle_norm', 'depth_front']
    for i in range(1, num_pixels + 1):
        header.extend([f'R{i}', f'G{i}', f'B{i}'])

    all_rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, run_dir in enumerate(subdirs):
        status_text.text(f"Processing {run_dir} ({i+1}/{len(subdirs)})")
        run_path = os.path.join(root_dir, run_dir)
        csv_path = os.path.join(run_path, "dataset.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        img_col = next((c for c in df.columns if any(x in c.lower() for x in ["rgb", "image", "path"])), None)
        if not img_col:
            continue

        for _, row in df.iterrows():
            rgb_path = row.get(img_col)
            if pd.isna(rgb_path):
                continue
            full_img = resolve_image_path(run_path, rgb_path)
            if not os.path.exists(full_img):
                continue
            try:
                img = Image.open(full_img).convert('RGB')
                if img.size != (width, height):
                    continue
                pixels = [str(v) for pixel in img.getdata() for v in pixel]
                data_row = [
                    str(row.get("timestamp", "")),
                    str(row.get("steer_us", "")),
                    str(row.get("throttle_us", "")),
                    str(row.get("steer_norm", 0)),
                    str(row.get("throttle_norm", 0)),
                    str(row.get("depth_front", ""))
                ] + pixels
                all_rows.append(data_row)
            except Exception as e:
                st.warning(f"Error reading {full_img}: {e}")

        progress_bar.progress((i + 1) / len(subdirs))

    # Write final CSV
    status_text.text("Writing combined_dataset.csv...")
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_rows)

    progress_bar.empty()
    status_text.empty()
    st.success(f"Combined dataset created!\n→ `{output_csv}`\nFrames: {len(all_rows)}")


# ────────────────────────────────
# Helper functions
# ────────────────────────────────
def is_csv_folder(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "dataset.csv"))

def load_csv_dataset(path: str):
    csv_path = os.path.join(path, "dataset.csv")
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df, csv_path

def find_image_column(df: pd.DataFrame):
    for col in ["rgb_path", "rgbpath", "image_path", "image"]:
        if col in df.columns:
            return col
    for col in df.columns:
        if any(k in col.lower() for k in ["rgb", "image", "path"]):
            return col
    return None

def delete_csv_frame(df, csv_path, idx, folder):
    row = df.iloc[idx]
    img_col = find_image_column(df)
    if img_col and pd.notna(row.get(img_col)):
        p = resolve_image_path(folder, row[img_col])
        if os.path.exists(p):
            try:
                os.remove(p)
            except:
                pass
    df = df.drop(idx).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    return df

def load_old_frame(folder: str, idx: int):
    img_p = os.path.join(folder, f"{idx}.jpg")
    json_p = os.path.join(folder, f"{idx}.json")
    if os.path.exists(img_p) and os.path.exists(json_p):
        try:
            img = Image.open(img_p)
            with open(json_p) as f:
                meta = json.load(f)
            return img, meta
        except:
            pass
    return None, None

def old_frame_count(folder: str):
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")])


# ────────────────────────────────
# Main page function (NO SIDEBAR!)
# ────────────────────────────────
def show():
    if "dm_current_folder" not in st.session_state:
        st.session_state.dm_current_folder = None
    if "dm_idx" not in st.session_state:
        st.session_state.dm_idx = 0

    st.title("RC Car Dataset Viewer & Cleaner")
    st.markdown("View • Clean • Delete frames • Bulk actions • Create pixel-flattened dataset")

    # ────────────────────────────────
    # Upload Section
    # ────────────────────────────────
    with st.expander("📂 Upload Dataset (Zip)", expanded=True):
        uploaded_zip = st.file_uploader("Upload a zip file containing your run folders (e.g. run_1, run_2)", type="zip")
        if uploaded_zip is not None:
            if st.button("Extract & Load Data", type="primary"):
                with st.spinner("Unzipping data..."):
                    # Create a 'data' folder in the current working directory
                    extract_root = os.path.join(os.getcwd(), "data")
                    os.makedirs(extract_root, exist_ok=True)
                    
                    try:
                        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                            zip_ref.extractall(extract_root)
                        st.success(f"Successfully extracted data to `{extract_root}`")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error extracting zip: {e}")

    st.markdown("---")

    # Scan for available directories in the current working directory
    base_dir = os.getcwd()
    # List directories, excluding hidden ones
    available_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]
    available_dirs = sorted(available_dirs)
    
    # Option to use the current directory itself
    available_dirs.insert(0, ".")

    # Default to 'data' if it exists, otherwise '.'
    default_ix = 0
    if "data" in available_dirs:
        default_ix = available_dirs.index("data")

    root_folder_name = st.selectbox(
        "Select Root Folder (containing run folders):",
        options=available_dirs,
        index=default_ix,
        key="dm_root_select"
    )
    
    root_folder = os.path.join(base_dir, root_folder_name)

    if not root_folder or not os.path.isdir(root_folder):
        st.info("Please select a valid folder.")
        st.stop()

    # Button to create combined CSV
    if st.button("Create combined_dataset.csv (pixel-flattened)", type="primary", use_container_width=True):
        create_combined_csv(root_folder)

    st.markdown("---")

    folders = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    if not folders:
        st.warning("No subfolders found.")
        st.stop()

    selected = st.selectbox("Select run/session:", sorted(folders), key="dm_sel")
    folder_path = os.path.join(root_folder, selected)

    # Reset index when folder changes
    if st.session_state.dm_current_folder != selected:
        st.session_state.dm_idx = 0
        st.session_state.dm_current_folder = selected

    is_csv = is_csv_folder(folder_path)

    # ───── CSV-based runs ─────
    if is_csv:
        df, csv_path = load_csv_dataset(folder_path)
        if df is None:
            st.stop()

        total = len(df)
        img_col = find_image_column(df)

        # Bulk delete zero throttle
        if "throttle_norm" in df.columns:
            if st.button("Delete all frames with throttle_norm == 0", type="secondary"):
                with st.spinner("Deleting..."):
                    before = len(df)
                    for i in reversed(df[df["throttle_norm"] == 0].index.tolist()):
                        df = delete_csv_frame(df, csv_path, i, folder_path)
                    st.success(f"Removed {before - len(df)} zero-throttle frames")
                    st.rerun()

        # Navigation
        c1, c2, c3 = st.columns([1, 3, 1])
        if c1.button("Previous", use_container_width=True):
            st.session_state.dm_idx = max(0, st.session_state.dm_idx - 1)
            st.rerun()
        st.session_state.dm_idx = c2.number_input("Frame", 0, total-1, st.session_state.dm_idx, step=1, key="dm_nav")
        if c3.button("Next", use_container_width=True):
            st.session_state.dm_idx = min(total-1, st.session_state.dm_idx + 1)
            st.rerun()

        idx = st.session_state.dm_idx
        row = df.iloc[idx]
        img_path = resolve_image_path(folder_path, row[img_col]) if img_col and pd.notna(row.get(img_col)) else None

        col_img, col_info = st.columns([4, 2])
        if img_path and os.path.exists(img_path):
            col_img.image(Image.open(img_path), caption=f"Frame {idx+1}/{total}", use_column_width=True)
        else:
            col_img.error("Image not found")
            if img_col:
                col_img.code(row.get(img_col, "N/A"))

        with col_info:
            st.subheader("Controls")
            t = float(row.get("throttle_norm", 0))
            s = float(row.get("steer_norm", 0))
            st.write(f"**Throttle:** {t:.3f}")
            st.progress(min(max(t, 0), 1))
            direction = "LEFT" if s > 0.05 else "RIGHT" if s < -0.05 else "STRAIGHT"
            st.write(f"**Steering:** {direction} ({s:+.3f})")

        if st.button("Delete This Frame", type="primary", use_container_width=True):
            df = delete_csv_frame(df, csv_path, idx, folder_path)
            st.success("Frame deleted!")
            if idx >= len(df) and len(df) > 0:
                st.session_state.dm_idx = len(df) - 1
            st.rerun()

        # Thumbnail strip
        st.markdown("### Thumbnail Strip")
        half = THUMB_WINDOW // 2
        start = max(0, idx - half)
        end = min(total, start + THUMB_WINDOW)
        if end == max(end, start + 1):  # at least one
            thumb_cols = st.columns(THUMB_WINDOW)
        for i in range(THUMB_WINDOW):
            pos = start + i
            if pos >= total:
                continue
            path = resolve_image_path(folder_path, df.iloc[pos][img_col]) if img_col else None
            if path and os.path.exists(path):
                thumb_cols[i].image(Image.open(path).resize((130, 90)), use_column_width=True)
            else:
                thumb_cols[i].write("missing")
            if thumb_cols[i].button(str(pos), key=f"thumb_{pos}"):
                st.session_state.dm_idx = pos
                st.rerun()

    # ───── Old format (idx.jpg + idx.json) ─────
    else:
        total = old_frame_count(folder_path)
        if total == 0:
            st.error("No frames found in old format.")
            st.stop()

        c1, c2, c3 = st.columns([1, 3, 1])
        if c1.button("Previous", use_container_width=True):
            st.session_state.dm_idx = max(0, st.session_state.dm_idx - 1)
            st.rerun()
        st.session_state.dm_idx = c2.number_input("Frame", 0, total-1, st.session_state.dm_idx, step=1, key="dm_old")
        if c3.button("Next", use_container_width=True):
            st.session_state.dm_idx = min(total-1, st.session_state.dm_idx + 1)
            st.rerun()

        img, meta = load_old_frame(folder_path, st.session_state.dm_idx)
        if not img:
            st.error("Frame missing")
            st.stop()

        col_img, col_info = st.columns([4, 2])
        col_img.image(img, caption=f"Frame {st.session_state.dm_idx}", use_column_width=True)

        with col_info:
            st.subheader("Metadata")
            steer = meta.get("steering", 0) if meta else 0
            throttle = meta.get("throttle", 0) if meta else 0
            st.write(f"**Steering:** {steer:+.3f}")
            st.write(f"**Throttle:** {throttle:.3f}")
            with st.expander("Full JSON"):
                st.json(meta)

        if st.button("Delete This Frame", type="primary", use_container_width=True):
            for ext in [".jpg", ".json"]:
                p = os.path.join(folder_path, f"{st.session_state.dm_idx}{ext}")
                if os.path.exists(p):
                    os.remove(p)
            st.success("Deleted")
            st.rerun()

    st.caption(f"Root: {root_folder} • Session: {selected} • {'CSV Run' if is_csv else 'Old Format'} • Frames: {total}")
