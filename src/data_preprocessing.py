import json
import shutil
from pathlib import Path
import pandas as pd
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime

# ---------- KONFİG ----------
PROJECT_ROOT = Path(__file__).parent.parent
RAW_IMAGES_DIR = PROJECT_ROOT / "data"
PROCESSED_IMAGES_DIR = PROJECT_ROOT / "data" / "processed" / "images_resized"
ANNOT_PATH = PROJECT_ROOT / "data" / "raw" / "annotations.json"
SPLITS_DIR = PROJECT_ROOT / "splits"
MEAN_STD_PATH = PROJECT_ROOT / "data" / "mean_std.json"

TARGET_SIZE = (256, 256)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------- 1. ANNOTATION YÜKLEME (COCO formatı) ----------
def load_annotations():
    with open(ANNOT_PATH, "r") as f:
        coco = json.load(f)
    
    # Haritalar: image_id -> (file_name, date)
    img_map = {}
    for img in coco["images"]:
        # date format: "2016:07:11 16:53:31" -> parse to "YYYY-MM-DD"
        date_str = img["date"].split()[0].replace(":", "-")  # 2016:07:11 -> 2016-07-11
        img_map[img["id"]] = {
            "file_path": RAW_IMAGES_DIR / img["path"],   # path zaten "images/t281/..."
            "date": date_str
        }
    
    records = []
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        identity = ann["identity"]     # birey etiketi, örn: "t145"
        if img_id not in img_map:
            continue
        file_path = img_map[img_id]["file_path"]
        date = img_map[img_id]["date"]
        records.append({
            "file_path": str(file_path),
            "individual_id": identity,
            "date": date
        })
    
    df = pd.DataFrame(records)
    print(f"Toplam görüntü: {len(df)}")
    print(f"Benzersiz birey: {df['individual_id'].nunique()}")
    print(f"Tarih aralığı: {df['date'].min()} -> {df['date'].max()}")
    return df

# ---------- 2. BOYUTLANDIRMA (processed klasörüne) ----------
def resize_images(df):
    PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Resizing"):
        src_path = Path(row["file_path"])
        # relative path: images/t281/example.JPG -> t281/example.JPG
        rel_path = src_path.relative_to(RAW_IMAGES_DIR)
        dst_path = PROCESSED_IMAGES_DIR / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img = cv2.imread(str(src_path))
        if img is None:
            print(f"Uyarı: okunamadı {src_path}")
            continue
        img_resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst_path), img_resized)
    print("Boyutlandırma tamamlandı.")

# ---------- 3. ZAMAN-AWARE SPLIT (birey+gün bazlı) ----------
def create_time_aware_split(df):
    df["group_key"] = df["individual_id"] + "_" + df["date"]
    group_df = df.groupby("group_key").agg({
        "date": "first",
        "file_path": list
    }).reset_index()
    group_df = group_df.sort_values("date").reset_index(drop=True)
    n_groups = len(group_df)
    train_cut = int(n_groups * TRAIN_RATIO)
    val_cut = int(n_groups * (TRAIN_RATIO + VAL_RATIO))
    train_groups = group_df.iloc[:train_cut]
    val_groups = group_df.iloc[train_cut:val_cut]
    test_groups = group_df.iloc[val_cut:]
    
    def flatten_paths(groups):
        paths = []
        for _, row in groups.iterrows():
            paths.extend(row["file_path"])
        return paths
    
    train_paths = flatten_paths(train_groups)
    val_paths = flatten_paths(val_groups)
    test_paths = flatten_paths(test_groups)
    
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_DIR / "train.txt", "w") as f:
        f.write("\n".join(train_paths))
    with open(SPLITS_DIR / "val.txt", "w") as f:
        f.write("\n".join(val_paths))
    with open(SPLITS_DIR / "test.txt", "w") as f:
        f.write("\n".join(test_paths))
    
    print(f"Train: {len(train_groups)} grup, {len(train_paths)} görüntü")
    print(f"Val: {len(val_groups)} grup, {len(val_paths)} görüntü")
    print(f"Test: {len(test_groups)} grup, {len(test_paths)} görüntü")
    return train_paths, val_paths, test_paths

# ---------- 4. MEAN/STD HESAPLAMA (train seti üzerinden) ----------
def compute_mean_std(train_paths):
    mean = np.zeros(3, dtype=np.float64)
    std = np.zeros(3, dtype=np.float64)
    n_pixels = 0
    for img_path in tqdm(train_paths, desc="Mean/Std"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        mean += img.sum(axis=(0,1))
        std += (img ** 2).sum(axis=(0,1))
        n_pixels += img.shape[0] * img.shape[1]
    mean /= n_pixels
    std = np.sqrt(std / n_pixels - mean ** 2)
    mean_std = {"mean": mean.tolist(), "std": std.tolist()}
    with open(MEAN_STD_PATH, "w") as f:
        json.dump(mean_std, f, indent=4)
    print(f"Mean: {mean}, Std: {std}")
    return mean_std

# ---------- ANA AKIŞ ----------
if __name__ == "__main__":
    df = load_annotations()
    if not PROCESSED_IMAGES_DIR.exists():
        resize_images(df)
    else:
        print("Boyutlandırılmış görüntüler zaten var, atlanıyor.")
    train_paths, val_paths, test_paths = create_time_aware_split(df)
    compute_mean_std(train_paths)
    print("Aşama 2 tamamlandı.")