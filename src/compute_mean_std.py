import cv2
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm

splits_dir = Path("C:/turtle_project/splits")
with open(splits_dir / "train.txt") as f:
    train_paths = [line.strip() for line in f.readlines()]

mean = np.zeros(3)
std = np.zeros(3)
n = 0

for p in tqdm(train_paths):
    img = cv2.imread(p)
    if img is None:
        print(f"Uyarı: {p} okunamadı")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    mean += img.sum(axis=(0,1))
    std += (img**2).sum(axis=(0,1))
    n += img.shape[0]*img.shape[1]

mean /= n
std = np.sqrt(std / n - mean**2)

mean_std = {"mean": mean.tolist(), "std": std.tolist()}
with open(Path("C:/turtle_project/data/mean_std.json"), "w") as f:
    json.dump(mean_std, f, indent=4)

print(f"Mean: {mean}, Std: {std}")