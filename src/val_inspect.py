import torch
import cv2
import json
import numpy as np
from pathlib import Path
import sys

project_root = Path("C:/turtle_project")
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from model import TurtleClassifier
from dataset import get_val_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
annot_path = project_root / "data/raw/annotations.json"
with open(annot_path, "r") as f:
    data = json.load(f)
unique_ids = sorted(set(ann["identity"] for ann in data["annotations"]))
label_map = {id: idx for idx, id in enumerate(unique_ids)}
inv_label_map = {v: k for k, v in label_map.items()}
num_classes = len(unique_ids)
print(f"Sınıf sayısı: {num_classes}")

# Model
model = TurtleClassifier(num_classes=num_classes, pretrained=False)
checkpoint_path = project_root / "outputs/checkpoints/best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

transform = get_val_transform()

# İlk validation görüntüsü
val_txt = project_root / "splits/val.txt"
with open(val_txt, "r") as f:
    first_path = f.readline().strip()

img = cv2.imread(first_path)
if img is None:
    print(f"HATA: Görüntü okunamadı: {first_path}")
    sys.exit(1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

augmented = transform(image=img)
img_array = augmented['image']   # numpy array (C, H, W) or (H,W,C)? Albumentations Normalize returns float32 array in channel-first? Actually Compose with Normalize returns HWC? We need to check.
# get_val_transform returns Resize + Normalize, Normalize returns same channel order. Typically albumentations keeps HWC. Then we need to convert to tensor and permute.

img_tensor = torch.from_numpy(img_array).permute(2,0,1).float().unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    pred_idx = output.argmax(dim=1).item()
    pred_identity = inv_label_map[pred_idx]

true_identity = Path(first_path).parent.name
print(f"Dosya: {first_path}")
print(f"Gerçek ID: {true_identity}")
print(f"Tahmin edilen ID: {pred_identity}")
print(f"Doğru mu? {true_identity == pred_identity}")