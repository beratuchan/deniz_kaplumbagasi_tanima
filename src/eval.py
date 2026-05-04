import torch
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).parent))
from dataset import TurtleDataset, get_val_transform
from model import TurtleClassifier
from utils import compute_metrics

# ---------- KONFİG ----------
DATA_ROOT = Path("C:/turtle_project/data/raw")
SPLITS_DIR = Path("C:/turtle_project/splits")
CHECKPOINT_DIR = Path("C:/turtle_project/outputs/checkpoints")
OUTPUT_DIR = Path("C:/turtle_project/outputs/eval_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {DEVICE}")

# ---------- ETİKET MAPPING ----------
def get_label_mapping():
    annot_path = Path("C:/turtle_project/data/raw/annotations.json")
    with open(annot_path, 'r') as f:
        data = json.load(f)
    unique_ids = sorted(set(ann["identity"] for ann in data["annotations"]))
    label_map = {id: idx for idx, id in enumerate(unique_ids)}
    inv_label_map = {idx: id for id, idx in label_map.items()}
    return label_map, inv_label_map, len(unique_ids)

label_map, inv_label_map, num_classes = get_label_mapping()
print(f"Sınıf sayısı: {num_classes}")

# ---------- MEAN/STD ----------
with open(Path("C:/turtle_project/data/mean_std.json"), 'r') as f:
    mean_std = json.load(f)
    mean = mean_std["mean"]
    std = mean_std["std"]

# ---------- TEST DATASET ----------
test_dataset = TurtleDataset(
    split_file=SPLITS_DIR / "test.txt",
    img_root=DATA_ROOT,
    label_map=label_map,
    transform=get_val_transform(mean=mean, std=std)
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

# ---------- MODEL YÜKLE ----------
model = TurtleClassifier(num_classes=num_classes, pretrained=False)
checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
if not checkpoint_path.exists():
    raise FileNotFoundError(f"Model checkpoint bulunamadı: {checkpoint_path}")
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("Model yüklendi.")

# ---------- TAHMİN TOPLA ----------
all_preds = []
all_labels = []
all_logits = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Test değerlendirmesi"):
        images = images.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_logits.extend(outputs.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_logits = np.array(all_logits)

# ---------- METRİKLER ----------
# Rank-1, Rank-5
test_metrics = compute_metrics(torch.from_numpy(all_logits), torch.from_numpy(all_labels))
rank1 = test_metrics['rank1']
rank5 = test_metrics['rank5']

# Balanced accuracy
bal_acc = balanced_accuracy_score(all_labels, all_preds)

# Classification report (sadece makale için özet)
class_report = classification_report(all_labels, all_preds, target_names=[inv_label_map[i] for i in range(num_classes)], zero_division=0)

# Confusion matrix (küçük boyutlu kaydet, çok büyük olabilir ama 400x400)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(20, 20))
sns.heatmap(cm, fmt='d', cmap='Blues', xticklabels=False, yticklabels=False)
plt.title("Confusion Matrix (Test Seti)")
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()

# Sonuçları JSON'a yaz
results = {
    "rank1": float(rank1),
    "rank5": float(rank5),
    "balanced_accuracy": float(bal_acc),
    "num_samples": len(all_labels),
    "num_classes": num_classes,
}
with open(OUTPUT_DIR / "test_results.json", 'w') as f:
    json.dump(results, f, indent=4)

# Ayrıca per-class doğrulukları hesapla istenirse
per_class_acc = {}
for c in range(num_classes):
    mask = (all_labels == c)
    if mask.sum() > 0:
        acc = (all_preds[mask] == c).sum() / mask.sum()
        per_class_acc[inv_label_map[c]] = float(acc)
    else:
        per_class_acc[inv_label_map[c]] = None

with open(OUTPUT_DIR / "per_class_accuracy.json", 'w') as f:
    json.dump(per_class_acc, f, indent=4)

print(f"Test Rank-1: {rank1:.4f}")
print(f"Test Rank-5: {rank5:.4f}")
print(f"Test Balanced Accuracy: {bal_acc:.4f}")
print(f"Sonuçlar {OUTPUT_DIR} klasörüne kaydedildi.")