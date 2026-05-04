import json
import pandas as pd
from pathlib import Path

annot_path = Path("C:/turtle_project/data/raw/annotations.json")
splits_dir = Path("C:/turtle_project/splits")
data_root = Path("C:/turtle_project/data/raw")   # not: images hassas

with open(annot_path) as f:
    coco = json.load(f)

img_info = {}
for img in coco['images']:
    # "images/t396/..." -> direkt kullan
    img_info[img['id']] = {
        'path': str(data_root / img['path']),   # artık doğru: .../raw/images/t396/...
        'date': img['date'].split()[0].replace(':', '-')
    }

records = []
for ann in coco['annotations']:
    img_id = ann['image_id']
    identity = ann['identity']
    records.append({
        'file_path': img_info[img_id]['path'],
        'identity': identity,
        'date': img_info[img_id]['date']
    })

df = pd.DataFrame(records)

train_paths, val_paths, test_paths = [], [], []
for identity, group in df.groupby('identity'):
    group = group.sort_values('date')
    n = len(group)
    if n == 1:
        train_paths.extend(group['file_path'].tolist())
    else:
        train_cut = max(1, int(n * 0.7))
        val_cut = int(n * 0.85)
        train_paths.extend(group.iloc[:train_cut]['file_path'].tolist())
        val_paths.extend(group.iloc[train_cut:val_cut]['file_path'].tolist())
        test_paths.extend(group.iloc[val_cut:]['file_path'].tolist())

splits_dir.mkdir(exist_ok=True)
for name, lst in [("train.txt", train_paths), ("val.txt", val_paths), ("test.txt", test_paths)]:
    with open(splits_dir / name, "w") as f:
        f.write("\n".join(lst))

print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")