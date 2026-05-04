import json
from pathlib import Path

annot = Path("C:/turtle_project/data/raw/annotations.json")
with open(annot) as f:
    data = json.load(f)

# image_id -> date mapping
img_date = {}
for img in data["images"]:
    date_str = img["date"].split()[0].replace(":", "-")
    img_date[img["id"]] = date_str

# annotation listesinden t392 için görüntüleri bul
t392_images = []
for ann in data["annotations"]:
    if ann["identity"] == "t392":
        img_id = ann["image_id"]
        t392_images.append(img_id)

print(f"t392 için toplam {len(t392_images)} görüntü var.")
print("İlk 5 image_id:", t392_images[:5])
print("Tarihleri:")
for img_id in t392_images[:10]:
    print(f"  {img_id}: {img_date.get(img_id, 'bilinmiyor')}")