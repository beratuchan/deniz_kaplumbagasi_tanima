import cv2
from pathlib import Path

val_txt = Path("C:/turtle_project/splits/val.txt")
with open(val_txt) as f:
    paths = [line.strip() for line in f.readlines()]

print(f"Toplam validation görüntüsü: {len(paths)}")

okunan = 0
okunamayan = 0
for p in paths[:20]:  # ilk 20'yi kontrol et
    img = cv2.imread(p)
    if img is not None:
        okunan += 1
        print(f"OK: {p}")
    else:
        okunamayan += 1
        print(f"HATA: {p} okunamadı")

print(f"Okunan: {okunan}, Okunamayan: {okunamayan}")