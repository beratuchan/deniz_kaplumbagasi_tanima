# Deniz Kaplumbağası Bireysel Tanıma

PyTorch ve ResNet50 transfer learning kullanarak deniz kaplumbağalarını bireysel düzeyde tanımlayan (re-identification) derin öğrenme projesi.

## Proje Hakkında

400 farklı kaplumbağa bireyi arasından kimlik tespiti yapan bir görüntü sınıflandırma sistemi. Tür tespitinin ötesinde, her kaplumbağayı ayrı bir birey olarak tanıyarak araştırmacılara popülasyon takibinde yardımcı olmayı hedefler.

## Özellikler

- **ResNet50 backbone** — ImageNet ağırlıklarıyla transfer learning
- **Diferansiyel öğrenme oranı** — backbone ve classifier katmanı için ayrı lr
- **Class-weighted loss** — dengesiz veri dağılımını dengelemek için
- **Grad-CAM görselleştirme** — modelin kararlarını yorumlamak için
- **Veri artırma** — albumentations ile renk, döndürme, kırpma dönüşümleri
- **Erken durdurma** — val loss'a göre en iyi checkpoint kaydedilir
- **GUI** — eğitilmiş modeli arayüzden test etme imkanı

## Mimari

```
Girdi Görüntüsü
      ↓
ResNet50 Backbone (freeze edilebilir)
      ↓
Lineer Sınıflandırıcı (400 sınıf)
      ↓
Birey Kimliği
```

## Teknolojiler

- Python 3.x
- PyTorch / torchvision
- OpenCV
- albumentations
- grad-cam
- scikit-learn

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım

### Eğitim
```bash
python src/train.py
```

### Değerlendirme
```bash
python src/eval.py
```

### Grad-CAM Görselleştirme
```bash
python src/grad_cam.py
```

### GUI ile Test
```bash
python src/gui.py
```

## Proje Yapısı

```
deniz_kaplumbagasi_tanima/
├── src/
│   ├── model.py           # ResNet50 mimarisi
│   ├── dataset.py         # Veri yükleme ve augmentation
│   ├── train.py           # Eğitim döngüsü
│   ├── eval.py            # Değerlendirme
│   ├── grad_cam.py        # Karar görselleştirme
│   ├── gui.py             # Test arayüzü
│   └── utils.py           # Metrik hesaplama
├── data/                  # Veri seti
├── splits/                # Train/val/test bölmeleri
└── requirements.txt
```

---

**Oluşturan:** Talha Berat Oruçhan
