import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
from albumentations import (
    HorizontalFlip, Rotate, RandomBrightnessContrast,
    HueSaturationValue, GaussNoise, Compose, Resize, Normalize
)

class TurtleDataset(Dataset):
    def __init__(self, split_file, img_root, label_map, transform=None, target_size=(256,256)):
        self.img_root = Path(img_root)
        self.transform = transform
        self.target_size = target_size
        
        with open(split_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        self.labels = [label_map[Path(p).parent.name] for p in self.image_paths]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Uyarı: {img_path} okunamadı, siyah görüntü")
            img = np.zeros((*self.target_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        label = self.labels[idx]
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        else:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            img = img / 255.0
        
        img = torch.from_numpy(img).permute(2,0,1).float()
        return img, label

def get_train_transform(mean, std, target_size=(256,256)):
    return Compose([
        Resize(height=target_size[0], width=target_size[1]),
        HorizontalFlip(p=0.5),
        Rotate(limit=15, p=0.5),            # RandomRotation ±15°
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        Normalize(mean=mean, std=std)
    ])

def get_val_transform(mean, std, target_size=(256,256)):
    return Compose([
        Resize(height=target_size[0], width=target_size[1]),
        Normalize(mean=mean, std=std)
    ])