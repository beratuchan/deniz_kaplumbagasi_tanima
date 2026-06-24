import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
import multiprocessing

from dataset import TurtleDataset, get_train_transform, get_val_transform
from model import TurtleClassifier
from utils import compute_metrics

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # ---------- KONFİG ----------
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_ROOT = PROJECT_ROOT / "data"
    SPLITS_DIR = PROJECT_ROOT / "splits"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 128
    NUM_EPOCHS = 50
    LEARNING_RATE_BACKBONE = 1e-4
    LEARNING_RATE_CLASSIFIER = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {DEVICE}")

    # ---------- ETİKET MAPPING (TÜM BİREYLER) ----------
    def get_label_mapping():
        annot_path = Path(__file__).parent.parent / "data" / "raw" / "annotations.json"
        with open(annot_path, 'r') as f:
            data = json.load(f)
        unique_ids = sorted(set(ann["identity"] for ann in data["annotations"]))
        label_map = {id: idx for idx, id in enumerate(unique_ids)}
        return label_map, len(unique_ids)

    label_map, num_classes = get_label_mapping()
    print(f"Toplam sınıf (birey) sayısı: {num_classes}")

    # ---------- CLASS WEIGHTS ----------
    def compute_class_weights(train_paths, label_map):
        counts = np.zeros(num_classes)
        for path in train_paths:
            label = label_map[Path(path).parent.name]
            counts[label] += 1
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * num_classes
        return torch.from_numpy(weights).float().to(DEVICE)

    with open(SPLITS_DIR / "train.txt", 'r') as f:
        train_paths = [line.strip() for line in f.readlines()]
    class_weights = compute_class_weights(train_paths, label_map)

    # ---------- DATALOADERS ----------
    train_dataset = TurtleDataset(
        split_file=SPLITS_DIR / "train.txt",
        img_root=DATA_ROOT,
        label_map=label_map,
        transform=get_train_transform()
    )
    val_dataset = TurtleDataset(
        split_file=SPLITS_DIR / "val.txt",
        img_root=DATA_ROOT,
        label_map=label_map,
        transform=get_val_transform()
    )

    # Windows'ta num_workers=0 yeterli, ama kod bloğu sayesinde artık worker kullanılabilir
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ---------- MODEL ----------
    model = TurtleClassifier(num_classes=num_classes, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # FC katmanı parametrelerini ayır
    backbone_params = []
    fc_params = []
    for name, param in model.backbone.named_parameters():
        if 'fc' in name:
            fc_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': LEARNING_RATE_BACKBONE},
        {'params': fc_params, 'lr': LEARNING_RATE_CLASSIFIER}
    ], weight_decay=WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

    # ---------- EĞİTİM DÖNGÜSÜ ----------
    best_val_rank1 = 0.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_rank1 = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for images, labels in progress:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            if DEVICE.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            metrics = compute_metrics(outputs, labels)
            train_rank1 += metrics['rank1'] * images.size(0)
            progress.set_postfix(loss=loss.item(), rank1=metrics['rank1'])
        
        train_loss /= len(train_loader.dataset)
        train_rank1 /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_rank1 = 0.0
        val_rank5 = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                metrics = compute_metrics(outputs, labels)
                val_rank1 += metrics['rank1'] * images.size(0)
                val_rank5 += metrics['rank5'] * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_rank1 /= len(val_loader.dataset)
        val_rank5 /= len(val_loader.dataset)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train R1={train_rank1:.4f} | Val Loss={val_loss:.4f}, Val R1={val_rank1:.4f}, Val R5={val_rank5:.4f}")
        
        scheduler.step(val_rank1)
        
        if val_rank1 > best_val_rank1:
            best_val_rank1 = val_rank1
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_model.pth")
            print(f"  -> Yeni best model kaydedildi (Val R1={val_rank1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Eğitim tamamlandı. En iyi validation Rank-1: {best_val_rank1:.4f}")