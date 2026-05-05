import torch.nn as nn
import torchvision.models as models

class TurtleClassifier(nn.Module):
    def __init__(self, num_classes=400, pretrained=True, dropout_rate=0.5):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        # Son fc katmanından önce dropout ekle
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)