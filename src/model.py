import torch.nn as nn
import torchvision.models as models

class TurtleClassifier(nn.Module):
    def __init__(self, num_classes=400, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)