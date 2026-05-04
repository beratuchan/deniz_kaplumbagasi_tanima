import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def compute_rank_k(logits, targets, k=1):
    """logits: (batch, num_classes), targets: (batch,)"""
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1)
        correct = pred.eq(targets.view(-1,1).expand_as(pred))
        return correct.any(dim=1).float().mean().item()

def compute_metrics(logits, targets):
    rank1 = compute_rank_k(logits, targets, k=1)
    rank5 = compute_rank_k(logits, targets, k=5)
    return {'rank1': rank1, 'rank5': rank5}