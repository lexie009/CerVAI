import torch
import torch.nn as nn

def compute_entropy(prob_map, dim=1, eps=1e-6):
    """Compute pixel-wise entropy for a probability map"""
    prob_map = torch.clamp(prob_map, min=eps)
    return -torch.sum(prob_map * torch.log(prob_map + eps), dim=dim)

def enable_dropout(module):
    for m in module.modules():
        if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
            m.train()

def compute_mc_stats(preds, activation='sigmoid'):
    # preds: [T, B, C, H, W]
    if activation == 'sigmoid':
        probs = torch.sigmoid(preds)
    elif activation == 'softmax':
        probs = torch.softmax(preds, dim=2)
    else:
        probs = preds

    mean = probs.mean(dim=0)             # [B, C, H, W]
    var = probs.var(dim=0)               # [B, C, H, W]
    entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=2).mean(dim=0)  # [B, H, W]

    return mean, var, entropy

