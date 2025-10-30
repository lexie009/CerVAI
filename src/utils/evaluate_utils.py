import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import os
from scipy.spatial.distance import directed_hausdorff
from skimage.transform import resize
import logging
from monai.metrics import HausdorffDistanceMetric
logger = logging.getLogger(__name__)

hd95_metric = HausdorffDistanceMetric(percentile=95, reduction="mean")


def dice_score(pred, target, smooth=1e-5):
    """Compute Dice coefficient"""
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    intersection = np.logical_and(pred, target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-5):
    """Compute IoU score"""
    pred = pred.astype(bool).flatten()
    target = target.astype(bool).flatten()
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return (intersection + smooth) / (union + smooth)


def hausdorff_95(pred, target):
    """Compute the 95th percentile Hausdorff Distance"""
    pred_points = np.argwhere(pred)
    target_points = np.argwhere(target)
    if pred_points.size == 0 or target_points.size == 0:
        return np.nan
    forward_hd = directed_hausdorff(pred_points, target_points)[0]
    backward_hd = directed_hausdorff(target_points, pred_points)[0]
    return np.percentile([forward_hd, backward_hd], 95)

def binary_stats(pred: torch.Tensor, gt: torch.Tensor):
    """
    pred, gt: (B,1,H,W) 0/1 tensor
    return TP, FP, FN, TN
    """
    tp = ((pred == 1) & (gt == 1)).sum().item()
    fp = ((pred == 1) & (gt == 0)).sum().item()
    fn = ((pred == 0) & (gt == 1)).sum().item()
    tn = ((pred == 0) & (gt == 0)).sum().item()
    return tp, fp, fn, tn

def _unpack_logits(y):
    # 兼容模型返回 (logits,) 或 (logits, aux...) 的情况
    if isinstance(y, (tuple, list)):
        return y[0]
    return y

@torch.no_grad()
def _tta_logits(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    水平翻转 TTA：平均 **logits**（不是平均softmax）。
    x: [B,C,H,W] on device
    return: logits [B,C,H,W] on same device
    """
    model.eval()
    y1 = _unpack_logits(model(x))
    x_flip = torch.flip(x, dims=[-1])
    y2 = _unpack_logits(model(x_flip))
    y2 = torch.flip(y2, dims=[-1])
    return (y1 + y2) / 2.0

def get_foreground_prob(tensor: torch.Tensor) -> torch.Tensor:
    """
    输入 tensor 可以是 logits [B,C,H,W]、概率 [B,C,H,W] 或 [B,H,W]。
    统一返回前景概率 [B,H,W]（二类）
    - C==1: 视为 logits → sigmoid → [:,0]
    - C==2: 视为 logits → softmax → [:,1]
    """
    if tensor.ndim == 4:
        C = tensor.shape[1]
        if C == 1:
            return torch.sigmoid(tensor[:, 0])
        elif C == 2:
            return torch.softmax(tensor, dim=1)[:, 1]
        else:
            raise ValueError(f"Unexpected channel count {C}. Expect 1 or 2 for binary seg.")
    elif tensor.ndim == 3:
        return tensor  # 已是概率 [B,H,W]
    else:
        raise ValueError(f"Unsupported tensor shape {tensor.shape}")

def binarize_probs(probs: torch.Tensor, thr: float) -> torch.Tensor:
    """(probs > thr) → uint8 0/1"""
    return (probs > float(thr)).to(torch.uint8)


@torch.no_grad()
def evaluate_basic(model, dataloader, device, threshold: float = 0.5) -> dict:
    """
    基础指标：Dice / IoU / HD95
    统一路径：TTA logits → 前景概率 → (probs>thr)
    """
    model.eval()
    dice_scores, iou_scores, hd95_scores = [], [], []

    for batch in dataloader:
        # 解包
        if isinstance(batch, (list, tuple)):
            images, masks = batch[0], batch[1]
        elif isinstance(batch, dict):
            images = batch.get('image', next(iter(batch.values())))
            masks  = batch.get('mask')
        else:
            raise ValueError("Unexpected batch type for evaluation.")

        images = images.to(device, non_blocking=True)
        logits = _tta_logits(model, images)                    # [B,C,H,W]
        probs  = get_foreground_prob(logits)                   # [B,H,W] on device
        preds  = binarize_probs(probs, threshold).cpu().numpy().astype(np.uint8)

        # GT
        masks_t = masks
        if masks_t.ndim == 4 and masks_t.size(1) == 1:
            masks_t = masks_t[:, 0]
        gts = (masks_t > 0).cpu().numpy().astype(np.uint8)

        # 逐图计算
        for pred, target in zip(preds, gts):
            pred_empty, tgt_empty = pred.sum() == 0, target.sum() == 0
            if pred_empty and tgt_empty:
                dice_scores.append(1.0); iou_scores.append(1.0); hd95_scores.append(np.nan); continue
            if pred_empty or tgt_empty:
                dice_scores.append(0.0); iou_scores.append(0.0); hd95_scores.append(np.nan); continue

            inter = np.logical_and(pred, target).sum()
            union = np.logical_or(pred, target).sum()
            dice_scores.append((2*inter + 1e-5) / (pred.sum()+target.sum() + 1e-5))
            iou_scores.append((inter + 1e-5) / (union + 1e-5))

            # HD95 with one-hot
            pred_t   = torch.from_numpy(pred[None,...]).to(device)
            target_t = torch.from_numpy(target[None,...]).to(device)
            pred_oh   = F.one_hot(pred_t.long(), 2).permute(0,3,1,2).float()
            target_oh = F.one_hot(target_t.long(), 2).permute(0,3,1,2).float()
            hd95_scores.append(float(hd95_metric(pred_oh, target_oh).item()))

    return {
        "dice": float(np.nanmean(dice_scores)),
        "iou" : float(np.nanmean(iou_scores)),
        "hd95": float(np.nanmean(hd95_scores)) if len(hd95_scores) > 0 else float("nan"),
    }


def _fg_prob_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """统一获取前景概率: 1通道→sigmoid，2通道→softmax取第1类"""
    if logits.shape[1] == 1:
        return torch.sigmoid(logits).squeeze(1)  # [B,H,W]
    else:
        return torch.softmax(logits, dim=1)[:, 1]  # [B,H,W]

@torch.no_grad()
def sweep_thresholds(model, loader, device, thr_list):
    """
    用 TTA + 统一通道规则，按 (probs>thr) 扫描阈值，返回每个 thr 的 P/R/Dice。
    """
    model.eval()
    all_probs, all_gts = [], []

    for batch in loader:
        if isinstance(batch, (list, tuple)):
            images, masks = batch[0], batch[1]
        elif isinstance(batch, dict):
            images = batch.get('image', next(iter(batch.values())))
            masks  = batch.get('mask')
        else:
            continue

        images = images.to(device, non_blocking=True)
        logits = _tta_logits(model, images)
        probs  = get_foreground_prob(logits).detach().cpu()   # [B,H,W] CPU
        if masks.ndim == 4 and masks.size(1) == 1:
            masks = masks[:, 0]
        gts = (masks > 0).to(torch.uint8).cpu()               # [B,H,W] CPU

        all_probs.append(probs)
        all_gts.append(gts)

    if not all_probs:
        return []

    probs = torch.cat(all_probs, dim=0)  # [N,H,W] CPU
    gts   = torch.cat(all_gts, dim=0)    # [N,H,W] CPU

    out = []
    for thr in thr_list:
        pred = (probs > float(thr)).to(torch.uint8)
        tp = int(((pred==1) & (gts==1)).sum().item())
        fp = int(((pred==1) & (gts==0)).sum().item())
        fn = int(((pred==0) & (gts==1)).sum().item())

        precision = tp/(tp+fp+1e-6)
        recall    = tp/(tp+fn+1e-6)
        dice      = 2*precision*recall/(precision+recall+1e-6)
        out.append({"thr": float(thr),
                    "precision": float(precision),
                    "recall": float(recall),
                    "dice": float(dice)})
    return out

@torch.no_grad()
def evaluate_full(model, dataloader, device, threshold: float = 0.5) -> dict:
    """
    完整评估：basic(dice/iou/hd95) + PR/F1/PixelAcc
    使用与 evaluate_basic 完全相同的推理路径
    """
    basic = evaluate_basic(model, dataloader, device, threshold=threshold)

    tp = fp = fn = tn = 0
    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images, masks = batch[0], batch[1]
        elif isinstance(batch, dict):
            images = batch.get('image', next(iter(batch.values())))
            masks  = batch.get('mask')
        else:
            raise ValueError("Unexpected batch type for evaluation.")

        images = images.to(device, non_blocking=True)
        logits = _tta_logits(model, images)
        probs  = get_foreground_prob(logits)                     # [B,H,W] device
        preds  = binarize_probs(probs, threshold)                # [B,H,W] device

        gts = masks
        if gts.ndim == 4 and gts.size(1) == 1:
            gts = gts[:, 0]
        gts = (gts > 0).to(torch.uint8).to(device)               # [B,H,W] device

        # 统一在 GPU 上统计，再 .item()
        tp += int(((preds == 1) & (gts == 1)).sum().item())
        fp += int(((preds == 1) & (gts == 0)).sum().item())
        fn += int(((preds == 0) & (gts == 1)).sum().item())
        tn += int(((preds == 0) & (gts == 0)).sum().item())

    precision   = tp / (tp + fp + 1e-6)
    recall      = tp / (tp + fn + 1e-6)
    f1_score    = 2 * precision * recall / (precision + recall + 1e-6)
    pixel_acc   = (tp + tn) / (tp + fp + fn + tn + 1e-6)
    mean_pa     = ((tp / (tp + fn + 1e-6)) + (tn / (tn + fp + 1e-6))) / 2

    out = {
        **basic,
        "precision":      float(precision),
        "recall":         float(recall),
        "f1":             float(f1_score),
        "pixel_acc":      float(pixel_acc),
        "mean_pixel_acc": float(mean_pa),
    }
    # 可选：统一保留 4 位小数
    out = {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v) for k, v in out.items()}
    return out



def save_round_metrics(round_id: int,
                       metrics: dict[str, float],
                       save_path: str,
                       strategy: str,
                       model_name: str,
                       pool_csv: str):
    """
    Save round evaluation metrics to a CSV file.

    Args:
        round_id (int): Current active learning round.
        metrics (dict): Dictionary of metrics (e.g., dice, iou, hd95).
        save_path (str): Path to CSV file.
    """
    import csv
    import os

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    num_labeled = pd.read_csv(pool_csv)["labeled"].sum()
    fieldnames = ["round","model","strategy","dice","iou","hd95","num_labeled"]
    row = { "round":round_id, "model":model_name, "strategy":strategy,
            "dice":metrics["dice"], "iou":metrics["iou"],
            "hd95":metrics["hd95"], "num_labeled":num_labeled }

    # Check if file exists
    file_exists = os.path.isfile(save_path)

    with open(save_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

@torch.no_grad()
def per_image_dice_list(model, loader, device, threshold=0.5):
    model.eval()
    dices = []
    eps = 1e-6
    for batch in loader:
        # 兼容你的 Dataset 返回 (image, mask, meta)
        if len(batch) == 3:
            images, masks, _ = batch
        else:
            images, masks = batch

        images = images.to(device)
        # masks: [B, H, W] （前景=1/背景=0）
        logits = model(images)

        if logits.shape[1] == 1:
            # 单通道互斥：logits → sigmoid → 前景概率
            prob_fg = torch.sigmoid(logits)[:, 0]
        else:
            # 双通道互斥：logits → softmax → [:,1] 前景
            prob_fg = torch.softmax(logits, dim=1)[:, 1]

        pred = (prob_fg >= threshold).float()

        # 逐图 Dice（不做 batch 归约）
        # masks 可能是 long/byte，统一到 float
        masks = (masks > 0).float().to(pred.device)
        B = masks.shape[0]
        for b in range(B):
            y = masks[b].reshape(-1)
            p = pred[b].reshape(-1)
            inter = (y * p).sum()
            dice = (2 * inter + eps) / (y.sum() + p.sum() + eps)
            dices.append(float(dice))
    return dices