# https://github.com/Minimel/StochasticBatchAL/blob/main/src/Utils/plot_utils.py

# This file includes utilities to visualize:
# - Input vs prediction vs uncertainty
# - Multiple MC Dropout inferences
# - Group-level score distributions (e.g., for active learning strategy debug)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from skimage.measure import find_contours
import pytorch_lightning as pl
from numpy.testing import assert_almost_equal
import csv
import json
import torch
import torch.nn.functional as F

def _unpack_logits(y):
    if isinstance(y, (tuple, list)):
        return y[0]
    return y

@torch.no_grad()
def _tta_logits(model, x):
    model.eval()
    y1 = _unpack_logits(model(x))
    x_flip = torch.flip(x, dims=[-1])
    y2 = _unpack_logits(model(x_flip))
    y2 = torch.flip(y2, dims=[-1])
    return (y1 + y2) / 2.0

def get_foreground_prob(tensor):
    if tensor.ndim == 4:
        C = tensor.shape[1]
        if C == 1:
            return torch.sigmoid(tensor[:, 0])
        elif C == 2:
            return torch.softmax(tensor, dim=1)[:, 1]
        else:
            raise ValueError(f"Unexpected channels: {C}")
    elif tensor.ndim == 3:
        return tensor
    else:
        raise ValueError(f"Unsupported tensor shape {tensor.shape}")


def plot_uncertainty_image(
    cur_data, cur_target, cur_pred, cur_uncertainty_map=None, img_indice="",
    plot_type="contour", vmin=0, vmax=None, save_path=None
):
    """
    Visualize input image, target, prediction, and optional uncertainty map.
    cur_data, cur_target, cur_pred 形状应为 (H, W)，数值 0/1 或 0~1。
    """
    # —— 1) 统一转 float，避免 imshow/等高线对 bool/int 的奇怪行为
    cur_data   = np.asarray(cur_data, dtype=float)
    cur_target = np.asarray(cur_target, dtype=float)
    cur_pred   = np.asarray(cur_pred, dtype=float)
    if cur_uncertainty_map is not None:
        cur_uncertainty_map = np.asarray(cur_uncertainty_map, dtype=float)
        if vmax is None:  # 给不确定性一眼直观的范围
            vmax = float(cur_uncertainty_map.max() if np.isfinite(cur_uncertainty_map).any() else 1.0)

    if plot_type == "contour":
        fig = plt.figure(figsize=(10, 5), layout="constrained")
        ncols = 2 if cur_uncertainty_map is not None else 1

        # ---- 左图：原图 + 轮廓（蓝=GT，红=Pred）----
        ax = fig.add_subplot(1, ncols, 1)
        ax.imshow(cur_data, cmap="gray")
        # skimage.find_contours 返回 (row, col)，绘图要 x=col, y=row —— 不需要转置 .T
        for contour in find_contours(cur_target, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], "-b", lw=3)
        for contour in find_contours(cur_pred, 0.5):
            ax.plot(contour[:, 1], contour[:, 0], "-r", lw=3)
        ax.set_title(f"Idx {img_indice}", fontsize=12)
        ax.axis("off")

        # ---- 右图：不确定性热图（可选）----
        if cur_uncertainty_map is not None:
            ax = fig.add_subplot(1, ncols, 2)
            im = ax.imshow(cur_uncertainty_map, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"Uncertainty: {float(cur_uncertainty_map.mean()):.4f}")
            ax.axis("off")
            # 可选：加 colorbar
            # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    else:
        fig = plt.figure(figsize=(15, 5), layout="constrained")
        ncols = 3 if cur_uncertainty_map is not None else 2

        # ---- 左：GT 叠加 ----
        ax = fig.add_subplot(1, ncols, 1)
        ax.imshow(cur_data, cmap="gray")
        ax.imshow(cur_target, cmap="viridis", alpha=0.6, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"GT - Idx {img_indice}")
        ax.axis("off")

        # ---- 中：Pred 叠加 ----
        ax = fig.add_subplot(1, ncols, 2)
        ax.imshow(cur_data, cmap="gray")
        ax.imshow(cur_pred, cmap="viridis", alpha=0.6, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title("Prediction")
        ax.axis("off")

        # ---- 右：不确定性（可选）----
        if cur_uncertainty_map is not None:
            ax = fig.add_subplot(1, ncols, 3)
            ax.imshow(cur_uncertainty_map, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"Uncertainty: {float(cur_uncertainty_map.mean()):.4f}")
            ax.axis("off")

    # —— 4) 保存要用 fig.savefig；并且关闭 fig（不是 plt）——防止返回了一个已关闭的句柄
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig



def plot_multiple_data_pred(cur_multiple_data, cur_multiple_pred, query_indice):
    """
    Visualize multiple inference predictions for a single sample.

    """
    fig = plt.figure(figsize=(15, 5), layout="constrained")
    ncols = int(np.ceil(len(cur_multiple_data) / 2))

    for i in range(len(cur_multiple_data)):
        ax = fig.add_subplot(4, ncols, i + 1)
        ax.imshow(cur_multiple_data[i], 'gray')
        ax.set_title('Input')
        plt.axis('off')
    for i in range(len(cur_multiple_pred)):
        ax = fig.add_subplot(4, ncols, 2 * ncols + i + 1)
        ax.imshow(cur_multiple_pred[i], 'viridis')
        ax.set_title('MC Prediction')
        plt.axis('off')
    return plt


def plot_all_uncertain_samples_from_lists(indice_list, data_list, logits_list, target_list, uncertainty_list,
                                          budget, trainer, title, model_out_channels,
                                          multiple_prob_list=None, multiple_data_list=None):
    """
    Visualize top-K uncertain samples.

    """
    data_array = np.stack(data_list)
    logits_array = np.stack(logits_list)
    target_array = np.stack(target_list)
    if isinstance(uncertainty_list[0], float):
        uncertainty_map_array = None
        mean_uncertainty_list = np.array(uncertainty_list)
    else:
        uncertainty_map_array = np.stack(uncertainty_list)
        mean_uncertainty_list = np.mean(uncertainty_map_array, axis=(1, 2))

    if multiple_prob_list is not None:
        multiple_prob_array = np.stack(multiple_prob_list)
    if multiple_data_list is not None:
        multiple_data_array = np.stack(multiple_data_list)

    arg = np.argsort(mean_uncertainty_list)
    query_pool_indices = list(torch.tensor(indice_list)[arg][-budget:].numpy())
    uncertainty_values = list(torch.tensor(mean_uncertainty_list)[arg][-budget:].numpy())

    for idx in range(budget):
        query_indice = query_pool_indices[idx]
        loader_pos = np.where(np.array(indice_list) == query_indice)[0][0]

        cur_data = data_array[loader_pos, 0, :, :]
        cur_target = target_array[loader_pos, 0, :, :]
        cur_logits = torch.from_numpy(logits_array[loader_pos]).unsqueeze(0)  # [1,C,H,W] (to torch)
        cur_probs = get_foreground_prob(cur_logits)[0].cpu().numpy()  # [H,W] 前景概率
        cur_pred = (cur_probs > 0.5).astype(np.uint8)
        cur_uncertainty_map = uncertainty_map_array[loader_pos] if uncertainty_map_array is not None else None

        if cur_uncertainty_map is not None:
            assert_almost_equal(np.mean(cur_uncertainty_map), uncertainty_values[idx], decimal=6)

        if multiple_prob_list is not None and multiple_data_list is not None:
            cur_multiple_pred = np.argmax(multiple_prob_array[loader_pos], axis=-3)
            cur_multiple_data = multiple_data_array[loader_pos, :, 0, :, :]
            plt = plot_multiple_data_pred(cur_multiple_data, cur_multiple_pred, query_indice)
            if isinstance(trainer.logger, pl.loggers.CometLogger):
                trainer.logger.experiment.log_figure(figure=plt, figure_name='Multiple Predictions', step=idx)

        plot_type = 'contour' if model_out_channels == 2 else 'image'
        plt = plot_uncertainty_image(cur_data, cur_target, cur_pred, cur_uncertainty_map,
                                     query_indice, plot_type, vmin=0, vmax=model_out_channels - 1)

        if isinstance(trainer.logger, pl.loggers.CometLogger):
            trainer.logger.experiment.log_figure(figure=plt, figure_name=title, step=idx)
        else:
            trainer.logger.experiment.add_image(f"{title}_data", cur_data, idx, dataformats="HW")
            trainer.logger.experiment.add_image(f"{title}_target", cur_target, idx, dataformats="HW")
            # 同步发 pred（以及不确定性）便于肉眼检查
            trainer.logger.experiment.add_image(f"{title}_pred", cur_pred.astype(np.uint8) * 255, idx,
                                                             dataformats="HW")

            if cur_uncertainty_map is not None:
                # 归一化到 0-255 再发
                u = cur_uncertainty_map
                u = ((u - u.min()) / (u.max() - u.min() + 1e-8) * 255).astype(np.uint8)
                trainer.logger.experiment.add_image(f"{title}_uncertainty", u, idx, dataformats="HW")

def plot_group_score_distribution(group_scores: dict,
                                   title: str = "Group Aggregated Scores",
                                   save_path: str = None,
                                   highlight_top_k: int = None,
                                   sort_descending: bool = True):
    """
    Visualize aggregated score per group (e.g., uncertainty mean) as bar plot.

    Args:
        group_scores (dict): Dictionary {group_id: score}
        title (str): Title of the plot
        save_path (str): Optional. If provided, saves the figure to this path
        highlight_top_k (int): Optional. If provided, highlight top-k groups with red
        sort_descending (bool): If True, groups with higher scores appear first
    """
    import matplotlib.pyplot as plt

    # Sort groups by score
    sorted_items = sorted(group_scores.items(), key=lambda x: x[1], reverse=sort_descending)
    group_ids, scores = zip(*sorted_items)

    # Prepare colors
    if highlight_top_k:
        bar_colors = ['red' if i < highlight_top_k else 'gray' for i in range(len(scores))]
    else:
        bar_colors = 'gray'

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(scores)), scores, color=bar_colors)
    plt.xlabel("Group Index (Sorted)")
    plt.ylabel("Aggregated Score")
    plt.title(title)
    plt.tight_layout()

    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def visualize_single_sample(image, target_mask, pred_mask, uncertainty_map=None,
                            save_dir="output/visualizations/", index='0000'):
    """
    A wrapper for visualizing and saving a single sample.
    """
    filename = f"{index}_viz.png"
    save_path = os.path.join(save_dir, filename)

    plot_uncertainty_image(cur_data=image,
                           cur_target=target_mask,
                           cur_pred=pred_mask,
                           cur_uncertainty_map=uncertainty_map,
                           img_indice=index,
                           plot_type='contour',
                           save_path=save_path)


def plot_metrics_curve(csv_path: str, save_path: str) -> None:
    """
    Plot Dice, IoU, and HD95 metrics over active learning rounds.

    Args:
        csv_path (str): Path to the CSV file containing metrics.
        save_path (str): Path to save the plot image.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    rounds = df['round']

    # helper
    def _single(metric: str, marker: str, x):
        plt.figure(figsize=(6, 4))
        plt.plot(x, df[metric], marker=marker, linewidth=2)
        plt.xlabel("Active-learning round")
        plt.ylabel(metric.upper())
        plt.title(f"{metric.upper()} vs round")
        plt.grid(True, ls="--", alpha=.4)
        plt.tight_layout()
        out_path = f"{save_path}_{metric}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"📊 {metric} curve saved → {out_path}")

    _single("dice", "o", rounds)
    _single("iou", "s", rounds)
    _single("hd95", "^", rounds)

    print(f"📊 Metrics plot saved to {save_path}")

# --- add: threshold sweep and overlay visualization utilities ---

def _denorm_image(t: torch.Tensor) -> np.ndarray:
    """
    t: [3,H,W] (normalized) or [H,W] (already 0-1)
    return: [H,W,3] in 0-1
    """
    if t.ndim == 3 and t.shape[0] == 3:
        mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        x = t.cpu().numpy()
        x = x * std + mean
        x = np.clip(x, 0, 1)
        return np.transpose(x, (1, 2, 0))
    elif t.ndim == 2:
        x = t.cpu().numpy()
        x = np.stack([x, x, x], axis=-1)
        return np.clip(x, 0, 1)
    else:
        # fallback
        x = t.squeeze().cpu().numpy()
        if x.ndim == 2:
            x = np.stack([x, x, x], axis=-1)
        return np.clip(x, 0, 1)

def _apply_cc_filter(mask_bool: np.ndarray,
                     keep_largest_cc: bool = False,
                     min_cc_area: int = 0) -> np.ndarray:
    """
    可选：仅保留最大连通块，或去掉小面积连通块。
    没有 scipy 时，直接返回原 mask（保证最小依赖）。
    """
    if not (keep_largest_cc or min_cc_area > 0):
        return mask_bool

    try:
        import scipy.ndimage as ndi
    except Exception:
        return mask_bool  # graceful degrade

    labeled, num = ndi.label(mask_bool.astype(np.uint8))
    if num == 0:
        return mask_bool

    result = mask_bool.copy()
    if keep_largest_cc:
        sizes = ndi.sum(mask_bool, labeled, index=range(1, num + 1))
        keep_label = int(np.argmax(sizes)) + 1
        result = (labeled == keep_label)

    if min_cc_area > 0:
        labeled2, num2 = ndi.label(result.astype(np.uint8))
        if num2 > 0:
            sizes2 = ndi.sum(result, labeled2, index=range(1, num2 + 1))
            for lb, sz in enumerate(sizes2, start=1):
                if sz < min_cc_area:
                    result[labeled2 == lb] = False

    return result

def visualize_predictions_overlay(model: torch.nn.Module,
                                  dataset,
                                  device: torch.device,
                                  save_dir: str,
                                  num_samples: int = 6,
                                  threshold: float = 0.5,
                                  keep_largest_cc: bool = False,
                                  min_cc_area: int = 0,
                                  seed: int = 42):
    """
    生成三色叠加图：绿色=TP、红色=FN、蓝色=FP，并输出 per-sample 连通块统计 CSV。
    仅用于分析/汇报，不影响训练。
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    n = min(num_samples, len(dataset))
    indices = rng.choice(len(dataset), n, replace=False)

    # 统计 CSV
    stats_rows = [["idx", "fp_cc_count", "fn_cc_count", "fp_area_sum", "fn_area_sum"]]

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask, _ = dataset[idx]  # image:[3,H,W], mask:[H,W]
            gt = (mask > 0).cpu().numpy().astype(bool)

            img_in = image.unsqueeze(0).to(device)
            logits = _tta_logits(model, img_in)  # ← 用 TTA
            prob = get_foreground_prob(logits)[0].cpu().numpy()  # ← 统一取前景概率
            # prob 是 [H,W]，用外部传入的 threshold 生成二值预测
            pred = (prob > float(threshold))
            pred = _apply_cc_filter(pred, keep_largest_cc=keep_largest_cc, min_cc_area=min_cc_area)

            tp = gt & pred
            fn = gt & (~pred)
            fp = (~gt) & pred

            # 连通块统计（可选）
            fp_cc_count = 0
            fn_cc_count = 0
            fp_area_sum = int(fp.sum())
            fn_area_sum = int(fn.sum())
            try:
                import scipy.ndimage as ndi
                fp_cc_count = int(ndi.label(fp.astype(np.uint8))[1])
                fn_cc_count = int(ndi.label(fn.astype(np.uint8))[1])
            except Exception:
                pass

            stats_rows.append([int(idx), fp_cc_count, fn_cc_count, fp_area_sum, fn_area_sum])

            base = _denorm_image(image)
            overlay = base.copy()

            # 三色叠加
            # 红色 FN
            overlay[fn] = np.clip(overlay[fn] * 0.4 + np.array([1, 0, 0]) * 0.6, 0, 1)
            # 绿色 TP
            overlay[tp] = np.clip(overlay[tp] * 0.4 + np.array([0, 1, 0]) * 0.6, 0, 1)
            # 蓝色 FP
            overlay[fp] = np.clip(overlay[fp] * 0.4 + np.array([0, 0, 1]) * 0.6, 0, 1)

            # 保存对比图：原图 / 叠加 / GT / Pred
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(base);    axes[0].set_title("Image");   axes[0].axis("off")
            axes[1].imshow(overlay); axes[1].set_title(f"TP/FN/FP @ thr={threshold:.2f}"); axes[1].axis("off")
            axes[2].imshow(gt, cmap="gray");   axes[2].set_title("GT");   axes[2].axis("off")
            axes[3].imshow(pred, cmap="gray"); axes[3].set_title("Pred"); axes[3].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"sample_{i:03d}.png"), dpi=150, bbox_inches="tight")
            plt.close()

    # 写 CSV
    with open(os.path.join(save_dir, "overlay_cc_stats.csv"), "w", newline="") as f:
        csv.writer(f).writerows(stats_rows)

def sweep_thresholds_and_plot(model: torch.nn.Module,
                              dataset,
                              device: torch.device,
                              save_dir: str,
                              thresholds=None,
                              keep_largest_cc: bool = False,
                              min_cc_area: int = 0):
    """
    在验证集上扫阈值，输出 PR 曲线、Dice-阈值曲线，并保存 CSV。
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    if thresholds is None:
        thresholds = np.linspace(0.30, 0.80, 21)

    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    all_tp, all_fp, all_fn, all_tn = [], [], [], []

    def _acc(gt_bool, pred_bool):
        tp = (gt_bool & pred_bool).sum()
        fp = ((~gt_bool) & pred_bool).sum()
        fn = (gt_bool & (~pred_bool)).sum()
        tn = ((~gt_bool) & (~pred_bool)).sum()
        return int(tp), int(fp), int(fn), int(tn)

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            gt = (masks.long().squeeze(1).cpu().numpy() > 0).astype(bool)[0]

            logits = _tta_logits(model, images)  # ← 用 TTA
            prob = get_foreground_prob(logits)[0].cpu().numpy()  # ← 统一取前景概率

            t_tp, t_fp, t_fn, t_tn = [], [], [], []
            for thr in thresholds:
                pred = prob > float(thr)
                pred = _apply_cc_filter(pred, keep_largest_cc, min_cc_area)
                tp, fp, fn, tn = _acc(gt, pred)
                t_tp.append(tp); t_fp.append(fp); t_fn.append(fn); t_tn.append(tn)

            all_tp.append(t_tp); all_fp.append(t_fp); all_fn.append(t_fn); all_tn.append(t_tn)

    tp = np.sum(np.array(all_tp), axis=0)
    fp = np.sum(np.array(all_fp), axis=0)
    fn = np.sum(np.array(all_fn), axis=0)
    tn = np.sum(np.array(all_tn), axis=0)

    precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    recall    = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
    dice      = np.where((2*tp + fp + fn) > 0, (2*tp) / (2*tp + fp + fn), 0.0)
    iou       = np.where((tp + fp + fn) > 0, tp / (tp + fp + fn), 0.0)

    best_idx = int(np.argmax(dice))
    best_thr = float(thresholds[best_idx])

    # CSV
    csv_path = os.path.join(save_dir, "val_threshold_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "precision", "recall", "dice", "iou", "tp", "fp", "fn", "tn"])
        for i, t in enumerate(thresholds):
            w.writerow([float(t), float(precision[i]), float(recall[i]),
                        float(dice[i]), float(iou[i]),
                        int(tp[i]), int(fp[i]), int(fn[i]), int(tn[i])])

    # PR
    plt.figure(figsize=(5,5))
    plt.plot(recall, precision, marker='.')
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR Curve (val)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "pr_curve.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # Dice vs thr
    plt.figure(figsize=(6,4))
    plt.plot(thresholds, dice, marker='o')
    plt.axvline(best_thr, linestyle='--')
    plt.xlabel("Threshold"); plt.ylabel("Dice")
    plt.title("Dice vs Threshold (val)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "dice_vs_threshold.png"), dpi=150, bbox_inches='tight')
    plt.close()

    # return summary (可按需使用)
    summary = {
        "best_thr": best_thr,
        "prec_at_best": float(precision[best_idx]),
        "rec_at_best": float(recall[best_idx]),
        "dice_at_best": float(dice[best_idx]),
        "iou_at_best": float(iou[best_idx]),
        "csv": csv_path,
    }
    with open(os.path.join(save_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


