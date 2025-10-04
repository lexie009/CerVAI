"""
Active Learning Training Pipeline for CerVAI Project
Supports multi-round active learning with various sampling strategies.

"""

import os
import sys
import yaml
import torch
import numpy as np
import json
import time
import logging
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser

from networkx import non_neighbors
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
from tqdm.auto import tqdm

# Add project root to path
file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(file_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from model.factory import build_model
from dataset import create_datasets, set_seed, CervixDataset
from trainer import Trainer, GradualWarmupScheduler
from sampling.active_sampling import sample_new_indices
from utils.sampling_recording_utils import record_sampling_info
from utils.visualization_utils import visualize_single_sample, plot_metrics_curve, visualize_predictions_overlay, sweep_thresholds_and_plot
from utils.evaluate_utils import evaluate_basic, evaluate_full, save_round_metrics, sweep_thresholds
from dataset import CervixUnlabeledImages

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ],
    force=True
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    with open(save_path, 'w') as f:
        yaml.safe_dump(config, f, indent=2)

def get_warm_ids_from_disk(pool_df: pd.DataFrame, data_cfg: Dict[str, Any]) -> List[int]:
    """Return CSV indices whose image files live in .../active_split/round0."""
    round0_dir = Path(data_cfg["round0_image_dir"])
    if not round0_dir.exists():
        return []

    disk_imgs = {p.name for p in round0_dir.iterdir()
                 if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}

    return pool_df[pool_df["new_image_name"].isin(disk_imgs)].index.tolist()


def visualize_predictions(model: torch.nn.Module,
                          dataset,
                          device: torch.device,
                          save_dir: str,
                          num_samples: int = 5,
                          threshold: float = 0.5,
                          keep_largest_cc: bool = False,
                          min_cc_area: int = 0) -> None:
    """
    Visualize prediction, superimpose TP/FP/FN and statistically analyze the information of connected blocks.

    threshold: Binary classification prospect threshold (default 0.5)
    keep_largest_cc: Whether to retain only the maximum connected component (for visualization/evaluation only)
    min_cc_area: The minimum pixel area for filtering out small connected components (0 indicates no filtering)
    """

    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # ---- helpers ----
    def _label_cc_numpy(binary_mask: np.ndarray):
        """return labels, num_labels, sizes(list)。"""
        h, w = binary_mask.shape
        labels = np.zeros((h, w), dtype=np.int32)
        curr = 0
        sizes = []
        # 8-neighbors
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for i in range(h):
            for j in range(w):
                if binary_mask[i, j] and labels[i, j] == 0:
                    curr += 1
                    stack = [(i, j)]
                    labels[i, j] = curr
                    sz = 0
                    while stack:
                        x, y = stack.pop()
                        sz += 1
                        for dx, dy in nbrs:
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < h and 0 <= ny < w and binary_mask[nx, ny] and labels[nx, ny] == 0:
                                labels[nx, ny] = curr
                                stack.append((nx, ny))
                    sizes.append(sz)
        return labels, curr, sizes

    def cc_stats(mask_bool: np.ndarray):
        """return (num_cc, sizes list)。use scipy.ndimage first，if unsuccessful use numpy """
        try:
            import scipy.ndimage as ndi  # 可用则更快
            labeled, num = ndi.label(mask_bool.astype(np.uint8))
            if num == 0:
                return 0, [], labeled
            sizes = ndi.sum(mask_bool, labeled, index=range(1, num + 1))
            sizes = list(np.asarray(sizes, dtype=int))
            return int(num), sizes, labeled
        except Exception:
            labels, num, sizes = _label_cc_numpy(mask_bool.astype(bool))
            return int(num), sizes, labels

    def keep_largest_component(mask_bool: np.ndarray) -> np.ndarray:
        num, sizes, labeled = cc_stats(mask_bool)
        if num <= 1:
            return mask_bool
        #  label（>= min_cc_area）
        sizes_arr = np.asarray(sizes)
        order = np.argsort(sizes_arr)[::-1]  # 大→小
        chosen = None
        for idx in order:
            if sizes_arr[idx] >= max(1, int(min_cc_area)):
                chosen = idx + 1  # labels 从 1 开始
                break
        if chosen is None:
            # 如果没有满足 min_cc_area 的，就保留原始 mask
            return mask_bool
        return (labeled == chosen)

    # 随机抽样
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask, _ = dataset[idx]  # image: [C,H,W], mask: [H,W] 0/1
            # 保留 GT
            mask_np = mask.cpu().numpy().astype(np.uint8)

            # 模型推理
            img_in = image.unsqueeze(0).to(device)  # [1,C,H,W]
            logits = model(img_in)

            # 取前景概率
            if logits.shape[1] == 1:
                prob_fg = torch.sigmoid(logits)[0, 0].cpu().numpy()
            else:
                prob_fg = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

            pred_mask = (prob_fg > threshold).astype(np.uint8)

            # 可选：只保留最大连通块 & 小块过滤
            if keep_largest_cc:
                pred_mask = keep_largest_component(pred_mask.astype(bool)).astype(np.uint8)

            if min_cc_area > 0:
                # 过滤 pred 中小连通块
                num_pred, sizes_pred, labeled_pred = cc_stats(pred_mask.astype(bool))
                for lbl in range(1, num_pred + 1):
                    if (labeled_pred == lbl).sum() < min_cc_area:
                        pred_mask[labeled_pred == lbl] = 0

            # 反归一化便于可视化
            image_np = image.cpu().numpy()
            if image_np.shape[0] == 3:
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)
            image_np = np.transpose(image_np, (1, 2, 0))  # [H,W,3]

            # 计算 TP/FP/FN
            gt = (mask_np > 0).astype(bool)
            pd = (pred_mask > 0).astype(bool)

            tp = gt & pd
            fn = gt & (~pd)
            fp = (~gt) & pd

            # 连通块统计（关注 FP/FN）
            num_fp, sizes_fp, _ = cc_stats(fp)
            num_fn, sizes_fn, _ = cc_stats(fn)

            # 叠加三色可视化
            overlay = image_np.copy()
            alpha = 0.5  # 透明度
            color = np.zeros_like(overlay)
            # TP=绿, FN=红, FP=蓝
            color[tp] = [0.0, 1.0, 0.0]
            color[fn] = [1.0, 0.0, 0.0]
            color[fp] = [0.0, 0.0, 1.0]
            mix_mask = tp | fn | fp
            overlay[mix_mask] = (1 - alpha) * overlay[mix_mask] + alpha * color[mix_mask]

            # 画图：原图 / GT / Pred / Overlay
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(image_np)
            axes[0].set_title('Input Image')
            axes[0].axis('off')

            axes[1].imshow(gt, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')

            axes[2].imshow(pd, cmap='gray')
            axes[2].set_title(f'Prediction (thr={threshold:.2f}'
                              f'{", LCC" if keep_largest_cc else ""})')
            axes[2].axis('off')

            axes[3].imshow(overlay)
            axes[3].set_title('TP(绿) / FN(红) / FP(蓝)')
            axes[3].axis('off')

            # 在最后一幅图上加统计文本
            txt = (f'FP 连通块: {num_fp}'
                   f'{", 平均面积: %.1f" % (np.mean(sizes_fp) if len(sizes_fp) else 0)}\n'
                   f'FN 连通块: {num_fn}'
                   f'{", 平均面积: %.1f" % (np.mean(sizes_fn) if len(sizes_fn) else 0)}')
            axes[3].text(0.02, 0.98, txt, color='w', va='top', ha='left',
                         transform=axes[3].transAxes,
                         bbox=dict(facecolor='black', alpha=0.4, pad=5, edgecolor='none'), fontsize=10)

            plt.tight_layout()
            out_png = os.path.join(save_dir, f'sample_{i:03d}.png')
            plt.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close()

            # 同步保存统计信息，便于后续量化分析
            stats = {
                "index_in_dataset": int(idx),
                "threshold": float(threshold),
                "keep_largest_cc": bool(keep_largest_cc),
                "min_cc_area": int(min_cc_area),
                "num_fp_cc": int(num_fp),
                "num_fn_cc": int(num_fn),
                "fp_areas_top5": list(sorted([int(x) for x in sizes_fp], reverse=True)[:5]),
                "fn_areas_top5": list(sorted([int(x) for x in sizes_fn], reverse=True)[:5]),
            }
            with open(os.path.join(save_dir, f'sample_{i:03d}_stats.json'), 'w') as f:
                json.dump(stats, f, indent=2)

    logger.info(f"Saved {len(indices)} prediction visualizations to {save_dir}")


def get_unlabeled_indices(csv_path: str) -> List[int]:
    df = pd.read_csv(csv_path)
    return df.query("labeled == 0 & set != 'val' & set != 'test'").index.tolist()

def sample_images_wrapper(strategy: str,
                          model: torch.nn.Module,
                          dataset,
                          config: Dict[str, Any],
                          budget: int,
                          device: torch.device,
                          unlabeled_indices: List[int],
                          csv_path: str,
                          seed: int = 42) -> List[str]:
    """
    Wrapper function to bridge the existing sampling module with the new train.py.
    
    Args:
        strategy: Sampling strategy name
        model: Trained model
        dataset: Dataset to sample from
        config: Training configuration
        budget: Number of samples to select
        device: Computing device
        unlabeled_indices: List of unlabeled sample indices
        csv_path: Path to CSV for getting image names
        
    Returns:
        List of selected image names
    """
    strategy_cfg = {"strategy": strategy.title(), "sampling_params": {}}
    # yaml_path = Path(
    #    "/Users/daidai/Documents/pythonProject_summer/CerVAI/src/configs/sampling_config") / f"{strategy.title()}.yaml"
    # yaml_path = Path(
       # "/home/mry/CerVAI/src/configs/sampling_config") / f"{strategy.title()}.yaml"
    yaml_path = Path(
     "/content/drive/MyDrive/CerVAI/src/configs/sampling_config") / f"{strategy.title()}.yaml"

    if yaml_path.exists():
        try:
            with open(yaml_path, "r") as f:
                yaml_cfg = yaml.safe_load(f) or {}
            # 允许 YAML 覆盖默认
            strategy_cfg.update(yaml_cfg)
        except Exception as e:
            logger.warning(f"Could not load YAML for {strategy}: {e}")

    unlabeled_ds = Subset(dataset, unlabeled_indices)
    unlabeled_loader = DataLoader(
        dataset=unlabeled_ds,
        batch_size=config.get("batch_size", 4),
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    mc_dropout_enabled = config.get('sampling', {}).get('enable_mc_dropout', False)
    T = config.get('sampling', {}).get('mc_iterations', 10)

    sampling_config = {
        "strategy": strategy_cfg["strategy"],
        "sampling_params": strategy_cfg.get("sampling_params", {}),
        "use_mc_dropout": mc_dropout_enabled,
        "mc_iterations": T,
    }

    # ---- 1. quick mock-trainer that looks like PyTorch-Lightning trainer ----
    import logging, time
    class MockTrainer:
        def __init__(self, model, device):
            self.model = model
            self.device_ids = [0]  # minimal attr PL expects
            self.current_epoch = 0
            # give it a *working* logger so SamplingLogger is happy
            self.logger = logging.getLogger("MockTrainer")
            if not self.logger.handlers:  # avoid dup handlers
                h = logging.StreamHandler()
                fmt = logging.Formatter("[Sampler] %(message)s")
                h.setFormatter(fmt)
                self.logger.addHandler(h)
                self.logger.setLevel(logging.INFO)

    trainer = MockTrainer(model, device)
    trainer.logger.info(f"Unlabeled dataloader size: {len(unlabeled_loader)} "
                        f"(samples: {len(unlabeled_indices)})")

    # ---- 3. construct sampling config ------------------------------------------------
    sampling_cfg = {
        "strategy": strategy_cfg["strategy"].lower(),  # e.g. "Random"
        "sampling_params": strategy_cfg.get("sampling_params", {}),
        "use_mc_dropout": config.get('sampling', {}).get('enable_mc_dropout', False),
        "mc_iterations": config.get('sampling', {}).get('mc_iterations', 10),
    }

    # ---- 4. call your generic sampler -------------------------------------------------
    query_indices, sampling_time = sample_new_indices(
        sampling_config=sampling_cfg,
        budget=budget,
        dataset=dataset,
        unlabeled_dataloader=unlabeled_loader,
        trainer=trainer,
        seed=42,
        device=device,
        unlabeled_indices=unlabeled_indices
    )
    trainer.logger.info(f" sampling finished in {sampling_time:.1f}s "
                        f"→ picked {len(query_indices)}")

    # ---- 5. convert global-index → file name ------------------------------------------
    df = pd.read_csv(csv_path)
    selected_names = [df.loc[idx, "new_image_name"] for idx in query_indices]

    return selected_names, sampling_time


def get_csv_path(data_config: Dict[str, Any], args_csv_path: str) -> str:
    """
    Centralized CSV path handling.
    
    Args:
        data_config: Data configuration dictionary
        args_csv_path: CSV path from command line arguments
        
    Returns:
        Final CSV path to use
    """
    # Priority: data_config.meta_csv > command line args > default
    if 'meta_csv' in data_config:
        csv_path = data_config['meta_csv']
        logger.info(f"Using CSV path from data_config: {csv_path}")
    else:
        csv_path = args_csv_path
        logger.info(f"Using CSV path from command line: {csv_path}")
    
    # Verify the file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    return csv_path


def create_round_datasets(data_config: Dict[str, Any],
                         csv_path: str,
                         current_round: int) -> Dict[str, Any]:
    """
    Create datasets for the current active learning round using enhanced data config.
    
    Args:
        data_config: Enhanced data configuration with explicit paths
        csv_path: Path to metadata CSV
        current_round: Current round number
        
    Returns:
        Dictionary containing train, val, and unlabeled datasets
    """
    round_name = f'round{current_round}'
    
    # Get image preprocessing parameters
    target_size = tuple(data_config.get('target_size', [512, 512]))
    if 'transform' in data_config and 'resize' in data_config['transform']:
        target_size = tuple(data_config['transform']['resize'])
    
    normalize = data_config.get('normalize', True)
    if 'transform' in data_config and 'normalize' in data_config['transform']:
        normalize = data_config['transform']['normalize']
    
    binary_mask = data_config.get('binary_mask', True)
    
    # Create training dataset for current round using labeled data only
    if 'train_image_dir' in data_config and 'train_mask_dir' in data_config:
        from dataset import CervixDataset
        train_dataset = CervixDataset(
            csv_path=csv_path,
            image_dir=data_config['train_image_dir'],
            mask_dir=data_config['train_mask_dir'],
            target_size=target_size,
            normalize=normalize,
            augment=True,
            set_filter='train',
            binary_mask=True,
            use_labeled_only=True,
            enable_roi=data_config.get('roi', {}).get('enable_train', True),
            pad_ratio=data_config.get('roi', {}).get('pad_ratio', 0.10),
            roi_mode=data_config.get('roi', {}).get('mode_train', 'image'),
            use_mask_on_valtest=False
        )
        
        train_datasets = {'train': train_dataset}
    else:
        # Fallback to standard dataset creation with labeled data only
        train_datasets = create_datasets(
            csv_path=csv_path,
            base_dir=data_config['base_dir_standard'],
            normalize=normalize,
            target_size=target_size,
            binary_mask=True,
            use_labeled_only=True
        )
    
    # Create validation dataset using explicit paths if available
    if 'val_image_dir' in data_config and 'val_mask_dir' in data_config:
        from dataset import CervixDataset
        
        val_dataset = CervixDataset(
            csv_path=csv_path,
            image_dir=data_config['val_image_dir'],
            mask_dir=data_config['val_mask_dir'],
            target_size=target_size,
            normalize=normalize,
            augment=False,
            set_filter='val',
            binary_mask=binary_mask,
            use_labeled_only=False,
            enable_roi=data_config.get('roi', {}).get('enable_val', False),
            pad_ratio=data_config.get('roi', {}).get('pad_ratio', 0.10),
            roi_mode=data_config.get('roi', {}).get('mode_val', 'image'),
            use_mask_on_valtest=False
        )
        
        val_datasets = {'val': val_dataset}
    else:
        # Fallback to standard split
        val_datasets = create_datasets(
            csv_path=csv_path,
            base_dir=data_config['base_dir_standard'],
            normalize=normalize,
            target_size=target_size,
            binary_mask=binary_mask,
            use_labeled_only=False
        )
    
    # Create unlabeled dataset for sampling
    unlabeled_indices = get_unlabeled_indices(csv_path)
    
    return {
        'train': train_datasets['train'],
        'val': val_datasets['val'],
        'unlabeled_indices': unlabeled_indices,
        "full_train": train_datasets['train']
    }


def setup_experiment_dirs(output_dir: str, experiment_name: str) -> Dict[str, str]:
    """Setup experiment directory structure."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(output_dir) / f"{experiment_name}_{timestamp}"
    
    dirs = {
        'experiment': str(exp_dir),
        'checkpoints': str(exp_dir / 'checkpoints'),
        'logs': str(exp_dir / 'logs'),
        'visualizations': str(exp_dir / 'visualizations'),
        'configs': str(exp_dir / 'configs'),
        'results': str(exp_dir / 'results')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def setup_experiment_logging(log_dir: str) -> None:
    """Setup logging to include file handler."""
    log_file = os.path.join(log_dir, 'active_learning.log')
    
    # Add file handler to existing logger
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add to root logger
    logging.getLogger().addHandler(file_handler)
    
    logger.info(f"Logging to file: {log_file}")


def update_config_from_args(config: Dict[str, Any], args: Any, prefix: str = '') -> Dict[str, Any]:
    """Update configuration with command line arguments."""
    for key, value in vars(args).items():
        if value is not None and key.startswith(prefix):
            config_key = key.replace(f'{prefix}__', '') if prefix else key
            if '__' in config_key:
                # Handle nested keys like 'optimizer__lr'
                keys = config_key.split('__')
                current = config
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                config[config_key] = value
    
    return config

def build_unlabeled_loader(data_config: Dict[str, Any],
                           csv_path: str,
                           batch_size: int,
                           num_workers: int = 0):
    image_dir = os.path.join(data_config['base_dir_standard'], 'train', 'images')
    target_size = tuple(data_config.get('target_size', [512, 512]))
    ds_u = CervixUnlabeledImages(
        csv_path=csv_path,
        image_dir=image_dir,
        target_size=target_size,
        normalize=True,
        set_filter='train',
        return_idx=False)

    if len(ds_u) == 0:
        return None
    return DataLoader(ds_u,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers,
                      pin_memory=False,
                      drop_last=True)

def validate_visualize_sample(
        rnd: int,
        total_iters: int,
        trainer: "Trainer",
        val_ds,
        full_train_ds,
        dirs: Dict[str, str],
        train_cfg: Dict[str, Any],
        data_cfg : Dict[str, Any],
        pool_csv : str,
        sampling_strategy: str,
        budget: int,
        device: torch.device,
        model_name: str,
        threshold: Optional[float] = None   # ← 入参保留，但本函数内会先做 sweep 覆盖它
    ):
    """
    One-stop post-training procedure (统一口径版本):

    A) sweep 阈值 → 取 best_thr
    B) 用 best_thr 做验证 evaluate_basic() 并落盘 round_metrics
    C) 用 best_thr 存可视化（与验证同阈值）
    D) (非最后一轮) 采样并回写 pool.csv
    E) 返回 (val_metrics@best_thr, best_thr)
    """
    round_name = f"round{rnd}"
    bs = train_cfg.get("batch_size", train_cfg.get("train", {}).get("batch_size", 4))
    v_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    # ---------- A) sweep 阈值（当前轮） ----------
    sweep_dir = os.path.join(dirs["results"], f"{round_name}_sweep")
    thr_list = np.linspace(0.10, 0.80, 21)
    sweep_info = sweep_thresholds_and_plot(
        model=trainer.model,
        dataset=val_ds,
        device=device,
        save_dir=sweep_dir,
        thresholds=thr_list,
        keep_largest_cc=False,
        min_cc_area=0
    )
    best_thr_from_val = float(sweep_info["best_thr"])
    logger.info(
        f"[Val sweep] Best Dice @ thr={best_thr_from_val:.2f} "
        f"→ P={sweep_info['prec_at_best']:.3f} "
        f"R={sweep_info['rec_at_best']:.3f} "
        f"Dice={sweep_info['dice_at_best']:.3f}"
    )

    # 统一本轮“推理阈值”：优先用本轮 sweep 的 best_thr
    infer_thr = best_thr_from_val

    # ---------- B) 用 best_thr 做验证并落盘 ----------
    val_metrics = evaluate_basic(
        trainer.model, v_loader, device,
        threshold=infer_thr,
    )
    # 把当轮阈值也写进 metrics 便于追踪
    val_metrics["thr"] = infer_thr

    save_round_metrics(
        round_id=rnd,
        metrics=val_metrics,
        save_path=os.path.join(dirs["results"], "round_metrics.csv"),
        strategy=sampling_strategy,
        model_name=model_name,
        pool_csv=pool_csv
    )

    # ---------- C) 用同一个阈值做可视化 ----------
    viz_dir = os.path.join(dirs["visualizations"], round_name)
    visualize_predictions_overlay(
        model=trainer.model,
        dataset=val_ds,
        device=device,
        save_dir=viz_dir,
        num_samples=data_cfg.get("validation", {}).get("val_visualization_samples", 6),
        threshold=infer_thr,               # ★ 与验证一致
        keep_largest_cc=False,
        min_cc_area=0
    )

    # ---------- D) 采样（非最后一轮） ----------
    if rnd < total_iters - 1:
        logger.info("🎯 Sampling new indices for next round …")

        global2local = {g: p for p, g in enumerate(full_train_ds.df.index)}
        pool_df = pd.read_csv(pool_csv)
        unlabeled_global = pool_df.query("set == 'train' and labeled == 0").index.tolist()
        unlabeled_local = [global2local[g] for g in unlabeled_global if g in global2local]

        new_names, _sampling_time = sample_images_wrapper(
            strategy=sampling_strategy,
            model=trainer.model,
            dataset=full_train_ds,
            config=train_cfg,
            budget=budget,
            device=device,
            unlabeled_indices=unlabeled_local,
            csv_path=pool_csv,
        )

        mask_new = (pool_df["new_image_name"].isin(new_names) & (pool_df["labeled"] == 0))
        pool_df.loc[mask_new, "labeled"] = 1
        pool_df.to_csv(pool_csv, index=False)
        logger.info(f"✅  added {mask_new.sum()} samples "
                    f"(total labeled = {pool_df['labeled'].sum()})")

    return val_metrics, best_thr_from_val



def active_learning_loop(
                            train_config: Dict[str, Any],
                            data_config : Dict[str, Any],
                            model_config: Dict[str, Any],
                            sampling_strategy: str,
                            pool_csv: str,
                            dirs: Dict[str, str],
                            device: torch.device,
                            model_name: str,
                            seed: int = 42,
                        ) -> Dict[str, Any]:
    """
    Run round-0 warm-start + N active-learning cycles.
    `train_config['sampling']['num_cycles']` == number of cycles **after** warm-start.
    """
    # ─────────────────────────── setup ────────────────────────────
    total_cycles = train_config["sampling"]["num_cycles"]          # user arg
    budget       = train_config["sampling"]["budget"]

    logger.info("🚀 Starting Active Learning Pipeline")
    logger.info(f"   strategy = {sampling_strategy}")
    logger.info(f"   cycles   = {total_cycles}  (+ warm-start round0)")
    logger.info(f"   budget   = {budget} / cycle")
    logger.info(f"   device   = {device}")
    logger.info(f"   pool     = {pool_csv}")

    results = {
        "rounds"                : [],
        "sampling_strategy"     : sampling_strategy,
        "total_samples_per_round": [],
        "val_metrics_per_round" : [],
    }

    # ─────────── basic vars ────────────
    cycles_after_warm = train_config["sampling"]["num_cycles"]  # e.g. 2
    budget = train_config["sampling"]["budget"]
    total_iters = cycles_after_warm + 1  # +1 gives us round0
    best_thr_last = None  # keep best threshold from last validation sweep

    # ─────────── warm-start once ────────────
    pool_df = pd.read_csv(pool_csv)
    if pool_df["labeled"].sum() == 0:  # first ever run
        warm_ids = get_warm_ids_from_disk(pool_df, data_config)
        if not warm_ids:
            init_budget = train_config["sampling"].get("init_budget", 8)
            warm_ids = (pool_df.query("set == 'train' and labeled == 0")
                        .sample(init_budget, random_state=seed).index.tolist())
            logger.warning(f"🌱 Warm-start fallback: random {init_budget} samples")
        else:
            logger.info(f"🌱 Warm-start from disk folder: {len(warm_ids)} samples")
        pool_df.loc[warm_ids, "labeled"] = 1
        pool_df.to_csv(pool_csv, index=False)

    # ------------------  round0 数据集  ------------------
    ds_round0 = create_round_datasets(data_config, pool_csv, current_round=0)
    train_ds0, val_ds0 = ds_round0["train"], ds_round0["val"]

    # ------------------  round0 模型 & Trainer -----------
    model = build_model(model_name, model_config)
    conf0   = train_config.copy()
    conf0["save_dir"] = os.path.join(dirs["checkpoints"], "round0")
    trainer = Trainer(model, train_ds0, val_ds0, conf0)

    full_train_ds = create_datasets(
        csv_path=pool_csv,
        base_dir=data_config['base_dir_standard'],
        normalize=True,
        target_size=(512, 512),
        binary_mask=True,
        use_labeled_only=False,

    )['train']

    trainer.train()

    if hasattr(train_ds0, "log_summary"): train_ds0.log_summary(reset=True)
    if hasattr(val_ds0, "log_summary"): val_ds0.log_summary(reset=True)

    semi_cfg = train_config.get('semi', {})
    if semi_cfg.get('enable', False):
        u_loader = build_unlabeled_loader(data_config, pool_csv, batch_size=train_config['batch_size'])
        if u_loader is not None:
            trainer.train_semi(
                unlabeled_loader=u_loader,
                num_epochs=int(semi_cfg.get('epochs', 6)),
                tau=float(semi_cfg.get('tau', 0.90)),
                lambda_pseudo=float(semi_cfg.get('lambda_pseudo', 1.0)),
                lambda_cons=float(semi_cfg.get('lambda_cons', 0.1)),
                min_area=int(semi_cfg.get('min_area', 50)),
                ignore_border_px=int(semi_cfg.get('ignore_border_px', 2)),
            )


    val_metrics0, best_thr_last = validate_visualize_sample(
        rnd=0, total_iters=total_iters, trainer=trainer,
        val_ds=val_ds0, full_train_ds=full_train_ds,
        dirs=dirs, train_cfg=train_config, data_cfg=data_config,
        pool_csv=pool_csv, sampling_strategy=sampling_strategy,
        budget=budget, device=device, model_name=model_name)

    # ─────────── iterate round0 … roundN ────────────
    for rnd in range(1, total_iters):  # 0-based
        round_name = f"round{rnd}"
        logger.info("\n" + "=" * 60)
        logger.info(f"🔁 {round_name} / {total_iters - 1}  —  {sampling_strategy}")
        logger.info("=" * 60)
        round_start = time.time()

        tqdm.write("=" * 60)
        tqdm.write(f"🔁 {round_name} / {total_iters - 1} — {sampling_strategy}")
        tqdm.write("=" * 60)

        pool_df = pd.read_csv(pool_csv)  # refresh
        unlabeled_idx = pool_df.query("set == 'train' and labeled == 0").index.tolist()

        # ---------- 1. datasets ----------
        datasets = create_round_datasets(data_config, pool_csv, current_round=rnd)
        train_ds, val_ds = (datasets["train"], datasets["val"])
        if len(train_ds) == 0:
            raise RuntimeError("train_dataset empty – check warm-start / CSV")

        # === 🔧 [ADD] 限制详细样本日志条数，避免日志爆炸 ===
        if hasattr(train_ds, "_set_dbg_limit"):
            train_ds._set_dbg_limit(20)
        else:
            setattr(train_ds, "_dbg_limit", 20);
            setattr(train_ds, "_dbg_seen", 0)
        if hasattr(val_ds, "_set_dbg_limit"):
            val_ds._set_dbg_limit(10)
        else:
            setattr(val_ds, "_dbg_limit", 10);
            setattr(val_ds, "_dbg_seen", 0)

        # ---------- 2. model + trainer ----------
        trainer.update_datasets(train_ds, val_ds, reset_early_stopping=True)
        trainer.save_dir = Path(os.path.join(dirs["checkpoints"], f"{round_name}"))
        trainer.save_dir.mkdir(parents=True, exist_ok=True)

        # （可选）跳过 warmup 调度器异常
        if hasattr(trainer, "scheduler") and isinstance(getattr(trainer, "scheduler"), GradualWarmupScheduler):
            if trainer.scheduler.multiplier <= 1.0:
                logger.warning("⚠️  warm-up multiplier ≤ 1 – skipping warm-up scheduler")
                trainer.scheduler = None

        # =========== ★★★ 关键：Warm-start + 二阶段训练  ★★★ ===========
        prev_best_path = Path(dirs["checkpoints"]) / f"round{rnd - 1}" / "best_model.pth"
        # 可调参数（也可做成 YAML 或 CLI）
        WARM_START = True
        PHASE1_EPOCHS = 5  # 先冻 3~5 个 epoch
        TOTAL_EPOCHS = int(train_config.get("num_epochs", 20))
        PHASE2_EPOCHS = max(0, TOTAL_EPOCHS - PHASE1_EPOCHS)
        LR_HEAD_P1 = float(train_config.get("optimizer", {}).get("lr_head_p1", 5e-4))
        LR_BB_P2 = float(train_config.get("optimizer", {}).get("lr_backbone_p2", 1e-4))
        LR_HEAD_P2 = float(train_config.get("optimizer", {}).get("lr_head_p2", 5e-4))

        if rnd > 0 and WARM_START and prev_best_path.exists():
            # 1) 仅加载上一轮权重
            trainer.load_weights_only(str(prev_best_path), strict=True)

            # 2) Phase-1：冻结 backbone，只训 head
            trainer.set_backbone_trainable(False)
            trainer.setup_optimizer_with_lrs(lr_backbone=0.0, lr_head=LR_HEAD_P1)
            trainer.train(num_epochs=PHASE1_EPOCHS)

            # 3) Phase-2：解冻，分层 lr 继续训练
            trainer.set_backbone_trainable(True)
            trainer.setup_optimizer_with_lrs(lr_backbone=LR_BB_P2, lr_head=LR_HEAD_P2)
            trainer.train(num_epochs=PHASE2_EPOCHS)
        else:
            # 第一轮或没有上一轮 best：按原始配置从头训练
            trainer.optimizer = trainer._setup_optimizer()
            trainer.scheduler = trainer._setup_scheduler()
            trainer.train()  # 用 Train.yaml 的 num_epochs

        semi_cfg = train_config.get('semi', {})
        if semi_cfg.get('enable', False):
            u_loader = build_unlabeled_loader(data_config, pool_csv, batch_size=train_config['batch_size'])
            if u_loader is not None:
                trainer.train_semi(
                    unlabeled_loader=u_loader,
                    num_epochs=int(semi_cfg.get('epochs', 6)),
                    tau=float(semi_cfg.get('tau', 0.90)),
                    lambda_pseudo=float(semi_cfg.get('lambda_pseudo', 1.0)),
                    lambda_cons=float(semi_cfg.get('lambda_cons', 0.1)),
                    min_area=int(semi_cfg.get('min_area', 50)),
                    ignore_border_px=int(semi_cfg.get('ignore_border_px', 2)),
                )

        # ---------- 验证、可视化、采样 ----------
        val_metrics, best_thr_last = validate_visualize_sample(
            rnd=rnd, total_iters=total_iters, trainer=trainer,
            val_ds=val_ds, full_train_ds=full_train_ds,
            dirs=dirs, train_cfg=train_config, data_cfg=data_config,
            pool_csv=pool_csv, sampling_strategy=sampling_strategy,
            budget=budget, device=device, model_name=model_name)

        # 让 full_train_ds 与最新 pool.csv 保持同步
        _pool_latest = pd.read_csv(pool_csv)[["new_image_name", "labeled"]]
        full_train_ds.df = full_train_ds.df.merge(_pool_latest, on="new_image_name", how="left", suffixes=("", "_new"))
        full_train_ds.df["labeled"] = full_train_ds.df["labeled_new"].fillna(full_train_ds.df["labeled"]).astype(int)
        full_train_ds.df.drop(columns=[c for c in full_train_ds.df.columns if c.endswith("_new")], inplace=True)

        # ---------- 记账 ----------
        results["rounds"].append({
            "round": round_name,
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "unlabeled_samples": len(unlabeled_idx),
            "best_dice": trainer.best_dice,
            "best_dice_round": trainer.best_dice_round,
            "time_min": (time.time() - round_start) / 60,
            "best_thr": float(best_thr_last),
        })
        results["total_samples_per_round"].append(len(train_ds))
        results["val_metrics_per_round"].append(val_metrics)

    # ─────────── finish ───────────
    with open(os.path.join(dirs["results"], "final_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    plot_metrics_curve(csv_path=os.path.join(dirs["results"], "round_metrics.csv"),
                       save_path=os.path.join(dirs["results"], "metrics_curve.png"))

    logger.info("🔬  Evaluating final model on TEST set ...")

    # create test-dataset / loader
    tmp_test_csv = Path(dirs["results"]) / "test_only.csv"
    pd.read_csv(pool_csv).query("set=='test'").to_csv(tmp_test_csv, index=False)

    test_ds = CervixDataset(
        csv_path=str(tmp_test_csv),
        image_dir=os.path.join(data_config['base_dir_standard'], 'test', 'images'),
        mask_dir=os.path.join(data_config['base_dir_standard'], 'test', 'masks'),
        normalize=True,
        target_size=(512, 512),
        binary_mask=True,
        enable_roi=data_config.get('roi', {}).get('enable_val', False),
        pad_ratio=data_config.get('roi', {}).get('pad_ratio', 0.10),
        roi_mode=data_config.get('roi', {}).get('mode_test', 'image'),
        use_mask_on_valtest=False
    )

    bs = train_config.get("batch_size",
                          train_config.get("train", {}).get("batch_size", 4))

    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=0
    )

    # Use the best-threshold found on the last validation sweep (fallback to config)
    test_thr = float(best_thr_last) if ("best_thr_last" in locals() and best_thr_last is not None) \
        else float(train_config.get("inference", {}).get("threshold", 0.50))

    test_metrics = evaluate_full(trainer.model, test_loader, device, threshold=test_thr)
    logger.info(f"Using inference threshold for TEST: {test_thr:.2f}")

    logger.info(
        ("TEST  Dice={dice:.4f}  IoU={iou:.4f}  HD95={hd95:.2f}  "
         "Prec={precision:.4f}  Rec={recall:.4f}  F1={f1:.4f}  "
         "PA={pixel_acc:.4f}  mPA={mean_pixel_acc:.4f}")
        .format(**test_metrics)
    )

    with open(os.path.join(dirs["results"], "final_test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)

    results["test_metrics"] = test_metrics
    logger.info("🏁 Active-learning pipeline finished.")

    return results


def main():
    """Main function to run active learning pipeline."""
    parser = ArgumentParser(description='Active Learning Training Pipeline')
    
    # Config file paths
    parser.add_argument('--train_config', type=str,
                        default="/content/drive/MyDrive/CerVAI/src/configs/train_config/Train.yaml",
                        help='Path to training config YAML')
    parser.add_argument('--data_config', type=str,
                        default='/content/drive/MyDrive/CerVAI/src/configs/data_config/Data.yaml',
                        help='Path to data config YAML')
    parser.add_argument('--model_config', type=str,
                        default='/content/drive/MyDrive/CerVAI/src/configs/model_config/DeepLabV3Plus.yaml',
                        help='Path to model config YAML')

    # parser.add_argument('--train_config', type=str,
    #                    default='/Users/daidai/Documents/pythonProject_summer/CerVAI/src/configs/train_config/Train.yaml',
    #                    help='Path to training config YAML')
    # parser.add_argument('--data_config', type=str,
    #                    default='/Users/daidai/Documents/pythonProject_summer/CerVAI/src/configs/data_config/Data.yaml',
    #                    help='Path to data config YAML')
    # parser.add_argument('--model_config', type=str,
    #                    default='/Users/daidai/Documents/pythonProject_summer/CerVAI/src/configs/model_config/DeepLabV3Plus.yaml',
    #                    help='Path to model config YAML')

    # parser.add_argument('--train_config', type=str,
    #                    default='/home/mry/CerVAI/src/configs/train_config/Train.yaml',
    #                    help='Path to training config YAML')
    # parser.add_argument('--data_config', type=str,
    #                    default='/home/mry/CerVAI/src/configs/data_config/Data.yaml',
    #                    help='Path to data config YAML')
    # parser.add_argument('--model_config', type=str,
    #                    default='/home/mry/CerVAI/src/configs/model_config/DeepLabV3Plus.yaml',
    #                    help='Path to model config YAML')
    
    # Active learning parameters
    parser.add_argument('--sampling_strategy', type=str, default='random',
                       choices=['random', 'entropy', 'entropy_mc', 'bordaimage','bordabatch','representative'],
                       help='Sampling strategy for active learning')
    parser.add_argument('--rounds', type=int, default=3,
                       help='Number of active learning rounds')
    parser.add_argument('--budget', type=int, default=25,
                       help='Number of samples to select per round')
    
    # Data paths
    parser.add_argument('--csv_path', type=str,
                       default='aceto_mask_check_split.csv',
                       help='Path to metadata CSV file')
    parser.add_argument('--output_dir', type=str, default='experiments',
                       help='Output directory for experiments')
    
    # Training overrides
    parser.add_argument('--train__batch_size', type=int, help='Batch size')
    parser.add_argument('--train__num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--train__optimizer__lr', type=float, help='Learning rate')
    parser.add_argument('--train__early_stopping_patience', type=int, help='Early stopping patience')
    parser.add_argument("--model_name", type=str, default="DeepLabV3Plus", help="Model name for logging")

    # General parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='auto', 
                       choices=['auto', 'cuda', 'cpu'], help='Device to use')

    parser.add_argument('--inference__threshold', type=float, default=0.5,
                        help='Probability threshold for foreground (eval/vis)')
    parser.add_argument('--semi__tau', type=float, default=0.75,
                        help='Pseudo-label confidence (keep if prob≥tau)')
    parser.add_argument('--semi__min_area', type=int, default=10,
                        help='Min connected-component area for pseudo labels')
    parser.add_argument('--semi__ignore_border_px', type=int, default=2,
                        help='Ignore border pixels around pseudo labels')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Load configurations
    logger.info("📋 Loading configurations...")
    train_config = load_config(args.train_config)
    data_config = load_config(args.data_config)
    model_config = load_config(args.model_config)

    train_config.setdefault('inference', {})
    train_config['inference']['threshold'] = args.inference__threshold

    train_config.setdefault('semi', {})
    if args.semi__tau is not None:
        train_config['semi']['tau'] = args.semi__tau
    if args.semi__min_area is not None:
        train_config['semi']['min_area'] = args.semi__min_area
    if args.semi__ignore_border_px is not None:
        train_config['semi']['ignore_border_px'] = args.semi__ignore_border_px

    # ---------- experiment & pool ----------
    exp_id   = f"{args.model_name}_{args.sampling_strategy}_{datetime.now():%Y%m%d_%H%M%S}"
    exp_dir  = Path("experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # master_csv     = Path("/Users/daidai/Documents/pythonProject_summer/CerVAI") / "master_pool.csv"    # single source-of-truth
    # master_csv = Path("/home/mry//CerVAI") / "master_pool.csv"
    master_csv = Path(load_config(args.data_config)['meta_csv']).parent / "master_pool.csv"

    # --- 如果 master_pool 不存在就用 meta CSV 生成一份 ---
    if not master_csv.exists():
        # meta_csv = Path("/Users/daidai/Documents/pythonProject_summer/CerVAI/aceto_mask_check_split_final.csv")  # e.g. aceto_mask_check_split.csv
        real_meta_csv = Path(load_config(args.data_config)['meta_csv'])
        df = pd.read_csv(real_meta_csv)


    # if not master_csv.exists():
        # meta_csv = Path("/home/mry/CerVAI/aceto_mask_check_split_final.csv")  # e.g. aceto_mask_check_split.csv
        # df = pd.read_csv(meta_csv)

        # 保证必须列
        if 'labeled' not in df.columns:
            df['labeled'] = 0  # 全部未标注
        if 'set' not in df.columns:
            raise ValueError("'set' column (train/val/test) is missing in meta csv")

        df.to_csv(master_csv, index=False)
        logger.info(f"🆕 Created master_pool.csv from {real_meta_csv}")

    run_pool_csv   = exp_dir / "pool.csv"
    shutil.copy(master_csv, run_pool_csv)
    logger.info(f"✅ Copied master_pool → {run_pool_csv}")

    # Update configs with command line arguments
    train_config = update_config_from_args(train_config, args, 'train')
    
    # Override with command line arguments
    if args.rounds:
        train_config['sampling']['num_cycles'] = args.rounds
    if args.budget:
        train_config['sampling']['budget'] = args.budget
    
    # Get centralized CSV path - now use the copied pool
    csv_path = str(run_pool_csv)
    
    # Setup experiment directories
    experiment_name = f"AL_{args.sampling_strategy}_rounds{args.rounds}_budget{args.budget}"
    dirs = setup_experiment_dirs(args.output_dir, experiment_name)

    
    # Setup experiment logging
    setup_experiment_logging(dirs['logs'])
    
    # Save configurations
    save_config(train_config, os.path.join(dirs['configs'], 'train_config.yaml'))
    save_config(data_config, os.path.join(dirs['configs'], 'data_config.yaml'))
    save_config(model_config, os.path.join(dirs['configs'], 'model_config.yaml'))
    
    # Save command line arguments
    with open(os.path.join(dirs['configs'], 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Log configuration summary
    logger.info("📋 Configuration Summary:")
    logger.info(f"   - Strategy: {args.sampling_strategy}")
    logger.info(f"   - Rounds: {args.rounds}")
    logger.info(f"   - Budget per round: {args.budget}")
    logger.info(f"   - CSV path: {csv_path}")
    logger.info(f"   - Experiment dir: {dirs['experiment']}")
    
    # Run active learning pipeline
    try:
        results = active_learning_loop(
            train_config=train_config,
            data_config=data_config,
            model_config=model_config,
            sampling_strategy=args.sampling_strategy,
            pool_csv=csv_path,
            dirs=dirs,
            device=device,
            model_name=args.model_name,
            seed=args.seed
        )
        
        logger.info("✅ Pipeline completed successfully!")
        logger.info("📊 Final results:")
        for rd in results["rounds"]:
            logger.info(f"  {rd['round']}: {rd['train_samples']} samples, "
                        f"Dice: {rd['best_dice']:.4f},"
                        f" Dice_round: {rd['best_dice_round']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
