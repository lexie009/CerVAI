import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import random
import os
import logging
from typing import Dict, Any, Optional, Tuple
from tqdm.auto import tqdm
import time
from pathlib import Path
from monai.losses import DiceCELoss, TverskyLoss, FocalLoss
from utils.evaluate_utils import _tta_logits, get_foreground_prob


# MONAI metrics
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.transforms import Compose, Activations, AsDiscrete

# === add near your imports ===
try:
    import scipy.ndimage as ndi
    _HAS_NDI = True
except Exception:
    _HAS_NDI = False

"""from https://github.com/jizongFox/deepclustering2/blob/master/deepclustering2/schedulers/warmup_scheduler.py"""
class GradualWarmupScheduler(lr_scheduler._LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier <= 1.0:
            raise ValueError("multiplier should be greater than 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = (
            epoch if epoch != 0 else 1
        )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group["lr"] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != lr_scheduler.ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

def remove_small_cc(mask_bool: torch.Tensor, min_area: int) -> torch.Tensor:
    """
    mask_bool: [H, W] bool
    return    : [H, W] bool, removing components < min_area
    """
    if min_area <= 0:
        return mask_bool
    m = mask_bool.cpu().numpy().astype(np.uint8)
    if _HAS_NDI:
        labeled, num = ndi.label(m)
        if num == 0:
            return mask_bool
        sizes = ndi.sum(m, labeled, index=range(1, num + 1))
        keep = np.zeros_like(m, dtype=np.uint8)
        for lbl, sz in enumerate(sizes, start=1):
            if sz >= min_area:
                keep[labeled == lbl] = 1
        out = torch.from_numpy(keep.astype(bool)).to(mask_bool.device)
        return out
    else:
        # 简易回退：用开运算近似抑制小块（不会“扩张”）
        k = 3
        pad = k // 2
        x = Fnn.max_pool2d(mask_bool.float().unsqueeze(0).unsqueeze(0), k, 1, pad)[0, 0]  # dilation
        y = Fnn.avg_pool2d(x.unsqueeze(0).unsqueeze(0), k, 1, pad)[0, 0] >= 1.0  # erosion after dilation
        return y.bool()

def build_loss():
    """
    返回 (dice_ce, tversky) 两个 loss 实例。
    适用于二分类单通道输出 (B,1,H,W)，mask 为 {0,1} 浮点。
    """
    dice_ce = DiceCELoss(
        sigmoid=True,          # 单通道 + sigmoid
        to_onehot_y=False,
        squared_pred=False,
        lambda_dice=1.0,
        lambda_ce=0.5,
    )
    tversky = TverskyLoss(
        sigmoid=True,
        alpha=0.75,            # 更重罚 FP
        beta=0.25,
    )
    return dice_ce, tversky

def make_pseudo_and_valid(prob_fg: torch.Tensor,
                           tau: float = 0.9,
                           min_area: int = 50,
                           ignore_border_px: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """
    prob_fg: [H, W] 前景概率
    return:
      pseudo: [H, W] long，取值 {0,1,255} (255=ignore，用于CE类);
      valid : [H, W] bool，有效监督位置（给 Dice/Tversky 用）
    规则：阈值化 -> 连通域过滤(只删小块，不做扩张) -> 边界像素标 ignore
    """
    # 1) hard threshold
    hard = (prob_fg > tau)

    # 2) remove small CC (收紧：只删，不增)
    hard = remove_small_cc(hard, min_area)

    # 3) 边界 ignore：用腐蚀来找边界
    if ignore_border_px > 0:
        kernel = torch.ones(1, 1, 3, 3, device=prob_fg.device)
        x = hard.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        eroded = x
        for _ in range(ignore_border_px):
            eroded = Fnn.max_pool2d(eroded, 3, stride=1, padding=1)  # 先“扩”后再与原图比较拿边缘
        border = (x != eroded).squeeze(0).squeeze(0).bool()  # 边界环
    else:
        border = torch.zeros_like(hard)

    # 伪标签 & 有效掩码
    pseudo = hard.long()
    pseudo[border] = 255  # ignore index for CE
    valid = ~border  # 给 Dice/Tversky 用的有效像素掩码

    return pseudo, valid

class Trainer:
    """
    Comprehensive Trainer class for binary semantic segmentation.
    
    Features:
    - Training loop with train_one_epoch() and validate()
    - CrossEntropyLoss for binary segmentation
    - Adam optimizer with configurable learning rate
    - Learning rate scheduler (StepLR or ReduceLROnPlateau)
    - Learning rate warmup support
    - Evaluation metrics (Dice, IoU, Hausdorff95) using MONAI
    - Early stopping
    - Checkpointing (saves best model)
    - Logging to console and file
    - Device compatibility (CUDA/CPU)
    - Random seed control
    """
    
    def __init__(self, 
                 model: nn.Module,
                 train_dataset,
                 val_dataset,
                 config: Dict[str, Any]):
        """
        Initialize the Trainer.
        
        Args:
            model: PyTorch model (e.g., DeepLabV3Plus)
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Configuration dictionary from Train.yaml
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Set random seed for reproducibility
        self._set_seed(42)
        
        # Device setup
        self.device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        self.repeat_k = int(config.get("repeat_aug_k", 1))

        if self.repeat_k <= 1:
                train_sampler = None
                train_shuffle = True
                effective_train_len = len(train_dataset)
        else:
        # 用“放回抽样”的 RandomSampler，把一个 epoch 的样本数扩展为 K * N
            train_sampler = RandomSampler(
                train_dataset, replacement=True, num_samples=self.repeat_k * len(train_dataset))

            train_shuffle = False  # sampler 生效时必须关闭 shuffle
            effective_train_len = self.repeat_k * len(train_dataset)

        self.logger = logging.getLogger(__name__) if hasattr(self, "logger") else logging.getLogger(__name__)
        self.logger.info(f"[RepeatAug] K={self.repeat_k} → effective samples/epoch = {effective_train_len}")

        self.train_loader = DataLoader(
                    train_dataset,
                    batch_size = config['batch_size'],
                    shuffle = train_shuffle,
                    sampler = train_sampler,
                    num_workers = config.get('num_workers', 4),
                    pin_memory = True if self.device.type == 'cuda' else False,
                    drop_last = True)
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 4),
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Loss function
        self.criterion = self._setup_loss()
        self.dice_ce, self.tversky = build_loss()

        # Optimizer
        self.optimizer = self._setup_optimizer()
        
        # Scheduler
        self.scheduler = self._setup_scheduler()
        
        # Metrics
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.iou_metric = MeanIoU(include_background=False, reduction="mean")
        self.hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction="mean", percentile=95)
        self.loss_name = self.config['loss']['name']
        self.loss_needs_channel = self.loss_name in {'DiceCE', 'Tversky', 'Focal', 'Dice'}
        
        # Post-processing for metrics (convert to discrete)
        self.post_pred = AsDiscrete(argmax=True)
        self.post_label = AsDiscrete(to_onehot=2)


        # Training state
        self.current_epoch = 0

        self.best_dice_global = -1.0  # 跨所有轮次的历史最佳
        self.best_dice_round = -1.0

        self.best_dice = 0.0
        self.best_loss = float('inf')
        self.early_stopping_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'val_dice': [], 'val_iou': [], 'val_hausdorff': []
        }
        
        # Setup logging and save directory
        self.save_dir = Path(config.get('save_dir', './checkpoints'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
    def _set_seed(self, seed: int = 42):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _setup_logging(self):
        """Setup logging to file and console."""
        log_file = self.save_dir / 'training.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)

    def _setup_loss(self):
        """Setup loss function."""
        loss_name = self.config['loss']['name']
        loss_params = self.config['loss'].get('params', {})

        if loss_name == 'CrossEntropyLoss':
            return nn.CrossEntropyLoss(**loss_params)

        elif loss_name == 'DiceCE':
            return DiceCELoss(
                sigmoid=False,
                softmax=True,
                to_onehot_y=True,
                include_background=False
            )

        elif loss_name == 'Tversky':
            alpha = loss_params.get("alpha", 0.7)
            beta = loss_params.get("beta", 0.3)
            return TverskyLoss(
                alpha=alpha,
                beta=beta,
                include_background=False,
                to_onehot_y=True,
                softmax=True
            )

        elif loss_name == 'Focal':
            return FocalLoss(
                gamma=2.0,
                to_onehot_y=True,
                softmax=True
            )

        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def _prep_target_for_loss(self, masks: torch.Tensor) -> torch.Tensor:
        """
        把 GT 统一成当前 loss 期望的形状/类型：
        - CrossEntropyLoss  : [B,H,W] 的 long index
        - DiceCE/Tversky等  : [B,1,H,W] 的 long（内部会 to_onehot_y）
        """
        masks = masks.long()
        if self.loss_needs_channel:
            # 需要 [B,1,H,W]
            if masks.dim() == 3:
                return masks.unsqueeze(1)
            if masks.dim() == 4 and masks.size(1) == 1:
                return masks
            # 若传来 one-hot（极少见），降为 index 再加 channel
            return masks.argmax(dim=1, keepdim=True)
        else:
            # 需要 [B,H,W]
            if masks.dim() == 4 and masks.size(1) == 1:
                return masks.squeeze(1)
            if masks.dim() == 3:
                return masks
            # 若为 one-hot，降为 index
            return masks.argmax(dim=1)

    def _setup_optimizer(self):
        """Setup optimizer."""
        opt_config = self.config['optimizer']
        opt_name = opt_config['name']
        
        if opt_name == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=float(opt_config.get('weight_decay', 0))
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_name}")
            
    def _setup_scheduler(self):
        """Setup learning rate scheduler with optional warmup."""
        sched_config = self.config['scheduler']
        sched_name = sched_config['name']
        
        # Create base scheduler
        if sched_name == 'StepLR':
            base_scheduler = lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        elif sched_name == 'ReduceLROnPlateau':
            base_scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('gamma', 0.5),
                patience=sched_config.get('patience', 10),
                verbose=True
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_name}")
        
        # Add warmup if enabled
        warmup_config = self.config.get('warmup', {})
        if warmup_config.get('enabled', False):
            return GradualWarmupScheduler(
                self.optimizer,
                multiplier=warmup_config['multiplier'],
                total_epoch=warmup_config['warmup_epochs'],
                after_scheduler=base_scheduler
            )
        else:
            return base_scheduler

    def load_weights_only(self, path: str, strict: bool = True):
        """
        只加载模型权重，不恢复优化器/调度器/历史指标。
        用于每轮 AL 前的 warm-start。
        """
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt)  # 兼容直接存的 state_dict
        self.model.load_state_dict(state, strict=strict)
        self.logger.info(f"[WarmStart] Loaded weights from: {path} (optimizer/scheduler will be reset)")

    def set_backbone_trainable(self, trainable: bool = True):
        """
        冻结/解冻 backbone（SMP 的 DeepLabV3+：一般 'encoder' 是骨干，'decoder/segmentation_head' 是头）。
        """
        for name, p in self.model.named_parameters():
            if name.startswith(("encoder", "backbone")):
                p.requires_grad = trainable

    def setup_optimizer_with_lrs(self, lr_backbone: float = 1e-4,
                                 lr_head: float = 5e-4,
                                 weight_decay: float = None):
        """
        用分层学习率重建优化器：backbone 与 head 使用不同 lr。
        会覆盖 self.optimizer，并**重建** self.scheduler（让 LR 从初值开始）。
        """
        if weight_decay is None:
            weight_decay = float(self.config.get("optimizer", {}).get("weight_decay", 1e-4))

        bb_params, head_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(("encoder", "backbone")):
                bb_params.append(p)
            else:
                head_params.append(p)

        param_groups = []
        if bb_params:
            param_groups.append({"params": bb_params, "lr": lr_backbone, "weight_decay": weight_decay})
        if head_params:
            param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})

        opt_name = self.config.get("optimizer", {}).get("name", "Adam").lower()
        if opt_name == "adam":
            self.optimizer = torch.optim.AdamW(param_groups)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(param_groups, momentum=0.9, nesterov=True)
        else:  # 默认
            self.optimizer = torch.optim.Adam(param_groups)

        # 重新创建调度器（保持你原来的 _setup_scheduler 逻辑 & 配置）
        self.scheduler = self._setup_scheduler()

        self.logger.info(
            f"[WarmStart] Rebuilt optimizer with LR(backbone={lr_backbone}, head={lr_head}), "
            f"weight_decay={weight_decay}, opt={opt_name}"
        )
        return self.optimizer

    def train_one_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} Training')

        for batch_idx, (images, masks, _) in enumerate(progress_bar):
            images = images.to(self.device)
            masks = masks.to(self.device).long()

            if self.loss_needs_channel:
                # Dice/Tversky/Focal：需要 [B,1,H,W]
                if masks.dim() == 3:
                    masks = masks.unsqueeze(1)
            else:
                # CrossEntropy：需要 [B,H,W]
                if masks.dim() == 4 and masks.size(1) == 1:
                    masks = masks.squeeze(1)

            self.optimizer.zero_grad()

            outputs = self.model(images)  # logits

            # --- 允许 1 通道或 2 通道 ---
            assert outputs.ndim == 4 and outputs.shape[1] in (1, 2), \
                f"bad logits shape: {outputs.shape} (expect [B,1,H,W] or [B,2,H,W])"
            C = outputs.shape[1]

            with torch.no_grad():
                # 统一前景概率 p：1通道→sigmoid(channel0)；2通道→softmax(channel1)
                if C == 2:
                    s = torch.softmax(outputs, dim=1)
                    # 软互斥性自检（仅在2通道时）
                    _sum = s[:, 0] + s[:, 1]
                    assert torch.allclose(_sum, torch.ones_like(_sum), atol=1e-5), "softmax sum != 1"
                    p = s[:, 1]
                else:
                    p = torch.sigmoid(outputs[:, 0])

                frac_over_05 = (p >= 0.5).float().mean().item()
                frac_over_08 = (p >= 0.8).float().mean().item()
                if batch_idx % 50 == 0:
                    self.logger.info(f"[PROBE] p_fg mean={p.mean():.3f} "
                                     f"p>0.5={frac_over_05:.3f} p>0.8={frac_over_08:.3f}")

            # === 统一得到 index 形式的 GT，做健壮性检查 ===
            if self.loss_needs_channel:
                masks_idx = masks.squeeze(1) if (masks.dim() == 4 and masks.size(1) == 1) else masks
            else:
                masks_idx = masks  # CrossEntropy 时已是 [B,H,W]

            assert masks_idx.dtype in (torch.int64, torch.long), f"target dtype must be Long, got {masks_idx.dtype}"
            with torch.no_grad():
                uniq = torch.unique(masks_idx)
                assert set(uniq.tolist()).issubset({0, 1}), f"target values {uniq.tolist()} not in {{0,1}}"

            if self.current_epoch == 0 and batch_idx < 2:
                with torch.no_grad():
                    if C == 2:
                        prob = torch.softmax(outputs, dim=1)[:, 1]
                        bg = torch.softmax(outputs, dim=1)[:, 0]
                    else:
                        prob = torch.sigmoid(outputs[:, 0])
                        bg = 1.0 - prob
                    self.logger.info(
                        "[PROB-DIST] ep=%d it=%d fg[mean=%.4f, std=%.4f] "
                        "bg[mean=%.4f, std=%.4f] p_fg>0.5=%.3f p_fg>0.8=%.3f" %
                        (self.current_epoch, batch_idx,
                         prob.mean().item(), prob.std().item(),
                         bg.mean().item(), bg.std().item(),
                         (prob > 0.5).float().mean().item(),
                         (prob > 0.8).float().mean().item())
                    )

            masks_for_loss = self._prep_target_for_loss(masks)

            # ---- 关键改动：损失一律用“前景单通道” ----
            outputs_for_loss = outputs if C == 1 else outputs[:, 1:2]
            main_loss = self.dice_ce(outputs_for_loss, masks_for_loss.float()) \
                        + 0.5 * self.tversky(outputs_for_loss, masks_for_loss.float())

            # 与损失口径一致，per-pixel 也用单通道的 sigmoid 概率
            per_pixel = torch.abs(torch.sigmoid(outputs_for_loss) - masks_for_loss.float())
            per_sample = per_pixel.mean(dim=(1, 2, 3))
            fg_ratio = masks_for_loss.float().mean(dim=(1, 2, 3))
            weights = torch.where(
                fg_ratio < 1e-6,
                torch.tensor(0.3, device=fg_ratio.device),
                torch.tensor(1.0, device=fg_ratio.device)
            )

            norm = (per_sample.detach() + 1e-8) / (per_sample.detach().mean() + 1e-8)
            weighted = (weights * norm)
            loss = (weighted * main_loss).mean() if main_loss.ndim == 0 else (
                    weighted * main_loss.mean(dim=(1, 2, 3))).mean()

            loss.backward()

            # --- 梯度探针 ---
            if (self.current_epoch == 0) and (batch_idx == 0):
                g_sum, n_has_grad = 0.0, 0
                for _, p_ in self.model.named_parameters():
                    if p_.grad is not None:
                        g_sum += p_.grad.detach().abs().mean().item()
                        n_has_grad += 1
                g_mean = g_sum / max(n_has_grad, 1)
                self.logger.info(f"[GRAD-PROBE] mean(|grad|) on first batch = {g_mean:.6e}")
                progress_bar.write(f"[GRAD-PROBE] mean(|grad|) on first batch = {g_mean:.6e}")

            self.optimizer.step()

            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        return total_loss / num_batches

    def validate(self, thr: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        """
        Validate the model and compute loss & metrics with a unified evaluation path:
        - logits -> get_foreground_prob() 统一取前景概率
        - 按 config.inference.threshold 二值化
        - 统一 one-hot 为 2 通道后喂给 MONAI 指标
        """
        self.model.eval()
        total_loss = 0.0

        self._logged_fg_probe = False

        # === 选用阈值：优先用调用者传入的 thr；否则回退到 config ===
        selected_thr = float(thr) if thr is not None else float(
            self.config.get("inference", {}).get("threshold", 0.5)
        )

        # === Reset metrics ===
        self.dice_metric.reset()
        self.iou_metric.reset()
        self.hausdorff_metric.reset()

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} Validation")

        with torch.no_grad():
            for b_idx, batch in enumerate(progress_bar):
                # 兼容 (images, masks, idx) / dict batch
                if isinstance(batch, (list, tuple)):
                    images, masks = batch[0], batch[1]
                elif isinstance(batch, dict):
                    images, masks = batch["data"], batch["mask"]
                else:
                    raise TypeError(f"Unexpected batch type: {type(batch)}")

                images = images.to(self.device)
                masks = masks.to(self.device).long()  # GT 始终为 long

                # === 统一：loss 输入形状 ===
                # 如果损失函数需要 [B,1,H,W] 的 channel 维
                if self.loss_needs_channel:
                    if masks.dim() == 3:
                        masks_for_loss = masks.unsqueeze(1)  # [B,1,H,W]
                    elif masks.dim() == 4 and masks.size(1) == 1:
                        masks_for_loss = masks  # already [B,1,H,W]
                    else:
                        # 若传入是 [B,H,W] 之外的形状（如已 one-hot），取 argmax 还原 index，再 unsqueeze
                        masks_for_loss = masks
                        if masks_for_loss.dim() == 4 and masks_for_loss.size(1) > 1:
                            masks_for_loss = masks_for_loss.argmax(dim=1, keepdim=True)
                else:
                    # 如果损失函数期望 [B,H,W] 的 index 图
                    if masks.dim() == 4 and masks.size(1) == 1:
                        masks_for_loss = masks.squeeze(1)  # [B,H,W]
                    elif masks.dim() == 3:
                        masks_for_loss = masks
                    else:
                        # 若是 one-hot，降成 index
                        masks_for_loss = masks.argmax(dim=1)

                # === forward + loss（只做一次，含 TTA logits） ===
                logits = _tta_logits(self.model, images)  # 你项目里已有
                loss = self.criterion(logits, masks_for_loss)
                total_loss += float(loss.item())

                # === 统一指标口径：概率 -> 二值 -> one-hot(2通道) ===
                probs = get_foreground_prob(logits)  # [B,H,W] 概率，项目里已有实现
                pred_discrete = (probs > selected_thr).long()   # [B,H,W] {0,1}

                # === [PROBE: channel-swap sanity] 只在前2个batch打印 ===
                try:
                    # 统一得到 “前景概率(ch1)” 和 “背景概率(ch0)”
                    if logits.shape[1] == 2:  # 双通道
                        prob2 = torch.softmax(logits, dim=1)  # [B,2,H,W]
                        fg1 = prob2[:, 1]  # 前景通道（你当前假设）
                        fg0 = prob2[:, 0]  # 背景通道
                    else:  # 单通道模型的容错写法（sigmoid 前景）
                        p1 = torch.sigmoid(logits[:, 0])  # [B,1,H,W] or [B,H,W]
                        p1 = p1 if p1.dim() == 3 else p1.squeeze(1)
                        fg1 = p1
                        fg0 = 1.0 - p1

                    thr_ = float(selected_thr)

                    # 统一得到 [B,H,W] 的GT索引图 y_index
                    if self.loss_needs_channel:
                        # 你上面为了loss可能构造了 masks_for_loss=[B,1,H,W]
                        if masks_for_loss.dim() == 4 and masks_for_loss.size(1) == 1:
                            y_index = masks_for_loss.squeeze(1)  # [B,H,W]
                        else:
                            # 若是 one-hot 就降成 index
                            y_index = masks_for_loss.argmax(dim=1) if masks_for_loss.dim() == 4 else masks_for_loss
                    else:
                        # 你上面让 masks_for_loss 已经是 [B,H,W]
                        y_index = masks_for_loss  # [B,H,W]

                    # 预测二值图（分别假设 ch1 与 ch0 为前景）
                    pred1 = (fg1 > thr_).float()  # [B,H,W]
                    pred0 = (fg0 > thr_).float()  # [B,H,W]
                    y = y_index.float()

                    eps = 1e-6
                    # 按batch算Dice（简化版）
                    inter1 = (pred1 * y).sum(dim=(1, 2))
                    dice1 = (2 * inter1) / (pred1.sum(dim=(1, 2)) + y.sum(dim=(1, 2)) + eps)
                    inter0 = (pred0 * y).sum(dim=(1, 2))
                    dice0 = (2 * inter0) / (pred0.sum(dim=(1, 2)) + y.sum(dim=(1, 2)) + eps)

                    # 简易像素级混淆（按 ch1 假设）
                    tp = (pred1 * y).sum().item()
                    fp = (pred1 * (1 - y)).sum().item()
                    fn = ((1 - pred1) * y).sum().item()

                    if b_idx < 2:  # 只打头两批
                        self.logger.info(
                            f"[VAL-PROBE] thr={thr_:.2f} "
                            f"Dice(fg=ch1)={dice1.mean().item():.4f} "
                            f"Dice(fg=ch0)={dice0.mean().item():.4f} "
                            f"TP={tp:.0f} FP={fp:.0f} FN={fn:.0f}"
                        )
                except Exception as e:
                    if b_idx < 2:
                        self.logger.warning(f"[VAL-PROBE] channel-swap probe failed: {e}")

                # [PROBE-2] foreground-prob quick stats (log once per validate)
                if (b_idx == 0) and (not getattr(self, "_logged_fg_probe", False)):
                    q = probs.detach().reshape(-1).float().cpu()
                    # 过滤 NaN/Inf，避免 .min/.mean 报错
                    q = q[torch.isfinite(q)]
                    if q.numel() > 0:
                        p50 = q.median().item()
                        frac = (q > selected_thr).float().mean().item()
                        self.logger.info(
                            f"[FG-prob] min={q.min():.3f} p50={p50:.3f} mean={q.mean():.3f} "
                            f"max={q.max():.3f} | >thr({selected_thr:.2f})={frac:.3f}"
                        )

                    else:
                        self.logger.info("[FG-prob] all probs invalid (empty after filtering)")

                    self._logged_fg_probe = True
                # label_index: [B,H,W] {0,1}
                if masks.dim() == 4 and masks.size(1) == 1:
                    label_index = masks[:, 0, ...]
                elif masks.dim() == 3:
                    label_index = masks
                else:
                    # 若万一传进来的是 one-hot
                    label_index = masks.argmax(dim=1)

                # one-hot 成 2 通道（背景+前景）
                pred_oh = torch.nn.functional.one_hot(pred_discrete, num_classes=2).permute(0, 3, 1, 2).float()
                label_oh = torch.nn.functional.one_hot(label_index.clamp(min=0, max=1), num_classes=2).permute(0, 3, 1,
                                                                                                               2).float()

                # === 累积指标 ===
                self.dice_metric(y_pred=pred_oh, y=label_oh)
                self.iou_metric(y_pred=pred_oh, y=label_oh)
                # HausdorffDistanceMetric 对空掩膜会有告警，交给 try/except 聚合时兜底
                self.hausdorff_metric(y_pred=pred_oh, y=label_oh)

                progress_bar.set_postfix({"Val Loss": f"{loss.item():.4f}", "thr": f"{selected_thr:.2f}"})

        # === 聚合 ===
        avg_loss = total_loss / max(len(self.val_loader), 1)

        dice_score = float(self.dice_metric.aggregate().item())
        iou_score = float(self.iou_metric.aggregate().item())
        try:
            hausdorff_score = float(self.hausdorff_metric.aggregate().item())
        except Exception:
            hausdorff_score = float("nan")

        metrics = {
            "dice": dice_score,
            "iou": iou_score,
            "hausdorff95": hausdorff_score,
            "thr": selected_thr,
        }

        # 维护 best 记录
        if dice_score > getattr(self, "best_dice_round", -1):
            self.best_dice_round = dice_score
        if dice_score > getattr(self, "best_dice_global", -1):
            self.best_dice_global = dice_score
            self.best_dice = dice_score

        return avg_loss, metrics

    def validate_with_threshold(self, best_thr: float) -> Tuple[float, Dict[str, float]]:
        """
        用外部扫出来的 best_thr 做一次验证（与可视化保持一致口径）。
        返回：与 validate 相同 (avg_loss, metrics)；metrics['thr']==best_thr
        """
        return self.validate(thr=best_thr)

    def save_checkpoint(self, epoch: int, is_best: bool = False,
                        interval: int = 5, max_to_keep: int = 3,
                        light_ckpt: bool = True):
        """
        保存训练快照。
        - interval: 每多少个 epoch 保存一次常规 ckpt（非 best）。
        - max_to_keep: 常规 ckpt 只保留最近 N 个。
        - light_ckpt: 常规 ckpt 是否轻量化（仅存模型权重），best 始终全量保存。
        """

        # -------- 1) 是否到保存间隔（best 始终保存） --------
        if not is_best and (epoch % interval != 0):
            return  # 非 best 且未到间隔，直接返回

        # -------- 2) 组装要保存的内容 --------
        if is_best or not light_ckpt:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'best_dice': getattr(self, 'best_dice', None),
                'best_loss': getattr(self, 'best_loss', None),
                'training_history': getattr(self, 'training_history', None),
            }
        else:
            # 轻量 ckpt：大幅节省空间
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
            }

        # -------- 3) 路径与实际保存 --------
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            if hasattr(self, 'logger'):
                self.logger.info(
                    f"New best model saved (epoch {epoch}) with Dice: {getattr(self, 'best_dice', float('nan')):.4f}")
        else:
            ckpt_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, ckpt_path)

            # -------- 4) 只保留最近 N 个常规 ckpt --------
            try:
                import re, os
                all_ckpts = [p for p in self.save_dir.glob('checkpoint_epoch_*.pth')]

                # 按 epoch 数字排序
                def _ep(p):
                    m = re.search(r'checkpoint_epoch_(\d+)\.pth$', p.name)
                    return int(m.group(1)) if m else -1

                all_ckpts.sort(key=_ep)

                # 删除多余的旧文件
                if len(all_ckpts) > max_to_keep:
                    to_del = all_ckpts[:len(all_ckpts) - max_to_keep]
                    for p in to_del:
                        try:
                            os.remove(p)
                            if hasattr(self, 'logger'):
                                self.logger.info(f"Pruned old checkpoint: {p.name}")
                        except Exception as e:
                            if hasattr(self, 'logger'):
                                self.logger.warning(f"Failed to delete {p}: {e}")
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Prune checkpoints failed: {e}")

    def train(self, num_epochs: int = None):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Training samples: {len(self.train_dataset)}")
        self.logger.info(f"Validation samples: {len(self.val_dataset)}")

        # 支持传入 num_epochs 覆盖配置
        total_epochs = int(num_epochs) if num_epochs is not None else int(self.config['num_epochs'])
        early_stopping_patience = int(self.config.get('early_stopping_patience', 10))

        start_time = time.time()

        for epoch in range(total_epochs):
            self.current_epoch = epoch

            # Training phase
            train_loss = self.train_one_epoch()
            
            # Validation phase
            val_loss, val_metrics = self.validate()
            
            # Update learning rate scheduler
            if isinstance(self.scheduler, GradualWarmupScheduler):
                if isinstance(self.scheduler.after_scheduler, lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step_ReduceLROnPlateau(val_loss, epoch)
                else:
                    self.scheduler.step()
            elif isinstance(self.scheduler, lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_dice'].append(val_metrics['dice'])
            self.training_history['val_iou'].append(val_metrics['iou'])
            self.training_history['val_hausdorff'].append(val_metrics['hausdorff95'])
            
            # Check for best model
            is_best = val_metrics['dice'] > self.best_dice
            if is_best:
                self.best_dice = val_metrics['dice']
                self.best_loss = val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Log epoch results
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Epoch [{epoch+1}/{total_epochs}] - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}, "
                f"Hausdorff95: {val_metrics['hausdorff95']:.4f}, LR: {current_lr:.6f}"
            )
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping check
            if self.early_stopping_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Best Dice score: {self.best_dice:.4f}")
        
        # Save final training history
        history_path = self.save_dir / 'training_history.pth'
        torch.save(self.training_history, history_path)
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load a saved checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_dice = checkpoint['best_dice']
        self.best_loss = checkpoint['best_loss']
        self.training_history = checkpoint['training_history']
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")

    def update_datasets(self,
                        train_ds,
                        val_ds,
                        reset_early_stopping: bool = True):
        """
       Replace internal dataloaders with new datasets.
        """
        self.train_dataset = train_ds
        self.val_dataset = val_ds

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.config.get("batch_size", 4),
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.config.get("batch_size", 4),
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )

        self.best_dice_round = -1.0

        if reset_early_stopping:
            self.early_stopping_counter = 0

    def _hflip(self, x: torch.Tensor) -> torch.Tensor:
        # 仅用于一致性：左右翻转
        return torch.flip(x, dims=[-1])

    @torch.no_grad()
    def _pseudo_from_logits(self, logits: torch.Tensor, tau: float):
        """
        从 logits 生成伪标签与“保留mask”（高置信像素）。
        返回:
          pseudo: [B,H,W] Long
          keep_mask: [B,H,W] Bool
        """
        probs = torch.softmax(logits, dim=1)  # [B,C,H,W]
        confs, pseudo = probs.max(dim=1)  # [B,H,W], [B,H,W]
        keep_mask = confs.ge(tau)  # 置信度过滤
        return pseudo, keep_mask

    def train_one_epoch_semi(self,
                             unlabeled_loader,
                             tau: float = 0.90,
                             lambda_pseudo: float = 1.0,
                             lambda_cons: float = 0.1) -> float:
        """
        半监督一个 epoch：
        监督损失(标注) + 伪标签CE损失(未标注, ignore低置信像素) + 一致性MSE(翻转前后)
        """
        self.model.train()
        ce_ignore = nn.CrossEntropyLoss(ignore_index=255)

        labeled_iter = iter(self.train_loader)
        total_loss, steps = 0.0, 0

        for images_u in unlabeled_loader:
            images_u = images_u.to(self.device)

            # ------- 取一个有标签 batch（循环取，不够就重启） -------
            try:
                images_l, masks_l, _ = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.train_loader)
                images_l, masks_l, _ = next(labeled_iter)

            images_l = images_l.to(self.device)
            masks_l = masks_l.to(self.device).long()
            if self.loss_needs_channel:
                if masks_l.dim() == 3:
                    masks_l = masks_l.unsqueeze(1)
            else:
                if masks_l.dim() == 4 and masks_l.size(1) == 1:
                    masks_l = masks_l.squeeze(1)

            # ------- 前向 + 各项损失 -------
            self.optimizer.zero_grad()

            # 1) 监督（标注）
            logits_l = self.model(images_l)
            loss_sup = self.criterion(logits_l, masks_l)

            # 2) 伪标签（未标注）
            logits_u = self.model(images_u)  # [B,C,H,W]
            pseudo_u, keep = self._pseudo_from_logits(logits_u, tau)
            # 低置信像素设为 ignore_index=255
            pseudo_ce = pseudo_u.clone()
            pseudo_ce[~keep] = 255
            loss_pseudo = ce_ignore(logits_u, pseudo_ce)

            # 3) 一致性（翻转增广前后）
            logits_u_flip = self.model(self._hflip(images_u))
            # 翻回到原方向再比
            probs_u = torch.softmax(logits_u, dim=1)
            probs_u_flip = torch.softmax(self._hflip(logits_u_flip), dim=1)
            loss_cons = Fnn.mse_loss(probs_u, probs_u_flip)

            # 合成总损失
            loss = loss_sup + lambda_pseudo * loss_pseudo + lambda_cons * loss_cons
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            steps += 1

        return total_loss / max(steps, 1)

    def _hflip(self, x: torch.Tensor) -> torch.Tensor:
        # 水平翻转（最后一维是 W）
        return torch.flip(x, dims=[-1])

    def train_semi(
            self,
            unlabeled_loader,
            num_epochs: int = 6,
            tau: float = 0.70,
            lambda_pseudo: float = 1.0,
            lambda_cons: float = 0.1,
            min_area: int = 30,
            ignore_border_px: int = 2,
    ):
        """
        半监督（只收紧不扩张）：
          1) 用模型概率做硬阈值 → 删除小连通域 → 边界若干像素设为 ignore
          2) 伪标签支路一律用 CrossEntropy(ignore_index=255) 监督（稳定且支持像素级忽略）
          3) 一致性：对同一张图做水平翻转，翻回后对概率做 MSE
        说明：
          - 统一用 CE 是为了避免 Tversky/Dice 对标签形状(one-hot 与否)的要求，且能方便地屏蔽低置信/边界像素。
          - 对 logits 为 1 通道的二分类分割：用 sigmoid；对 2 通道：用 softmax 取前景通道。
        """
        self.model.train()
        ce_ignore_index = 255

        def _foreground_prob(logits: torch.Tensor) -> torch.Tensor:
            # 返回 [B,H,W] 概率图
            if logits.shape[1] == 1:
                return torch.sigmoid(logits).squeeze(1)
            else:
                return torch.softmax(logits, dim=1)[:, 1]

        for ep in range(num_epochs):
            pbar = tqdm(unlabeled_loader, desc=f" Semi E{ep + 1}/{num_epochs} ")
            for batch in pbar:
                # ---- 解包 batch，兼容多种 DataLoader 输出形态 ----
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                elif isinstance(batch, dict):
                    # 常见键名：'image'，否则取第一个值
                    images = batch.get("image", next(iter(batch.values())))
                else:
                    images = batch
                images = images.to(self.device, non_blocking=True)

                # ---- 生成伪标签（只收紧不扩张）----
                with torch.no_grad():
                    logits_u = self.model(images)  # [B,C,H,W]
                    probs_u = _foreground_prob(logits_u)  # [B,H,W]

                    pseudo_list, valid_list = [], []
                    for b in range(images.size(0)):
                        pseudo_b, valid_b = make_pseudo_and_valid(
                            probs_u[b],
                            tau=float(tau),
                            min_area=int(min_area),
                            ignore_border_px=int(ignore_border_px),
                        )
                        pseudo_list.append(pseudo_b)  # {0,1} 或 {0,1,255}
                        valid_list.append(valid_b)  # bool

                    pseudo = torch.stack(pseudo_list)  # [B,H,W]
                    valid = torch.stack(valid_list)  # [B,H,W] bool

                    # 统一构造 CE 的标签：无效像素 → ignore_index=255
                    pseudo_ce = pseudo.clone().long()  # [B,H,W] int64
                    pseudo_ce[~valid] = ce_ignore_index

                # ---- 前向 + 各项损失 ----
                self.optimizer.zero_grad()

                logits_s = self.model(images)  # student [B,C,H,W]

                # 监督项（伪标签 + CE + ignore）
                loss_sup = Fnn.cross_entropy(logits_s, pseudo_ce, ignore_index=ce_ignore_index)

                # 一致性项（teacher=student(with flip)，不反传梯度）
                with torch.no_grad():
                    logits_t = self.model(self._hflip(images))  # [B,C,H,W]
                if logits_s.shape[1] == 1:
                    probs_s = torch.sigmoid(logits_s).squeeze(1)  # [B,H,W]
                    probs_t = torch.sigmoid(logits_t).squeeze(1)  # [B,H,W]
                    probs_t = self._hflip(probs_t.unsqueeze(1)).squeeze(1)
                else:
                    probs_s = torch.softmax(logits_s, dim=1)  # [B,2,H,W]
                    probs_t = torch.softmax(logits_t, dim=1)  # [B,2,H,W]
                    probs_t = self._hflip(probs_t)

                loss_cons = Fnn.mse_loss(probs_s, probs_t)

                loss = lambda_pseudo * loss_sup + lambda_cons * loss_cons
                loss.backward()
                self.optimizer.step()

                pbar.set_postfix({"semi_loss": f"{float(loss):.4f}"})


