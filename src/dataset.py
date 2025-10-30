import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from typing import Optional, Tuple, Dict, Any, Union
import torch
import sys
import os
import numpy as np
import random
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_preprocessing.resizing import resize_with_padding
from utils.roi_utils import crop_by_mask, crop_by_auto


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transform(target_size: Tuple[int, int] = (512, 512),
                  normalize: bool = True,
                  augment: bool = False,
                  is_mask: bool = False) -> T.Compose:
    """
    Get transform pipeline based on settings.
    """
    transform_list = []
    transform_list.append(T.ToTensor())
    if normalize and not is_mask:
        transform_list.append(
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        )
    return T.Compose(transform_list)


class CervixDataset(Dataset):
    """
    PyTorch Dataset for loading cervix image-mask pairs (semantic segmentation).
    """

    def __init__(self,
                 csv_path: Union[str, pd.DataFrame],
                 image_dir: str,
                 mask_dir: str,
                 target_size: Tuple[int, int] = (512, 512),
                 normalize: bool = True,
                 augment: bool = False,
                 set_filter: Optional[str] = None,
                 binary_mask: bool = False,
                 round_column_filter: Optional[str] = None,
                 use_labeled_only: bool = False,
                 enable_roi: bool = False,
                 pad_ratio: float = 0.1,
                 roi_mode: str = "auto",
                 use_mask_on_valtest: bool = False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.target_size = target_size
        self.binary_mask = binary_mask
        self.enable_roi = enable_roi
        self.pad_ratio = pad_ratio
        self.augment = augment
        self.roi_mode = roi_mode
        self.use_mask_on_valtest = use_mask_on_valtest
        self.split = set_filter or ""

        # === Sanity/Debug 统计 & 限流 ===
        self.logger = logging.getLogger(f"Dataset[{self.split}]")
        self._dbg_limit = 5      # 只打印前 N 个样本的详细日志（可外部修改）
        self._dbg_seen = 0       # 已经详细打印过的样本数

        self._seen = 0           # 访问样本总数
        self._cnt_allzero = 0    # 掩膜全 0 的样本数
        self._cnt_allone = 0     # 掩膜全 1 的样本数
        self._pos_sum = 0.0      # 正类像素比例累计（用于均值）

        self.transform = get_transform(target_size, normalize, augment=False, is_mask=False)
        self.mask_transform = get_transform(target_size, normalize=False, augment=False, is_mask=True)

        # 载入 CSV
        self.df = pd.read_csv(csv_path) if isinstance(csv_path, str) else csv_path.copy()

        if use_labeled_only:
            self.df = self.df[self.df.get("labeled", 1) == 1]

        if set_filter:
            self.df = self.df[self.df['set'] == set_filter]

        if round_column_filter and round_column_filter in self.df.columns:
            self.df = self.df[self.df[round_column_filter] == True]

        if not hasattr(self, "_dbg_limit"):
            self._dbg_limit = int(os.environ.get("DATA_DBG_LIMIT", 20))  # 只详细打印前 N 个样本
        if not hasattr(self, "_dbg_seen"):
            self._dbg_seen = 0

        if not hasattr(self, "_seen"):
            self._seen = 0
        if not hasattr(self, "_pos_sum"):
            self._pos_sum = 0.0
        if not hasattr(self, "_cnt_allzero"):
            self._cnt_allzero = 0
        if not hasattr(self, "_cnt_allone"):
            self._cnt_allone = 0

        # split 名称用于日志 tag（尽量从 set_filter 推断）
        split_name = getattr(self, "set_filter", None) or getattr(self, "split", None) or "unknown"
        if not hasattr(self, "logger"):
            self.logger = logging.getLogger(f"CervixDataset[{split_name}]")

        # 必需列检查
        required_cols = ['new_image_name', 'new_mask_name']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        返回预处理后的 (image, mask) 对
        """
        row = self.df.iloc[idx]
        global_id = row.name
        img_path = os.path.join(self.image_dir, row['new_image_name'])
        mask_path = os.path.join(self.mask_dir, row['new_mask_name'])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # ==== 单样本调试总开关（该样本是否允许打印）====
        dbg_ok = (str(self.split).lower() == 'train') and (self._dbg_seen < self._dbg_limit)

        if dbg_ok:
            self.logger.info(f"[DATA/IO] Opened mask: {mask_path}, original size: {mask.size}")

        if self.enable_roi:
            # 掩膜全 0 时，crop_by_auto 会回退为原图
            img_np = np.array(image)
            mask_np = np.array(mask)
            img_np, mask_np = crop_by_auto(
                img_np, mask_np,
                mode=self.roi_mode,
                use_mask_on_valtest=self.use_mask_on_valtest,
                split=self.split,
                pad_ratio=self.pad_ratio,
                out_size=self.target_size
            )
            image = Image.fromarray(img_np)
            mask = Image.fromarray(mask_np)
        else:
            image = image.resize(self.target_size, resample=Image.BILINEAR)
            mask = mask.resize(self.target_size, resample=Image.NEAREST)
            if dbg_ok:
                self.logger.info(f"[DATA/IO] Resized mask size (PIL): {mask.size}")

        # === [PROBE-1] 变换前检查（并做限流打印） ===
        m = np.array(mask, dtype=np.uint8)
        u = np.unique(m)

        self._seen += 1
        pos_ratio_pre = float((m > 0).mean())
        if pos_ratio_pre < 1e-6:
            self._cnt_allzero += 1
        if pos_ratio_pre > 1.0 - 1e-6:
            self._cnt_allone += 1
        self._pos_sum += pos_ratio_pre

        if dbg_ok:
            self.logger.info(
                f"[DATA/SANITY:pre] {self.split} sample={row['new_image_name']} "
                f"img_size={image.size} mask_size={mask.size} "
                f"mask_uniq={u.tolist()[:6]} pos_ratio={pos_ratio_pre:.4f}"
            )

        if not set(u.tolist()) <= {0, 1, 255}:
            self.logger.warning(f"[DATA] Non-binary mask values {u[:6]} ... binarizing (>0) @ {mask_path}")
            m = (m > 0).astype(np.uint8) * 255
            mask = Image.fromarray(m, mode="L")
        elif m.max() == 0 and dbg_ok:
            self.logger.info(f"[DATA] All-zero mask @ {mask_path}")

        mask_tensor_before = T.ToTensor()(mask)
        if dbg_ok:
            self.logger.info(f"[DATA/TENSOR] Mask tensor before squeeze: {tuple(mask_tensor_before.shape)}")

        if mask_tensor_before.shape[1:] != self.target_size:
            # 这种错误应无条件提示
            print(f"[ERROR] ❌ Mask tensor shape mismatch: {tuple(mask_tensor_before.shape)} from {mask_path}")

        # === 同步几何增强（只在 train）===
        if self.augment:
            image, mask = self._sync_geom_aug(image, mask)
            # 颜色增强（只作用在 image）
            if random.random() < 0.5:
                cj = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                image = cj(image)

        # === 转 Tensor / 归一化 ===
        image = self.transform(image)
        mask = self.mask_transform(mask)

        # === [PROBE-2] 变换后检查（限流打印） ===
        mask_after_t = mask.clone()
        uniq_after = torch.unique(mask_after_t)
        pos_ratio_after = float((mask_after_t > 0.5).float().mean().item())

        if dbg_ok:
            self.logger.info(
                f"[DATA/SANITY:postT] {self.split} sample={row['new_image_name']} "
                f"img[tensor] min/max/mean=({image.min():.3f},{image.max():.3f},{image.mean():.3f}) "
                f"mask_uniq={uniq_after.tolist()} pos_ratio={pos_ratio_after:.4f} "
                f"tensor_shape(img,mask)={[tuple(image.shape), tuple(mask_after_t.shape)]}"
            )

        # 将 mask→long，二类分割阈值 0.5
        if self.binary_mask:
            mask = (mask > 0.5).long().squeeze(0)
        else:
            mask = mask.long().squeeze(0)

        assert mask.dtype == torch.long and mask.ndim == 2 and mask.max() <= 1, \
            f"Invalid mask: {mask.unique()}"

        # 该样本的详细日志已输出完毕，增加计数
        if dbg_ok:
            self._dbg_seen += 1

        return image, mask, global_id

    def log_summary(self, reset: bool = False):
        """
        打印本数据集在 __getitem__ 过程中积累的简单统计。
        Args:
            reset: True 则打印后清零计数器；False 则保留继续累计。
        """
        # 柔性兜底（防止属性不存在）
        if not hasattr(self, "_seen"):
            self._reset_counters()

        n = int(self._seen)
        if n <= 0:
            self.logger.info("[DATA/SUMMARY] no samples seen yet.")
            return

        avg_pos = self._pos_sum / max(1, n)
        self.logger.info(
            "[DATA/SUMMARY] seen=%d  all_zero=%d  all_one=%d  avg_pos_ratio=%.4f",
            n, int(self._cnt_allzero), int(self._cnt_allone), float(avg_pos)
        )

        if reset:
            self._reset_counters()
            self.logger.info("[DATA/SUMMARY] counters reset.")

    def _reset_counters(self):
        """Reset sanity-check counters (and limited debug counter)."""
        self._seen = 0
        self._pos_sum = 0.0
        self._cnt_allzero = 0
        self._cnt_allone = 0
        self._dbg_seen = 0

    def _sync_geom_aug(self, image_pil, mask_pil):
        """同步几何增强（仅 train 使用）：翻转 + 旋转 + 等比缩放 + 平移"""
        # 50% 概率水平翻转
        if random.random() < 0.5:
            image_pil = F.hflip(image_pil)
            mask_pil = F.hflip(mask_pil)
        # 50% 概率垂直翻转
        if random.random() < 0.5:
            image_pil = F.vflip(image_pil)
            mask_pil = F.vflip(mask_pil)

        # 旋转（±25°）+ 轻度平移（±10%）+ 等比缩放（0.9~1.1）
        angle = random.uniform(-25, 25)
        W, H = image_pil.size
        max_dx, max_dy = 0.10 * W, 0.10 * H
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)
        scale = random.uniform(0.9, 1.1)

        image_pil = F.affine(
            image_pil, angle=angle, translate=(tx, ty), scale=scale, shear=(0.0, 0.0),
            interpolation=InterpolationMode.BILINEAR, fill=0
        )
        mask_pil = F.affine(
            mask_pil, angle=angle, translate=(tx, ty), scale=scale, shear=(0.0, 0.0),
            interpolation=InterpolationMode.NEAREST, fill=0
        )
        return image_pil, mask_pil

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """Return paths and metadata for a sample."""
        if idx in self.df.index:
            row = self.df.loc[idx]
        elif 0 <= idx < len(self.df):
            row = self.df.iloc[idx]
        else:
            raise IndexError(f"Index {idx} invalid (0-{len(self.df)-1} or a valid df.index label).")
        return {
            "image_path": os.path.join(self.image_dir, row["new_image_name"]),
            "mask_path": os.path.join(self.mask_dir, row["new_mask_name"]),
            "set": row.get("set", ""),
            "swede_category": row.get("swede_category", "")
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        """Compute summary statistics for the dataset."""
        stats = {
            'total_samples': len(self.df),
            'image_dir': self.image_dir,
            'mask_dir': self.mask_dir,
            'target_size': self.target_size
        }
        if 'set' in self.df.columns:
            stats['set_distribution'] = self.df['set'].value_counts().to_dict()
        if 'swede_category' in self.df.columns:
            stats['swede_category_distribution'] = self.df['swede_category'].value_counts().to_dict()
        return stats


def create_datasets(csv_path: str,
                    base_dir: str,
                    normalize: bool = True,
                    target_size: Tuple[int, int] = (512, 512),
                    binary_mask: bool = False,
                    active_round: Optional[str] = None,
                    use_labeled_only: bool = False,
                    enable_roi: bool = False,
                    pad_ratio: float = 0.10) -> Dict[str, CervixDataset]:
    """
    Create train/val/test datasets.
    """
    datasets = {}
    set_seed(42)

    for split in ['train', 'val', 'test']:
        image_dir = os.path.join(base_dir, split, "images")
        mask_dir = os.path.join(base_dir, split, "masks")

        datasets[split] = CervixDataset(
            csv_path=csv_path,
            image_dir=image_dir,
            mask_dir=mask_dir,
            target_size=target_size,
            normalize=normalize,
            augment=(split == 'train'),
            set_filter=split,
            binary_mask=True,
            use_labeled_only=(split == 'train' and use_labeled_only),
            enable_roi=enable_roi if split == 'train' else False,
            pad_ratio=pad_ratio
        )

    return datasets


class CervixUnlabeledImages(Dataset):
    def __init__(self,
                 csv_path: Union[str, pd.DataFrame],
                 image_dir: str,
                 target_size: Tuple[int, int] = (512, 512),
                 normalize: bool = True,
                 set_filter: str = 'train',
                 return_idx: bool = False):
        self.image_dir = image_dir
        self.target_size = target_size
        self.transform = get_transform(target_size, normalize, augment=False, is_mask=False)
        self.return_idx = return_idx

        df = pd.read_csv(csv_path) if isinstance(csv_path, str) else csv_path.copy()
        if set_filter:
            df = df[df['set'] == set_filter]
        self.df = df[df.get('labeled', 1) == 0]

        if 'new_image_name' not in self.df.columns:
            raise ValueError("CSV 缺少 new_image_name 列")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['new_image_name'])
        image = Image.open(img_path).convert("RGB")
        image = image.resize(self.target_size, resample=Image.BILINEAR)
        image = self.transform(image)
        if self.return_idx:
            return image, int(self.df.index[idx])
        else:
            return image
