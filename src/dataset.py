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
    
    Args:
        target_size: Target size for resizing
        normalize: Whether to apply normalization
        augment: Whether to apply data augmentation (only for training)
        is_mask: Whether this transform is for a mask
        
    Returns:
        Composed transform pipeline
    """
    transform_list = []
    
    # Basic transforms
    transform_list.append(T.ToTensor())
    
    if normalize and not is_mask:
        # Only normalize images, not masks
        transform_list.append(
            T.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        )
    
    return T.Compose(transform_list)


class CervixDataset(Dataset):
    """
    PyTorch Dataset for loading cervix image-mask pairs.
    Designed for semantic segmentation with optional active learning support.

    Features:
        - Supports binary masks for 2-class segmentation
        - Optional augmentation and normalization
        - Filters by dataset split (train/val/test)
        - Filters by active learning round (e.g., round0 = True)
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
        """
        Initialize the dataset with CSV metadata and image/mask directories.

        Args:
            csv_path: Path to metadata CSV or preloaded DataFrame
            image_dir: Directory containing input images
            mask_dir: Directory containing corresponding masks
            target_size: Final (H, W) size for resizing
            normalize: Whether to apply ImageNet normalization to images
            augment: Whether to use data augmentation
            set_filter: Dataset split filter ('train', 'val', 'test')
            binary_mask: Whether to binarize masks to 0/1 values
            round_column_filter: Active learning round filter (e.g., 'round0')
            use_labeled_only: Whether to filter for labeled samples only
        """
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

        self.transform = get_transform(target_size, normalize, augment=False, is_mask=False)
        self.mask_transform = get_transform(target_size, normalize=False, augment=False, is_mask=True)

        # Load metadata from CSV or DataFrame
        self.df = pd.read_csv(csv_path) if isinstance(csv_path, str) else csv_path.copy()

        if use_labeled_only:
            self.df = self.df[self.df["labeled"] == 1]

        # Apply dataset split filtering (e.g., only use 'train' set)
        if set_filter:
            self.df = self.df[self.df['set'] == set_filter]

        # Filter by active learning round flag if provided
        if round_column_filter and round_column_filter in self.df.columns:
            self.df = self.df[self.df[round_column_filter] == True]

        # Check that required columns exist
        required_cols = ['new_image_name', 'new_mask_name']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Load and return a preprocessed (image, mask) pair.

        Args:
            idx: Sample index

        Returns:
            image: Tensor of shape [C, H, W]
            mask: Tensor of shape [1, H, W] with optional binarization
        """
        row = self.df.iloc[idx]
        global_id = row.name
        img_path = os.path.join(self.image_dir, row['new_image_name'])
        mask_path = os.path.join(self.mask_dir, row['new_mask_name'])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        logging.info(f"Opened mask: {mask_path}, original size: {mask.size}")


        if self.enable_roi:
            # 若掩码全 0，crop_by_mask 会直接 return 原图
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
            logging.info(f"Resized mask size (PIL): {mask.size}")


        # [PROBE-1] ensure mask ∈ {0,1} (or {0,255}); if not, warn & binarize (>0)
        m = np.array(mask, dtype=np.uint8)
        u = np.unique(m)

        if not set(u.tolist()) <= {0, 1, 255}:
            logging.warning(f"[DATA] Non-binary mask values {u[:6]} ... binarizing (>0) @ {mask_path}")
            m = (m > 0).astype(np.uint8) * 255  # → {0,255}
            mask = Image.fromarray(m, mode="L")
        elif m.max() == 0:
            logging.info(f"[DATA] All-zero mask @ {mask_path}")

        mask_tensor_before = T.ToTensor()(mask)  # don't use self.mask_transform here yet
        logging.info(f"Tensor shape before squeeze: {mask_tensor_before.shape}")

        # If you want to catch mismatch early
        if mask_tensor_before.shape[1:] != self.target_size:
            print(f"[ERROR] ❌ Mask tensor shape mismatch: {mask_tensor_before.shape} from {mask_path}")

        if self.augment:
            # 几何增强（同步）
            image, mask = self._sync_geom_aug(image, mask)

            # 可选：只对图像做“光照/颜色”增强（不影响 mask）
            if random.random() < 0.5:
                cj = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                image = cj(image)

        image = self.transform(image)
        mask = self.mask_transform(mask)

        if self.binary_mask:
            # Ensure binary mask (0 or 1), required by CrossEntropyLoss
            mask = (mask > 0.5).long().squeeze(0)  # Tensor [H, W] with values 0 or 1
        else:
            mask = mask.long().squeeze(0)  # just in case of multi-class mask

        assert mask.dtype == torch.long and mask.ndim == 2 and mask.max() <= 1, f"Invalid mask: {mask.unique()}"

        return image, mask, global_id

    def _sync_geom_aug(self, image_pil, mask_pil):
        """ data augumentation on both image and mask return PIL.Image"""
        # 1) 水平翻转
        if random.random() < 0.5:
            image_pil = F.hflip(image_pil)
            mask_pil = F.hflip(mask_pil)

        # 2) 随机仿射（含旋转+平移；不缩放、不剪切）
        angle = random.uniform(-10, 10)  # 旋转角度
        W, H = image_pil.size
        max_dx, max_dy = 0.10 * W, 0.10 * H
        tx = random.uniform(-max_dx, max_dx)
        ty = random.uniform(-max_dy, max_dy)

        image_pil = F.affine(
            image_pil, angle=angle, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0),
            interpolation=InterpolationMode.BILINEAR, fill=0
        )
        mask_pil = F.affine(
            mask_pil, angle=angle, translate=(tx, ty), scale=1.0, shear=(0.0, 0.0),
            interpolation=InterpolationMode.NEAREST, fill=0
        )

        return image_pil, mask_pil

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        """
        Return paths and metadata for a sample.

        Args
        ----
        idx : int
            Either the *global CSV index* or the *positional index* inside this
            filtered DataFrame.
        """
        if idx in self.df.index:
            row = self.df.loc[idx]
        elif 0 <= idx < len(self.df):
            row = self.df.iloc[idx]
        else:
            raise IndexError( f"Index {idx} neither a label in df.index",
                f"nor a valid positional idx (0-{len(self.df) - 1})"
                )
        return {
            "image_path": os.path.join(self.image_dir, row["new_image_name"]),
            "mask_path": os.path.join(self.mask_dir, row["new_mask_name"]),
            "set": row.get("set", ""),
            "swede_category": row.get("swede_category", "")
        }

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Compute summary statistics for the dataset.

        Returns:
            Dictionary with sample count and optional distribution info
        """
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

    Args:
        csv_path: Path to the metadata CSV
        base_dir: Base directory where split folders (train/val/test) reside
        normalize: Whether to normalize images using ImageNet stats
        target_size: Final size for image/mask
        binary_mask: Whether to binarize the masks
        active_round: Legacy parameter, kept for compatibility
        use_labeled_only: If True, only use labeled samples for training set

    Returns:
        Dictionary with datasets {'train': ..., 'val': ..., 'test': ...}
    """
    datasets = {}
    set_seed(42)  # Ensure reproducibility

    # Standard 3-way split (train/val/test)
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
            enable_roi=enable_roi if split == 'train' else False,  # 只给 train 开
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
                 return_idx: bool = False
                 ):
        self.image_dir = image_dir
        self.target_size = target_size
        self.transform = get_transform(target_size, normalize, augment=False, is_mask=False)
        self.return_idx = return_idx

        df = pd.read_csv(csv_path) if isinstance(csv_path, str) else csv_path.copy()
        if set_filter:
            df = df[df['set'] == set_filter]
        # 只取未标注
        self.df = df[df['labeled'] == 0]

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
        # 只返回图像即可（无 mask）
        if self.return_idx:
            # 给“采样器”用：不要返回 None！最多返回 (image, idx) 或 dict
            return image, int(self.df.index[idx])
        else:
            # 给“半监督训练”用：只要图像
            return image

