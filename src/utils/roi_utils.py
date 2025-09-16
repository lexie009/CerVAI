import numpy as np
import cv2

def crop_by_mask(img, msk, pad_ratio: float = 0.1, out_size=(512, 512)):
    ys, xs = np.where(msk > 0)
    if len(xs) == 0:
        h, w = img.shape[:2]
        dh, dw = int(h * 0.10), int(w * 0.10)
        img_roi = img[dh:h-dh, dw:w-dw]
        msk_roi = msk[dh:h-dh, dw:w-dw]
    else:
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        pad_x = int((x2 - x1) * pad_ratio)
        pad_y = int((y2 - y1) * pad_ratio)
        x1, x2 = max(0, x1 - pad_x), min(img.shape[1], x2 + pad_x)
        y1, y2 = max(0, y1 - pad_y), min(img.shape[0], y2 + pad_y)
        img_roi = img[y1:y2, x1:x2]
        msk_roi = msk[y1:y2, x1:x2]

    img_roi = cv2.resize(img_roi, out_size, interpolation=cv2.INTER_LINEAR)
    msk_roi = cv2.resize(msk_roi, out_size, interpolation=cv2.INTER_NEAREST)
    return img_roi, msk_roi

# === 新增：仅用图像估计 ROI（给 val/test 用，避免掩码泄漏） ===
def _pad_bbox(y1, x1, y2, x2, H, W, pad_ratio=0.30, min_box_frac=0.60):
    h = y2 - y1 + 1
    w = x2 - x1 + 1
    # 先 pad
    pad_h = int(h * pad_ratio)
    pad_w = int(w * pad_ratio)
    ny1 = max(0, y1 - pad_h)
    nx1 = max(0, x1 - pad_w)
    ny2 = min(H - 1, y2 + pad_h)
    nx2 = min(W - 1, x2 + pad_w)

    # 再保证最小尺寸（避免几乎全前景）
    min_h = int(H * min_box_frac)
    min_w = int(W * min_box_frac)
    if (ny2 - ny1 + 1) < min_h:
        extra = (min_h - (ny2 - ny1 + 1)) // 2
        ny1 = max(0, ny1 - extra)
        ny2 = min(H - 1, ny2 + extra)
    if (nx2 - nx1 + 1) < min_w:
        extra = (min_w - (nx2 - nx1 + 1)) // 2
        nx1 = max(0, nx1 - extra)
        nx2 = min(W - 1, nx2 + extra)

    return ny1, nx1, ny2, nx2

def find_bbox_by_image(img_np: np.ndarray,
                       min_area: int = 3000,
                       pad_ratio: float = 0.30,
                       min_box_frac: float = 0.60):
    H, W = img_np.shape[:2]
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    Hch, Sch, Vch = cv2.split(hsv)

    # 粗筛：偏红/粉 + R 显著（阈值保守些）
    m1 = ((Hch <= 25) & (Sch >= 35) & (Vch >= 40)).astype(np.uint8)
    R, G, B = img_np[..., 0], img_np[..., 1], img_np[..., 2]
    m2 = ((R.astype(np.int16) > G.astype(np.int16) + 10) &
          (R.astype(np.int16) > B.astype(np.int16) + 10)).astype(np.uint8)
    m = cv2.bitwise_or(m1, m2)

    # 形态学净化
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return None

    # 最大连通域
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_id = np.argmax(areas) + 1
    if stats[max_id, cv2.CC_STAT_AREA] < min_area:
        return None

    x, y, w, h = stats[max_id, cv2.CC_STAT_LEFT], stats[max_id, cv2.CC_STAT_TOP], \
                 stats[max_id, cv2.CC_STAT_WIDTH], stats[max_id, cv2.CC_STAT_HEIGHT]
    y1, x1, y2, x2 = y, x, y + h - 1, x + w - 1
    return _pad_bbox(y1, x1, y2, x2, H, W, pad_ratio=pad_ratio, min_box_frac=min_box_frac)

def crop_by_image(img_np: np.ndarray,
                  mask_np: np.ndarray,
                  pad_ratio: float = 0.15,
                  fallback_ratio: float = 0.8,
                  out_size=(512, 512)):  # <- add this
    H, W = img_np.shape[:2]
    bbox = find_bbox_by_image(img_np, pad_ratio=pad_ratio)
    if bbox is None:
        side = int(min(H, W) * fallback_ratio)
        cy, cx = H // 2, W // 2
        y1 = max(0, cy - side // 2); x1 = max(0, cx - side // 2)
        y2 = min(H - 1, y1 + side - 1); x2 = min(W - 1, x1 + side - 1)
    else:
        y1, x1, y2, x2 = bbox

    img_crop  = img_np[y1:y2 + 1, x1:x2 + 1]
    mask_crop = mask_np[y1:y2 + 1, x1:x2 + 1]
    # use out_size for final resize
    img_crop  = cv2.resize(img_crop,  out_size, interpolation=cv2.INTER_LINEAR)
    mask_crop = cv2.resize(mask_crop, out_size, interpolation=cv2.INTER_NEAREST)
    return img_crop, mask_crop


# === 统一入口：根据 mode 路由 ===
def crop_by_auto(img_np, mask_np,
                 mode: str = "auto",  # "mask" / "image" / "auto"
                 use_mask_on_valtest: bool = False,
                 split: str = "train",
                 pad_ratio: float = 0.10,
                 out_size=(512, 512)):
    """
    auto: train 且 mask 有前景 -> 用 mask；否则用 image。
    """
    mode = (mode or "auto").lower()
    if mode == "mask":
        return crop_by_mask(img_np, mask_np, pad_ratio=pad_ratio, out_size=out_size)
    if mode == "image":
        return crop_by_image(img_np, mask_np, pad_ratio=pad_ratio, out_size=out_size)

    # auto
    if split == "train" or use_mask_on_valtest:
        if (mask_np > 0).any():
            return crop_by_mask(img_np, mask_np, pad_ratio=pad_ratio, out_size=out_size)
    return crop_by_image(img_np, mask_np, pad_ratio=pad_ratio, out_size=out_size)
