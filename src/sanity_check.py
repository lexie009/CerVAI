# sanity_check.py
import os, json, logging, torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import CervixDataset
from models.unet import UNet  # 你的模型构建函数
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def scan_dataset(csv_path, image_dir, mask_dir, split_name, max_samples=200):
    ds = CervixDataset(
        csv_path, image_dir, mask_dir,
        normalize=True, target_size=(512,512),
        binary_mask=True, enable_roi=False, use_mask_on_valtest=False,
        set_filter=split_name         # ★ 传入 split
    )
    zero, one, ratios = 0, 0, []
    for i in range(min(len(ds), max_samples)):
        img, msk, meta = ds[i]
        p = float((msk > 0).float().mean().item())   # msk 已经是 long 二值，[H,W]
        ratios.append(p)
        if p < 1e-6: zero += 1
        if p > 1 - 1e-6: one += 1
    if ratios:
        q = np.quantile(ratios, [0, .25, .5, .75, 1.0]).round(4)
        logging.info(f"[SCAN] split={split_name}  n={len(ds)}  sampled={len(ratios)}  "
                     f"all_zero={zero}  all_one={one}  pos_ratio_q={q.tolist()}")
    return ds

@torch.no_grad()
def probe_model(model, ds, device, thr=0.8, bs=2):
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    for i,(x,y,_) in enumerate(loader):
        x,y = x.to(device), y.to(device)
        out = model(x)
        if out.shape[1]==2:
            prob = torch.softmax(out, dim=1)
            fg1, fg0 = prob[:,1], prob[:,0]
        else:
            fg1 = torch.sigmoid(out[:,0]); fg0 = 1-fg1
        def dice_of(p,y):
            p = (p>thr).float(); eps=1e-6
            inter = (p*y).sum((1,2,3))
            return ((2*inter)/(p.sum((1,2,3))+y.sum((1,2,3))+eps)).mean().item()
        d1, d0 = dice_of(fg1,y), dice_of(fg0,y)
        logging.info(f"[FWD-PROBE] batch={i} fg=ch1 mean={fg1.mean():.4f} std={fg1.std():.4f} "
                     f"Dice(ch1)={d1:.4f}  Dice(ch0)={d0:.4f}  "
                     f"p_fg>0.5={float((fg1>0.5).float().mean()):.3f} p_fg>0.8={float((fg1>0.8).float().mean()):.3f}")
        break

def main():
    # 用你实际的路径替换 ↓↓↓
    base = "/Users/daidai/Documents/pythonProject_summer/CerVAI/dataset/dataset_split_final"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) 扫描数据
    tr = scan_dataset(os.path.join(base,"train.csv"),
                      os.path.join(base,"train","images"),
                      os.path.join(base,"train","masks"), "train")
    va = scan_dataset(os.path.join(base,"val.csv"),
                      os.path.join(base,"val","images"),
                      os.path.join(base,"val","masks"), "val")
    te = scan_dataset(os.path.join(base,"test.csv"),
                      os.path.join(base,"test","images"),
                      os.path.join(base,"test","masks"), "test")

    # 2) 构建模型并载入权重（可选）
    model = UNet(num_classes=2)  # 与你配置保持一致
    # model.load_state_dict(torch.load(".../best_model.pth")["state_dict"], strict=False)
    model.to(device).eval()

    # 3) 前向探测（优先在 val 上）
    probe_model(model, va, device, thr=0.8)

if __name__ == "__main__":
    main()
