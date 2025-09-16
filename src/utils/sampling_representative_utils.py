from sklearn.cluster import KMeans
import numpy as np
import torch
import torch.nn.functional as F

def run_kmeans(features, budget):
    """
    Cluster feature vectors into `budget` clusters and return the index of the closest sample to each centroid.
    """
    kmeans = KMeans(n_clusters=budget, random_state=0).fit(features)
    centers = kmeans.cluster_centers_

    selected_indices = []
    for center in centers:
        # 找到距离 center 最近的样本索引
        dists = np.linalg.norm(features - center, axis=1)
        selected_idx = np.argmin(dists)
        selected_indices.append(selected_idx)

    return selected_indices

def _forward_backbone(model, x):
    """
    Try to obtain a single feature map (B,C,H,W) from different model variants.
    """
    if hasattr(model, "backbone"):                       # e.g. torchvision models
        feats = model.backbone(x)                        # (B,C,H,W)
        if isinstance(feats, dict):                      # SegFormer 等
            feats = feats["out"]
        return feats

    if hasattr(model, "encoder"):                        # smp.Unet 等
        feats_list = model.encoder(x)                    # list of stage outputs
        return feats_list[-1]                            # 取最后一级 (B,C,H,W)

    if hasattr(model, "base_model") and hasattr(model.base_model, "encoder"):
        feats_list = model.base_model.encoder(x)
        return feats_list[-1]

    raise AttributeError(
        "No backbone/encoder found – please adjust _forward_backbone()"
    )

def get_features(model, dataloader, device):
    model.eval()
    feats_all, idx_all = [], []

    with torch.no_grad():
        for batch_id, batch in enumerate(dataloader):

            # -------- ① unpack ----------
            if isinstance(batch, (list, tuple)):
                images, _, gid = batch            # (B,C,H,W)
                idx_batch = gid.cpu().tolist()    # 用真实 global_id
            elif isinstance(batch, dict):
                images = batch["data"]
                idx_batch = batch.get("idx")
                if idx_batch is None:
                    # fallback: fabricate running indices
                    idx_batch = [batch_id * len(images) + i
                                 for i in range(len(images))]
                elif isinstance(idx_batch, torch.Tensor):
                    idx_batch = idx_batch.cpu().tolist()
                else:
                    idx_batch = [int(i) for i in idx_batch]
            else:
                raise TypeError(f"Unsupported batch type {type(batch)}")

            # -------- ② forward ----------
            images = images.to(device)
            feats  = _forward_backbone(model, images)          # (B,C,H,W)
            feats  = F.adaptive_avg_pool2d(feats, 1).flatten(1)  # (B,C)

            feats_all.append(feats.cpu())      # ← 只追加一次
            idx_all.extend(idx_batch)          # ← 与 feats 对齐

    # -------- ③ sanity / return ----------
    if not feats_all:                          # dataloader 为空
        return [], np.empty((0, 1), dtype=np.float32)

    feats_np = torch.cat(feats_all, dim=0).numpy()
    assert len(idx_all) == feats_np.shape[0], "idx/features length mismatch"
    return idx_all, feats_np
