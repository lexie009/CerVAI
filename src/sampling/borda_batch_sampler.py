import time
import numpy as np
import torch
import torch.nn.functional as F
import os
import cv2
import pandas as pd

from utils.uncertainty_utils import compute_entropy, enable_dropout
from utils.texture_utils import compute_texture_score
from utils.sampling_logger_utils import SamplingLogger
from utils.sampling_representative_utils import get_features
from utils.borda_count_utils import borda_count_ranking
from utils.stochastic_batch_utils import generate_random_groups, aggregate_group_uncertainty
from sampling.base_sampler import BaseSampler

class BordaBatchSampler(BaseSampler):
    def __init__(self, trainer, **kwargs):
        super().__init__(
            budget=kwargs.get('budget'),
            device=kwargs.get('device', 'cuda'),
            log_dir=kwargs.get('log_dir', './sampling_logs')
        )

        self.trainer = trainer
        self.model = trainer.model.to(self.device)

        self.num_inferences = kwargs.get('num_inferences',10)
        self.texture_image_dir = kwargs.get('texture_image_dir')
        self.mask_dir = kwargs.get('mask_dir')
        self.group_size = kwargs.get('group_size', 4)
        self.num_groups = kwargs.get('num_groups', 40)
        self.aggregation = kwargs.get('aggregation', "mean")
        self.use_dropout = kwargs.get('use_dropout', True)

        self.sampling_logger = SamplingLogger(self.log_dir, strategy_name="BordaBatchSampler")

    def select(self, unlabeled_dataloader):
        t0 = time.time()
        self.model.eval();
        self.model.apply(enable_dropout)

        all_ids, entropy_maps = [], []

        # ---------------- 1. 逐图像不确定性 (MC-Dropout Entropy) ----------------
        with torch.no_grad():
            for batch in unlabeled_dataloader:
                # unpack batch
                if isinstance(batch, dict):
                    x, gid = batch["data"].to(self.device), batch["idx"]
                elif isinstance(batch, (list, tuple)):
                    x, _, gid = batch;
                    x = x.to(self.device)
                else:
                    raise TypeError(f"Unsupported batch type {type(batch)}")

                gid_list = (gid.cpu().tolist() if isinstance(gid, torch.Tensor)
                            else [int(gid)] if isinstance(gid, int)
                else [int(k) for k in gid])

                probs_stack = torch.stack([
                    F.softmax((self.model(x)[0] if isinstance(self.model(x), (list, tuple))
                               else self.model(x)), dim=1)
                    for _ in range(self.num_inferences)
                ], dim=0)  # (N,B,C,H,W)

                ent = compute_entropy(probs_stack.mean(0)).cpu().numpy()  # (B,H,W)

                all_ids.extend(gid_list)
                entropy_maps.extend(list(ent))

        all_unc_scores = [m.mean() for m in entropy_maps]  # image-level entropy

        # ---------------- 2. 代表性 (特征距离) ----------------
        idx_feat, feat_mat = get_features(self.model, unlabeled_dataloader, self.device)
        rep_raw = dict(zip(idx_feat,
                           np.linalg.norm(feat_mat - feat_mat.mean(0), axis=1)))

        # ---------------- 3. 纹理 (如启用) ----------------
        tex_raw = {}
        if self.texture_image_dir and self.mask_dir:
            base_ds = getattr(self.trainer, "train_dataset", unlabeled_dataloader.dataset)
            while isinstance(base_ds, torch.utils.data.Subset):
                base_ds = base_ds.dataset
            df = base_ds.df

            for gid in all_ids:
                row = df.loc[gid]
                ip = os.path.join(self.texture_image_dir, row["new_image_name"])
                mp = os.path.join(self.mask_dir, row["new_mask_name"])
                if os.path.exists(ip) and os.path.exists(mp):
                    img = cv2.cvtColor(cv2.imread(ip), cv2.COLOR_BGR2RGB)
                    msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    tex_raw[gid] = compute_texture_score(img, msk)["score"]
                else:
                    tex_raw[gid] = 0.0

        # ---------- 对齐键集合 ----------
        unique_ids = sorted(set(all_ids))
        unc_scores = [dict(zip(all_ids, all_unc_scores)).get(i, 0.0) for i in unique_ids]
        rep_scores = [rep_raw.get(i, 0.0) for i in unique_ids]
        tex_scores = [tex_raw.get(i, 0.0) for i in unique_ids]

        # ---------- 保存 per-image 分数 ----------
        self.sampling_logger.save_scores(unique_ids, unc_scores, prefix="entropy")
        self.sampling_logger.save_scores(unique_ids, rep_scores, prefix="representativeness")
        if tex_raw:
            self.sampling_logger.save_scores(unique_ids, tex_scores, prefix="texture")

        # ---------------- 4. 生成随机 group & 聚合 ----------------
        groups = generate_random_groups(unique_ids,
                                        num_groups=self.num_groups,
                                        group_size=self.group_size)  # list[list[id]]

        def agg(metric_scores):
            return aggregate_group_uncertainty(
                groups, dict(zip(unique_ids, metric_scores)),
                aggregation=self.aggregation)

        entropy_group_scores = agg(unc_scores)
        rep_group_scores = agg(rep_scores)
        tex_group_scores = agg(tex_scores)

        score_dict_group = {
            "uncertainty": dict(zip(range(len(groups)), entropy_group_scores)),
            "representativeness": dict(zip(range(len(groups)), rep_group_scores)),
        }
        if tex_raw:
            score_dict_group["texture"] = dict(zip(range(len(groups)), tex_group_scores))

        # ---------------- 5. Borda 在 group 级别排序 ----------------
        top_g_idx = borda_count_ranking(score_dict_group,
                                        top_k=self.budget // self.group_size,
                                        ascending=False)
        selected_ids = [img for gi in top_g_idx for img in groups[gi]][: self.budget]
        sel_unc = [dict(zip(unique_ids, unc_scores))[i] for i in selected_ids]

        # ---------------- 6. 日志 & 返回 ----------------
        elapsed = time.time() - t0
        self.sampling_logger.save_indices(selected_ids)
        self.sampling_logger.save_metadata({
            "strategy": "BordaBatchSampler",
            "budget": self.budget,
            "group_size": self.group_size,
            "num_groups": self.num_groups,
            "aggregation": self.aggregation,
            "sampling_time_sec": elapsed,
            "num_samples": len(unique_ids),
            "num_inferences": self.num_inferences
        })

        return selected_ids, sel_unc, elapsed

    def save_all_group_scores(group_ids, score_dict, aggregated_scores_dict, save_dir, prefix="group_scores"):
        """
        save the scores for each group + scores after aggregation。

        Args:
            group_ids: List[List[int]]
            score_dict: dict {metric_name: {sample_id: score, ...}}
            aggregated_scores_dict: dict {metric_name: List[float]}，the score after aggregation
            save_dir: save directory
            prefix: the prefix of the saved file
        """
        os.makedirs(save_dir, exist_ok=True)
        all_rows = []
        for group_idx, group in enumerate(group_ids):
            for sample_id in group:
                row = {"group": group_idx, "sample_id": sample_id}
                for metric in score_dict:
                    row[metric] = score_dict[metric].get(sample_id, np.nan)
                all_rows.append(row)

        # Group-level scores
        for metric, values in aggregated_scores_dict.items():
            for group_idx, score in enumerate(values):
                all_rows.append({"group": group_idx, "sample_id": "AGGREGATED", metric: score})

        df = pd.DataFrame(all_rows)
        csv_path = os.path.join(save_dir, f"{prefix}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved group scores to {csv_path}")
        return csv_path
