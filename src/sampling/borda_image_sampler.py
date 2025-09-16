import time
import numpy as np
import torch
import torch.nn.functional as F
import os
import cv2
import pandas as pd

from utils.uncertainty_utils import compute_entropy, enable_dropout
from utils.texture_utils import compute_texture_score
from utils.sampling_representative_utils import get_features
from utils.borda_count_utils import borda_count_ranking
from utils.sampling_logger_utils import SamplingLogger
from sampling.base_sampler import BaseSampler


class BordaSampler(BaseSampler):
    def __init__(self, trainer, **kwargs):
        super().__init__(
            budget=kwargs.get('budget'),
            device=kwargs.get('device', 'cuda'),
            log_dir=kwargs.get('log_dir', './sampling_logs')
        )
        self.trainer = trainer
        self.model = self.trainer.model.to(self.device)

        self.metrics = kwargs.get("metrics", ["uncertainty", "representativeness", "texture"])
        self.normalize = kwargs.get("normalize", True)
        self.fusion_weight = kwargs.get("fusion_weight", None)

        self.num_inferences = kwargs.get("num_inferences", 10)
        self.texture_image_dir = kwargs.get("texture_image_dir")
        self.mask_dir = kwargs.get("mask_dir")

        self.sampling_logger = SamplingLogger(self.log_dir, strategy_name="BordaImage")

    def select(self, unlabeled_dataloader):
        self.model.eval()
        self.model.apply(enable_dropout)
        tic = time.time()

        # --------------------------------------------------
        # 1) 收集  id  &  entropy maps
        # --------------------------------------------------
        all_ids, entropy_map_list = [], []

        with torch.no_grad():
            for batch in unlabeled_dataloader:
                # -------- unpack --------
                if isinstance(batch, dict):
                    x, gid = batch["data"].to(self.device), batch["idx"]
                elif isinstance(batch, (list, tuple)):
                    x, _, gid = batch
                    x = x.to(self.device)
                else:
                    raise TypeError(f"Unexpected batch type {type(batch)}")

                gid = (gid.cpu().tolist()
                       if isinstance(gid, torch.Tensor)
                       else [int(gid)] if isinstance(gid, int)
                else [int(k) for k in gid])

                # -------- entropy --------
                probs_stack = torch.stack(
                    [F.softmax(self.model(x)[0], dim=1)
                     for _ in range(self.num_inferences)],
                    dim=0)
                ent = compute_entropy(probs_stack.mean(0)).cpu().numpy()

                all_ids.extend(gid)
                entropy_map_list.extend(list(ent))

        all_entropy_scores = [np.mean(m) for m in entropy_map_list]

        # --------------------------------------------------
        # 2) representativeness
        # --------------------------------------------------
        idx_feat, feat_mat = get_features(self.model, unlabeled_dataloader, self.device)
        rep_dict = dict(zip(idx_feat,
                            np.linalg.norm(feat_mat - feat_mat.mean(0), axis=1)))

        # --------------------------------------------------
        # 3) texture
        # --------------------------------------------------
        texture_enabled = (
                "texture" in self.metrics
                and self.texture_image_dir
                and self.mask_dir
        )
        if not texture_enabled and "texture" in self.metrics:
            self.metrics.remove("texture")
            self.sampling_logger.log_message("⚠️ texture metric disabled (dirs missing)")

        all_tex_scores = []

        if texture_enabled:
            base_ds = getattr(self.trainer, "train_dataset",
                              unlabeled_dataloader.dataset)
            while isinstance(base_ds, torch.utils.data.Subset):
                base_ds = base_ds.dataset
            train_df = base_ds.df

            img_root, msk_root = self.texture_image_dir, self.mask_dir

            for gid in all_ids:
                row = train_df.loc[gid]
                img_path = os.path.join(img_root, row["new_image_name"])
                msk_path = os.path.join(msk_root, row["new_mask_name"])

                if os.path.exists(img_path) and os.path.exists(msk_path):
                    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
                    tex = compute_texture_score(img, msk)["score"]
                else:
                    self.sampling_logger.log_message(
                        f"⚠️ missing img/mask for gid {gid}, texture=0")
                    tex = 0.0
                all_tex_scores.append(tex)
        else:
            all_tex_scores = [0.0] * len(all_ids)

        # ---------- 统一唯一 ID ----------
        unique_ids = sorted(set(all_ids))  # 随意排序即可

        # ——— metric 1: uncertainty ———
        unc_dict = {i: s for i, s in zip(all_ids, all_entropy_scores)}
        unc_scores = [unc_dict.get(i, 0.0) for i in unique_ids]

        # ——— metric 2: representativeness ———
        rep_scores = [rep_dict.get(i, 0.0) for i in unique_ids]

        # ——— metric 3: texture (如果启用) ———
        if texture_enabled:
            tex_dict = dict(zip(all_ids, all_tex_scores))
            tex_scores = [tex_dict.get(i, 0.0) for i in unique_ids]
        else:
            tex_scores = [0.0] * len(unique_ids)

        # -------- 保存原始分数（完全对齐） --------
        self.sampling_logger.save_scores(unique_ids, unc_scores, prefix="entropy")
        self.sampling_logger.save_scores(unique_ids, rep_scores, prefix="representativeness")
        if texture_enabled:
            self.sampling_logger.save_scores(unique_ids, tex_scores, prefix="texture")

        # -------- 构造 score_dict --------
        score_dict = {
            "uncertainty": dict(zip(unique_ids, unc_scores)),
            "representativeness": dict(zip(unique_ids, rep_scores)),
        }
        if texture_enabled:
            score_dict["texture"] = dict(zip(unique_ids, tex_scores))

        # -------- Borda 排序 --------
        selected_ids = borda_count_ranking(score_dict,
                                           top_k=self.budget,
                                           ascending=False)

        # -------- 一致性断言 --------
        assert len(unique_ids) == len(unc_scores) == len(rep_scores) == len(tex_scores)
        for md in score_dict.values():
            assert set(md.keys()) == set(unique_ids)


        # --------------------------------------------------
        # 6) 日志 & 返回
        # --------------------------------------------------
        elapsed = time.time() - tic
        self.sampling_logger.log_message("Borda-image sampling completed")
        self.sampling_logger.save_indices(selected_ids)
        self.sampling_logger.save_metadata({
            "strategy": "BordaSampler",
            "budget": self.budget,
            "sampling_time_sec": elapsed,
            "num_samples": len(all_ids),
            "num_inferences": self.num_inferences
        })

        sel_unc_scores = [score_dict["uncertainty"][i] for i in selected_ids]

        return selected_ids, sel_unc_scores, elapsed


