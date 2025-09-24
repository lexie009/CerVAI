import time
import numpy as np
import torch
import torch.nn.functional as F
import logging

from utils.evaluate_utils import get_foreground_prob

logger = logging.getLogger(__name__)

from utils.uncertainty_utils import compute_entropy
from utils.sampling_logger_utils import SamplingLogger
from sampling.base_sampler import BaseSampler

class EntropySampler(BaseSampler):
    """
    Entropy-based active learning sampler for semantic segmentation.
    Selects top-K uncertain images based on mean pixel-wise entropy.
    """

    def __init__(self, trainer, **kwargs):
        super().__init__(
            budget=kwargs.get("budget"),
            device=kwargs.get("device", "cuda"),
            log_dir=kwargs.get("log_dir", "./sampling_logs")
        )

        self.trainer = trainer
        self.model = self.trainer.model.to(self.device)
        self.logger = SamplingLogger(self.log_dir, strategy_name="Entropy")

    def select(self, unlabeled_dataloader):
        """
        Computes mean entropy per image and selects top-K most uncertain samples.

        Returns
        -------
        selected_indices : list[int]
            Global CSV row-ids of chosen images
        mean_uncertainty : list[float]
            Corresponding mean-entropy scores
        sampling_time    : float
            Time elapsed (s)
        """
        self.model.eval()
        tic = time.time()

        all_scores: list[float] = []
        all_indices: list[int] = []

        try:
            with torch.no_grad():
                for batch in unlabeled_dataloader:

                    # -------- ① unpack batch & get idx -----------
                    if isinstance(batch, (list, tuple)):
                        x, _, gid = batch  # (B,C,H,W) ...
                        idx = gid  # Tensor[int] (global row-ids)
                    elif isinstance(batch, dict):
                        x = batch["data"]
                        idx = batch.get("idx")  # Tensor | list | int | None
                        if idx is None:
                            raise ValueError("Dataset must return 'idx' for entropy sampler")
                    else:
                        raise TypeError(f"Unexpected batch type {type(batch)}")

                    # -------- ② normalize idx → list[int] -------
                    if isinstance(idx, torch.Tensor):
                        idx = idx.cpu().tolist()
                    elif isinstance(idx, int):
                        idx = [idx]
                    elif isinstance(idx, list):
                        idx = [int(i) for i in idx]  # 保证纯 int
                    else:
                        raise TypeError(f"idx has unsupported type {type(idx)}")

                    # -------- ③ forward pass & entropy ----------
                    x = x.to(self.device)
                    logits = self.model(x)  # (B,C,H,W)
                    # probs = F.softmax(logits, dim=1)
                    probs = get_foreground_prob(logits)
                    entmap = compute_entropy(probs)  # (B,H,W)
                    entvec = entmap.mean(dim=(1, 2))  # (B,)

                    # -------- ④ collect -------------------------
                    all_indices.extend(idx)
                    all_scores.extend(entvec.cpu().tolist())

            # -------- ⑤ sanity check ---------------------------
            assert len(all_indices) == len(all_scores), \
                f"idx/score length mismatch: {len(all_indices)} vs {len(all_scores)}"

            # -------- ⑥ top-K selection ------------------------
            all_scores_np = np.asarray(all_scores)
            all_indices_np = np.asarray(all_indices)

            topk = np.argsort(-all_scores_np)[: self.budget]
            sel_idx = all_indices_np[topk].tolist()
            sel_score = all_scores_np[topk].tolist()

            # 去重（理论上不会重复，但加保险）
            sel_unique, sel_score_unique = [], []
            seen = set()
            for i, s in zip(sel_idx, sel_score):
                if i not in seen:
                    sel_unique.append(i)
                    sel_score_unique.append(s)
                    seen.add(i)

            elapsed = time.time() - tic

            # -------- ⑦ logging -------------------------------
            self.logger.log_message(
                f"Entropy sampling chose {len(sel_unique)}/{self.budget} imgs "
                f"in {elapsed:.2f}s (pool={len(all_indices)})"
            )
            self.logger.save_indices(sel_unique)
            self.logger.save_scores(sel_unique, sel_score_unique, prefix="entropy_scores")
            self.logger.save_metadata({
                "strategy": "Entropy",
                "budget": self.budget,
                "num_unlabeled": len(all_indices),
                "sampling_time_s": elapsed
            })

            # 如果仍少于 budget，给个警告
            if len(sel_unique) < self.budget:
                self.logger.log_message(
                    f"⚠️  Budget={self.budget},但仅选到{len(sel_unique)}条(可能分数重复)"
                )

            return sel_unique, sel_score_unique, elapsed

        except Exception as e:
            self.logger.log_error("Entropy sampling failed", exc_info=True)
            raise
