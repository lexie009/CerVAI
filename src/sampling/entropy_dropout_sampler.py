import time
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.uncertainty_utils import compute_entropy, enable_dropout
from utils.sampling_logger_utils import SamplingLogger
from sampling.base_sampler import BaseSampler


class EntropyDropoutSampler(BaseSampler):
    """
    Entropy-based active learning sampler using MC Dropout.
    Selects top-K uncertain images based on predictive entropy.
    """

    def __init__(self, trainer, **kwargs):
        super().__init__(
            budget=kwargs.get("budget"),
            device=kwargs.get("device", "cuda"),
            log_dir=kwargs.get("log_dir", "./sampling_logs")
        )
        self.trainer = trainer
        self.model = self.trainer.model.to(self.device)

        self.num_inferences = kwargs.get("num_inferences", 10)
        self.use_dropout = kwargs.get('use_dropout')

        self.sampling_logger = SamplingLogger(self.log_dir, strategy_name="entropy_mc")

    def select(self, unlabeled_dataloader):
        """
        MC-Dropout entropy sampling.

        Returns
        -------
        selected_indices : list[int]
            Global row-ids of chosen images
        """
        self.model.eval()
        self.model.apply(enable_dropout)  # turn on dropout layers
        tic = time.time()

        all_scores: list[float] = []
        all_indices: list[int] = []

        try:
            with torch.no_grad():
                for batch in unlabeled_dataloader:

                    # ---------- ① unpack batch ----------
                    if isinstance(batch, (list, tuple)):
                        x, _, gid = batch  # tuple path
                        idx = gid  # Tensor[int]
                    elif isinstance(batch, dict):
                        x = batch["data"]
                        idx = batch["idx"]
                    else:
                        raise TypeError(f"Unexpected batch type {type(batch)}")

                    # ---------- ② normalize idx --------
                    if isinstance(idx, torch.Tensor):
                        idx = idx.cpu().tolist()
                    elif isinstance(idx, int):
                        idx = [idx]
                    elif isinstance(idx, list):
                        idx = [int(i) for i in idx]
                    else:
                        raise TypeError(f"idx has unsupported type {type(idx)}")

                    # ---------- ③ MC-Dropout ----------
                    x = x.to(self.device)
                    mc_probs = []
                    for _ in range(self.num_inferences):
                        logits = self.model(x)  # model already returns logits
                        if isinstance(logits, tuple):  # 有些实现返回 (logits, aux)
                            logits = logits[0]
                        probs = F.softmax(logits, dim=1)
                        mc_probs.append(probs)

                    probs_stack = torch.stack(mc_probs, dim=0)  # (T,B,C,H,W)
                    probs_mean = probs_stack.mean(dim=0)  # (B,C,H,W)
                    entmap = compute_entropy(probs_mean)  # (B,H,W)
                    entvec = entmap.mean(dim=(1, 2))  # (B,)

                    # ---------- ④ collect ---------------
                    all_indices.extend(idx)
                    all_scores.extend(entvec.cpu().tolist())

            # ---------- ⑤ sanity -----------------------
            assert len(all_indices) == len(all_scores), \
                f"idx/score mismatch {len(all_indices)} vs {len(all_scores)}"

            scores_np = np.asarray(all_scores)
            indices_np = np.asarray(all_indices)

            topk = np.argsort(-scores_np)[: self.budget]
            sel_idx = indices_np[topk].tolist()
            sel_score = scores_np[topk].tolist()

            elapsed = time.time() - tic

            # ---------- ⑥ logging ----------------------
            self.sampling_logger.log_message(
                f"Entropy MC-Dropout sampling chose {len(sel_idx)} / {self.budget} "
                f"in {elapsed:.2f}s (pool={len(all_indices)})"
            )
            self.sampling_logger.save_indices(sel_idx)
            self.sampling_logger.save_scores(sel_idx, sel_score,prefix="entropy_mc_scores")
            self.sampling_logger.save_metadata({
                "strategy": "entropy_mc",
                "budget": self.budget,
                "num_samples": len(all_indices),
                "sampling_time_s": elapsed,
                "num_inferences": self.num_inferences
            })

            return sel_idx, sel_score, elapsed

        except Exception:
            self.sampling_logger.log_error("Entropy MC sampling failed", exc_info=True)
            raise
