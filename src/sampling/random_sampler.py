# sampling/random_sampler.py
import time, random, numpy as np
from sampling.base_sampler import BaseSampler
from utils.sampling_logger_utils import SamplingLogger


class RandomSampler(BaseSampler):
    """
    最简单的随机采样：
        - 从未标注索引列表 (unlabeled_indices) 随机抽取 budget 个
        - 与 DataLoader / batch 结构彻底解耦
    """

    def __init__(self, trainer=None, **kwargs):
        """
        kwargs 中必须包含：
            • budget              : 每轮要选多少张
            • unlabeled_indices   : 未标注样本在 “主 CSV” 的行索引列表
        可选：
            • device / log_dir
        """
        super().__init__(
            budget=kwargs.get("budget"),
            trainer=trainer,
            device=kwargs.get("device", "cpu"),
            log_dir=kwargs.get("log_dir", "./sampling_logs"),
        )
        self.trainer            = trainer
        self.unlabeled_indices  = kwargs["unlabeled_indices"]       # ← 关键！
        self.sampling_logger             = SamplingLogger(self.log_dir, strategy_name="Random")

    # ---------------------------------------------------------
    # Core API  ------------------------------------------------
    # ---------------------------------------------------------
    def select(self, unlabeled_dataloader):
        """
        参数 `unlabeled_dataloader` 仍然保留，但这里需要用它来做 local -> global 映射。
        """
        tic = time.time()

        subset = unlabeled_dataloader.dataset
        base_dataset = subset.dataset  # full_train_ds
        local_pool = list(subset.indices)  # local index in base_dataset

        n_pool = len(local_pool)

        if self.budget is None:
            raise ValueError("RandomSampler expects a non-None budget.")
        if self.budget > n_pool:
            raise ValueError(f"Budget {self.budget} > unlabeled pool size {n_pool}")

        # 先从 local indices 里随机抽
        selected_local = random.sample(local_pool, self.budget)

        # 再映射成 global CSV index
        selected_indices = [int(base_dataset.df.iloc[i].name) for i in selected_local]

        elapsed = time.time() - tic

        self.sampling_logger.log_message(
            f"Randomly selected {len(selected_indices)} / {n_pool} samples "
            f"in {elapsed:.2f} s (budget={self.budget})"
        )
        self.sampling_logger.save_indices(selected_indices)
        self.sampling_logger.save_metadata(
            {
                "strategy": "Random",
                "budget": self.budget,
                "num_unlabeled": n_pool,
                "sampling_time_s": elapsed,
            }
        )

        return selected_indices, elapsed
