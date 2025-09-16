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
        参数 `unlabeled_dataloader` 仍然保留，但本策略并不真正使用它。
        这样可以保持与其他采样器的统一接口。
        """
        tic = time.time()
        n_pool = len(self.unlabeled_indices)

        if self.budget is None:
            raise ValueError("RandomSampler expects a non-None budget.")
        if self.budget > n_pool:
            raise ValueError(f"Budget {self.budget} > unlabeled pool size {n_pool}")

        # ------------ 真正随机抽样 ------------
        selected_indices = random.sample(self.unlabeled_indices, self.budget)
        elapsed = time.time() - tic

        # ------------ 记录日志 ------------
        self.sampling_logger.log_message(
            f"Randomly selected {len(selected_indices)} / {n_pool} samples "
            f"in {elapsed:.2f} s (budget={self.budget})"
        )
        self.sampling_logger.save_indices(selected_indices)
        self.sampling_logger.save_metadata(
            {
                "strategy"        : "Random",
                "budget"          : self.budget,
                "num_unlabeled"   : n_pool,
                "sampling_time_s" : elapsed,
            }
        )

        return selected_indices, elapsed
