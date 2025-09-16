from abc import ABC, abstractmethod
import os
import logging

class BaseSampler(ABC):
    """
    所有主动学习采样器的基类，统一构造函数签名：
        budget         : 需要选多少个样本
        trainer        : 可选，若采样器需要访问 trainer 中的模型、epoch 等信息
        device         : 设备
        log_dir        : 采样日志保存目录
    """

    def __init__(
        self,
        budget: int,
        trainer=None,
        *,
        device: str = "cuda",
        log_dir: str | None = "./sampling_logs",
        **kwargs,               # 允许子类传递其他自定义参数
    ) -> None:
        if budget is None:
            raise ValueError("`budget` cannot be none, number of sample cannot be decided")

        self.budget  = budget
        self.trainer = trainer
        self.device  = device
        self.log_dir = log_dir

        # 日志器 —— 子类可直接 self.logger.info(...)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:    # 避免重复添加 handler
            fh = logging.FileHandler(os.path.join(self.log_dir, f"{self.__class__.__name__}.log"))
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(fh)
            self.logger.setLevel(logging.INFO)

        # 可选：保存所有额外 kwargs 供子类使用
        self.extra_cfg = kwargs

    # --------------------------- 抽象接口 --------------------------- #
    @abstractmethod
    def select(self, dataloader):
        """
        由各具体采样器实现核心逻辑：
            给定 `dataloader`（未标注样本），
            返回一个 (indices, [optional_uncertainties], elapsed_time) 元组。

        必须至少返回:
            indices (List[int])      选中的全局索引
            elapsed_time (float)     采样耗时（秒）

        若不需要额外信息，可用下划线占位：
            return indices, None, elapsed
        """
        raise NotImplementedError

    # --------------------------- 便捷打印 --------------------------- #
    def __str__(self) -> str:
        return (f"{self.__class__.__name__}(budget={self.budget}, "
                f"device='{self.device}', trainer={type(self.trainer).__name__})")
