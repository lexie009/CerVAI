import os, logging, json
from datetime import datetime
from typing import Any

class SamplingLogger:
    """
    既兼容 Lightning-Trainer 风格 (trainer.logger.logger.info)
    也兼容普通 logging.Logger 以及本地实例化的 SamplingLogger
    """

    def __init__(self, log_dir: str, strategy_name: str = "unknown") -> None:
        self.strategy_name = strategy_name
        self.log_dir = os.path.join(log_dir, strategy_name)
        os.makedirs(self.log_dir, exist_ok=True)

        _base = logging.getLogger(f"{strategy_name}_Sampler")
        _base.setLevel(logging.INFO)
        logfile = os.path.join(self.log_dir,
                               f"{strategy_name.lower()}_log.txt")
        if not _base.handlers:
            fh = logging.FileHandler(logfile)
            fh.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'))
            _base.addHandler(fh)

        # 关键：让外部能找到 `.logger` 属性
        # (Lightning 内部 trainer.logger.logger.info ...)
        self.logger = _base

    # ---------- 静态工具：解决 logger ----------
    @staticmethod
    def _resolve_logger(obj: Any) -> logging.Logger:
        """
        * 如果传进来的是 Lightning-Trainer.logger ，再向下取 .logger
        * 如果本身就是 logging.Logger，则直接返回
        """
        return getattr(obj, "logger", obj)

    # ---------- 兼容 sample_new_indices 的 classmethod ----------
    @staticmethod
    def log_generic_data(trainer_logger: Any,
                         data_type    : str,
                         data         : Any,
                         name         : str,
                         epoch        : int | None = None) -> None:
        """
        保持原来调用顺序：SamplingLogger.log_generic_data(
            trainer.logger, 'hyperparameter', strategy, 'sampling_strategy')
        """
        logger = SamplingLogger._resolve_logger(trainer_logger)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if data_type == "metric":
            logger.info(f"[{ts}] METRIC - {name}: {data} (epoch={epoch})")
        elif data_type == "list":
            logger.info(f"[{ts}] LIST   - {name}: {data} (len={len(data)})")
        elif data_type == "hyperparameter":
            logger.info(f"[{ts}] HYPERPARAM - {name}: {json.dumps(data)}")
        else:
            logger.warning(f"[{ts}] Unknown data_type '{data_type}' for {name}")

    # ---------- 下面这几段是原本实例方法，不变 ----------
    def log_message(self, msg: str):
        self.logger.info(msg)

    def log_error(self, msg: str, exc_info: bool = False):
        self.logger.error(msg, exc_info=exc_info)

    def save_scores(self, indices, scores, prefix="scores"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = os.path.join(self.log_dir, f"{prefix}_{ts}.txt")
        with open(fn, "w") as f:
            for i, s in zip(indices, scores):
                f.write(f"{i}\t{s:.6f}\n")
        self.logger.info(f"Saved scores → {fn}")
        return fn

    def save_indices(self, indices, prefix="selected"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = os.path.join(self.log_dir, f"{prefix}_indices_{ts}.txt")
        with open(fn, "w") as f:
            for i in indices:
                f.write(f"{i}\n")
        self.logger.info(f"Saved indices → {fn}")
        return fn

    def save_metadata(self, meta: dict):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = os.path.join(self.log_dir, f"metadata_{ts}.json")
        with open(fn, "w") as f:
            json.dump(meta, f, indent=2)
        self.logger.info(f"Saved metadata → {fn}")
        return fn
