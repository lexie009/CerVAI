import os
import time
from datetime import datetime

from utils.sampling_representative_utils import get_features, run_kmeans
from utils.sampling_logger_utils import SamplingLogger
from sampling.base_sampler import BaseSampler

class RepresentativeSampler(BaseSampler):
    """
    Diversity-based sampling using KMeans clustering on feature space.
    Selects the samples closest to each cluster center.
    """

    def __init__(self, trainer, **kwargs):  # ← 接收 **kwargs
        budget = kwargs.get("budget")
        device = kwargs.get("device", "cuda")
        log_dir = kwargs.get("log_dir", "./sampling_logs")

        super().__init__(budget=budget, device=device, log_dir=log_dir)
        self.trainer = trainer
        self.model = trainer.model.to(self.device)
        self.sampling_logger = SamplingLogger(log_dir, strategy_name="Representative")


    def select(self, unlabeled_dataloader):
        sampling_start = time.time()
        self.sampling_logger.log_message(f"Start sampling with budget {self.budget}")

        try:
            # extract features
            idx_list, feature_matrix = get_features(
                self.model, unlabeled_dataloader, self.device
            )
            self.sampling_logger.log_message(f"Extracted features for {len(idx_list)} samples with dim {feature_matrix.shape[1]}")

            # run KMeans and select representative samples
            selected_indices = run_kmeans(feature_matrix, self.budget)
            global_ids = [idx_list[i] for i in selected_indices]

            elapsed = time.time() - sampling_start
            self.sampling_logger.log_message(f"Selected {len(selected_indices)} representative samples in {elapsed:.2f} seconds")

            # save selected indices
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.log_dir, f"representative_indices_{timestamp}.txt")
            with open(save_path, "w") as f:
                for idx in selected_indices:
                    f.write(f"{idx}\n")

            self.sampling_logger.log_message(f"Selected {len(selected_indices)} representative samples in {elapsed:.2f} seconds")

            self.sampling_logger.save_metadata({
                "strategy": "Representative",
                "budget": self.budget,
                "num_unlabeled": len(selected_indices),
                "sampling_time_sec": elapsed
            })

            return global_ids, elapsed


        except Exception as e:
            self.sampling_logger.log_error(f"Representative sampling failed", exc_info=True)
            raise
