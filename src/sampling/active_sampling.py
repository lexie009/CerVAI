import time
import numpy as np
import torch
from pytorch_lightning import seed_everything

from sampling.random_sampler import  RandomSampler
from sampling.entropy_sampler import EntropySampler
from sampling.entropy_dropout_sampler import EntropyDropoutSampler
from sampling.borda_image_sampler import BordaSampler
from sampling.borda_batch_sampler import BordaBatchSampler
from sampling.representative_sampler import RepresentativeSampler
from utils.stochastic_batch_utils import (
    generate_random_groups,
    aggregate_group_uncertainty,
    select_top_positions_with_highest_uncertainty
)
from utils.sampling_logger_utils import SamplingLogger
from torch.utils.data import Subset, DataLoader

def sample_new_indices(
        sampling_config,
        budget,
        trainer,
        dataset,
        unlabeled_indices,
        **kwargs):
    """
    Active-Learning sampling entry-point.
    Returns:
        query_indices : list[int]  ← 行号(=全局 CSV index)
        sampling_time : float (sec)
    """
    # ------------ build DataLoader only on unlabeled set ------------
    unlabeled_dataset   = Subset(dataset, unlabeled_indices)
    unlabeled_loader    = DataLoader(
        unlabeled_dataset,
        batch_size = kwargs.get("batch_size", 4),
        shuffle    = False,
        num_workers= 0,
    )

    # --------- dispatch ---------
    strategy = sampling_config["strategy"]
    seed     = kwargs.get("seed", 42)
    seed_everything(seed)

    # Strategy dispatch
    if strategy == 'random':
        kwargs.setdefault("unlabeled_indices", unlabeled_indices)
        sampler = RandomSampler(trainer, budget=budget, **kwargs)
        query_indices, sampling_time = sampler.select(unlabeled_loader)

    elif strategy == 'entropy':
        sampler = EntropySampler(trainer, budget=budget, **kwargs)
        query_indices, uncertainty_values, sampling_time = sampler.select(unlabeled_loader)

    elif strategy == 'entropy_mc':
        sampler = EntropyDropoutSampler(trainer, budget=budget, **kwargs)
        query_indices, uncertainty_values, sampling_time = sampler.select(unlabeled_loader)

    elif strategy == 'bordaimage':
        sampler = BordaSampler(trainer, budget=budget, **kwargs)
        query_indices, uncertainty_values, sampling_time = sampler.select(unlabeled_loader)

    elif strategy == 'bordabatch':
        sampler = BordaBatchSampler(trainer, budget=budget, **kwargs)
        query_indices, uncertainty_values, sampling_time = sampler.select(unlabeled_loader)

    elif strategy == 'representative':
        sampler = RepresentativeSampler(trainer, budget=budget, **kwargs)
        query_indices, sampling_time = sampler.select(unlabeled_loader)

    else:
        raise NotImplementedError(f"Sampling strategy '{strategy}' not implemented.")

        # --------- logging ---------
    SamplingLogger.log_generic_data(
        trainer.logger, "metric",
        round(sampling_time / 60, 2), "sampling_time_min",
        epoch=trainer.current_epoch
    )
    SamplingLogger.log_generic_data(
        trainer.logger, "list", query_indices, "query_indices"
    )

    return query_indices, sampling_time
