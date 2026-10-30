"""Random-seed helpers for repeatable training runs."""

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep benchmark-based kernel selection from changing between runs.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python RNGs inside a PyTorch DataLoader worker."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
