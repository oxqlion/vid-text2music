"""Shared training utilities: seeding, class weights, early stopping, checkpointing."""
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights: w_k = N / (K * n_k)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1  # guard div-by-zero for an absent class
    n = counts.sum()
    weights = n / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    def __init__(self, patience: int = 10, mode: str = "max"):
        assert mode in ("max", "min")
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """Returns True if `score` is a new best."""
        improved = self.best_score is None or (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(model, path, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt
