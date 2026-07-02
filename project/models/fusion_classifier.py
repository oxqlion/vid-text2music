"""Phase 5 — fusion classifier on frozen VideoMAE + CLIP embeddings.
concat(768 + 512 = 1280) -> Linear -> GELU -> Dropout -> Linear -> 5 classes.
"""
import torch
from torch import nn


class FusionClassifier(nn.Module):
    def __init__(
        self,
        videomae_dim: int = 768,
        clip_dim: int = 512,
        hidden_dim: int = 512,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(videomae_dim + clip_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused_embedding: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """return_embedding=True returns the 512-d penultimate activation (post GELU+Dropout,
        pre final Linear) instead of the 5-class logits -- the task-supervised video embedding
        Stage B's cross-modal alignment projects from. Indexes into `self.net` directly so the
        state_dict layout (and therefore existing checkpoints) is unchanged."""
        h = self.net[2](self.net[1](self.net[0](fused_embedding)))  # Linear -> GELU -> Dropout
        if return_embedding:
            return h
        return self.net[3](h)  # Linear -> logits
