"""Phase 5 — baseline classifier on the shipped, pre-extracted SlowFast
(N,2304) features. Established as a reference point before investing in the
frozen VideoMAE+CLIP fusion model.
"""
from torch import nn


class SlowFastBaselineMLP(nn.Module):
    def __init__(self, input_dim: int = 2304, hidden_dim: int = 256, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)
