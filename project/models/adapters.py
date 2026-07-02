"""Stage B — projection adapters mapping each frozen, task-supervised
penultimate embedding into a shared, L2-normalized latent space.

video_adapter: 512-d (FusionClassifier penultimate) -> D
text_adapter:  768-d (FrozenTextMoodEncoder penultimate) -> D
"""
import math

import torch
from torch import nn


class ProjectionAdapter(nn.Module):
    def __init__(self, input_dim: int, shared_dim: int, dropout: float = 0.15):
        super().__init__()
        hidden_dim = 2 * shared_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, shared_dim),
        )

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        z = self.net(x)
        if normalize:
            z = nn.functional.normalize(z, p=2, dim=-1)
        return z


class VideoAdapter(ProjectionAdapter):
    def __init__(self, shared_dim: int = 256, input_dim: int = 512, dropout: float = 0.15):
        super().__init__(input_dim, shared_dim, dropout)


class TextAdapter(ProjectionAdapter):
    def __init__(self, shared_dim: int = 256, input_dim: int = 768, dropout: float = 0.15):
        super().__init__(input_dim, shared_dim, dropout)


class LogitScale(nn.Module):
    """Learnable contrastive temperature, parameterized as CLIP does (log-scale, clamped)."""

    def __init__(self, init_value: float = 1 / 0.07, max_value: float = 100.0):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(init_value)))
        self.log_max = math.log(max_value)

    def forward(self) -> torch.Tensor:
        return self.log_scale.clamp(max=self.log_max).exp()


class AlignmentModel(nn.Module):
    """Bundles the two Stage B adapters + learnable temperature into one module,
    so training/evaluation only need to checkpoint/load a single object."""

    def __init__(
        self,
        shared_dim: int = 256,
        video_input_dim: int = 512,
        text_input_dim: int = 768,
        dropout: float = 0.15,
        logit_scale_init: float = 1 / 0.07,
        logit_scale_max: float = 100.0,
    ):
        super().__init__()
        self.video_adapter = VideoAdapter(shared_dim, video_input_dim, dropout)
        self.text_adapter = TextAdapter(shared_dim, text_input_dim, dropout)
        self.logit_scale = LogitScale(logit_scale_init, logit_scale_max)

    def encode_video(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.video_adapter(x, normalize=normalize)

    def encode_text(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        return self.text_adapter(x, normalize=normalize)
