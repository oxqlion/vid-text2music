"""Stage C, Milestone 1 — bridge adapter mapping a single shared-space vector
(from the frozen, already-trained Stage B AlignmentModel) into MusicGen's
T5-shaped cross-attention conditioning sequence.

MusicGen (`transformers.MusicgenForConditionalGeneration`) never consumes a
single vector -- its conditioner normally is a T5 encoder producing a
(B, T, 768) sequence that the decoder cross-attends to (see
`forward(..., encoder_outputs=...)` in modeling_musicgen.py, confirmed by
direct inspection of the installed transformers source). This bridge learns
that reshaping: a single 256-d point becomes a short learned-query sequence
in the same (B, T, 768) shape, which is passed to MusicGen verbatim as
`encoder_outputs=(bridge_output,)` -- bypassing the internal T5 call
entirely. Only this bridge is trained; MusicGen itself stays fully frozen
throughout Milestone 1.
"""
import torch
from torch import nn


class EmbeddingToConditioningBridge(nn.Module):
    def __init__(
        self,
        shared_dim: int = 256,
        seq_len: int = 8,
        cond_dim: int = 768,
        hidden_dim: int = 512,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.cond_dim = cond_dim

        self.input_proj = nn.Linear(shared_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(seq_len, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(hidden_dim, cond_dim)
        self.output_norm = nn.LayerNorm(cond_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, shared_dim) -> (B, seq_len, cond_dim) conditioning sequence."""
        b = z.shape[0]
        memory = self.input_proj(z).unsqueeze(1)  # (B, 1, hidden_dim) -- single summary token
        queries = self.queries.unsqueeze(0).expand(b, -1, -1)  # (B, T, hidden_dim)
        decoded = self.decoder(tgt=queries, memory=memory)  # learned queries attend to the summary
        return self.output_norm(self.output_proj(decoded))

    def attention_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """All-ones mask -- the bridge always emits a fixed-length sequence, no padding."""
        return torch.ones(batch_size, self.seq_len, dtype=torch.long, device=device)
