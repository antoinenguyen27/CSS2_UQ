"""
HALT Bi-GRU model.

Input:  (N, T, 25) feature sequences
Output: (N,) logits (sigmoid => probability model answer is correct)
"""

import torch
import torch.nn as nn


class HALTModel(nn.Module):
    def __init__(
        self,
        input_dim: int = 25,
        proj_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 5,
        dropout: float = 0.4,
        top_q: float = 0.15,
    ):
        super().__init__()
        self.top_q = top_q
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )
        self.bigru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        n, t, _ = x.shape
        x = self.input_norm(x)
        x = self.input_proj(x)
        out, _ = self.bigru(x)

        norms = out.norm(dim=-1)
        if lengths is not None:
            mask = torch.arange(t, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            norms = norms.masked_fill(~mask, float("-inf"))

        k = max(1, int(self.top_q * t))
        topk_idx = norms.topk(k, dim=1).indices
        topk_out = out.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, out.size(-1)))
        pooled = topk_out.mean(dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

    def predict_proba(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.forward(x, lengths))

