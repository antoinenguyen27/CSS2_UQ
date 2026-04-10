"""
HALT Bi-GRU model — replication of Shapiro et al. (2026)

Input:  (N, 192, 25)  — preprocessed log-prob feature sequences
Output: (N,)          — raw logits (apply sigmoid for UQ probability)
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

        # 1. Input LayerNorm
        self.input_norm = nn.LayerNorm(input_dim)

        # 2. MLP projection: 25 -> 128 -> 128
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

        # 3. Bidirectional GRU: 128 -> 512 (256 * 2)
        self.bigru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Linear classifier: 512 -> 1
        self.classifier = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x:       (N, 192, 25) input feature sequences
            lengths: (N,) actual sequence lengths before padding (optional)

        Returns:
            logits: (N,) raw logits — pass through sigmoid for UQ probability
        """
        N, T, _ = x.shape

        # 1. LayerNorm
        x = self.input_norm(x)

        # 2. MLP projection
        x = self.input_proj(x)                          # (N, 192, 128)

        # 3. Bidirectional GRU
        out, _ = self.bigru(x)                          # (N, 192, 512)

        # 4. Top-q pooling
        norms = out.norm(dim=-1)                        # (N, 192)

        if lengths is not None:
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            norms = norms.masked_fill(~mask, float("-inf"))

        k = max(1, int(self.top_q * T))
        topk_idx = norms.topk(k, dim=1).indices         # (N, k)
        topk_out = out.gather(
            1, topk_idx.unsqueeze(-1).expand(-1, -1, out.size(-1))
        )
        pooled = topk_out.mean(dim=1)                   # (N, 512)

        # 5. Linear classifier
        logits = self.classifier(pooled).squeeze(-1)    # (N,)

        return logits

    def predict_proba(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """Returns UQ probability in [0, 1] — probability that the LLM was correct."""
        return torch.sigmoid(self.forward(x, lengths))