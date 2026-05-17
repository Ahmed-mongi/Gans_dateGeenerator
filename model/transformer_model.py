"""
Model 4 (out-of-course): Transformer-based Conditional Date Generator
──────────────────────────────────────────────────────
Architecture:
  - Treat each of the 4 conditions as a token in a sequence.
  - Pass through a Transformer encoder (self-attention over conditions).
  - Pool the encoder output and feed three classification heads.

This allows the model to learn cross-condition interactions (e.g. the
interaction between decade and leap-year constraints) via attention.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import math


class TransformerDateGenerator(nn.Module):
    """
    conditions → (dd_logits, mm_logits, yyyy_logits)

    Parameters
    ----------
    n_days, n_months, n_leaps, n_decades : condition vocab sizes
    embed_dim   : embedding dimension (must be divisible by nhead)
    nhead       : number of attention heads
    num_layers  : number of TransformerEncoder layers
    ff_dim      : feedforward dimension inside each encoder layer
    dropout     : dropout probability
    dd_vocab_size, mm_vocab_size, yyyy_vocab_size : output vocab sizes
    """

    def __init__(
        self,
        n_days: int,
        n_months: int,
        n_leaps: int,
        n_decades: int,
        embed_dim: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        dropout: float = 0.1,
        dd_vocab_size: int = 35,
        mm_vocab_size: int = 16,
        yyyy_vocab_size: int = 404,
    ) -> None:
        super().__init__()

        # One learned embedding per condition type
        self.day_emb    = nn.Embedding(n_days,    embed_dim)
        self.month_emb  = nn.Embedding(n_months,  embed_dim)
        self.leap_emb   = nn.Embedding(n_leaps,   embed_dim)
        self.decade_emb = nn.Embedding(n_decades, embed_dim)

        # Positional encoding for the 4-token sequence
        self.pos_enc = nn.Parameter(torch.zeros(4, embed_dim))
        nn.init.normal_(self.pos_enc, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,       # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                                  enable_nested_tensor=False)

        # Classification heads applied to the mean-pooled encoder output
        self.dd_head   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, dd_vocab_size),
        )
        self.mm_head   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, mm_vocab_size),
        )
        self.yyyy_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, yyyy_vocab_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        conditions : LongTensor (batch, 4)

        Returns
        -------
        dd_logits, mm_logits, yyyy_logits
        """
        # Embed each condition: (batch, 4, embed_dim)
        day_e    = self.day_emb(conditions[:, 0]).unsqueeze(1)
        month_e  = self.month_emb(conditions[:, 1]).unsqueeze(1)
        leap_e   = self.leap_emb(conditions[:, 2]).unsqueeze(1)
        decade_e = self.decade_emb(conditions[:, 3]).unsqueeze(1)

        seq = torch.cat([day_e, month_e, leap_e, decade_e], dim=1)   # (B, 4, D)
        seq = seq + self.pos_enc.unsqueeze(0)                          # add pos

        enc_out = self.transformer(seq)                                # (B, 4, D)
        pooled  = enc_out.mean(dim=1)                                  # (B, D)

        return self.dd_head(pooled), self.mm_head(pooled), self.yyyy_head(pooled)
