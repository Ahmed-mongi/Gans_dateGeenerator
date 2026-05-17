"""
Model 1 (in-course): Conditional GAN (cGAN) for Date Generation
─────────────────────────────────────────────────────
Architecture:
  Generator:
    - Input: noise vector z (latent_dim) + embedded conditions
    - Output: soft logits over (dd, mm, yyyy) via Gumbel-Softmax
              so the discriminator can receive differentiable "fake" inputs.

  Discriminator:
    - Input: embedded conditions + one-hot / soft date representation
    - Output: real/fake scalar score (no sigmoid – uses BCEWithLogitsLoss)

Training follows the standard minimax GAN objective.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANGenerator(nn.Module):
    """
    (z, conditions) → (dd_logits, mm_logits, yyyy_logits)

    During inference we take the argmax of each head to get the predicted index.
    During training we use Gumbel-Softmax for a differentiable discrete sample.
    """

    def __init__(
        self,
        latent_dim: int,
        n_days: int,
        n_months: int,
        n_leaps: int,
        n_decades: int,
        embed_dim: int,
        hidden_dim: int,
        dd_vocab_size: int,
        mm_vocab_size: int,
        yyyy_vocab_size: int,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.day_emb    = nn.Embedding(n_days,    embed_dim)
        self.month_emb  = nn.Embedding(n_months,  embed_dim)
        self.leap_emb   = nn.Embedding(n_leaps,   embed_dim)
        self.decade_emb = nn.Embedding(n_decades, embed_dim)

        in_dim = latent_dim + embed_dim * 4

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
        )

        self.dd_head   = nn.Linear(hidden_dim // 2, dd_vocab_size)
        self.mm_head   = nn.Linear(hidden_dim // 2, mm_vocab_size)
        self.yyyy_head = nn.Linear(hidden_dim // 2, yyyy_vocab_size)

    def forward(
        self,
        conditions: torch.Tensor,
        z: torch.Tensor | None = None,
        temperature: float = 1.0,
        hard: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        conditions  : LongTensor (batch, 4)
        z           : FloatTensor (batch, latent_dim) – sampled from N(0,1).
                      If None, samples internally.
        temperature : Gumbel-Softmax temperature (lower = closer to one-hot)
        hard        : if True returns straight-through hard one-hot

        Returns
        -------
        dd_soft, mm_soft, yyyy_soft : soft one-hots (batch, vocab_size)
        """
        if z is None:
            z = torch.randn(conditions.size(0), self.latent_dim,
                            device=conditions.device)

        day_e    = self.day_emb(conditions[:, 0])
        month_e  = self.month_emb(conditions[:, 1])
        leap_e   = self.leap_emb(conditions[:, 2])
        decade_e = self.decade_emb(conditions[:, 3])

        x = torch.cat([z, day_e, month_e, leap_e, decade_e], dim=-1)
        h = self.net(x)

        dd_logits   = self.dd_head(h)
        mm_logits   = self.mm_head(h)
        yyyy_logits = self.yyyy_head(h)

        dd_soft   = F.gumbel_softmax(dd_logits,   tau=temperature, hard=hard)
        mm_soft   = F.gumbel_softmax(mm_logits,   tau=temperature, hard=hard)
        yyyy_soft = F.gumbel_softmax(yyyy_logits, tau=temperature, hard=hard)

        return dd_soft, mm_soft, yyyy_soft

    @torch.no_grad()
    def predict(self, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return argmax indices for each component (no Gumbel noise)."""
        z = torch.randn(conditions.size(0), self.latent_dim,
                        device=conditions.device)

        day_e    = self.day_emb(conditions[:, 0])
        month_e  = self.month_emb(conditions[:, 1])
        leap_e   = self.leap_emb(conditions[:, 2])
        decade_e = self.decade_emb(conditions[:, 3])

        x = torch.cat([z, day_e, month_e, leap_e, decade_e], dim=-1)
        h = self.net(x)

        dd_idx   = self.dd_head(h).argmax(dim=-1)
        mm_idx   = self.mm_head(h).argmax(dim=-1)
        yyyy_idx = self.yyyy_head(h).argmax(dim=-1)
        return dd_idx, mm_idx, yyyy_idx


class GANDiscriminator(nn.Module):
    """
    (conditions, dd_soft, mm_soft, yyyy_soft) → real/fake scalar

    Receives soft one-hot date representations so gradients flow through
    the generator via Gumbel-Softmax.
    """

    def __init__(
        self,
        n_days: int,
        n_months: int,
        n_leaps: int,
        n_decades: int,
        embed_dim: int,
        hidden_dim: int,
        dd_vocab_size: int,
        mm_vocab_size: int,
        yyyy_vocab_size: int,
    ) -> None:
        super().__init__()

        self.day_emb    = nn.Embedding(n_days,    embed_dim)
        self.month_emb  = nn.Embedding(n_months,  embed_dim)
        self.leap_emb   = nn.Embedding(n_leaps,   embed_dim)
        self.decade_emb = nn.Embedding(n_decades, embed_dim)

        # Project each soft one-hot to embed_dim
        self.dd_proj   = nn.Linear(dd_vocab_size,   embed_dim)
        self.mm_proj   = nn.Linear(mm_vocab_size,   embed_dim)
        self.yyyy_proj = nn.Linear(yyyy_vocab_size, embed_dim)

        in_dim = embed_dim * 7  # 4 condition embeds + 3 date projections

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        conditions: torch.Tensor,
        dd_soft: torch.Tensor,
        mm_soft: torch.Tensor,
        yyyy_soft: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns raw logit (batch, 1) – apply BCEWithLogitsLoss outside.
        """
        day_e    = self.day_emb(conditions[:, 0])
        month_e  = self.month_emb(conditions[:, 1])
        leap_e   = self.leap_emb(conditions[:, 2])
        decade_e = self.decade_emb(conditions[:, 3])

        dd_e   = self.dd_proj(dd_soft)
        mm_e   = self.mm_proj(mm_soft)
        yyyy_e = self.yyyy_proj(yyyy_soft)

        x = torch.cat([day_e, month_e, leap_e, decade_e, dd_e, mm_e, yyyy_e], dim=-1)
        return self.net(x)
