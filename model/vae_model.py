"""
Model 2 (in-course): Conditional VAE for Date Generation
──────────────────────────────────────────────────────────
Architecture:
  Encoder (recognition network):
    Input : condition embeddings + one-hot date → μ, log σ²
    Used only during training to provide the variational posterior q(z|x,c).

  Decoder (generator):
    Input : z sampled from q(z|x,c) during training,
            z ~ N(0,I) at inference time
            + condition embeddings
    Output: (dd_logits, mm_logits, yyyy_logits)

Loss = Reconstruction loss (cross-entropy on dd / mm / yyyy)
     + β · KL( q(z|x,c) ‖ N(0,I) )     [KL annealing applied in train.py]
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Encoder(nn.Module):
    """q(z | date, conditions) → (μ, log σ²)"""

    def __init__(
        self,
        cond_dim: int,
        dd_vocab_size: int,
        mm_vocab_size: int,
        yyyy_vocab_size: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        in_dim = cond_dim + dd_vocab_size + mm_vocab_size + yyyy_vocab_size
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head      = nn.Linear(hidden_dim, latent_dim)
        self.log_var_head = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        cond_emb: torch.Tensor,
        dd_oh: torch.Tensor,
        mm_oh: torch.Tensor,
        yyyy_oh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([cond_emb, dd_oh, mm_oh, yyyy_oh], dim=-1)
        h = self.net(x)
        return self.mu_head(h), self.log_var_head(h)


class _Decoder(nn.Module):
    """p(date | z, conditions) → (dd_logits, mm_logits, yyyy_logits)"""

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        hidden_dim: int,
        dd_vocab_size: int,
        mm_vocab_size: int,
        yyyy_vocab_size: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.dd_head   = nn.Linear(hidden_dim, dd_vocab_size)
        self.mm_head   = nn.Linear(hidden_dim, mm_vocab_size)
        self.yyyy_head = nn.Linear(hidden_dim, yyyy_vocab_size)

    def forward(
        self, z: torch.Tensor, cond_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(torch.cat([z, cond_emb], dim=-1))
        return self.dd_head(h), self.mm_head(h), self.yyyy_head(h)


class VAE(nn.Module):
    """
    Conditional VAE for date generation.

    Parameters
    ----------
    n_days, n_months, n_leaps, n_decades : condition vocabulary sizes
    embed_dim   : per-condition embedding size
    hidden_dim  : encoder / decoder hidden width
    latent_dim  : latent space dimensionality
    beta        : KL weight (β-VAE; 1.0 = standard VAE)
    dd_vocab_size, mm_vocab_size, yyyy_vocab_size : output vocab sizes
    """

    def __init__(
        self,
        n_days: int,
        n_months: int,
        n_leaps: int,
        n_decades: int,
        embed_dim: int,
        hidden_dim: int,
        latent_dim: int,
        beta: float,
        dd_vocab_size: int,
        mm_vocab_size: int,
        yyyy_vocab_size: int,
    ) -> None:
        super().__init__()
        self.latent_dim      = latent_dim
        self.beta            = beta
        self.dd_vocab_size   = dd_vocab_size
        self.mm_vocab_size   = mm_vocab_size
        self.yyyy_vocab_size = yyyy_vocab_size

        self.day_emb    = nn.Embedding(n_days,    embed_dim)
        self.month_emb  = nn.Embedding(n_months,  embed_dim)
        self.leap_emb   = nn.Embedding(n_leaps,   embed_dim)
        self.decade_emb = nn.Embedding(n_decades, embed_dim)

        cond_dim = embed_dim * 4

        self.encoder = _Encoder(
            cond_dim, dd_vocab_size, mm_vocab_size, yyyy_vocab_size,
            hidden_dim, latent_dim,
        )
        self.decoder = _Decoder(
            latent_dim, cond_dim, hidden_dim,
            dd_vocab_size, mm_vocab_size, yyyy_vocab_size,
        )

    def _embed_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.day_emb(conditions[:, 0]),
            self.month_emb(conditions[:, 1]),
            self.leap_emb(conditions[:, 2]),
            self.decade_emb(conditions[:, 3]),
        ], dim=-1)

    @staticmethod
    def _reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """z = μ + ε · σ,  ε ~ N(0, I)"""
        return mu + torch.randn_like(mu) * torch.exp(0.5 * log_var)

    def forward(
        self,
        conditions: torch.Tensor,
        dd_idx: torch.Tensor,
        mm_idx: torch.Tensor,
        yyyy_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Returns
        -------
        dd_logits, mm_logits, yyyy_logits, kl_loss
        """
        cond_emb = self._embed_conditions(conditions)

        dd_oh   = F.one_hot(dd_idx,   self.dd_vocab_size).float()
        mm_oh   = F.one_hot(mm_idx,   self.mm_vocab_size).float()
        yyyy_oh = F.one_hot(yyyy_idx, self.yyyy_vocab_size).float()

        mu, log_var = self.encoder(cond_emb, dd_oh, mm_oh, yyyy_oh)
        z           = self._reparameterise(mu, log_var)

        dd_logits, mm_logits, yyyy_logits = self.decoder(z, cond_emb)

        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

        return dd_logits, mm_logits, yyyy_logits, kl_loss

    @torch.no_grad()
    def predict(
        self, conditions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample z ~ N(0, I) and decode (inference time)."""
        cond_emb = self._embed_conditions(conditions)
        z = torch.randn(conditions.size(0), self.latent_dim,
                        device=conditions.device)
        dd_logits, mm_logits, yyyy_logits = self.decoder(z, cond_emb)
        return (
            dd_logits.argmax(dim=-1),
            mm_logits.argmax(dim=-1),
            yyyy_logits.argmax(dim=-1),
        )
