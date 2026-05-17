"""
VQE-inspired Date Generator (classical variational ansatz)
────────────────────────────────────────────────────────────
This module replaces the previous CNN-based date generator with a
VQE-style variational (classical) ansatz implemented in PyTorch. The
interface is compatible with the original model: calling the module with
`conditions` returns `(dd_logits, mm_logits, yyyy_logits)`.

The implementation intentionally avoids quantum dependencies and instead
provides a parameterized variational block that mimics a quantum
variational circuit's role: a set of layer-wise, trainable rotations
followed by entangling (mixing) linear transforms. This keeps the model
lightweight and easy to run on CPU/GPU while being conceptually similar
to VQE.
"""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn


class VQEDateGenerator(nn.Module):
    """VQE-inspired variational model for date generation.

    Keeps the same constructor signature as the previous CNN-based model
    for compatibility with the rest of the codebase. A compatibility
    alias is added at the bottom so imports of `CNNDateGenerator` keep
    working.
    """

    def __init__(
        self,
        n_days: int,
        n_months: int,
        n_leaps: int,
        n_decades: int,
        embed_dim: int,
        num_filters: int = 0,
        kernel_sizes: List[int] | None = None,
        dropout: float = 0.1,
        dd_vocab_size: int = 31,
        mm_vocab_size: int = 12,
        yyyy_vocab_size: int = 100,
        n_layers: int = 3,
    ) -> None:
        super().__init__()

        self.day_emb    = nn.Embedding(n_days,    embed_dim)
        self.month_emb  = nn.Embedding(n_months,  embed_dim)
        self.leap_emb   = nn.Embedding(n_leaps,   embed_dim)
        self.decade_emb = nn.Embedding(n_decades, embed_dim)

        # Flattened input dimension (4 conditions)
        self.embed_dim = embed_dim
        self.flat_dim = embed_dim * 4

        # Variational ansatz parameters: one angle parameter per element
        # per layer. These act like tunable rotation angles in a VQE.
        self.n_layers = n_layers
        thetas = torch.randn(n_layers, self.flat_dim) * 0.01
        self.register_parameter("thetas", nn.Parameter(thetas))

        # Entangling (mixing) linear layers to simulate multi-qubit gates
        self.mixers = nn.ModuleList([
            nn.Linear(self.flat_dim, self.flat_dim) for _ in range(n_layers)
        ])

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        # Classification heads (same outputs as before)
        self.dd_head   = nn.Linear(self.flat_dim, dd_vocab_size)
        self.mm_head   = nn.Linear(self.flat_dim, mm_vocab_size)
        self.yyyy_head = nn.Linear(self.flat_dim, yyyy_vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def variational_block(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a sequence of variational layers to input `x`.

        Args:
            x: Tensor of shape (B, flat_dim)

        Returns:
            Tensor of shape (B, flat_dim)
        """
        for l in range(self.n_layers):
            theta = self.thetas[l].unsqueeze(0)   # (1, flat_dim)
            # Elementwise 'rotation' using learned angles. We combine a
            # simple trig transform with a learned mixing linear layer to
            # emulate a parameterized quantum circuit step.
            rot = torch.cos(theta) * x + torch.sin(theta) * self.activation(self.mixers[l](x))
            # Residual connection for stability
            x = x + rot
        return x

    def forward(self, conditions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            conditions: LongTensor (batch, 4)

        Returns:
            dd_logits, mm_logits, yyyy_logits
        """
        day_e    = self.day_emb(conditions[:, 0])
        month_e  = self.month_emb(conditions[:, 1])
        leap_e   = self.leap_emb(conditions[:, 2])
        decade_e = self.decade_emb(conditions[:, 3])

        # Concatenate embeddings to create a single 'quantum register' vector
        seq = torch.cat([day_e, month_e, leap_e, decade_e], dim=-1)  # (B, flat_dim)

        # Variational processing
        h = self.variational_block(seq)
        h = self.dropout(h)

        return self.dd_head(h), self.mm_head(h), self.yyyy_head(h)


# Compatibility alias: keep the original import name working.
CNNDateGenerator = VQEDateGenerator
