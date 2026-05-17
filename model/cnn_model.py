"""
Vector-Quantised GAN (VQGAN) Date Generator
────────────────────────────────────────────
Model 3 (out-of-course): replaces the VQE-style variational model with a
proper VQGAN.  The interface is identical to the previous file:
    forward(conditions) → (dd_logits, mm_logits, yyyy_logits)
The CNNDateGenerator alias is preserved so the rest of the codebase
(train.py, predict.py, evaluate.py) requires zero changes.

Architecture summary
────────────────────
  Encoder   – embeds the 4 condition indices, projects to a latent vector
              z_e ∈ ℝ^(latent_dim).
  Codebook  – vector-quantises z_e against K learnable embedding vectors,
              producing z_q via the straight-through estimator.
  Decoder   – maps z_q back to three classification heads (dd, mm, yyyy).
  Discriminator – a lightweight MLP that distinguishes real from generated
              latent codes; trained with the hinge loss.

Training signals (combined in train.py via vqgan_loss())
──────────────────────────────────────────────────────────
  L_recon    – cross-entropy on (dd, mm, yyyy) predictions
  L_vq       – vector-quantisation commitment + codebook losses
  L_adv_G    – hinge adversarial loss (generator side)
  L_adv_D    – hinge adversarial loss (discriminator side)

Only VQGANDateGenerator and Discriminator need to be instantiated.
vqgan_loss() is a convenience helper consumed by train.py.
"""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Vector-Quantisation layer
# ─────────────────────────────────────────────────────────────────────────────

class VectorQuantiser(nn.Module):
    """Discretises a continuous latent vector via nearest-codebook-entry lookup.

    Uses the straight-through estimator so gradients flow back through
    the encoder even though the argmin operation is non-differentiable.

    Args:
        n_embeddings:  number of codebook entries  (K)
        embedding_dim: dimension of each entry     (D)
        commitment_cost: weight β on the commitment loss term
    """

    def __init__(
        self,
        n_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_embeddings   = n_embeddings
        self.embedding_dim  = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook: (K, D)
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        nn.init.uniform_(
            self.embedding.weight,
            -1.0 / n_embeddings,
             1.0 / n_embeddings,
        )

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantise z_e.

        Args:
            z_e: (B, D) – encoder output

        Returns:
            z_q:      (B, D) – quantised vector (straight-through)
            vq_loss:  scalar – combined codebook + commitment loss
            indices:  (B,)   – chosen codebook index per sample
        """
        # Pairwise L2 distances: (B, K)
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)          # (B, 1)
            - 2.0 * z_e @ self.embedding.weight.t()      # (B, K)
            + self.embedding.weight.pow(2).sum(dim=1)    # (K,)
        )
        indices = distances.argmin(dim=1)                 # (B,)

        # Look up nearest embeddings
        z_q = self.embedding(indices)                     # (B, D)

        # VQ loss: codebook moves toward encoder outputs
        codebook_loss  = F.mse_loss(z_q.detach(), z_e)
        # Commitment loss: encoder stays close to chosen codebook vector
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through: copy gradients from z_q to z_e
        z_q = z_e + (z_q - z_e).detach()

        return z_q, vq_loss, indices


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Embeds the 4 condition tokens and projects to the latent space.

    Args:
        n_days, n_months, n_leaps, n_decades: vocabulary sizes for each
            condition index.
        embed_dim:   per-token embedding dimension
        latent_dim:  output dimensionality (= codebook entry dimension D)
        dropout:     dropout probability
    """

    def __init__(
        self,
        n_days: int,
        n_months: int,
        n_leaps: int,
        n_decades: int,
        embed_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.day_emb    = nn.Embedding(n_days,    embed_dim)
        self.month_emb  = nn.Embedding(n_months,  embed_dim)
        self.leap_emb   = nn.Embedding(n_leaps,   embed_dim)
        self.decade_emb = nn.Embedding(n_decades, embed_dim)

        flat_dim = embed_dim * 4

        self.net = nn.Sequential(
            nn.Linear(flat_dim, flat_dim * 2),
            nn.LayerNorm(flat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(flat_dim * 2, flat_dim),
            nn.LayerNorm(flat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(flat_dim, latent_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            conditions: LongTensor (B, 4) – [day_idx, month_idx, leap_idx, decade_idx]

        Returns:
            z_e: (B, latent_dim)
        """
        day_e    = self.day_emb(conditions[:, 0])
        month_e  = self.month_emb(conditions[:, 1])
        leap_e   = self.leap_emb(conditions[:, 2])
        decade_e = self.decade_emb(conditions[:, 3])

        x = torch.cat([day_e, month_e, leap_e, decade_e], dim=-1)  # (B, flat_dim)
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Decoder
# ─────────────────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """Maps a quantised latent vector to three output logit heads.

    Args:
        latent_dim:     input dimension (D, matches encoder & codebook)
        hidden_dim:     width of hidden layers
        dropout:        dropout probability
        dd_vocab_size:  number of day   classes (default 31)
        mm_vocab_size:  number of month classes (default 12)
        yyyy_vocab_size: number of year  classes (default 100)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        dd_vocab_size: int = 31,
        mm_vocab_size: int = 12,
        yyyy_vocab_size: int = 100,
    ) -> None:
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.dd_head   = nn.Linear(hidden_dim, dd_vocab_size)
        self.mm_head   = nn.Linear(hidden_dim, mm_vocab_size)
        self.yyyy_head = nn.Linear(hidden_dim, yyyy_vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_q: (B, latent_dim) – quantised latent

        Returns:
            dd_logits:   (B, dd_vocab_size)
            mm_logits:   (B, mm_vocab_size)
            yyyy_logits: (B, yyyy_vocab_size)
        """
        h = self.trunk(z_q)
        return self.dd_head(h), self.mm_head(h), self.yyyy_head(h)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Discriminator
# ─────────────────────────────────────────────────────────────────────────────

class Discriminator(nn.Module):
    """Latent-space MLP discriminator for the adversarial training signal.

    Operates on latent vectors (real encoder outputs vs. sampled/generated
    ones) and returns a scalar score per sample (hinge loss convention:
    no sigmoid).

    Args:
        latent_dim: dimension of input latent vectors
        hidden_dim: width of hidden layers
        dropout:    dropout probability
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),   # raw score, no activation
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim)

        Returns:
            scores: (B, 1) – higher = more 'real'
        """
        return self.net(z)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Top-level generator module  (public API)
# ─────────────────────────────────────────────────────────────────────────────

class VQGANDateGenerator(nn.Module):
    """VQGAN-based conditional date generator.

    This is the module that train.py / predict.py should instantiate.
    The Discriminator is a separate object (see below) so it can be given
    its own optimiser in train.py.

    Args:
        n_days, n_months, n_leaps, n_decades: condition vocabulary sizes
        embed_dim:       per-condition embedding dimension  (default 32)
        latent_dim:      VQ bottleneck dimension            (default 128)
        n_embeddings:    codebook size K                    (default 512)
        commitment_cost: β weight on commitment loss        (default 0.25)
        hidden_dim:      decoder hidden width               (default 256)
        dropout:         shared dropout rate                (default 0.1)
        dd_vocab_size:   day   output classes               (default 31)
        mm_vocab_size:   month output classes               (default 12)
        yyyy_vocab_size: year  output classes               (default 100)

    Ignored kwargs (num_filters, kernel_sizes, n_layers) are accepted for
    drop-in compatibility with the old CNN/VQE constructor signature.
    """

    def __init__(
        self,
        n_days: int,
        n_months: int,
        n_leaps: int,
        n_decades: int,
        embed_dim: int = 32,
        # VQE-era args silently ignored for API compatibility
        num_filters: int = 0,
        kernel_sizes: List[int] | None = None,
        n_layers: int = 3,
        # VQGAN-specific
        latent_dim: int = 128,
        n_embeddings: int = 512,
        commitment_cost: float = 0.25,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        dd_vocab_size: int = 31,
        mm_vocab_size: int = 12,
        yyyy_vocab_size: int = 100,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = Encoder(
            n_days=n_days,
            n_months=n_months,
            n_leaps=n_leaps,
            n_decades=n_decades,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )

        self.vq = VectorQuantiser(
            n_embeddings=n_embeddings,
            embedding_dim=latent_dim,
            commitment_cost=commitment_cost,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            dd_vocab_size=dd_vocab_size,
            mm_vocab_size=mm_vocab_size,
            yyyy_vocab_size=yyyy_vocab_size,
        )

    # ------------------------------------------------------------------
    # Convenience: build the matching discriminator with correct latent_dim
    # ------------------------------------------------------------------
    def make_discriminator(
        self,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> "Discriminator":
        """Return a Discriminator sized to match this generator's latent_dim."""
        return Discriminator(
            latent_dim=self.latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    # Core forward  (used at inference time and in the reconstruction step)
    # ------------------------------------------------------------------
    def forward(
        self,
        conditions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode → quantise → decode.  Returns logits only (no VQ loss).

        Suitable for inference and for predict.py which only needs logits.

        Args:
            conditions: LongTensor (B, 4)

        Returns:
            dd_logits, mm_logits, yyyy_logits
        """
        z_e = self.encoder(conditions)
        z_q, _, _ = self.vq(z_e)
        return self.decoder(z_q)

    # ------------------------------------------------------------------
    # Training forward  (used inside train.py)
    # ------------------------------------------------------------------
    def forward_train(
        self,
        conditions: torch.Tensor,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Encode → quantise → decode, also returning vq_loss and z_q.

        Use this variant in train.py so the VQ loss can be added to the
        total training objective and z_q can be passed to the discriminator.

        Args:
            conditions: LongTensor (B, 4)

        Returns:
            (dd_logits, mm_logits, yyyy_logits): prediction heads
            vq_loss:  scalar tensor – codebook + commitment loss
            z_q:      (B, latent_dim) – quantised latent (for discriminator)
        """
        z_e = self.encoder(conditions)
        z_q, vq_loss, _ = self.vq(z_e)
        logits = self.decoder(z_q)
        return logits, vq_loss, z_q

    # ------------------------------------------------------------------
    # Codebook lookup (useful for analysis / evaluation)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_indices(self, conditions: torch.Tensor) -> torch.Tensor:
        """Return the codebook index each sample is mapped to.

        Args:
            conditions: LongTensor (B, 4)

        Returns:
            indices: LongTensor (B,)
        """
        z_e = self.encoder(conditions)
        _, _, indices = self.vq(z_e)
        return indices


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Loss helper  (import in train.py)
# ─────────────────────────────────────────────────────────────────────────────

def vqgan_loss(
    logits: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    vq_loss: torch.Tensor,
    disc_fake_scores: torch.Tensor,
    adv_weight: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    """Compute the combined generator-side VQGAN loss.

    L_total = L_recon + L_vq + adv_weight * L_adv_G

    Args:
        logits:           (dd_logits, mm_logits, yyyy_logits)
        targets:          (dd_targets, mm_targets, yyyy_targets) – LongTensors
        vq_loss:          scalar from VectorQuantiser
        disc_fake_scores: (B, 1) discriminator scores on generated samples
        adv_weight:       weight for the adversarial term

    Returns:
        total_loss: scalar
        breakdown:  dict with individual loss components for logging
    """
    dd_logits, mm_logits, yyyy_logits = logits
    dd_tgt,    mm_tgt,    yyyy_tgt    = targets

    l_dd   = F.cross_entropy(dd_logits,   dd_tgt)
    l_mm   = F.cross_entropy(mm_logits,   mm_tgt)
    l_yyyy = F.cross_entropy(yyyy_logits, yyyy_tgt)
    l_recon = (l_dd + l_mm + l_yyyy) / 3.0

    # Hinge loss generator term: want discriminator to score fake as high
    l_adv_g = -disc_fake_scores.mean()

    total = l_recon + vq_loss + adv_weight * l_adv_g

    return total, {
        "recon": l_recon.item(),
        "vq":    vq_loss.item(),
        "adv_g": l_adv_g.item(),
        "total": total.item(),
    }


def discriminator_loss(
    real_scores: torch.Tensor,
    fake_scores: torch.Tensor,
) -> torch.Tensor:
    """Hinge loss for the discriminator.

    L_D = max(0, 1 - D(real)) + max(0, 1 + D(fake))

    Args:
        real_scores: (B, 1) – discriminator output on real latents (z_e from data)
        fake_scores: (B, 1) – discriminator output on fake latents (z_q from generator)

    Returns:
        loss: scalar
    """
    l_real = F.relu(1.0 - real_scores).mean()
    l_fake = F.relu(1.0 + fake_scores).mean()
    return l_real + l_fake


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Compatibility alias  (keeps existing imports working)
# ─────────────────────────────────────────────────────────────────────────────

CNNDateGenerator = VQGANDateGenerator
