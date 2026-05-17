"""
train.py – Train one model at a time.

Usage:
    python train.py --model cnn       [options]
    python train.py --model gan       [options]
    python train.py --model transformer [options]
    python train.py --model vae       [options]

All hyperparameters can be overridden via CLI flags.
Trained weights are saved to model/weights/<model_name>_best.pt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tokenizer  import DateTokenizer
from utils.dataset    import DatesDataset
from utils.validation import all_conditions_met, check_conditions

from model.cnn_model import CNNDateGenerator
from model.gan_model         import GANGenerator, GANDiscriminator
from model.transformer_model import TransformerDateGenerator
from model.vae_model         import VAE


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ── Condition satisfaction rate (evaluation metric) ───────────────────────────

def evaluate_satisfaction(
    model: nn.Module,
    loader: DataLoader,
    tokenizer: DateTokenizer,
    device: torch.device,
    model_type: str,
) -> float:
    """
    Returns the fraction of predictions that satisfy ALL four conditions.
    This is the primary metric – not accuracy – because multiple outputs
    are valid for each input.
    """
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for batch in loader:
            conditions, dd_gt, mm_gt, yyyy_gt = [b.to(device) for b in batch]

            if model_type == "vae":
                dd_idx, mm_idx, yyyy_idx = model.predict(conditions)
            elif model_type == "gan":
                dd_idx, mm_idx, yyyy_idx = model.predict(conditions)
            else:
                dd_logits, mm_logits, yyyy_logits = model(conditions)
                dd_idx   = dd_logits.argmax(dim=-1)
                mm_idx   = mm_logits.argmax(dim=-1)
                yyyy_idx = yyyy_logits.argmax(dim=-1)

            for i in range(conditions.size(0)):
                cond = conditions[i].tolist()
                # Decode condition indices back to string tokens
                day_tok    = tokenizer.idx2day.get(cond[0], "?")
                month_tok  = tokenizer.idx2month.get(cond[1], "?")
                leap_tok   = tokenizer.idx2leap.get(cond[2], "?")
                decade_tok = tokenizer.idx2decade.get(cond[3], "?")

                date_str = tokenizer.decode_date(
                    dd_idx[i].item(), mm_idx[i].item(), yyyy_idx[i].item()
                )
                if all_conditions_met(date_str, day_tok, month_tok, leap_tok, decade_tok):
                    correct += 1
                total += 1

    model.train()
    return correct / total if total > 0 else 0.0


# ── Individual trainers ────────────────────────────────────────────────────────

def train_discriminative(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: DateTokenizer,
    args: argparse.Namespace,
    device: torch.device,
    model_type: str,
    save_path: str,
) -> None:
    """
    Shared trainer for CNN and Transformer (standard cross-entropy).
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    best_sat = -1.0
    train_losses, val_sats = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for conditions, dd_gt, mm_gt, yyyy_gt in train_loader:
            conditions = conditions.to(device)
            dd_gt      = dd_gt.to(device)
            mm_gt      = mm_gt.to(device)
            yyyy_gt    = yyyy_gt.to(device)

            optimizer.zero_grad()
            dd_logits, mm_logits, yyyy_logits = model(conditions)

            loss = (
                criterion(dd_logits,   dd_gt)
                + criterion(mm_logits, mm_gt)
                + criterion(yyyy_logits, yyyy_gt)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if epoch % args.eval_every == 0:
            sat = evaluate_satisfaction(model, val_loader, tokenizer, device, model_type)
            val_sats.append((epoch, sat))
            print(f"[{model_type}] Epoch {epoch:3d} | Loss {avg_loss:.4f} | "
                  f"Val satisfaction {sat:.4f}")
            if sat > best_sat:
                best_sat = sat
                torch.save(model.state_dict(), save_path)
                print(f"  → Saved best model (sat={best_sat:.4f})")

    print(f"[{model_type}] Best validation satisfaction: {best_sat:.4f}")
    # Save loss log
    _save_loss_log(train_losses, val_sats, model_type, args.output_dir)


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: DateTokenizer,
    args: argparse.Namespace,
    device: torch.device,
    save_path: str,
) -> None:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()
    best_sat  = -1.0
    train_losses, val_sats = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        # KL annealing: linearly warm up β over the first 20% of training
        kl_weight = min(1.0, epoch / (0.2 * args.epochs)) * model.beta

        for conditions, dd_gt, mm_gt, yyyy_gt in train_loader:
            conditions = conditions.to(device)
            dd_gt      = dd_gt.to(device)
            mm_gt      = mm_gt.to(device)
            yyyy_gt    = yyyy_gt.to(device)

            optimizer.zero_grad()
            dd_logits, mm_logits, yyyy_logits, kl_loss = model(
                conditions, dd_gt, mm_gt, yyyy_gt
            )

            recon_loss = (
                criterion(dd_logits,   dd_gt)
                + criterion(mm_logits, mm_gt)
                + criterion(yyyy_logits, yyyy_gt)
            )
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if epoch % args.eval_every == 0:
            sat = evaluate_satisfaction(model, val_loader, tokenizer, device, "vae")
            val_sats.append((epoch, sat))
            print(f"[vae] Epoch {epoch:3d} | Loss {avg_loss:.4f} | "
                  f"Val satisfaction {sat:.4f}")
            if sat > best_sat:
                best_sat = sat
                torch.save(model.state_dict(), save_path)
                print(f"  → Saved best model (sat={best_sat:.4f})")

    print(f"[vae] Best validation satisfaction: {best_sat:.4f}")
    _save_loss_log(train_losses, val_sats, "vae", args.output_dir)


def train_gan(
    generator: GANGenerator,
    discriminator: GANDiscriminator,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer: DateTokenizer,
    args: argparse.Namespace,
    device: torch.device,
    save_path: str,
) -> None:
    opt_g = torch.optim.AdamW(generator.parameters(),     lr=args.lr_g,
                               weight_decay=args.weight_decay, betas=(0.5, 0.999))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=args.lr_d,
                               weight_decay=args.weight_decay, betas=(0.5, 0.999))
    adv_criterion  = nn.BCEWithLogitsLoss()
    aux_criterion  = nn.CrossEntropyLoss()  # auxiliary supervision on G

    best_sat = -1.0
    g_losses, d_losses, val_sats = [], [], []

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        epoch_g = epoch_d = 0.0

        # Gumbel-Softmax temperature annealing
        temperature = max(0.5, 1.0 - (epoch / args.epochs) * 0.5)

        for conditions, dd_gt, mm_gt, yyyy_gt in train_loader:
            conditions = conditions.to(device)
            dd_gt      = dd_gt.to(device)
            mm_gt      = mm_gt.to(device)
            yyyy_gt    = yyyy_gt.to(device)
            batch_size = conditions.size(0)

            real_label = torch.ones(batch_size,  1, device=device)
            fake_label = torch.zeros(batch_size, 1, device=device)

            # ── Train Discriminator ────────────────────────────────────────
            opt_d.zero_grad()

            # Real samples: convert GT indices to one-hot
            dd_real   = F.one_hot(dd_gt,   generator.dd_head.out_features).float()
            mm_real   = F.one_hot(mm_gt,   generator.mm_head.out_features).float()
            yyyy_real = F.one_hot(yyyy_gt, generator.yyyy_head.out_features).float()

            d_real = discriminator(conditions, dd_real, mm_real, yyyy_real)
            loss_d_real = adv_criterion(d_real, real_label)

            dd_fake, mm_fake, yyyy_fake = generator(
                conditions, temperature=temperature, hard=False
            )
            d_fake = discriminator(
                conditions,
                dd_fake.detach(), mm_fake.detach(), yyyy_fake.detach()
            )
            loss_d_fake = adv_criterion(d_fake, fake_label)

            loss_d = (loss_d_real + loss_d_fake) * 0.5
            loss_d.backward()
            opt_d.step()
            epoch_d += loss_d.item()

            # ── Train Generator ────────────────────────────────────────────
            opt_g.zero_grad()

            dd_fake, mm_fake, yyyy_fake = generator(
                conditions, temperature=temperature, hard=False
            )
            d_fake = discriminator(conditions, dd_fake, mm_fake, yyyy_fake)
            loss_g_adv = adv_criterion(d_fake, real_label)

            # Auxiliary cross-entropy loss to stabilise training
            dd_idx_pred, mm_idx_pred, yyyy_idx_pred = generator.predict(conditions)
            loss_g_aux = (
                aux_criterion(dd_fake,   dd_gt)
                + aux_criterion(mm_fake, mm_gt)
                + aux_criterion(yyyy_fake, yyyy_gt)
            )
            loss_g = loss_g_adv + args.aux_weight * loss_g_aux
            loss_g.backward()
            opt_g.step()
            epoch_g += loss_g.item()

        g_losses.append(epoch_g / len(train_loader))
        d_losses.append(epoch_d / len(train_loader))

        if epoch % args.eval_every == 0:
            sat = evaluate_satisfaction(generator, val_loader, tokenizer, device, "gan")
            val_sats.append((epoch, sat))
            print(f"[gan] Epoch {epoch:3d} | G loss {g_losses[-1]:.4f} | "
                  f"D loss {d_losses[-1]:.4f} | Val sat {sat:.4f} | temp {temperature:.3f}")
            if sat > best_sat:
                best_sat = sat
                torch.save(generator.state_dict(), save_path)
                print(f"  → Saved best generator (sat={best_sat:.4f})")

    print(f"[gan] Best validation satisfaction: {best_sat:.4f}")
    _save_loss_log(g_losses, val_sats, "gan", args.output_dir, d_losses=d_losses)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _save_loss_log(
    train_losses, val_sats, model_type, output_dir, d_losses=None
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_type}_loss_log.txt")
    with open(path, "w") as f:
        f.write("epoch,train_loss,val_epoch,val_satisfaction\n")
        val_dict = {ep: s for ep, s in val_sats}
        for i, loss in enumerate(train_losses, start=1):
            sat = val_dict.get(i, "")
            if d_losses is not None:
                f.write(f"{i},{loss:.6f},{d_losses[i-1]:.6f},{sat}\n")
            else:
                f.write(f"{i},{loss:.6f},{sat}\n")
    print(f"Loss log saved to {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train date generator models")
    parser.add_argument("--model",       type=str, default="cnn",
                        choices=["cnn", "gan", "transformer", "vae"])
    parser.add_argument("--data",        type=str, default="data/data.txt")
    parser.add_argument("--output_dir",  type=str, default="model/weights")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=512)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--lr_g",        type=float, default=2e-4)
    parser.add_argument("--lr_d",        type=float, default=2e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--embed_dim",   type=int, default=64)
    parser.add_argument("--hidden_dim",  type=int, default=256)
    parser.add_argument("--latent_dim",  type=int, default=64)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--val_split",   type=float, default=0.1)
    parser.add_argument("--eval_every",  type=int, default=5)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--aux_weight",  type=float, default=1.0,
                        help="Weight of auxiliary CE loss in GAN generator")
    parser.add_argument("--beta",        type=float, default=1.0,
                        help="KL weight for VAE (beta-VAE)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = DateTokenizer()
    dataset   = DatesDataset(args.data, tokenizer)

    # Train / val split
    n_val   = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )
    print(f"Dataset: {len(dataset)} samples | Train {n_train} | Val {n_val}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    os.makedirs(args.output_dir, exist_ok=True)
    save_path = os.path.join(args.output_dir, f"{args.model}_best.pt")

    # ── Build and train the chosen model ──────────────────────────────────────
    if args.model == "cnn":
        model = CNNDateGenerator(
            n_days=tokenizer.n_days, n_months=tokenizer.n_months,
            n_leaps=tokenizer.n_leaps, n_decades=tokenizer.n_decades,
            embed_dim=args.embed_dim, num_filters=args.hidden_dim,
            kernel_sizes=[2, 3, 4], dropout=args.dropout,
            dd_vocab_size=tokenizer.dd_vocab_size,
            mm_vocab_size=tokenizer.mm_vocab_size,
            yyyy_vocab_size=tokenizer.yyyy_vocab_size,
        ).to(device)
        train_discriminative(model, train_loader, val_loader, tokenizer,
                             args, device, "cnn", save_path)

    elif args.model == "transformer":
        model = TransformerDateGenerator(
            n_days=tokenizer.n_days, n_months=tokenizer.n_months,
            n_leaps=tokenizer.n_leaps, n_decades=tokenizer.n_decades,
            embed_dim=args.embed_dim, nhead=4, num_layers=2,
            ff_dim=args.hidden_dim, dropout=args.dropout,
            dd_vocab_size=tokenizer.dd_vocab_size,
            mm_vocab_size=tokenizer.mm_vocab_size,
            yyyy_vocab_size=tokenizer.yyyy_vocab_size,
        ).to(device)
        train_discriminative(model, train_loader, val_loader, tokenizer,
                             args, device, "transformer", save_path)

    elif args.model == "vae":
        model = VAE(
            n_days=tokenizer.n_days, n_months=tokenizer.n_months,
            n_leaps=tokenizer.n_leaps, n_decades=tokenizer.n_decades,
            embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim, beta=args.beta,
            dd_vocab_size=tokenizer.dd_vocab_size,
            mm_vocab_size=tokenizer.mm_vocab_size,
            yyyy_vocab_size=tokenizer.yyyy_vocab_size,
        ).to(device)
        train_vae(model, train_loader, val_loader, tokenizer,
                   args, device, save_path)

    elif args.model == "gan":
        generator = GANGenerator(
            latent_dim=args.latent_dim,
            n_days=tokenizer.n_days, n_months=tokenizer.n_months,
            n_leaps=tokenizer.n_leaps, n_decades=tokenizer.n_decades,
            embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
            dd_vocab_size=tokenizer.dd_vocab_size,
            mm_vocab_size=tokenizer.mm_vocab_size,
            yyyy_vocab_size=tokenizer.yyyy_vocab_size,
        ).to(device)
        discriminator = GANDiscriminator(
            n_days=tokenizer.n_days, n_months=tokenizer.n_months,
            n_leaps=tokenizer.n_leaps, n_decades=tokenizer.n_decades,
            embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
            dd_vocab_size=tokenizer.dd_vocab_size,
            mm_vocab_size=tokenizer.mm_vocab_size,
            yyyy_vocab_size=tokenizer.yyyy_vocab_size,
        ).to(device)
        train_gan(generator, discriminator, train_loader, val_loader, tokenizer,
                  args, device, save_path)


if __name__ == "__main__":
    main()
