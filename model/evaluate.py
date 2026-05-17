"""
evaluate.py – Post-training evaluation & result visualisation.

Usage:
    python evaluate.py --model gan|vae|cnn|transformer [--data data/data.txt] [--n_samples 200]

Produces:
  - Per-condition satisfaction breakdown table
  - Sample predictions (both successes and failures)
  - Saves a summary to model/weights/<model>_eval.txt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tokenizer  import DateTokenizer
from utils.dataset    import DatesDataset
from utils.validation import check_conditions, all_conditions_met

from model.cnn_model import CNNDateGenerator
from model.gan_model         import GANGenerator
from model.transformer_model import TransformerDateGenerator
from model.vae_model         import VAE

WEIGHTS_DIR = Path(__file__).parent / "weights"


def _build_and_load(model_type: str, tokenizer: DateTokenizer,
                    device: torch.device):
    common = dict(
        n_days=tokenizer.n_days, n_months=tokenizer.n_months,
        n_leaps=tokenizer.n_leaps, n_decades=tokenizer.n_decades,
        dd_vocab_size=tokenizer.dd_vocab_size,
        mm_vocab_size=tokenizer.mm_vocab_size,
        yyyy_vocab_size=tokenizer.yyyy_vocab_size,
    )
    if model_type == "cnn":
        m = CNNDateGenerator(embed_dim=64, num_filters=256, kernel_sizes=[2, 3, 4], dropout=0.1, **common)
    elif model_type == "transformer":
        m = TransformerDateGenerator(embed_dim=64, nhead=4, num_layers=2,
                                     ff_dim=256, dropout=0.1, **common)
    elif model_type == "vae":
        m = VAE(embed_dim=64, hidden_dim=256, latent_dim=64, beta=1.0, **common)
    elif model_type == "gan":
        m = GANGenerator(latent_dim=64, embed_dim=64, hidden_dim=256, **common)
    else:
        raise ValueError(f"Unknown model: {model_type}")

    weight_path = WEIGHTS_DIR / f"{model_type}_best.pt"
    m.load_state_dict(torch.load(weight_path, map_location=device))
    m.to(device).eval()
    return m


@torch.no_grad()
def run_eval(
    model,
    model_type: str,
    loader: DataLoader,
    tokenizer: DateTokenizer,
    device: torch.device,
    n_examples: int = 10,
) -> Dict:
    stats: Dict[str, int] = defaultdict(int)
    examples_ok:   List[str] = []
    examples_fail: List[str] = []

    for batch in loader:
        conditions, dd_gt, mm_gt, yyyy_gt = [b.to(device) for b in batch]

        if model_type in ("cnn", "transformer"):
            dd_logits, mm_logits, yyyy_logits = model(conditions)
            dd_idx   = dd_logits.argmax(dim=-1)
            mm_idx   = mm_logits.argmax(dim=-1)
            yyyy_idx = yyyy_logits.argmax(dim=-1)
        else:
            dd_idx, mm_idx, yyyy_idx = model.predict(conditions)

        for i in range(conditions.size(0)):
            cond    = conditions[i].tolist()
            day_t   = tokenizer.idx2day.get(cond[0],    "?")
            month_t = tokenizer.idx2month.get(cond[1],  "?")
            leap_t  = tokenizer.idx2leap.get(cond[2],   "?")
            dec_t   = tokenizer.idx2decade.get(cond[3], "?")

            date_str = tokenizer.decode_date(
                dd_idx[i].item(), mm_idx[i].item(), yyyy_idx[i].item()
            )
            valid, day_ok, month_ok, leap_ok, decade_ok = check_conditions(
                date_str, day_t, month_t, leap_t, dec_t
            )

            stats["total"] += 1
            if valid:        stats["valid_dates"]  += 1
            if day_ok:       stats["day_ok"]       += 1
            if month_ok:     stats["month_ok"]     += 1
            if leap_ok:      stats["leap_ok"]      += 1
            if decade_ok:    stats["decade_ok"]    += 1
            all_ok = valid and day_ok and month_ok and leap_ok and decade_ok
            if all_ok:       stats["all_ok"]       += 1

            cond_str = f"[{day_t}] [{month_t}] [{leap_t}] [{dec_t}]"
            line = f"  Input: {cond_str}  →  Predicted: {date_str}"
            if all_ok and len(examples_ok) < n_examples:
                examples_ok.append(line)
            elif not all_ok and len(examples_fail) < n_examples:
                fail_detail = (
                    f"  Input: {cond_str}  →  Predicted: {date_str}"
                    f"  [valid={valid} day={day_ok} month={month_ok} "
                    f"leap={leap_ok} decade={decade_ok}]"
                )
                examples_fail.append(fail_detail)

    return {
        "stats":        dict(stats),
        "examples_ok":  examples_ok,
        "examples_fail":examples_fail,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    type=str, default="cnn",
                        choices=["cnn", "transformer", "vae", "gan"])
    parser.add_argument("--data",     type=str, default="data/data.txt")
    parser.add_argument("--n_examples", type=int, default=10)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = DateTokenizer()
    dataset   = DatesDataset(args.data, tokenizer)

    # Use 10% as held-out test set (same split logic as train.py)
    n_val   = int(len(dataset) * 0.1)
    n_train = len(dataset) - n_val
    _, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    model = _build_and_load(args.model, tokenizer, device)
    print(f"\nEvaluating model: {args.model} on {len(val_ds)} val samples\n")

    result = run_eval(model, args.model, loader, tokenizer, device, args.n_examples)
    s = result["stats"]
    n = s["total"]

    report_lines = [
        f"Model: {args.model}",
        f"Evaluated on {n} samples",
        "",
        "── Per-condition satisfaction ───────────────────────────────",
        f"  Valid calendar dates : {s['valid_dates']:>6} / {n}  ({100*s['valid_dates']/n:.1f}%)",
        f"  Day-of-week correct  : {s['day_ok']:>6} / {n}  ({100*s['day_ok']/n:.1f}%)",
        f"  Month correct        : {s['month_ok']:>6} / {n}  ({100*s['month_ok']/n:.1f}%)",
        f"  Leap-year correct    : {s['leap_ok']:>6} / {n}  ({100*s['leap_ok']/n:.1f}%)",
        f"  Decade correct       : {s['decade_ok']:>6} / {n}  ({100*s['decade_ok']/n:.1f}%)",
        f"  ALL conditions met   : {s['all_ok']:>6} / {n}  ({100*s['all_ok']/n:.1f}%)",
        "",
        "── Successful predictions (sample) ─────────────────────────",
        *result["examples_ok"],
        "",
        "── Failed predictions (sample) ──────────────────────────────",
        *result["examples_fail"],
    ]

    report = "\n".join(report_lines)
    print(report)

    out_path = WEIGHTS_DIR / f"{args.model}_eval.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
