"""
predict.py – Inference entry point.

Usage (as required by the assignment):
    python predict.py -i $path_to_input_file -o $path_to_output_file

The script loads the best available trained model (priority order:
transformer → cnn → vae → gan) and runs inference on the input file.
Output format mirrors data.txt exactly: conditions + predicted date on each line.

Override the model with --model:
    python predict.py -i data/example_input.txt -o out.txt --model gan
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

# Allow imports from parent when running as model/predict.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.tokenizer  import DateTokenizer
from utils.dataset    import InferenceDataset
from utils.validation import all_conditions_met

from model.cnn_model import CNNDateGenerator
from model.gan_model         import GANGenerator
from model.transformer_model import TransformerDateGenerator
from model.vae_model         import VAE

WEIGHTS_DIR = Path(__file__).parent / "weights"

# Model priority when --model is not explicitly specified
MODEL_PRIORITY = ["transformer", "cnn", "vae", "gan"]


def _build_model(model_type: str, tokenizer: DateTokenizer):
    """Instantiate the model architecture (no weights loaded yet)."""
    common = dict(
        n_days=tokenizer.n_days,
        n_months=tokenizer.n_months,
        n_leaps=tokenizer.n_leaps,
        n_decades=tokenizer.n_decades,
        dd_vocab_size=tokenizer.dd_vocab_size,
        mm_vocab_size=tokenizer.mm_vocab_size,
        yyyy_vocab_size=tokenizer.yyyy_vocab_size,
    )

    if model_type == "cnn":
        return CNNDateGenerator(embed_dim=64, num_filters=256, kernel_sizes=[2, 3, 4], dropout=0.1, **common)

    elif model_type == "transformer":
        return TransformerDateGenerator(
            embed_dim=64, nhead=4, num_layers=2, ff_dim=256, dropout=0.1, **common
        )

    elif model_type == "vae":
        return VAE(embed_dim=64, hidden_dim=256, latent_dim=64, beta=1.0, **common)

    elif model_type == "gan":
        return GANGenerator(
            latent_dim=64, embed_dim=64, hidden_dim=256, **common
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _load_model(model_type: str, tokenizer: DateTokenizer, device: torch.device):
    """Build model and load weights from disk."""
    weight_path = WEIGHTS_DIR / f"{model_type}_best.pt"
    if not weight_path.exists():
        return None, None

    model = _build_model(model_type, tokenizer)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device).eval()
    return model, model_type


@torch.no_grad()
def predict_batch(
    model,
    model_type: str,
    conditions: torch.Tensor,
    tokenizer: DateTokenizer,
) -> list[str]:
    """Run a single batch and return decoded date strings."""
    if model_type in ("cnn", "transformer"):
        dd_logits, mm_logits, yyyy_logits = model(conditions)
        dd_idx   = dd_logits.argmax(dim=-1)
        mm_idx   = mm_logits.argmax(dim=-1)
        yyyy_idx = yyyy_logits.argmax(dim=-1)
    else:
        # vae and gan both expose .predict()
        dd_idx, mm_idx, yyyy_idx = model.predict(conditions)

    results = []
    for i in range(conditions.size(0)):
        date_str = tokenizer.decode_date(
            dd_idx[i].item(), mm_idx[i].item(), yyyy_idx[i].item()
        )
        results.append(date_str)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Date Generator – Inference")
    parser.add_argument("-i", "--input",  required=True,
                        help="Path to input conditions file")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to write predictions")
    parser.add_argument("--model", type=str, default=None,
                        choices=["cnn", "transformer", "vae", "gan"],
                        help="Which model to use (auto-selects best available if omitted)")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer = DateTokenizer()

    # Load model
    if args.model is not None:
        model, model_type = _load_model(args.model, tokenizer, device)
        if model is None:
            print(f"ERROR: No weights found for model '{args.model}' "
                  f"at {WEIGHTS_DIR / (args.model + '_best.pt')}")
            sys.exit(1)
    else:
        model = model_type = None
        for mt in MODEL_PRIORITY:
            model, model_type = _load_model(mt, tokenizer, device)
            if model is not None:
                break
        if model is None:
            print("ERROR: No trained weights found. "
                  "Please run train.py first for at least one model.")
            sys.exit(1)

    print(f"Using model: {model_type} | Device: {device}")

    # Load input
    inf_dataset = InferenceDataset(args.input, tokenizer)
    print(f"Running inference on {len(inf_dataset)} samples …")

    # Predict
    all_conditions = [inf_dataset[i][0] for i in range(len(inf_dataset))]
    all_raw        = [inf_dataset[i][1] for i in range(len(inf_dataset))]

    predictions: list[str] = []
    for start in range(0, len(all_conditions), args.batch_size):
        batch_cond = torch.stack(all_conditions[start:start + args.batch_size]).to(device)
        preds = predict_batch(model, model_type, batch_cond, tokenizer)
        predictions.extend(preds)

    # Write output (format: conditions + predicted date, matching data.txt)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for raw_line, date_str in zip(all_raw, predictions):
            f.write(f"{raw_line} {date_str}\n")

    # Quick summary: how many predictions satisfy all conditions?
    satisfied = 0
    for raw_line, date_str in zip(all_raw, predictions):
        parts = raw_line.strip().split()
        day, month, leap, decade = (parts[0].strip("[]"), parts[1].strip("[]"),
                                    parts[2].strip("[]"), parts[3].strip("[]"))
        if all_conditions_met(date_str, day, month, leap, decade):
            satisfied += 1

    total = len(predictions)
    print(f"Output written to: {args.output}")
    print(f"Condition satisfaction: {satisfied}/{total} ({100*satisfied/total:.1f}%)")


if __name__ == "__main__":
    main()
