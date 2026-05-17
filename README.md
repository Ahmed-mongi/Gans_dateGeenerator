# Dates Generator – DSAI 490 Assignment 2

## Project structure

```
dates_generator/
├── data/
│   ├── data.txt              ← full dataset (146 461 samples)
│   └── example_input.txt     ← inference example (conditions only)
├── model/
│   ├── gan_model.py          ← Model 1 (in-course):  Conditional GAN
│   ├── vae_model.py          ← Model 2 (in-course):  Conditional VAE
│   ├── cnn_model.py          ← Model 3 (out-of-course): 1D CNN
│   ├── transformer_model.py  ← Model 4 (out-of-course): Transformer Encoder
│   ├── train.py              ← unified training script
│   ├── predict.py            ← inference entry point (as per spec)
│   ├── evaluate.py           ← per-condition evaluation + examples
│   └── weights/              ← saved model checkpoints (created after training)
├── utils/
│   ├── tokenizer.py          ← custom tokenizer (input + output)
│   ├── dataset.py            ← PyTorch Dataset classes
│   └── validation.py         ← condition-satisfaction checker
├── environment.yml           ← conda environment spec
└── README.md
```

## Models

| # | Model | Type | File |
|---|-------|------|------|
| 1 | Conditional GAN | In-course (required) | `gan_model.py` |
| 2 | Conditional VAE | In-course | `vae_model.py` |
| 3 | 1D CNN | Out-of-course | `cnn_model.py` |
| 4 | Transformer Encoder | Out-of-course | `transformer_model.py` |

## Setup

```bash
conda env create -f environment.yml
conda activate dates_generator
```

## Training

Run from the project root (`dates_generator/`):

```bash
# Model 1 – GAN (in-course, required)
python model/train.py --model gan --epochs 80 --batch_size 512

# Model 2 – VAE (in-course)
python model/train.py --model vae --epochs 60 --batch_size 512

# Model 3 – CNN (out-of-course)
python model/train.py --model cnn --epochs 60 --batch_size 512

# Model 4 – Transformer (out-of-course)
python model/train.py --model transformer --epochs 60 --batch_size 512
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 512 | Mini-batch size |
| `--lr` | 1e-3 | Learning rate (CNN / Transformer / VAE) |
| `--lr_g` | 2e-4 | Generator learning rate (GAN) |
| `--lr_d` | 2e-4 | Discriminator learning rate (GAN) |
| `--embed_dim` | 64 | Condition embedding dimension |
| `--hidden_dim` | 256 | Hidden layer width / num filters (CNN) |
| `--latent_dim` | 64 | Latent space size (GAN / VAE) |
| `--beta` | 1.0 | KL weight for VAE (β-VAE) |
| `--aux_weight` | 1.0 | Auxiliary CE loss weight for GAN generator |
| `--seed` | 42 | Random seed for reproducibility |
| `--val_split` | 0.1 | Fraction of data held out for validation |
| `--eval_every` | 5 | Evaluate satisfaction rate every N epochs |

## Inference (as per assignment spec)

```bash
python model/predict.py -i data/example_input.txt -o predictions.txt
```

Override which model to use:

```bash
python model/predict.py -i data/example_input.txt -o predictions.txt --model gan
python model/predict.py -i data/example_input.txt -o predictions.txt --model vae
python model/predict.py -i data/example_input.txt -o predictions.txt --model cnn
python model/predict.py -i data/example_input.txt -o predictions.txt --model transformer
```

Output format matches `data.txt` exactly: each line is `[conditions] predicted_date`.

## Evaluation

```bash
python model/evaluate.py --model gan
python model/evaluate.py --model vae
python model/evaluate.py --model cnn
python model/evaluate.py --model transformer
```

Prints per-condition satisfaction rates and sample successes/failures,
and saves a report to `model/weights/<model>_eval.txt`.

## Evaluation metric

Because multiple dates satisfy a given set of conditions, accuracy against
a single ground-truth date is a poor metric. Instead we measure the
**condition satisfaction rate**: the fraction of predictions where **all
four conditions** (day-of-week, month, leap-year, decade) are met by the
predicted date.

Loss curves (train loss vs epoch) are saved as `model/weights/<model>_loss_log.txt`.

## Notes

- Date range: 1-1-1800 to 31-12-2200 (as required).
- Leap-year rule: standard Gregorian (÷4 yes, ÷100 no, ÷400 yes).
- All models are seeded with `--seed 42` by default for reproducibility.
- All code is PyTorch (no TensorFlow).
