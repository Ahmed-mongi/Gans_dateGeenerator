"""
Custom tokenizer for the dates generation task.
Handles encoding/decoding of input conditions and output date tokens.
"""

from __future__ import annotations
from typing import List, Tuple, Dict


# ── Vocabulary definitions ─────────────────────────────────────────────────────

DAY_TOKENS: List[str] = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
MONTH_TOKENS: List[str] = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                            "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
LEAP_TOKENS: List[str] = ["False", "True"]
# Decades from 180 (1800–1809) to 220 (2200)
DECADE_TOKENS: List[str] = [str(d) for d in range(180, 221)]

# Date output digits
DD_TOKENS: List[str] = [str(d) for d in range(1, 32)]    # day-of-month  1–31
MM_TOKENS: List[str] = [str(m) for m in range(1, 13)]    # month         1–12
# Year digits: 1800–2200 → encode as offset from 1800 (0–400)
YEAR_TOKENS: List[str] = [str(y) for y in range(1800, 2201)]

SPECIAL_TOKENS: List[str] = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]


class DateTokenizer:
    """
    Encodes input conditions into integer indices and decodes predicted
    integer sequences back into date strings (dd-mm-yyyy).

    Input sequence layout (fixed length = 4):
        [day_idx, month_idx, leap_idx, decade_idx]

    Output sequence layout (variable, teacher-forced during training):
        <SOS>  dd  mm  yyyy  <EOS>
    """

    def __init__(self) -> None:
        # ── Build input vocab ──────────────────────────────────────────────
        self.day2idx: Dict[str, int]    = {t: i for i, t in enumerate(DAY_TOKENS)}
        self.month2idx: Dict[str, int]  = {t: i for i, t in enumerate(MONTH_TOKENS)}
        self.leap2idx: Dict[str, int]   = {t: i for i, t in enumerate(LEAP_TOKENS)}
        self.decade2idx: Dict[str, int] = {t: i for i, t in enumerate(DECADE_TOKENS)}

        self.idx2day:    Dict[int, str] = {v: k for k, v in self.day2idx.items()}
        self.idx2month:  Dict[int, str] = {v: k for k, v in self.month2idx.items()}
        self.idx2leap:   Dict[int, str] = {v: k for k, v in self.leap2idx.items()}
        self.idx2decade: Dict[int, str] = {v: k for k, v in self.decade2idx.items()}

        # ── Build output vocab (dd / mm / yyyy each independent) ───────────
        self.dd_vocab   = SPECIAL_TOKENS + DD_TOKENS
        self.mm_vocab   = SPECIAL_TOKENS + MM_TOKENS
        self.yyyy_vocab = SPECIAL_TOKENS + YEAR_TOKENS

        self.dd2idx   = {t: i for i, t in enumerate(self.dd_vocab)}
        self.mm2idx   = {t: i for i, t in enumerate(self.mm_vocab)}
        self.yyyy2idx = {t: i for i, t in enumerate(self.yyyy_vocab)}

        self.idx2dd   = {v: k for k, v in self.dd2idx.items()}
        self.idx2mm   = {v: k for k, v in self.mm2idx.items()}
        self.idx2yyyy = {v: k for k, v in self.yyyy2idx.items()}

        # Sizes
        self.n_days    = len(DAY_TOKENS)
        self.n_months  = len(MONTH_TOKENS)
        self.n_leaps   = len(LEAP_TOKENS)
        self.n_decades = len(DECADE_TOKENS)

        self.dd_vocab_size   = len(self.dd_vocab)
        self.mm_vocab_size   = len(self.mm_vocab)
        self.yyyy_vocab_size = len(self.yyyy_vocab)

        # Special token indices (shared across dd/mm/yyyy vocabs since they
        # all start with the same SPECIAL_TOKENS prefix)
        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3

    # ── Input encoding ─────────────────────────────────────────────────────────

    def encode_conditions(self, day: str, month: str,
                          leap: str, decade: str) -> List[int]:
        """Return [day_idx, month_idx, leap_idx, decade_idx]."""
        return [
            self.day2idx.get(day, self.UNK_IDX),
            self.month2idx.get(month, self.UNK_IDX),
            self.leap2idx.get(leap, self.UNK_IDX),
            self.decade2idx.get(decade, self.UNK_IDX),
        ]

    # ── Output encoding ────────────────────────────────────────────────────────

    def encode_date(self, date_str: str) -> Tuple[int, int, int]:
        """
        Parse 'dd-mm-yyyy' and return (dd_idx, mm_idx, yyyy_idx)
        using each sub-vocab (including the SPECIAL_TOKENS offset).
        """
        parts = date_str.strip().split("-")
        dd, mm, yyyy = parts[0], parts[1], parts[2]
        return (
            self.dd2idx.get(dd, self.UNK_IDX),
            self.mm2idx.get(mm, self.UNK_IDX),
            self.yyyy2idx.get(yyyy, self.UNK_IDX),
        )

    def decode_date(self, dd_idx: int, mm_idx: int, yyyy_idx: int) -> str:
        """Convert predicted indices back to 'dd-mm-yyyy' string."""
        dd   = self.idx2dd.get(dd_idx, "?")
        mm   = self.idx2mm.get(mm_idx, "?")
        yyyy = self.idx2yyyy.get(yyyy_idx, "?")
        return f"{dd}-{mm}-{yyyy}"

    # ── Line parsing ───────────────────────────────────────────────────────────

    @staticmethod
    def parse_line(line: str) -> Tuple[str, str, str, str, str | None]:
        """
        Parse a data line into (day, month, leap, decade, date|None).
        Works for both training lines (with date) and inference lines (without).

        Example input:  '[WED] [JAN] [False] [180] 1-1-1800'
        Example output: ('WED', 'JAN', 'False', '180', '1-1-1800')
        """
        tokens = line.strip().split()
        # Tokens: [DAY] [MONTH] [LEAP] [DECADE] [optional: date]
        day    = tokens[0].strip("[]")
        month  = tokens[1].strip("[]")
        leap   = tokens[2].strip("[]")
        decade = tokens[3].strip("[]")
        date   = tokens[4] if len(tokens) > 4 else None
        return day, month, leap, decade, date
