"""
PyTorch Dataset for the Dates Generator task.
"""

from __future__ import annotations
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from utils.tokenizer import DateTokenizer


class DatesDataset(Dataset):
    """
    Loads data.txt and returns (conditions_tensor, dd_idx, mm_idx, yyyy_idx).

    conditions_tensor : LongTensor of shape (4,)  – [day, month, leap, decade]
    dd_idx            : int – day-of-month index in tokenizer.dd_vocab
    mm_idx            : int – month index in tokenizer.mm_vocab
    yyyy_idx          : int – year index in tokenizer.yyyy_vocab
    """

    def __init__(self, filepath: str, tokenizer: DateTokenizer) -> None:
        self.tokenizer = tokenizer
        self.samples: List[Tuple[List[int], int, int, int]] = []
        self._load(filepath)

    def _load(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                day, month, leap, decade, date = DateTokenizer.parse_line(line)
                if date is None:
                    continue  # skip inference-only lines
                conditions = self.tokenizer.encode_conditions(day, month, leap, decade)
                dd_idx, mm_idx, yyyy_idx = self.tokenizer.encode_date(date)
                self.samples.append((conditions, dd_idx, mm_idx, yyyy_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        conditions, dd_idx, mm_idx, yyyy_idx = self.samples[idx]
        return (
            torch.tensor(conditions, dtype=torch.long),
            torch.tensor(dd_idx,    dtype=torch.long),
            torch.tensor(mm_idx,    dtype=torch.long),
            torch.tensor(yyyy_idx,  dtype=torch.long),
        )


class InferenceDataset(Dataset):
    """
    Loads example_input.txt (no date column) for inference.
    Returns (conditions_tensor, raw_line).
    """

    def __init__(self, filepath: str, tokenizer: DateTokenizer) -> None:
        self.tokenizer = tokenizer
        self.raw_lines: List[str] = []
        self.conditions_list: List[List[int]] = []
        self._load(filepath)

    def _load(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                day, month, leap, decade, _ = DateTokenizer.parse_line(line)
                conditions = self.tokenizer.encode_conditions(day, month, leap, decade)
                self.raw_lines.append(line)
                self.conditions_list.append(conditions)

    def __len__(self) -> int:
        return len(self.raw_lines)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        conditions = torch.tensor(self.conditions_list[idx], dtype=torch.long)
        return conditions, self.raw_lines[idx]
