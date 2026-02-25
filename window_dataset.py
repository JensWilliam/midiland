#!/usr/bin/env python3
"""
Dataset utilities for `data_windows/` produced by make_windows.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependency: numpy. Install with: pip install -r requirements.txt") from e
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True, slots=True)
class WindowRecord:
    source: str | None
    rel_source: str
    source_tokens: str
    window_tokens: str
    doc_length: int
    start: int
    end: int
    length: int


def iter_window_records(manifest_path: str | Path) -> Iterator[WindowRecord]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            yield WindowRecord(
                source=obj.get("source"),
                rel_source=str(obj.get("rel_source", "")),
                source_tokens=str(obj["source_tokens"]),
                window_tokens=str(obj["window_tokens"]),
                doc_length=int(obj["doc_length"]),
                start=int(obj["start"]),
                end=int(obj["end"]),
                length=int(obj["length"]),
            )


class NpyWindowDataset(Dataset[torch.Tensor]):
    def __init__(self, windows_manifest: str | Path) -> None:
        self.records = list(iter_window_records(windows_manifest))
        if not self.records:
            raise ValueError(f"No window records found in {windows_manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rec = self.records[int(idx)]
        arr = np.load(rec.window_tokens, allow_pickle=False)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D token array in {rec.window_tokens}")
        return torch.from_numpy(arr.astype(np.int64, copy=False))
