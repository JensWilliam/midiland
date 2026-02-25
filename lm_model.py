#!/usr/bin/env python3
"""
Tiny decoder-only Transformer for next-token prediction over MIDI token IDs.

This is intentionally minimal and readable. It is NOT optimized.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class GPTConfig:
    vocab_size: int
    max_seq_len: int = 2048
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        if cfg.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0")
        if cfg.max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")

        self.cfg = cfg
        self.tok_emb = nn.Embedding(int(cfg.vocab_size), int(cfg.d_model))
        self.pos_emb = nn.Embedding(int(cfg.max_seq_len), int(cfg.d_model))
        self.drop = nn.Dropout(float(cfg.dropout))

        layer = nn.TransformerEncoderLayer(
            d_model=int(cfg.d_model),
            nhead=int(cfg.n_heads),
            dim_feedforward=int(cfg.d_model) * 4,
            dropout=float(cfg.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=int(cfg.n_layers))
        self.ln_f = nn.LayerNorm(int(cfg.d_model))
        self.lm_head = nn.Linear(int(cfg.d_model), int(cfg.vocab_size), bias=False)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, *, pad_id: int = 0) -> torch.Tensor:
        """
        Args:
          idx: LongTensor [B, T] token ids
        Returns:
          logits: FloatTensor [B, T, vocab_size]
        """
        if idx.ndim != 2:
            raise ValueError("idx must be [B, T]")
        bsz, t = idx.shape
        if t > int(self.cfg.max_seq_len):
            raise ValueError(f"Sequence length {t} > max_seq_len {self.cfg.max_seq_len}")

        pos = torch.arange(t, device=idx.device, dtype=torch.long).unsqueeze(0).expand(bsz, t)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)

        # Causal mask: prevent attention to future tokens.
        # Shape expected by TransformerEncoder with batch_first=True: [T, T]
        causal_mask = torch.triu(
            torch.ones((t, t), device=idx.device, dtype=torch.bool),
            diagonal=1,
        )
        key_padding_mask = idx.eq(int(pad_id))  # True where PAD

        x = self.blocks(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

