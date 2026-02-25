#!/usr/bin/env python3
"""
Train a decoder-only Transformer LM on windowed token sequences.

This expects `data_windows/` created by make_windows.py.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependency: numpy. Install with: pip install -r requirements.txt") from e
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lm_model import GPT, GPTConfig
from tokenizer import MidiEventTokenizer, TokenizerConfig
from window_dataset import NpyWindowDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_windows_config(data_windows: Path) -> dict:
    cfg_path = data_windows / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"Missing {cfg_path} (did you run make_windows.py?)")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    idxs = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    n_val = int(round(n * float(val_fraction)))
    val = idxs[:n_val]
    train = idxs[n_val:]
    return train, val


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: GPT,
    optim: torch.optim.Optimizer,
    cfg: GPTConfig,
    tokenizer_cfg: TokenizerConfig,
    best_val_loss: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict(),
        "gpt_config": asdict(cfg),
        "tokenizer_config": asdict(tokenizer_cfg),
        "best_val_loss": best_val_loss,
    }
    torch.save(payload, str(path))


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: torch.device, pad_id: int = 0) -> float:
    model.eval()
    losses: list[float] = []
    for batch in loader:
        x = batch.to(device, non_blocking=True)
        logits = model(x, pad_id=pad_id)
        # Next-token prediction: shift by 1.
        targets = x[:, 1:].contiguous()
        pred = logits[:, :-1, :].contiguous()
        # Ignore PAD targets.
        targets_masked = targets.clone()
        targets_masked[targets_masked.eq(int(pad_id))] = -100
        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), targets_masked.view(-1), ignore_index=-100)
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def main() -> None:
    p = argparse.ArgumentParser(description="Train a Transformer LM on token windows.")
    p.add_argument("data_windows", help="Folder created by make_windows.py")
    p.add_argument("--out", default="checkpoints", help="Checkpoint output folder.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-fraction", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--steps", type=int, default=10_000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Model size knobs
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)

    # Device
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(int(args.seed))
    device = torch.device(str(args.device))

    data_windows = Path(args.data_windows)
    windows_cfg = load_windows_config(data_windows)

    tokenizer_cfg_dict = windows_cfg["tokenizer_config"]
    if isinstance(tokenizer_cfg_dict.get("ts_denominators"), list):
        tokenizer_cfg_dict["ts_denominators"] = tuple(tokenizer_cfg_dict["ts_denominators"])
    tokenizer_cfg = TokenizerConfig(**tokenizer_cfg_dict)
    tokenizer = MidiEventTokenizer(tokenizer_cfg)
    vocab_size = int(windows_cfg["vocab_size"])
    seq_len = int(windows_cfg["seq_len"])
    if int(tokenizer.vocab_size) != vocab_size:
        raise SystemExit(f"vocab mismatch: data says {vocab_size}, tokenizer says {tokenizer.vocab_size}")

    ds = NpyWindowDataset(data_windows / "manifest.jsonl")
    train_idx, val_idx = split_indices(len(ds), float(args.val_fraction), int(args.seed))
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx) if val_idx else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        if val_ds is not None
        else None
    )

    cfg = GPTConfig(
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        d_model=int(args.d_model),
        n_heads=int(args.n_heads),
        n_layers=int(args.n_layers),
        dropout=float(args.dropout),
    )
    model = GPT(cfg).to(device)

    optim = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        betas=(0.9, 0.95),
    )

    out_dir = Path(args.out)
    best_val_loss: float | None = None

    step = 0
    it = iter(train_loader)
    while step < int(args.steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)

        x = batch.to(device, non_blocking=True)
        logits = model(x, pad_id=tokenizer.PAD)
        targets = x[:, 1:].contiguous()
        pred = logits[:, :-1, :].contiguous()
        targets_masked = targets.clone()
        targets_masked[targets_masked.eq(int(tokenizer.PAD))] = -100
        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), targets_masked.view(-1), ignore_index=-100)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        optim.step()

        step += 1
        if step % 50 == 0:
            print(f"step {step} loss {loss.item():.4f}")

        if val_loader is not None and step % int(args.eval_every) == 0:
            val_loss = evaluate(model, val_loader, device, pad_id=tokenizer.PAD)
            print(f"eval step {step} val_loss {val_loss:.4f}")
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                save_checkpoint(
                    out_dir / "best.pt",
                    step=step,
                    model=model,
                    optim=optim,
                    cfg=cfg,
                    tokenizer_cfg=tokenizer_cfg,
                    best_val_loss=best_val_loss,
                )

        if step % int(args.save_every) == 0:
            save_checkpoint(
                out_dir / "latest.pt",
                step=step,
                model=model,
                optim=optim,
                cfg=cfg,
                tokenizer_cfg=tokenizer_cfg,
                best_val_loss=best_val_loss,
            )

    save_checkpoint(
        out_dir / "latest.pt",
        step=step,
        model=model,
        optim=optim,
        cfg=cfg,
        tokenizer_cfg=tokenizer_cfg,
        best_val_loss=best_val_loss,
    )
    print(f"Done. Saved: {out_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
