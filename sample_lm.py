#!/usr/bin/env python3
"""
Sample tokens from a trained LM checkpoint and print them as readable token strings.

This does NOT convert tokens back to MIDI yet. It's for quick sanity-checking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependency: numpy. Install with: pip install -r requirements.txt") from e
import torch
import torch.nn.functional as F

from lm_model import GPT, GPTConfig
from tokenizer import MidiEventTokenizer, TokenizerConfig


def _load_checkpoint(path: Path) -> dict:
    return torch.load(str(path), map_location="cpu")


def top_p_sample(probs: torch.Tensor, p: float) -> int:
    # probs: [V]
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=0)
    mask = cdf <= float(p)
    # keep at least 1 token
    if not bool(mask.any()):
        mask[0] = True
    keep_probs = sorted_probs[mask]
    keep_idx = sorted_idx[mask]
    keep_probs = keep_probs / keep_probs.sum()
    choice = torch.multinomial(keep_probs, 1).item()
    return int(keep_idx[choice].item())


@torch.no_grad()
def generate(
    model: GPT,
    prompt: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[int]:
    model.eval()
    ids = prompt[:]
    for _ in range(int(max_new_tokens)):
        x = torch.tensor([ids[-model.cfg.max_seq_len :]], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :]  # [1, V]
        logits = logits / max(1e-6, float(temperature))
        probs = F.softmax(logits.squeeze(0), dim=0)
        next_id = top_p_sample(probs, float(top_p))
        ids.append(int(next_id))
        if int(next_id) == 2:  # EOS
            break
    return ids


def main() -> None:
    p = argparse.ArgumentParser(description="Sample tokens from a trained LM checkpoint.")
    p.add_argument("checkpoint", help="Path to checkpoint .pt (e.g. checkpoints/best.pt).")
    p.add_argument("--max-new", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-npy", help="Optional output .npy path to save generated token ids.")
    args = p.parse_args()

    ckpt = _load_checkpoint(Path(args.checkpoint))
    gpt_cfg = GPTConfig(**ckpt["gpt_config"])
    tok_cfg_dict = ckpt["tokenizer_config"]
    if isinstance(tok_cfg_dict.get("ts_denominators"), list):
        tok_cfg_dict["ts_denominators"] = tuple(tok_cfg_dict["ts_denominators"])
    tok_cfg = TokenizerConfig(**tok_cfg_dict)
    tokenizer = MidiEventTokenizer(tok_cfg)

    model = GPT(gpt_cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    device = torch.device(str(args.device))
    model.to(device)

    prompt = [tokenizer.BOS]
    ids = generate(
        model,
        prompt,
        max_new_tokens=int(args.max_new),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        device=device,
    )

    print(" ".join(tokenizer.token_to_str(t) for t in ids[:300]))
    print(f"\nTotal tokens: {len(ids)}")

    if args.save_npy:
        out = Path(args.save_npy)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, np.asarray(ids, dtype=np.int32), allow_pickle=False)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
