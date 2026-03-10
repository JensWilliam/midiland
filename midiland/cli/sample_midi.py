#!/usr/bin/env python3
"""
Sample a trained LM checkpoint and write the result to a MIDI file.

This is the practical "use the model" path:
checkpoint -> sampled token ids -> canonical events -> MIDI
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependency: numpy. Install with: pip install -r requirements.txt") from e
import torch

from midiland.cli.sample_lm import _load_checkpoint, generate
from midiland.lm_model import GPT, GPTConfig
from midiland.midi_io import canonical_events_to_midi
from midiland.tokenizer import MidiEventTokenizer, TokenizerConfig


def _build_model_and_tokenizer(checkpoint_path: Path, device: torch.device) -> tuple[GPT, MidiEventTokenizer]:
    ckpt = _load_checkpoint(checkpoint_path)
    gpt_cfg = GPTConfig(**ckpt["gpt_config"])
    tok_cfg_dict = ckpt["tokenizer_config"]
    if isinstance(tok_cfg_dict.get("ts_denominators"), list):
        tok_cfg_dict["ts_denominators"] = tuple(tok_cfg_dict["ts_denominators"])
    tok_cfg = TokenizerConfig(**tok_cfg_dict)

    tokenizer = MidiEventTokenizer(tok_cfg)
    model = GPT(gpt_cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Sample a trained LM checkpoint and write a MIDI file.")
    p.add_argument("checkpoint", help="Path to checkpoint .pt (e.g. checkpoints/best.pt).")
    p.add_argument("out_mid", help="Output MIDI path.")
    p.add_argument("--max-new", type=int, default=512, help="Maximum new sampled tokens.")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--ticks-per-beat",
        type=int,
        default=480,
        help="Ticks per beat for the written MIDI file (default: 480).",
    )
    p.add_argument(
        "--attempts",
        type=int,
        default=4,
        help="How many sampling attempts to try before giving up if decode fails.",
    )
    p.add_argument(
        "--print-tokens",
        type=int,
        default=120,
        help="Print the first N generated tokens for inspection.",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-npy", help="Optional output .npy path to save the successful token ids.")
    args = p.parse_args()

    checkpoint = Path(args.checkpoint)
    out_mid = Path(args.out_mid)
    device = torch.device(str(args.device))

    model, tokenizer = _build_model_and_tokenizer(checkpoint, device)

    last_error: Exception | None = None
    for attempt in range(1, max(1, int(args.attempts)) + 1):
        ids = generate(
            model,
            [tokenizer.BOS],
            max_new_tokens=int(args.max_new),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            device=device,
        )
        try:
            events = tokenizer.decode(ids)
        except Exception as e:  # noqa: BLE001
            last_error = e
            print(f"Attempt {attempt}/{int(args.attempts)} failed to decode: {type(e).__name__}: {e}")
            continue

        out_mid.parent.mkdir(parents=True, exist_ok=True)
        canonical_events_to_midi(
            events,
            cfg=tokenizer.config,
            ticks_per_beat=int(args.ticks_per_beat),
            out_path=out_mid,
        )

        if int(args.print_tokens) > 0:
            preview = ids[: int(args.print_tokens)]
            print(" ".join(tokenizer.token_to_str(t) for t in preview))
            print()

        print(f"Wrote: {out_mid}")
        print(f"Generated tokens: {len(ids)}")
        print(f"Decoded events: {len(events)}")

        if args.save_npy:
            npy_path = Path(args.save_npy)
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, np.asarray(ids, dtype=np.int32), allow_pickle=False)
            print(f"Saved tokens: {npy_path}")
        return

    raise SystemExit(
        f"Failed to decode a sampled sequence after {int(args.attempts)} attempts. "
        f"Last error: {type(last_error).__name__}: {last_error}"
    )


if __name__ == "__main__":
    main()
