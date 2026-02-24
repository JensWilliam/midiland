#!/usr/bin/env python3
"""
Round-trip test: MIDI -> (quantized+binned events) -> tokens -> events -> MIDI.

This is meant as a "does my representation actually work?" sanity check.

Important idea: the representation is *lossy* (quantization + velocity bins + tempo bins),
so the right notion of "strict" is:
  canonical(midi_in) == canonical(midi_out)

Where canonical() means "parse MIDI, quantize to steps, and bin values according to tokenizer config".
"""

from __future__ import annotations

import argparse
from pathlib import Path

from midi_io import canonical_events_to_midi, diff_events, midi_to_canonical_events
from tokenizer import MidiEventTokenizer, TokenizerConfig


def main() -> None:
    p = argparse.ArgumentParser(description="Round-trip test MIDI <-> tokens <-> MIDI.")
    p.add_argument("input_mid")
    p.add_argument(
        "--out",
        help="Optional output .mid path (default: input filename + _out).",
    )
    p.add_argument("--steps-per-beat", type=int, default=4)
    p.add_argument("--keep-drums", action="store_true", help="Do not filter MIDI channel 10.")
    p.add_argument("--print-tokens", type=int, default=80)
    args = p.parse_args()

    cfg = TokenizerConfig(steps_per_beat=int(args.steps_per_beat))
    tok = MidiEventTokenizer(cfg)

    input_path = Path(args.input_mid)
    if args.out:
        output_path = Path(args.out)
    else:
        suffix = input_path.suffix if input_path.suffix else ".mid"
        output_path = input_path.with_name(f"{input_path.stem}_out{suffix}")

    try:
        events1, tpb = midi_to_canonical_events(
            input_path, cfg=cfg, ignore_drums=not bool(args.keep_drums)
        )
    except ModuleNotFoundError as e:
        msg = str(e)
        if "mido" in msg.lower():
            raise SystemExit("Missing dependency: mido. Install with: pip install mido") from e
        raise
    tokens = tok.encode(events1)  # type: ignore[arg-type]
    events2 = tok.decode(tokens)

    # tokenizer strictness check (should match exactly because decode(encode(canonical)) is deterministic)
    if list(events2) != list(events1):
        print("Tokenizer mismatch (unexpected):")
        print(diff_events(list(events1), list(events2)))
        raise SystemExit(2)

    # write MIDI, then canonicalize again and compare
    canonical_events_to_midi(events2, cfg=cfg, ticks_per_beat=tpb, out_path=output_path)
    events3, _tpb2 = midi_to_canonical_events(
        output_path, cfg=cfg, ignore_drums=not bool(args.keep_drums)
    )
    print(diff_events(list(events1), list(events3)))
    print(f"\nWrote: {output_path}")

    if args.print_tokens > 0:
        preview = tokens[: int(args.print_tokens)]
        print("\nToken preview:")
        print(" ".join(tok.token_to_str(t) for t in preview))


if __name__ == "__main__":
    main()
