#!/usr/bin/env python3
"""
Preprocess a folder of .mid files into per-file token arrays (.npy).

Outputs:
  out_dir/
    config.json          # frozen tokenizer config used for this export
    manifest.jsonl       # one line per MIDI: paths, lengths, etc.
    tokens/<...>.npy     # tokens for each MIDI (dtype configurable; default uint16)

This script does NOT do windowing. It only creates "documents" (one token sequence per MIDI).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as e:
    raise SystemExit("Missing dependency: numpy. Install with: pip install -r requirements.txt") from e

from midi_io import midi_to_canonical_events
from tokenizer import MidiEventTokenizer, TokenizerConfig


def _iter_mid_files(root: Path) -> list[Path]:
    exts = {".mid", ".midi"}
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            files.append(path)
    files.sort()
    return files


def _safe_relpath(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return Path(path.name)


def _dtype_from_name(name: str) -> np.dtype:
    name = name.lower().strip()
    if name in ("u16", "uint16"):
        return np.dtype(np.uint16)
    if name in ("i32", "int32"):
        return np.dtype(np.int32)
    if name in ("i64", "int64"):
        return np.dtype(np.int64)
    raise ValueError("dtype must be one of: uint16, int32, int64")


def main() -> None:
    p = argparse.ArgumentParser(description="Tokenize a folder of MIDI files into .npy token arrays.")
    p.add_argument("input_dir", help="Folder containing .mid/.midi files (recursive).")
    p.add_argument("out_dir", help="Output folder.")
    p.add_argument("--steps-per-beat", type=int, default=8)
    p.add_argument("--dtype", default="uint16", help="Token dtype: uint16 (default), int32, int64.")
    p.add_argument("--keep-drums", action="store_true", help="Do not filter MIDI channel 10.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing token files.")
    p.add_argument("--limit", type=int, help="Only process first N MIDI files.")
    p.add_argument("--print-every", type=int, default=200, help="Progress logging interval.")
    args = p.parse_args()

    input_root = Path(args.input_dir)
    out_root = Path(args.out_dir)
    tokens_root = out_root / "tokens"

    if not input_root.exists():
        raise SystemExit(f"Input folder does not exist: {input_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    tokens_root.mkdir(parents=True, exist_ok=True)

    cfg = TokenizerConfig(steps_per_beat=int(args.steps_per_beat))
    tok = MidiEventTokenizer(cfg)
    dtype = _dtype_from_name(str(args.dtype))
    if dtype == np.dtype(np.uint16) and int(tok.vocab_size) >= 2**16:
        raise SystemExit(
            f"Tokenizer vocab_size={tok.vocab_size} does not fit in uint16; use --dtype int32."
        )

    # Freeze config used for this export.
    (out_root / "config.json").write_text(
        json.dumps(
            {
                "tokenizer_config": asdict(cfg),
                "vocab_size": int(tok.vocab_size),
                "dtype": str(dtype),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest_path = out_root / "manifest.jsonl"
    midi_files = _iter_mid_files(input_root)
    if args.limit is not None:
        midi_files = midi_files[: int(args.limit)]

    processed = 0
    skipped = 0
    failed = 0
    lengths: list[int] = []

    with manifest_path.open("w", encoding="utf-8") as mf:
        for idx, midi_path in enumerate(midi_files, start=1):
            rel = _safe_relpath(midi_path, input_root)
            out_path = (tokens_root / rel).with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists() and not bool(args.overwrite):
                skipped += 1
                continue

            try:
                events, ticks_per_beat = midi_to_canonical_events(
                    midi_path,
                    cfg=cfg,
                    ignore_drums=not bool(args.keep_drums),
                )
                token_ids = tok.encode(events)
                arr = np.asarray(token_ids, dtype=dtype)
                np.save(out_path, arr, allow_pickle=False)

                rec = {
                    "source": str(midi_path),
                    "rel_source": str(rel),
                    "tokens": str(out_path),
                    "length": int(arr.shape[0]),
                    "ticks_per_beat": int(ticks_per_beat),
                }
                mf.write(json.dumps(rec) + "\n")

                processed += 1
                lengths.append(int(arr.shape[0]))
            except ModuleNotFoundError as e:
                msg = str(e)
                if "mido" in msg.lower():
                    raise SystemExit("Missing dependency: mido. Install with: pip install mido") from e
                raise
            except Exception as e:  # noqa: BLE001
                failed += 1
                rec = {
                    "source": str(midi_path),
                    "rel_source": str(rel),
                    "error": f"{type(e).__name__}: {e}",
                }
                mf.write(json.dumps(rec) + "\n")

            if int(args.print_every) > 0 and idx % int(args.print_every) == 0:
                avg = (sum(lengths) / len(lengths)) if lengths else 0.0
                print(
                    f"[{idx}/{len(midi_files)}] processed={processed} skipped={skipped} failed={failed} "
                    f"avg_len={avg:.1f}"
                )

    stats = {
        "input_dir": str(input_root),
        "out_dir": str(out_root),
        "files_total": int(len(midi_files)),
        "processed": int(processed),
        "skipped": int(skipped),
        "failed": int(failed),
        "length_min": int(min(lengths)) if lengths else None,
        "length_max": int(max(lengths)) if lengths else None,
        "length_mean": (sum(lengths) / len(lengths)) if lengths else None,
        "length_p95": float(np.percentile(np.asarray(lengths), 95)) if lengths else None,
    }
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Done. processed={processed} skipped={skipped} failed={failed}")
    print(f"Wrote: {manifest_path}")


if __name__ == "__main__":
    main()
