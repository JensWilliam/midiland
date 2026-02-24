#!/usr/bin/env python3
"""
Convert per-document token arrays into fixed-length training windows.

Input:
  - a `manifest.jsonl` produced by `preprocess_dataset.py`
  - each line points at a per-MIDI `.npy` token array

Output:
  out_dir/
    windows/**/*.npy     # each is shape (seq_len,), padded with PAD=0
    manifest.jsonl       # one line per window: source + offsets + lengths
    stats.json           # summary stats

Rules:
  - windows never cross document (MIDI) boundaries
  - window *start and end* positions are aligned to "safe boundaries"
    (we don't cut inside the argument tokens of EV_* events)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from tokenizer import MidiEventTokenizer, TokenizerConfig


ARG_TOKEN_TYPES = {
    "ch",
    "pitch",
    "dur",
    "vel",
    "prog",
    "bpm",
    "ts_num",
    "ts_den",
}

SAFE_START_TYPES = {
    "pad",
    "bos",
    "eos",
    "time_shift",
    "time_shift_coarse",
    "ev_note",
    "ev_prog",
    "ev_tempo",
    "ev_ts",
    "bar",
    "pos",
}


def _load_cfg(config_path: Path) -> tuple[TokenizerConfig, int, str]:
    obj = json.loads(config_path.read_text(encoding="utf-8"))
    cfg_dict = obj["tokenizer_config"]
    # json has lists; TokenizerConfig expects tuples.
    if "ts_denominators" in cfg_dict and isinstance(cfg_dict["ts_denominators"], list):
        cfg_dict["ts_denominators"] = tuple(cfg_dict["ts_denominators"])
    cfg = TokenizerConfig(**cfg_dict)
    return cfg, int(obj["vocab_size"]), str(obj.get("dtype", "uint16"))


def _iter_docs(manifest_path: Path) -> list[dict]:
    docs: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "error" in obj:
                continue
            docs.append(obj)
    return docs


def _compute_next_safe(tokenizer: MidiEventTokenizer, tokens: np.ndarray) -> list[int]:
    n = int(tokens.shape[0])
    safe = [False] * (n + 1)
    safe[n] = True
    for i in range(n):
        ttype = tokenizer.token_type(int(tokens[i]))
        safe[i] = ttype in SAFE_START_TYPES
    next_safe = [n] * (n + 1)
    next_idx = n
    for i in range(n, -1, -1):
        if safe[i]:
            next_idx = i
        next_safe[i] = next_idx
    return next_safe


def _compute_prev_safe(tokenizer: MidiEventTokenizer, tokens: np.ndarray) -> list[int]:
    n = int(tokens.shape[0])
    safe = [False] * (n + 1)
    safe[n] = True
    for i in range(n):
        ttype = tokenizer.token_type(int(tokens[i]))
        safe[i] = ttype in SAFE_START_TYPES
    prev_safe = [0] * (n + 1)
    prev_idx = 0
    for i in range(n + 1):
        if safe[i]:
            prev_idx = i
        prev_safe[i] = prev_idx
    return prev_safe


def main() -> None:
    p = argparse.ArgumentParser(description="Create fixed-length token windows from per-doc token arrays.")
    p.add_argument("tokens_root", help="The output folder from preprocess_dataset.py (contains config.json).")
    p.add_argument("out_dir", help="Output folder for windows.")
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument(
        "--stride",
        type=int,
        help="Token stride between window starts (default: seq_len//2).",
    )
    p.add_argument(
        "--min-len",
        type=int,
        default=256,
        help="Minimum unpadded tokens to keep a window (default: 256).",
    )
    p.add_argument("--overwrite", action="store_true", help="Overwrite out_dir if it exists.")
    p.add_argument("--limit-docs", type=int, help="Only process first N documents.")
    p.add_argument("--print-every", type=int, default=200, help="Progress logging interval.")
    args = p.parse_args()

    tokens_root = Path(args.tokens_root)
    config_path = tokens_root / "config.json"
    manifest_path = tokens_root / "manifest.jsonl"
    if not config_path.exists():
        raise SystemExit(f"Missing {config_path} (did you run preprocess_dataset.py?)")
    if not manifest_path.exists():
        raise SystemExit(f"Missing {manifest_path} (did you run preprocess_dataset.py?)")

    out_root = Path(args.out_dir)
    out_windows = out_root / "windows"
    if out_root.exists() and bool(args.overwrite):
        # remove only our expected outputs, not arbitrary paths
        for path in (out_windows, out_root / "manifest.jsonl", out_root / "stats.json", out_root / "config.json"):
            if path.is_dir():
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_file() or child.is_symlink():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                path.rmdir()
            elif path.exists():
                path.unlink()

    out_root.mkdir(parents=True, exist_ok=True)
    out_windows.mkdir(parents=True, exist_ok=True)

    seq_len = int(args.seq_len)
    if seq_len <= 0:
        raise SystemExit("--seq-len must be > 0")
    stride = int(args.stride) if args.stride is not None else max(1, seq_len // 2)
    if stride <= 0:
        raise SystemExit("--stride must be > 0")
    min_len = int(args.min_len)
    if min_len < 0:
        min_len = 0

    cfg, vocab_size, dtype_name = _load_cfg(config_path)
    tok = MidiEventTokenizer(cfg)
    if int(tok.vocab_size) != int(vocab_size):
        raise SystemExit(
            f"Tokenizer vocab mismatch: config.json says {vocab_size}, tokenizer computed {tok.vocab_size}"
        )

    # Save config used for windows too (helps keep things reproducible).
    (out_root / "config.json").write_text(
        json.dumps(
            {
                "source_tokens_root": str(tokens_root),
                "tokenizer_config": asdict(cfg),
                "vocab_size": int(tok.vocab_size),
                "seq_len": int(seq_len),
                "stride": int(stride),
                "min_len": int(min_len),
                "dtype": dtype_name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    docs = _iter_docs(manifest_path)
    if args.limit_docs is not None:
        docs = docs[: int(args.limit_docs)]

    out_manifest = out_root / "manifest.jsonl"

    total_windows = 0
    total_docs = 0
    total_tokens_unpadded = 0
    max_doc_len = 0

    with out_manifest.open("w", encoding="utf-8") as mf:
        for doc_idx, doc in enumerate(docs, start=1):
            doc_tokens_path = Path(doc["tokens"])
            if not doc_tokens_path.exists():
                # skip missing token arrays
                continue

            tokens = np.load(doc_tokens_path, allow_pickle=False)
            if tokens.ndim != 1:
                continue
            n = int(tokens.shape[0])
            if n <= 0:
                continue

            max_doc_len = max(max_doc_len, n)
            total_docs += 1

            next_safe = _compute_next_safe(tok, tokens)
            prev_safe = _compute_prev_safe(tok, tokens)

            start = next_safe[0]
            win_i = 0
            while start < n:
                desired_end = min(n, start + seq_len)
                end = prev_safe[desired_end]
                if end <= start:
                    # If we're somehow not on a safe boundary, jump forward.
                    start = next_safe[min(n, start + 1)]
                    continue

                length = int(end - start)
                if length < min_len:
                    break

                window = np.full((seq_len,), 0, dtype=tokens.dtype)  # PAD=0
                window[:length] = tokens[start:end]

                rel_source = doc.get("rel_source") or doc.get("relpath") or ""
                rel_source = str(rel_source)
                src_stem = Path(rel_source).with_suffix("").name or "doc"
                src_parent = Path(rel_source).with_suffix("").parent

                out_path = out_windows / src_parent / f"{src_stem}__w{win_i:05d}.npy"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, window, allow_pickle=False)

                mf.write(
                    json.dumps(
                        {
                            "source": doc.get("source"),
                            "rel_source": rel_source,
                            "source_tokens": str(doc_tokens_path),
                            "window_tokens": str(out_path),
                            "doc_length": int(n),
                            "start": int(start),
                            "end": int(end),
                            "length": int(length),
                        }
                    )
                    + "\n"
                )

                total_windows += 1
                total_tokens_unpadded += length
                win_i += 1

                desired_next = min(n, start + stride)
                start = next_safe[desired_next]

            if int(args.print_every) > 0 and doc_idx % int(args.print_every) == 0:
                avg_len = (total_tokens_unpadded / total_windows) if total_windows else 0.0
                print(
                    f"[{doc_idx}/{len(docs)}] docs={total_docs} windows={total_windows} avg_win_len={avg_len:.1f}"
                )

    stats = {
        "source_tokens_root": str(tokens_root),
        "out_dir": str(out_root),
        "docs_total": int(len(docs)),
        "docs_processed": int(total_docs),
        "windows_total": int(total_windows),
        "seq_len": int(seq_len),
        "stride": int(stride),
        "min_len": int(min_len),
        "avg_window_unpadded_len": (total_tokens_unpadded / total_windows) if total_windows else None,
        "max_doc_len": int(max_doc_len),
    }
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Done. docs={total_docs} windows={total_windows}")
    print(f"Wrote: {out_manifest}")


if __name__ == "__main__":
    main()

