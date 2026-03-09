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
  - optional: prepend a fixed-size "header" that restates state at the window start
    (time signature, tempo, program per channel, and starting bar/pos)
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

from midiland.tokenizer import MidiEventTokenizer, TokenizerConfig, bpm_to_bin


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
    p.add_argument(
        "--no-header",
        action="store_true",
        help="Disable state headers (by default, headers are prepended to every window).",
    )
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

    add_header = not bool(args.no_header)
    default_ts_num, default_ts_den = 4, 4
    default_tempo_bin = int(
        bpm_to_bin(
            120.0,
            min_bpm=int(cfg.tempo_min_bpm),
            max_bpm=int(cfg.tempo_max_bpm),
            bins=int(cfg.tempo_bins),
        )
    )

    def _header_tokens(
        *,
        cursor_steps: int,
        ts_num: int,
        ts_den: int,
        tempo_bin: int,
        programs: list[int],
    ) -> list[int]:
        # Fixed-size header. Keeping it fixed makes it easier for the model to learn.
        bar_steps = int(tok._bar_steps(int(ts_num), int(ts_den)))  # type: ignore[attr-defined]
        if bar_steps <= 0:
            bar_steps = int(tok._bar_steps(default_ts_num, default_ts_den))  # type: ignore[attr-defined]
        pos = int(cursor_steps % bar_steps)

        out: list[int] = []
        out.append(tok.BOS)

        out.append(tok._ev_ts_id)  # type: ignore[attr-defined]
        out.append(tok.ts_num_id(int(ts_num)))
        out.append(tok.ts_den_id(int(ts_den)))

        out.append(tok._ev_tempo_id)  # type: ignore[attr-defined]
        out.append(tok.bpm_id(int(tempo_bin)))

        for ch in range(16):
            out.append(tok._ev_prog_id)  # type: ignore[attr-defined]
            out.append(tok.ch_id(int(ch)))
            out.append(tok.prog_id(int(programs[ch])))

        out.append(tok._bar_id)  # type: ignore[attr-defined]
        out.append(tok.pos_id(int(pos)))
        return out

    header_len = (
        len(
            _header_tokens(
                cursor_steps=0,
                ts_num=default_ts_num,
                ts_den=default_ts_den,
                tempo_bin=default_tempo_bin,
                programs=[0] * 16,
            )
        )
        if add_header
        else 0
    )
    if header_len >= seq_len:
        raise SystemExit(f"Header length {header_len} >= seq_len {seq_len}; increase --seq-len.")

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
                "add_header": bool(add_header),
                "header_len": int(header_len),
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
    total_body_tokens = 0
    total_window_tokens = 0
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

            # Compute start indices first (aligned to safe boundaries, never cross documents).
            starts: list[int] = []
            start = int(next_safe[0])
            while start < n:
                starts.append(int(start))
                desired_next = min(n, start + stride)
                start = int(next_safe[desired_next])

            # For the header, we need the state at the start of each window.
            # Capture at `body_start` (if start token is BOS, skip it so we don't get BOS BOS).
            body_starts: list[int] = []
            for s in starts:
                bs = int(s)
                if bs < n and int(tokens[bs]) == int(tok.BOS):
                    bs = int(next_safe[min(n, bs + 1)])
                body_starts.append(bs)

            # Scan the document once and record state at each unique body_start index.
            need = sorted(set(body_starts))
            state_at: dict[int, tuple[int, int, int, int, list[int]]] = {}
            # value: (cursor_steps, ts_num, ts_den, tempo_bin, programs[16])

            cursor_steps = 0
            ts_num, ts_den = default_ts_num, default_ts_den
            tempo_bin = default_tempo_bin
            programs = [0] * 16

            need_i = 0
            next_need = need[need_i] if need else None

            i = 0
            while i <= n and next_need is not None:
                if i == next_need:
                    state_at[i] = (int(cursor_steps), int(ts_num), int(ts_den), int(tempo_bin), programs.copy())
                    need_i += 1
                    next_need = need[need_i] if need_i < len(need) else None
                    if next_need is None:
                        break
                if i >= n:
                    break

                tok_id = int(tokens[i])
                ttype = tok.token_type(tok_id)
                if ttype == "time_shift":
                    cursor_steps += int(tok.time_shift_from_id(tok_id))
                    i += 1
                    continue
                if ttype == "time_shift_coarse":
                    cursor_steps += int(tok.time_shift_coarse_from_id(tok_id)) * int(cfg.coarse_time_shift_unit)
                    i += 1
                    continue
                if ttype == "ev_ts" and i + 2 < n:
                    ts_num = int(tok.ts_num_from_id(int(tokens[i + 1])))
                    ts_den = int(tok.ts_den_from_id(int(tokens[i + 2])))
                    i += 3
                    continue
                if ttype == "ev_tempo" and i + 1 < n:
                    tempo_bin = int(tok.bpm_bin_from_id(int(tokens[i + 1])))
                    i += 2
                    continue
                if ttype == "ev_prog" and i + 2 < n:
                    ch = int(tok.ch_from_id(int(tokens[i + 1])))
                    prog = int(tok.prog_from_id(int(tokens[i + 2])))
                    if 0 <= ch < 16:
                        programs[ch] = prog
                    i += 3
                    continue
                i += 1

            win_i = 0
            for s, bs in zip(starts, body_starts, strict=True):
                body_start = int(bs)
                if body_start >= n:
                    continue

                if add_header:
                    cur, tsn, tsd, tb, progs = state_at.get(
                        body_start,
                        (0, default_ts_num, default_ts_den, default_tempo_bin, [0] * 16),
                    )
                    header = _header_tokens(
                        cursor_steps=int(cur),
                        ts_num=int(tsn),
                        ts_den=int(tsd),
                        tempo_bin=int(tb),
                        programs=progs,
                    )
                else:
                    header = []

                available = int(seq_len - len(header))
                desired_end = min(n, body_start + available)
                body_end = int(prev_safe[desired_end])
                if body_end <= body_start:
                    continue

                body_len = int(body_end - body_start)
                if body_len < min_len:
                    continue

                window = np.full((seq_len,), 0, dtype=tokens.dtype)  # PAD=0
                if header:
                    window[: len(header)] = np.asarray(header, dtype=tokens.dtype)
                window[len(header) : len(header) + body_len] = tokens[body_start:body_end]
                length = int(len(header) + body_len)

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
                            "start": int(body_start),
                            "end": int(body_end),
                            "length": int(length),
                            "header_len": int(len(header)),
                            "body_len": int(body_len),
                        }
                    )
                    + "\n"
                )

                total_windows += 1
                total_body_tokens += body_len
                total_window_tokens += length
                win_i += 1

            if int(args.print_every) > 0 and doc_idx % int(args.print_every) == 0:
                avg_len = (total_window_tokens / total_windows) if total_windows else 0.0
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
        "add_header": bool(add_header),
        "header_len": int(header_len),
        "avg_body_len": (total_body_tokens / total_windows) if total_windows else None,
        "avg_window_unpadded_len": (total_window_tokens / total_windows) if total_windows else None,
        "max_doc_len": int(max_doc_len),
    }
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Done. docs={total_docs} windows={total_windows}")
    print(f"Wrote: {out_manifest}")


if __name__ == "__main__":
    main()
