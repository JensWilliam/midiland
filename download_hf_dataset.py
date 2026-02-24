#!/usr/bin/env python3
"""
Download a Hugging Face *dataset repo* that contains MIDI files.

This is intentionally simple: it downloads the dataset repository snapshot and then
copies/links any *.mid/*.midi files into a local folder.

Example:
  python download_hf_dataset.py drengskapur/midi-classical-music data_midi

If the repo is private, set an auth token via environment variables, e.g.:
  export HF_TOKEN=...
or put it in a local `.env` file (this script can read it).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ModuleNotFoundError:
    snapshot_download = None


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _get_token(token_env: str | None, explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    if token_env:
        return os.environ.get(token_env)
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_ACCESS_TOKEN")
    )


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _materialize_file(src: Path, dst: Path, *, mode: str) -> None:
    _ensure_parent(dst)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return
    raise ValueError("mode must be one of: copy, hardlink, symlink")


def main() -> None:
    p = argparse.ArgumentParser(description="Download MIDI files from a HF dataset repo snapshot.")
    p.add_argument("dataset_id", help="HF dataset repo id, e.g. drengskapur/midi-classical-music")
    p.add_argument("out_dir", help="Output folder (will create midis/ + manifest/stats).")
    p.add_argument("--revision", help="Optional git revision (branch/tag/commit).")
    p.add_argument(
        "--mode",
        choices=["hardlink", "copy", "symlink"],
        default="hardlink",
        help="How to place MIDI files into out_dir/midis (default: hardlink, falls back to copy).",
    )
    p.add_argument("--limit", type=int, help="Only materialize first N MIDI files.")
    p.add_argument(
        "--env-file",
        default=".env",
        help="Optional .env file to read for tokens (default: .env).",
    )
    p.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable name to read the HF token from (default: HF_TOKEN).",
    )
    p.add_argument("--token", help="Explicit HF token (not recommended; prefer env/.env).")
    args = p.parse_args()

    if snapshot_download is None:
        raise SystemExit("Missing dependency: huggingface_hub. Install with: pip install huggingface_hub")

    _load_dotenv(Path(args.env_file))
    token = _get_token(str(args.token_env) if args.token_env else None, args.token)

    out_root = Path(args.out_dir)
    out_midis = out_root / "midis"
    out_root.mkdir(parents=True, exist_ok=True)
    out_midis.mkdir(parents=True, exist_ok=True)

    allow_patterns = ["**/*.mid", "**/*.midi", "*.mid", "*.midi"]
    snapshot_path = Path(
        snapshot_download(
            repo_id=str(args.dataset_id),
            repo_type="dataset",
            revision=str(args.revision) if args.revision else None,
            allow_patterns=allow_patterns,
            token=token,
        )
    )

    exts = {".mid", ".midi"}
    midi_files = [p for p in snapshot_path.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    midi_files.sort()
    if args.limit is not None:
        midi_files = midi_files[: int(args.limit)]

    manifest_path = out_root / "manifest.jsonl"
    written = 0

    with manifest_path.open("w", encoding="utf-8") as mf:
        for src in midi_files:
            rel = src.relative_to(snapshot_path)
            dst = out_midis / rel
            _materialize_file(src, dst, mode=str(args.mode))
            written += 1
            mf.write(
                json.dumps(
                    {
                        "dataset_id": str(args.dataset_id),
                        "revision": str(args.revision) if args.revision else None,
                        "snapshot_root": str(snapshot_path),
                        "relpath": str(rel),
                        "path": str(dst),
                    }
                )
                + "\n"
            )

    stats = {
        "dataset_id": str(args.dataset_id),
        "revision": str(args.revision) if args.revision else None,
        "snapshot_root": str(snapshot_path),
        "out_dir": str(out_root),
        "mode": str(args.mode),
        "midi_files_found": int(len(midi_files)),
        "midi_files_written": int(written),
    }
    (out_root / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Found: {len(midi_files)} MIDI files in snapshot")
    print(f"Wrote: {written} files into {out_midis}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

