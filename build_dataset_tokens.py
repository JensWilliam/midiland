#!/usr/bin/env python3
"""
Build one contiguous training token file from the Hugging Face dataset:
drengskapur/midi-classical-music
"""

from __future__ import annotations

import argparse
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np

from tokenizer import VOCAB, tokenize_midi_path

DATASET_ID = "drengskapur/midi-classical-music"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tokenize MIDI files from drengskapur/midi-classical-music into one "
            "contiguous 1D int32 token array."
        )
    )
    parser.add_argument(
        "--out",
        type=str,
        default="tokens.npy",
        help="Output .npy path for the concatenated token stream.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of files to process after ordering/shuffling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed used by --shuffle-files.",
    )
    parser.add_argument(
        "--shuffle-files",
        action="store_true",
        help="Shuffle file processing order before tokenization.",
    )
    return parser.parse_args()


def _try_load_dataset_train_split():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "datasets is required. Install it with: pip install datasets"
        ) from exc
    return load_dataset(DATASET_ID, split="train")


def _try_snapshot_root() -> Path | None:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError:
        return None

    try:
        snapshot_dir = snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            allow_patterns=["data/*.mid", "data/**/*.mid", "metadata.csv", "README.md"],
        )
    except Exception as exc:
        print(f"Warning: snapshot download failed, falling back to cache search: {exc}")
        return None

    return Path(snapshot_dir)


def _candidate_roots_from_cache(dataset) -> list[Path]:
    roots: list[Path] = [Path.cwd()]

    cache_files = getattr(dataset, "cache_files", None) or []
    for cache_entry in cache_files:
        filename = cache_entry.get("filename")
        if not filename:
            continue
        cache_path = Path(str(filename)).resolve()
        roots.append(cache_path.parent)
        for parent in cache_path.parents:
            roots.append(parent)
            if parent.name == "datasets":
                break

    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    snapshots_root = hf_home / "hub" / "datasets--drengskapur--midi-classical-music" / "snapshots"
    if snapshots_root.exists():
        for snapshot in sorted(snapshots_root.iterdir()):
            if snapshot.is_dir():
                roots.append(snapshot)

    extracted_root = hf_home / "datasets" / "downloads" / "extracted"
    if extracted_root.exists():
        roots.append(extracted_root)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _normalize_file_name(file_name: str) -> Path:
    raw = str(file_name).strip()
    path = Path(raw)
    if path.is_absolute():
        return path
    clean_parts = [part for part in path.parts if part not in ("", ".")]
    return Path(*clean_parts) if clean_parts else path


def _resolve_midi_path(
    file_name: str,
    *,
    snapshot_root: Path | None,
    cache_roots: Iterable[Path],
) -> Path | None:
    relative = _normalize_file_name(file_name)
    if relative.is_absolute() and relative.is_file():
        return relative

    if snapshot_root is not None:
        candidate = snapshot_root / relative
        if candidate.is_file():
            return candidate

    for root in cache_roots:
        candidate = root / relative
        if candidate.is_file():
            return candidate

    basename = relative.name
    if not basename:
        return None

    search_roots: list[Path] = []
    if snapshot_root is not None:
        search_roots.append(snapshot_root)
    for root in cache_roots:
        if root not in search_roots:
            search_roots.append(root)
        if len(search_roots) >= 8:
            break

    relative_posix = relative.as_posix()
    for root in search_roots:
        if not root.exists() or not root.is_dir():
            continue
        try:
            for candidate in root.rglob(basename):
                if not candidate.is_file():
                    continue
                if candidate.as_posix().endswith(relative_posix):
                    return candidate
        except OSError:
            continue

    return None


class _StreamingTokenWriter:
    def __init__(self, out_path: Path) -> None:
        self.out_path = out_path
        self.tmp_path = Path(f"{out_path}.tmp.int32.bin")
        self.tmp_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.tmp_path.open("wb")
        self.total_tokens = 0

    def append(self, token_ids: np.ndarray) -> None:
        if token_ids.ndim != 1:
            token_ids = token_ids.reshape(-1)
        if token_ids.dtype != np.int32:
            token_ids = token_ids.astype(np.int32, copy=False)
        token_ids.tofile(self._file)
        self.total_tokens += int(token_ids.size)

    def finalize(self) -> None:
        self._file.flush()
        self._file.close()

        if self.total_tokens == 0:
            np.save(self.out_path, np.asarray([], dtype=np.int32))
        else:
            tokens_memmap = np.memmap(
                self.tmp_path,
                dtype=np.int32,
                mode="r",
                shape=(self.total_tokens,),
            )
            np.save(self.out_path, tokens_memmap)
            del tokens_memmap

        self.tmp_path.unlink(missing_ok=True)


def main() -> None:
    args = _parse_args()

    if args.limit is not None and int(args.limit) < 0:
        raise ValueError("--limit must be >= 0")

    print(f"Loading dataset split: {DATASET_ID} [train]")
    dataset = _try_load_dataset_train_split()

    if "file_name" not in dataset.column_names:
        raise KeyError("Dataset is missing required column: file_name")

    file_names = [str(name) for name in dataset["file_name"]]
    if args.shuffle_files:
        rng = random.Random(int(args.seed))
        rng.shuffle(file_names)
        print(f"Shuffled file order with seed={int(args.seed)}")

    if args.limit is not None:
        file_names = file_names[: int(args.limit)]

    total_files = len(file_names)
    print(f"Files scheduled: {total_files}")
    if total_files == 0:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, np.asarray([], dtype=np.int32))
        print(f"Saved empty token array to: {out_path}")
        print("Total token count: 0")
        return

    snapshot_root = _try_snapshot_root()
    if snapshot_root is not None:
        print(f"Using dataset snapshot root: {snapshot_root}")
    else:
        print("No snapshot root available; using local cache search.")

    cache_roots = _candidate_roots_from_cache(dataset)
    out_path = Path(args.out)
    writer = _StreamingTokenWriter(out_path)

    processed = 0
    skipped = 0
    exception_counts: Counter[str] = Counter()
    global_min: int | None = None
    global_max: int | None = None

    try:
        for index, file_name in enumerate(file_names, start=1):
            midi_path = _resolve_midi_path(
                file_name,
                snapshot_root=snapshot_root,
                cache_roots=cache_roots,
            )

            if midi_path is None:
                skipped += 1
                exception_counts["FileNotFoundError"] += 1
                print(f"Skipping {file_name}: FileNotFoundError")
                processed += 1
                if processed % 25 == 0 or processed == total_files:
                    print(
                        f"Progress: files processed={processed}/{total_files}, "
                        f"tokens accumulated={writer.total_tokens}, skipped files={skipped}"
                    )
                continue

            try:
                piece_tokens = tokenize_midi_path(midi_path)
                token_array = np.asarray(piece_tokens, dtype=np.int64).reshape(-1)

                if token_array.size > 0:
                    piece_min = int(token_array.min())
                    piece_max = int(token_array.max())
                    if piece_min < 0 or piece_max >= int(VOCAB.vocab_size):
                        raise ValueError(
                            "Token ID out of range "
                            f"(min={piece_min}, max={piece_max}, vocab={VOCAB.vocab_size})"
                        )

                    if global_min is None or piece_min < global_min:
                        global_min = piece_min
                    if global_max is None or piece_max > global_max:
                        global_max = piece_max

                writer.append(token_array.astype(np.int32, copy=False))
            except Exception as exc:
                skipped += 1
                exception_counts[type(exc).__name__] += 1
                print(f"Skipping {file_name}: {type(exc).__name__}: {exc}")

            processed += 1
            if processed % 25 == 0 or processed == total_files:
                print(
                    f"Progress: files processed={processed}/{total_files}, "
                    f"tokens accumulated={writer.total_tokens}, skipped files={skipped}"
                )
    finally:
        writer.finalize()

    print(f"Saved tokens to: {out_path}")
    if writer.total_tokens == 0:
        print("Token ID range: N/A (no tokens)")
        print("Range check: no tokens to validate")
    else:
        assert global_min is not None and global_max is not None
        in_vocab = global_max < int(VOCAB.vocab_size)
        print(f"Token ID range: min={global_min}, max={global_max}")
        print(f"Range check: max < VOCAB.vocab_size ({VOCAB.vocab_size}) -> {in_vocab}")
    print(f"Total token count: {writer.total_tokens}")
    print(f"Skipped files: {skipped}")

    if exception_counts:
        print("Exception counts:")
        for name, count in exception_counts.most_common():
            print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
