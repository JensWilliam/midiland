import argparse
import os
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model import TinyTransformerLM


class TokenSequenceDataset(Dataset):
    """
    Builds overlapping next-token samples from a 1D token array.
    Example with seq_len=4:
      input  = tokens[i : i+4]
      target = tokens[i+1 : i+5]
    """

    def __init__(self, tokens: np.ndarray, seq_len: int) -> None:
        tokens = np.asarray(tokens)
        if tokens.ndim != 1:
            tokens = tokens.reshape(-1)
        if tokens.size <= seq_len:
            raise ValueError(
                f"Need more than seq_len ({seq_len}) tokens, but got {tokens.size}."
            )
        if not np.issubdtype(tokens.dtype, np.integer):
            raise TypeError(
                f"Expected integer token IDs, but got dtype={tokens.dtype}."
            )
        if tokens.min() < 0:
            raise ValueError("Token IDs must be non-negative.")

        self.tokens = tokens.astype(np.int64, copy=False)
        self.seq_len = seq_len
        self.num_samples = self.tokens.size - seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return torch.from_numpy(x), torch.from_numpy(y)


class SegmentedTokenSequenceDataset(Dataset):
    """
    Builds next-token samples from segments without crossing segment boundaries.

    The original TokenSequenceDataset uses fully-overlapping samples with stride=1.
    This dataset supports a configurable stride to reduce redundancy and runtime.
    """

    def __init__(
        self,
        tokens: np.ndarray,
        *,
        segments: list[tuple[int, int]],
        seq_len: int,
        stride: int,
    ) -> None:
        tokens = np.asarray(tokens)
        if tokens.ndim != 1:
            tokens = tokens.reshape(-1)
        if tokens.size <= seq_len:
            raise ValueError(
                f"Need more than seq_len ({seq_len}) tokens, but got {tokens.size}."
            )
        if not np.issubdtype(tokens.dtype, np.integer):
            raise TypeError(f"Expected integer token IDs, but got dtype={tokens.dtype}.")
        if tokens.min() < 0:
            raise ValueError("Token IDs must be non-negative.")

        stride_i = int(stride)
        if stride_i <= 0:
            raise ValueError("--stride must be >= 1")

        normalized: list[tuple[int, int]] = []
        for start, end in segments:
            s = int(start)
            e = int(end)
            if s < 0 or e < 0 or e < s:
                continue
            if s >= tokens.size:
                continue
            e = min(e, int(tokens.size))
            if e - s <= 0:
                continue
            normalized.append((s, e))
        if not normalized:
            raise ValueError("No valid segments available for training.")

        seq_len_i = int(seq_len)
        prefix: list[int] = [0]
        for start, end in normalized:
            max_start = int(end) - seq_len_i - 1
            if max_start < int(start):
                count = 0
            else:
                count = int((max_start - int(start)) // stride_i) + 1
            prefix.append(prefix[-1] + count)

        total = int(prefix[-1])
        if total <= 0:
            raise ValueError(
                "No training samples after applying segments/seq_len/stride. "
                "Try reducing --seq-len or --stride, or increase data."
            )

        self.tokens = tokens.astype(np.int64, copy=False)
        self.segments = normalized
        self.seq_len = seq_len_i
        self.stride = stride_i
        self._prefix = prefix
        self.num_samples = total

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(idx)

        seg_i = bisect_right(self._prefix, int(idx)) - 1
        local_i = int(idx) - int(self._prefix[seg_i])
        start, end = self.segments[seg_i]
        pos = int(start) + local_i * int(self.stride)

        # pos is guaranteed to be valid by construction.
        x = self.tokens[pos : pos + self.seq_len]
        y = self.tokens[pos + 1 : pos + self.seq_len + 1]
        return torch.from_numpy(x), torch.from_numpy(y)


def _tokenizer_vocab_size() -> int | None:
    """
    Return tokenizer.VOCAB.vocab_size when tokenizer.py is importable.
    Falls back to None so training can still run in non-tokenizer contexts.
    """
    try:
        from tokenizer import VOCAB  # Local import keeps startup robust.
    except Exception:
        return None
    return int(VOCAB.vocab_size)


def _segments_from_bos(tokens: np.ndarray, *, bos_id: int) -> list[tuple[int, int]]:
    """
    Split a concatenated token stream into segments starting at BOS tokens.
    Assumes each piece begins with BOS (tokenizer.VOCAB.bos_id).
    """
    flat = np.asarray(tokens).reshape(-1)
    bos = int(bos_id)
    bos_positions = np.flatnonzero(flat == bos).astype(np.int64)

    if bos_positions.size == 0:
        return [(0, int(flat.size))]

    if int(bos_positions[0]) != 0:
        bos_positions = np.concatenate([np.asarray([0], dtype=np.int64), bos_positions])

    segments: list[tuple[int, int]] = []
    for i in range(int(bos_positions.size)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < int(bos_positions.size) else int(flat.size)
        if end > start:
            segments.append((start, end))
    return segments


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokens = np.load(args.tokens_path)
    if args.no_cross_bos:
        try:
            from tokenizer import VOCAB

            bos_id = int(VOCAB.bos_id)
        except Exception:
            bos_id = 1  # tokenizer.py default

        segments = _segments_from_bos(tokens, bos_id=bos_id)
        dataset = SegmentedTokenSequenceDataset(
            tokens=tokens,
            segments=segments,
            seq_len=args.seq_len,
            stride=args.stride,
        )
        print(
            f"Segmented training: segments={len(segments)}, stride={int(args.stride)}, samples={len(dataset)}"
        )
    else:
        dataset = TokenSequenceDataset(tokens=tokens, seq_len=args.seq_len)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

    tokenizer_vocab_size = _tokenizer_vocab_size()
    vocab_size = int(args.vocab_size)

    if tokenizer_vocab_size is not None and vocab_size != tokenizer_vocab_size:
        raise ValueError(
            f"Model vocab size ({vocab_size}) must match tokenizer.VOCAB.vocab_size ({tokenizer_vocab_size})."
        )
    if int(dataset.tokens.max()) >= vocab_size:
        raise ValueError(
            "Provided --vocab-size is too small for the token IDs in the dataset."
        )

    device = torch.device(args.device)
    model = TinyTransformerLM(
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {dataset.tokens.size} tokens")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Vocab size: {vocab_size}")
    print(f"Training on device: {device}")

    max_steps = None if args.max_steps is None else int(args.max_steps)
    if max_steps is not None and max_steps <= 0:
        raise ValueError("--max-steps must be >= 1")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for x, y in dataloader:
            if max_steps is not None and global_step >= max_steps:
                break

            x = x.to(device)
            y = y.to(device)

            logits = model(x)  # [B, T, V]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1
            global_step += 1

        avg_loss = epoch_loss / max(steps, 1)
        print(
            f"Epoch {epoch:03d} | loss: {avg_loss:.4f} | steps: {steps} | global_step: {global_step}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "vocab_size": vocab_size,
        }
        ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, ckpt_path)

        if max_steps is not None and global_step >= max_steps:
            print(f"Reached --max-steps={max_steps}. Stopping early.")
            break

    print("Training complete.")


def parse_args() -> argparse.Namespace:
    default_vocab_size = _tokenizer_vocab_size()
    if default_vocab_size is None:
        default_vocab_size = 2396

    parser = argparse.ArgumentParser(
        description="Train a tiny decoder-only transformer on token IDs saved in .npy."
    )
    parser.add_argument("tokens_path", type=str, help="Path to tokens.npy")
    parser.add_argument("--seq-len", type=int, default=128, dest="seq_len")
    parser.add_argument("--batch-size", type=int, default=32, dest="batch_size")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    parser.add_argument("--d-model", type=int, default=128, dest="d_model")
    parser.add_argument("--n-heads", type=int, default=4, dest="n_heads")
    parser.add_argument("--n-layers", type=int, default=4, dest="n_layers")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--no-cross-bos",
        action="store_true",
        help=(
            "Prevent training samples from crossing piece boundaries by splitting the token "
            "stream on BOS tokens (tokenizer.VOCAB.bos_id)."
        ),
        dest="no_cross_bos",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for segmented training overlap reduction (only used with --no-cross-bos).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional cap on total optimizer steps across all epochs.",
        dest="max_steps",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=int(default_vocab_size),
        help=(
            "Vocabulary size for embeddings/output head. "
            "Defaults to tokenizer.VOCAB.vocab_size when tokenizer.py is importable "
            "(fallback: 2396)."
        ),
        dest="vocab_size",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", dest="checkpoint_dir")
    parser.add_argument("--device", type=str, default="cpu", help="Use 'cpu' (default) or 'cuda'.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.tokens_path):
        raise FileNotFoundError(f"Could not find token file: {args.tokens_path}")
    train(args)
