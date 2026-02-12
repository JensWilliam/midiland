import argparse
import os
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


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokens = np.load(args.tokens_path)
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

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for x, y in dataloader:
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

        avg_loss = epoch_loss / max(steps, 1)
        print(f"Epoch {epoch:03d} | loss: {avg_loss:.4f}")

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(args),
            "vocab_size": vocab_size,
        }
        ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        torch.save(checkpoint, ckpt_path)

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
