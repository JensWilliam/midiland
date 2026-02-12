#!/usr/bin/env python3
"""
Sample token sequences from a trained decoder-only transformer checkpoint.

Example:
    python sample.py \
      --ckpt checkpoints/checkpoint_epoch_010.pt \
      --max-new 2000 \
      --temperature 1.0 \
      --top-k 50 \
      --device cpu \
      --out generated_tokens.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from midi_parser import parse_midi_file
from model import TinyTransformerLM
from tokenizer import VOCAB, tokenize_piece


def _default_prompt() -> list[int]:
    """Build a minimal header prompt for unconditional generation."""
    ts_num, ts_den = VOCAB.default_time_signature
    return [
        int(VOCAB.bos_id),
        int(VOCAB.time_signature_token(ts_num, ts_den)),
        int(VOCAB.tempo_token_from_bpm(VOCAB.default_tempo_bpm)),
        int(VOCAB.key_signature_token(VOCAB.default_key)),
        int(VOCAB.program_change_token(0, 0)),
    ]


def _prompt_from_midi(midi_path: Path, *, prompt_len: int, steps_per_beat: int) -> list[int]:
    """
    Build a prompt by tokenizing a real MIDI and taking its first prompt_len tokens.
    """
    try:
        import mido
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mido is required for --prompt-midi. Install it with: pip install mido"
        ) from exc

    midi = mido.MidiFile(str(midi_path))
    events = parse_midi_file(
        midi_path,
        ignore_drums=True,
        include_meta=True,
        notes_only=False,
    )
    tokens = tokenize_piece(
        events,
        ticks_per_beat=int(midi.ticks_per_beat),
        steps_per_beat=int(steps_per_beat),
    )
    if not tokens:
        raise ValueError(f"No tokens produced from prompt MIDI: {midi_path}")
    return [int(token) for token in tokens[: max(1, int(prompt_len))]]


def _build_model_from_checkpoint(
    ckpt: dict,
    *,
    device: torch.device,
) -> tuple[TinyTransformerLM, int]:
    """
    Recreate the trained model architecture and load weights.
    For legacy checkpoints trained with smaller inferred vocab, use checkpoint vocab.
    """
    config = ckpt.get("config", {})
    tokenizer_vocab_size = int(VOCAB.vocab_size)
    ckpt_vocab_size = ckpt.get("vocab_size")
    if ckpt_vocab_size is None:
        model_vocab_size = tokenizer_vocab_size
    else:
        model_vocab_size = int(ckpt_vocab_size)
        if model_vocab_size > tokenizer_vocab_size:
            raise ValueError(
                f"Checkpoint vocab_size ({model_vocab_size}) is larger than "
                f"tokenizer.VOCAB.vocab_size ({tokenizer_vocab_size})."
            )
        if model_vocab_size < tokenizer_vocab_size:
            print(
                "Warning: loading legacy checkpoint with smaller vocab "
                f"({model_vocab_size} < {tokenizer_vocab_size}). "
                "Generation will only use the checkpoint vocabulary range."
            )

    model = TinyTransformerLM(
        vocab_size=model_vocab_size,
        max_seq_len=int(config.get("seq_len", 128)),
        d_model=int(config.get("d_model", 128)),
        n_heads=int(config.get("n_heads", 4)),
        n_layers=int(config.get("n_layers", 4)),
        dropout=float(config.get("dropout", 0.1)),
    ).to(device)

    state_dict = ckpt.get("model_state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint is missing 'model_state_dict'.")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint weights into model. "
            "This usually means architecture or vocab mismatch."
        ) from exc

    model.eval()
    return model, model_vocab_size


@torch.no_grad()
def _generate_tokens(
    model: TinyTransformerLM,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> list[int]:
    if not prompt_tokens:
        raise ValueError("Prompt must contain at least one token.")

    prompt = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated = prompt

    for _ in range(int(max_new_tokens)):
        # Crop context if sequence is longer than the model window.
        context = generated[:, -model.max_seq_len :]
        logits = model(context)
        next_logits = logits[:, -1, :]

        if temperature <= 0.0:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        else:
            next_logits = next_logits / float(temperature)
            k = int(top_k)
            if k > 0:
                k = min(k, next_logits.size(-1))
                topk_values, _ = torch.topk(next_logits, k=k, dim=-1)
                cutoff = topk_values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

    return [int(token) for token in generated.squeeze(0).tolist()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate token IDs from a trained checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to training checkpoint (.pt).")
    parser.add_argument("--max-new", type=int, default=2000, dest="max_new")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50, dest="top_k")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="generated_tokens.npy")
    parser.add_argument(
        "--prompt-midi",
        type=str,
        default=None,
        help="Optional MIDI file to build prompt from.",
        dest="prompt_midi",
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        default=256,
        help="When using --prompt-midi, use the first N tokenized tokens.",
        dest="prompt_len",
    )
    parser.add_argument(
        "--steps-per-beat",
        type=int,
        default=4,
        help="Tokenization grid used when --prompt-midi is provided.",
        dest="steps_per_beat",
    )
    parser.add_argument("--preview", type=int, default=200, help="Number of decoded tokens to print.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device(args.device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    model, model_vocab_size = _build_model_from_checkpoint(checkpoint, device=device)

    if args.prompt_midi:
        prompt_tokens = _prompt_from_midi(
            Path(args.prompt_midi),
            prompt_len=int(args.prompt_len),
            steps_per_beat=int(args.steps_per_beat),
        )
    else:
        prompt_tokens = _default_prompt()

    for token in prompt_tokens:
        if token < 0 or token >= int(model_vocab_size):
            raise ValueError(
                "Prompt token out of range for checkpoint vocabulary size "
                f"{model_vocab_size}: {token}"
            )

    generated_tokens = _generate_tokens(
        model,
        prompt_tokens,
        max_new_tokens=int(args.max_new),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        device=device,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, np.asarray(generated_tokens, dtype=np.int64))

    preview_count = min(max(0, int(args.preview)), len(generated_tokens))
    print(f"Generated token count: {len(generated_tokens)}")
    print(f"Saved tokens to: {out_path}")
    print(f"Decoded preview (first {preview_count} tokens):")
    for idx, token in enumerate(generated_tokens[:preview_count]):
        print(f"{idx:04d}: {token:4d}  {VOCAB.token_to_string(int(token))}")


if __name__ == "__main__":
    main()
