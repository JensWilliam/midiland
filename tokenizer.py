#!/usr/bin/env python3
"""
Simple event tokenizer for symbolic MIDI note sequences.

Token format per note:
  BAR, POS_<step>, PROG_<program>, NOTE_<pitch>, DUR_<steps>, VEL_<velocity>

This file is intentionally minimal and educational.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence


class MidiTokenizer:
    """Tokenizer for note dicts with fields: pitch/start_step/duration_step/velocity/program."""

    SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    def __init__(self, steps_per_beat: int = 4, beats_per_bar: int = 4) -> None:
        self.steps_per_beat = steps_per_beat
        self.beats_per_bar = beats_per_bar
        self.bar_steps = steps_per_beat * beats_per_bar

        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    @staticmethod
    def _read_value(note: dict[str, Any] | Any, key: str) -> Any:
        if isinstance(note, dict):
            return note[key]
        return getattr(note, key)

    @staticmethod
    def _clamp(value: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(value)))

    @staticmethod
    def _parse_token_value(token: str, prefix: str) -> int | None:
        if not token.startswith(prefix):
            return None
        try:
            return int(token[len(prefix) :])
        except ValueError:
            return None

    def notes_to_tokens(self, notes: Sequence[dict[str, Any] | Any]) -> list[str]:
        """Convert quantized note events to a flat token stream."""
        if not notes:
            return []

        # Stable order is important for reproducible datasets.
        sorted_notes = sorted(
            notes,
            key=lambda n: (
                int(self._read_value(n, "start_step")),
                int(self._read_value(n, "pitch")),
                int(self._read_value(n, "program")),
            ),
        )

        tokens: list[str] = []
        current_bar = -1

        for note in sorted_notes:
            start_step = max(0, int(self._read_value(note, "start_step")))
            duration_step = max(1, int(self._read_value(note, "duration_step")))
            pitch = self._clamp(int(self._read_value(note, "pitch")), 0, 127)
            velocity = self._clamp(int(self._read_value(note, "velocity")), 1, 127)
            program = self._clamp(int(self._read_value(note, "program")), 0, 127)

            bar_index = start_step // self.bar_steps
            pos = start_step % self.bar_steps

            # Emit one BAR token for each advanced bar (preserves long rests).
            while current_bar < bar_index:
                tokens.append("BAR")
                current_bar += 1

            tokens.extend(
                [
                    f"POS_{pos}",
                    f"PROG_{program}",
                    f"NOTE_{pitch}",
                    f"DUR_{duration_step}",
                    f"VEL_{velocity}",
                ]
            )

        return tokens

    def tokens_to_notes(self, tokens: Iterable[str]) -> list[dict[str, int]]:
        """
        Convert token stream back to note events.
        Assumes note fields arrive in a mostly valid pattern.
        """
        notes: list[dict[str, int]] = []
        current_bar = -1

        pos: int | None = None
        program: int | None = None
        pitch: int | None = None
        duration: int | None = None

        for token in tokens:
            token = token.strip()
            if not token:
                continue

            if token == "BAR":
                current_bar += 1
                continue

            pos_val = self._parse_token_value(token, "POS_")
            if pos_val is not None:
                pos = max(0, min(self.bar_steps - 1, pos_val))
                continue

            prog_val = self._parse_token_value(token, "PROG_")
            if prog_val is not None:
                program = self._clamp(prog_val, 0, 127)
                continue

            note_val = self._parse_token_value(token, "NOTE_")
            if note_val is not None:
                pitch = self._clamp(note_val, 0, 127)
                continue

            dur_val = self._parse_token_value(token, "DUR_")
            if dur_val is not None:
                duration = max(1, dur_val)
                continue

            vel_val = self._parse_token_value(token, "VEL_")
            if vel_val is not None:
                velocity = self._clamp(vel_val, 1, 127)
                if (
                    current_bar >= 0
                    and pos is not None
                    and program is not None
                    and pitch is not None
                    and duration is not None
                ):
                    start_step = current_bar * self.bar_steps + pos
                    notes.append(
                        {
                            "pitch": pitch,
                            "start_step": start_step,
                            "duration_step": duration,
                            "velocity": velocity,
                            "program": program,
                        }
                    )
                # Reset only per-note state. Bar state is continuous.
                pos = None
                program = None
                pitch = None
                duration = None

        notes.sort(key=lambda n: (n["start_step"], n["pitch"], n["program"]))
        return notes

    @staticmethod
    def _token_sort_key(token: str) -> tuple[int, int | str]:
        if token == "BAR":
            return (0, 0)
        prefixes = ["POS_", "PROG_", "NOTE_", "DUR_", "VEL_"]
        for idx, prefix in enumerate(prefixes, start=1):
            if token.startswith(prefix):
                try:
                    return (idx, int(token[len(prefix) :]))
                except ValueError:
                    return (idx, token)
        return (99, token)

    def build_vocab(self, token_sequences: Sequence[Sequence[str]]) -> dict[str, int]:
        """Build token/id mappings from dataset token sequences."""
        unique_tokens = set()
        for sequence in token_sequences:
            unique_tokens.update(sequence)

        ordered_tokens = self.SPECIAL_TOKENS + sorted(
            unique_tokens, key=self._token_sort_key
        )

        self.token_to_id = {token: idx for idx, token in enumerate(ordered_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        return self.token_to_id

    def encode(self, tokens: Sequence[str]) -> list[int]:
        if not self.token_to_id:
            raise ValueError("Vocabulary is empty. Run build_vocab() first.")
        unk_id = self.token_to_id["<UNK>"]
        return [self.token_to_id.get(token, unk_id) for token in tokens]

    def decode(self, ids: Sequence[int]) -> list[str]:
        if not self.id_to_token:
            raise ValueError("Vocabulary is empty. Run build_vocab() first.")
        return [self.id_to_token.get(int(idx), "<UNK>") for idx in ids]

    def save_vocab(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    def load_vocab(self, path: str | Path) -> dict[str, int]:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.token_to_id = {str(k): int(v) for k, v in data.items()}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        return self.token_to_id


def load_token_file(path: str | Path) -> list[str]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        return list(json.loads(text))
    return [line.strip() for line in text.splitlines() if line.strip()]


def save_token_file(path: str | Path, tokens: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize or detokenize MIDI note events.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tokenize_parser = subparsers.add_parser("tokenize", help="notes.json -> tokens")
    tokenize_parser.add_argument("notes_json", help="Input notes JSON (list of dicts).")
    tokenize_parser.add_argument("--tokens-out", required=True, help="Output token file.")
    tokenize_parser.add_argument("--vocab-out", help="Optional output vocab JSON.")
    tokenize_parser.add_argument("--ids-out", help="Optional output token-id sequence JSON.")
    tokenize_parser.add_argument("--steps-per-beat", type=int, default=4)
    tokenize_parser.add_argument("--beats-per-bar", type=int, default=4)

    detokenize_parser = subparsers.add_parser("detokenize", help="tokens -> notes.json")
    detokenize_parser.add_argument("tokens_in", help="Input token file (.txt or .json).")
    detokenize_parser.add_argument("--notes-out", required=True, help="Output notes JSON.")
    detokenize_parser.add_argument("--steps-per-beat", type=int, default=4)
    detokenize_parser.add_argument("--beats-per-bar", type=int, default=4)

    args = parser.parse_args()
    tokenizer = MidiTokenizer(
        steps_per_beat=args.steps_per_beat, beats_per_bar=args.beats_per_bar
    )

    if args.command == "tokenize":
        notes = json.loads(Path(args.notes_json).read_text(encoding="utf-8"))
        tokens = tokenizer.notes_to_tokens(notes)
        save_token_file(args.tokens_out, tokens)
        print(f"Wrote {len(tokens)} tokens to {args.tokens_out}")

        if args.vocab_out or args.ids_out:
            tokenizer.build_vocab([tokens])

        if args.vocab_out:
            tokenizer.save_vocab(args.vocab_out)
            print(f"Wrote vocab ({len(tokenizer.token_to_id)} tokens) to {args.vocab_out}")

        if args.ids_out:
            ids = tokenizer.encode(tokens)
            ids_path = Path(args.ids_out)
            ids_path.parent.mkdir(parents=True, exist_ok=True)
            ids_path.write_text(json.dumps(ids), encoding="utf-8")
            print(f"Wrote {len(ids)} token IDs to {args.ids_out}")

    elif args.command == "detokenize":
        tokens = load_token_file(args.tokens_in)
        notes = tokenizer.tokens_to_notes(tokens)
        out_path = Path(args.notes_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(notes, indent=2), encoding="utf-8")
        print(f"Wrote {len(notes)} notes to {args.notes_out}")


if __name__ == "__main__":
    main()
