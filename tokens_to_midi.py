#!/usr/bin/env python3
"""
Convert a generated token stream (.npy) back to a MIDI file.

Example:
    python tokens_to_midi.py generated_tokens.npy --out generated.mid --steps-per-beat 4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from midi_parser import (
    KeySignatureEvent,
    NoteEvent,
    ProgramChangeEvent,
    QuantizedNoteEvent,
    TempoEvent,
    TimeSignatureEvent,
)
from midi_writer import write_midi
from tokenizer import VOCAB


@dataclass(slots=True)
class _ParsedTokens:
    notes: list[QuantizedNoteEvent]
    program_changes: list[tuple[int, int, int]]  # (step, channel, program)
    time_signature: tuple[int, int]
    tempo_bpm: int
    key: str
    skipped_malformed_notes: int


def _bucket_to_velocity(bucket: int, bins: int) -> int:
    """Map a velocity bucket index back to an approximate MIDI velocity [1, 127]."""
    bucket_i = max(0, min(int(bins) - 1, int(bucket)))
    center = (bucket_i + 0.5) / float(bins)
    return max(1, min(127, int(round(center * 127.0))))


def _bpm_to_tempo_us_per_beat(bpm: int) -> int:
    bpm_i = int(bpm)
    if bpm_i <= 0:
        bpm_i = int(VOCAB.default_tempo_bpm)
    return int(round(60_000_000.0 / float(bpm_i)))


def _parse_tokens(tokens: np.ndarray) -> _ParsedTokens:
    """
    Decode token stream into quantized notes + metadata.

    Rules:
    - SHIFT_n advances current_step.
    - PROGRAM_CHx_Py updates current program per channel.
    - NOTE + CH + PITCH + DUR + VEL emits QuantizedNoteEvent at current_step.
    - First seen TS/TEMPO/KEY are used as global metadata.
    """
    flat_tokens = np.asarray(tokens).reshape(-1)
    if not np.issubdtype(flat_tokens.dtype, np.integer):
        raise TypeError(f"Expected integer tokens, got dtype={flat_tokens.dtype}")

    current_step = 0
    current_program_by_channel = {channel: 0 for channel in range(16)}
    notes: list[QuantizedNoteEvent] = []
    program_changes: list[tuple[int, int, int]] = []

    first_ts: tuple[int, int] | None = None
    first_tempo_bpm: int | None = None
    first_key: str | None = None
    skipped_malformed_notes = 0

    i = 0
    while i < len(flat_tokens):
        token = int(flat_tokens[i])
        kind, payload = VOCAB.decode_token(token)

        if kind == "step_shift":
            current_step += int(payload)
            i += 1
            continue

        if kind == "program_change":
            channel, program = payload
            channel_i = int(channel)
            program_i = int(program)
            if current_program_by_channel.get(channel_i, 0) != program_i:
                program_changes.append((int(current_step), channel_i, program_i))
            current_program_by_channel[channel_i] = program_i
            i += 1
            continue

        if kind == "time_signature":
            if first_ts is None:
                first_ts = (int(payload[0]), int(payload[1]))
            i += 1
            continue

        if kind == "tempo":
            if first_tempo_bpm is None:
                first_tempo_bpm = int(payload)
            i += 1
            continue

        if kind == "key_signature":
            if first_key is None:
                first_key = str(payload)
            i += 1
            continue

        if kind == "note":
            # Defensive parsing: NOTE must be followed by CH + PITCH + DUR + VEL.
            if i + 4 >= len(flat_tokens):
                skipped_malformed_notes += 1
                i += 1
                continue

            k1, p1 = VOCAB.decode_token(int(flat_tokens[i + 1]))
            k2, p2 = VOCAB.decode_token(int(flat_tokens[i + 2]))
            k3, p3 = VOCAB.decode_token(int(flat_tokens[i + 3]))
            k4, p4 = VOCAB.decode_token(int(flat_tokens[i + 4]))
            if not (
                k1 == "note_channel"
                and k2 == "note_pitch"
                and k3 == "note_duration"
                and k4 == "note_velocity"
            ):
                skipped_malformed_notes += 1
                i += 1
                continue

            channel = int(p1)
            pitch = int(p2)
            duration_step = max(1, int(p3))
            velocity = _bucket_to_velocity(int(p4), int(VOCAB.velocity_bins))
            program = int(current_program_by_channel.get(channel, 0))
            notes.append(
                QuantizedNoteEvent(
                    pitch=pitch,
                    start_step=int(current_step),
                    duration_step=duration_step,
                    velocity=velocity,
                    program=program,
                    channel=channel,
                )
            )
            i += 5
            continue

        i += 1

    return _ParsedTokens(
        notes=notes,
        program_changes=program_changes,
        time_signature=first_ts or tuple(VOCAB.default_time_signature),
        tempo_bpm=int(first_tempo_bpm if first_tempo_bpm is not None else VOCAB.default_tempo_bpm),
        key=str(first_key if first_key is not None else VOCAB.default_key),
        skipped_malformed_notes=skipped_malformed_notes,
    )


def _quantized_to_events(
    parsed: _ParsedTokens,
    *,
    steps_per_beat: int,
    ticks_per_beat: int,
) -> list[object]:
    if int(steps_per_beat) <= 0:
        raise ValueError("steps_per_beat must be >= 1")
    if int(ticks_per_beat) <= 0:
        raise ValueError("ticks_per_beat must be >= 1")

    ticks_per_step = float(ticks_per_beat) / float(steps_per_beat)

    def step_to_tick(step: int) -> int:
        return int(round(float(int(step)) * ticks_per_step))

    events: list[object] = [
        TempoEvent(time=0, tempo=_bpm_to_tempo_us_per_beat(parsed.tempo_bpm)),
        TimeSignatureEvent(
            time=0,
            numerator=int(parsed.time_signature[0]),
            denominator=int(parsed.time_signature[1]),
        ),
        KeySignatureEvent(time=0, key=str(parsed.key)),
    ]

    for step, channel, program in parsed.program_changes:
        events.append(
            ProgramChangeEvent(
                time=step_to_tick(int(step)),
                channel=int(channel),
                program=int(program),
            )
        )

    for note in parsed.notes:
        start_tick = step_to_tick(int(note.start_step))
        duration_ticks = max(1, step_to_tick(int(note.duration_step)))
        events.append(
            NoteEvent(
                time=start_tick,
                duration_ticks=duration_ticks,
                channel=int(note.channel),
                program=int(note.program),
                pitch=int(note.pitch),
                velocity=int(note.velocity),
            )
        )

    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert generated token IDs (.npy) to MIDI.")
    parser.add_argument("tokens_path", type=str, help="Path to generated token .npy file.")
    parser.add_argument("--out", type=str, required=True, help="Output MIDI path.")
    parser.add_argument("--steps-per-beat", type=int, default=4, dest="steps_per_beat")
    parser.add_argument("--ticks-per-beat", type=int, default=480, dest="ticks_per_beat")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens_path = Path(args.tokens_path)
    if not tokens_path.exists():
        raise FileNotFoundError(f"Token file not found: {tokens_path}")

    tokens = np.load(tokens_path)
    parsed = _parse_tokens(tokens)
    events = _quantized_to_events(
        parsed,
        steps_per_beat=int(args.steps_per_beat),
        ticks_per_beat=int(args.ticks_per_beat),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_midi(
        str(out_path),
        ticks_per_beat=int(args.ticks_per_beat),
        events=events,
    )

    print(f"Loaded tokens: {np.asarray(tokens).reshape(-1).size}")
    print(f"Parsed notes: {len(parsed.notes)}")
    print(f"Program changes: {len(parsed.program_changes)}")
    print(f"Skipped malformed NOTE groups: {parsed.skipped_malformed_notes}")
    print(
        "Metadata: "
        f"time_signature={parsed.time_signature[0]}/{parsed.time_signature[1]}, "
        f"tempo_bpm={parsed.tempo_bpm}, key={parsed.key}"
    )
    print(f"Wrote MIDI: {out_path}")


if __name__ == "__main__":
    main()
