#!/usr/bin/env python3
"""
Minimal MIDI parser for a drum-free symbolic dataset.

What this file does:
1. Loads a MIDI file with `mido`.
2. Ignores all channel-10 events (channel index 9).
3. Tracks program changes per channel.
4. Extracts note events with pitch/start/duration/velocity/program.
5. Quantizes timing to a fixed step grid (default: 1/16 notes).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict

try:
    import mido
except ModuleNotFoundError:
    mido = None


@dataclass
class NoteEvent:
    """Single symbolic note event after quantization."""

    pitch: int
    start_step: int
    duration_step: int
    velocity: int
    program: int


def quantize_tick_to_step(tick: int, ticks_per_step: float) -> int:
    """Convert absolute MIDI ticks to integer grid steps."""
    return int(round(tick / ticks_per_step))


def parse_midi_file(midi_path: str | Path, steps_per_beat: int = 4) -> list[NoteEvent]:
    """
    Parse a MIDI file into quantized note events.

    Args:
        midi_path: Input .mid file path.
        steps_per_beat: Grid resolution. 4 means 1/16 notes in 4/4.
    """
    if mido is None:
        raise ModuleNotFoundError(
            "mido is required for MIDI parsing. Install it with: pip install mido"
        )

    midi = mido.MidiFile(str(midi_path))
    ticks_per_step = midi.ticks_per_beat / float(steps_per_beat)

    # Program is tracked independently for each channel (MIDI default program = 0).
    channel_program = {channel: 0 for channel in range(16)}

    # Handles overlapping same-pitch notes: key=(channel, pitch), value=list of active note-ons.
    active_notes: DefaultDict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)

    notes: list[NoteEvent] = []
    abs_tick = 0

    # `merge_tracks` gives a single time-ordered stream (delta times preserved).
    for msg in mido.merge_tracks(midi.tracks):
        abs_tick += msg.time

        if msg.is_meta or not hasattr(msg, "channel"):
            continue

        channel = msg.channel

        # Hard drum filter: channel 10 in MIDI UI is index 9.
        if channel == 9:
            continue

        if msg.type == "program_change":
            channel_program[channel] = msg.program
            continue

        if msg.type == "note_on" and msg.velocity > 0:
            active_notes[(channel, msg.note)].append(
                (abs_tick, msg.velocity, channel_program[channel])
            )
            continue

        is_note_off = msg.type == "note_off" or (
            msg.type == "note_on" and msg.velocity == 0
        )
        if not is_note_off:
            continue

        key = (channel, msg.note)
        if not active_notes[key]:
            # Note-off with no matching note-on (can happen in imperfect MIDI files).
            continue

        # FIFO pairing is simple and works well for educational pipelines.
        start_tick, velocity, program = active_notes[key].pop(0)
        end_tick = max(start_tick + 1, abs_tick)

        start_step = quantize_tick_to_step(start_tick, ticks_per_step)
        end_step = quantize_tick_to_step(end_tick, ticks_per_step)
        duration_step = max(1, end_step - start_step)

        notes.append(
            NoteEvent(
                pitch=int(msg.note),
                start_step=int(start_step),
                duration_step=int(duration_step),
                velocity=int(velocity),
                program=int(program),
            )
        )

    notes.sort(key=lambda n: (n.start_step, n.pitch, n.program))
    return notes


def note_events_to_dicts(notes: list[NoteEvent]) -> list[dict]:
    """Convert dataclass notes to plain dicts (easy JSON serialization)."""
    return [asdict(note) for note in notes]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse MIDI into quantized note events.")
    parser.add_argument("midi_path", help="Path to input MIDI file.")
    parser.add_argument(
        "--steps-per-beat",
        type=int,
        default=4,
        help="Grid steps per beat. 4 = 1/16-note grid in 4/4.",
    )
    parser.add_argument("--out", help="Optional output JSON path.")
    parser.add_argument(
        "--print-limit",
        type=int,
        default=20,
        help="How many parsed note events to print when --out is not set.",
    )
    args = parser.parse_args()

    notes = parse_midi_file(args.midi_path, steps_per_beat=args.steps_per_beat)
    note_dicts = note_events_to_dicts(notes)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(note_dicts, indent=2), encoding="utf-8")
        print(f"Saved {len(note_dicts)} note events to {out_path}")
    else:
        preview = note_dicts[: args.print_limit]
        print(json.dumps(preview, indent=2))
        print(f"\nTotal note events: {len(note_dicts)}")


if __name__ == "__main__":
    main()
