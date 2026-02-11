#!/usr/bin/env python3
"""
Minimal MIDI parser for a drum-free symbolic dataset.

What this file does:
1. Loads a MIDI file with `mido`.
2. Ignores all channel-10 events (channel index 9).
3. Tracks program changes per channel.
4. Extracts typed events in absolute MIDI ticks (Note + Program Change by default).
5. Optionally converts notes to a quantized step grid (default: 1/16 notes).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Iterable, Literal, TypeAlias

try:
    import mido
except ModuleNotFoundError:
    mido = None


@dataclass(frozen=True, slots=True)
class Event:
    """Base event with absolute time in MIDI ticks."""

    time: int


@dataclass(frozen=True, slots=True)
class NoteEvent(Event):
    """A note with onset time and duration in MIDI ticks."""

    duration_ticks: int
    channel: int
    program: int
    pitch: int
    velocity: int


@dataclass(frozen=True, slots=True)
class ProgramChangeEvent(Event):
    """Program change (instrument) event in MIDI ticks."""

    channel: int
    program: int


@dataclass(frozen=True, slots=True)
class TempoEvent(Event):
    """Tempo (microseconds per beat) meta event."""

    tempo: int


@dataclass(frozen=True, slots=True)
class TimeSignatureEvent(Event):
    """Time signature meta event."""

    numerator: int
    denominator: int


@dataclass(frozen=True, slots=True)
class KeySignatureEvent(Event):
    """Key signature meta event."""

    key: str


MidiEvent: TypeAlias = (
    NoteEvent | ProgramChangeEvent | TempoEvent | TimeSignatureEvent | KeySignatureEvent
)


@dataclass(frozen=True, slots=True)
class QuantizedNoteEvent:
    """Single symbolic note event after quantization."""

    pitch: int
    start_step: int
    duration_step: int
    velocity: int
    program: int
    channel: int


def quantize_tick_to_step(tick: int, ticks_per_step: float) -> int:
    """Convert absolute MIDI ticks to integer grid steps."""
    return int(round(tick / ticks_per_step))


def parse_midi_file(
    midi_path: str | Path,
    *,
    ignore_drums: bool = True,
    include_meta: bool = False,
    notes_only: bool = False,
) -> list[MidiEvent] | list[NoteEvent]:
    """
    Parse a MIDI file into typed events with absolute MIDI tick timing.

    Args:
        midi_path: Input .mid file path.
        ignore_drums: If True, drop channel 10 (index 9) events.
        include_meta: If True, also include tempo/time/key-signature meta events.
        notes_only: If True, return only NoteEvent objects.
    """
    if mido is None:
        raise ModuleNotFoundError(
            "mido is required for MIDI parsing. Install it with: pip install mido"
        )

    midi = mido.MidiFile(str(midi_path))

    # Program is tracked independently for each channel (MIDI default program = 0).
    channel_program = {channel: 0 for channel in range(16)}

    # Handles overlapping same-pitch notes: key=(channel, pitch), value=list of active note-ons.
    active_notes: DefaultDict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)

    events: list[MidiEvent] = []
    abs_tick = 0

    # `merge_tracks` gives a single time-ordered stream (delta times preserved).
    for msg in mido.merge_tracks(midi.tracks):
        abs_tick += msg.time

        if msg.is_meta:
            if include_meta:
                if msg.type == "set_tempo" and hasattr(msg, "tempo"):
                    events.append(TempoEvent(time=int(abs_tick), tempo=int(msg.tempo)))
                elif msg.type == "time_signature" and hasattr(msg, "numerator"):
                    events.append(
                        TimeSignatureEvent(
                            time=int(abs_tick),
                            numerator=int(msg.numerator),
                            denominator=int(msg.denominator),
                        )
                    )
                elif msg.type == "key_signature" and hasattr(msg, "key"):
                    events.append(KeySignatureEvent(time=int(abs_tick), key=str(msg.key)))
            continue

        if not hasattr(msg, "channel"):
            continue

        channel = msg.channel

        # Hard drum filter: channel 10 in MIDI UI is index 9.
        if ignore_drums and channel == 9:
            continue

        if msg.type == "program_change":
            channel_program[channel] = msg.program
            events.append(
                ProgramChangeEvent(
                    time=int(abs_tick), channel=int(channel), program=int(msg.program)
                )
            )
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
        duration_ticks = max(1, int(end_tick - start_tick))

        events.append(
            NoteEvent(
                time=int(start_tick),
                duration_ticks=int(duration_ticks),
                channel=int(channel),
                program=int(program),
                pitch=int(msg.note),
                velocity=int(velocity),
            )
        )

    def _event_sort_key(e: MidiEvent) -> tuple[int, int, int, int]:
        order: dict[type, int] = {
            TempoEvent: 0,
            TimeSignatureEvent: 1,
            KeySignatureEvent: 2,
            ProgramChangeEvent: 3,
            NoteEvent: 4,
        }
        if isinstance(e, NoteEvent):
            return (int(e.time), order[NoteEvent], int(e.channel), int(e.pitch))
        if isinstance(e, ProgramChangeEvent):
            return (int(e.time), order[ProgramChangeEvent], int(e.channel), int(e.program))
        if isinstance(e, TempoEvent):
            return (int(e.time), order[TempoEvent], int(e.tempo), 0)
        if isinstance(e, TimeSignatureEvent):
            return (int(e.time), order[TimeSignatureEvent], int(e.numerator), int(e.denominator))
        return (int(e.time), order[KeySignatureEvent], 0, 0)

    events.sort(key=_event_sort_key)
    if notes_only:
        return [e for e in events if isinstance(e, NoteEvent)]
    return events


def quantize_note_events(
    notes: Iterable[NoteEvent],
    *,
    ticks_per_beat: int,
    steps_per_beat: int = 4,
) -> list[QuantizedNoteEvent]:
    """
    Convert tick-timed note events to a fixed step grid.

    Args:
        notes: NoteEvent items with absolute tick timing.
        ticks_per_beat: MIDI ticks per beat (from MidiFile.ticks_per_beat).
        steps_per_beat: Grid resolution. 4 means 1/16 notes in 4/4.
    """
    ticks_per_step = ticks_per_beat / float(steps_per_beat)

    quantized: list[QuantizedNoteEvent] = []
    for note in notes:
        start_step = quantize_tick_to_step(int(note.time), ticks_per_step)
        end_step = quantize_tick_to_step(int(note.time + note.duration_ticks), ticks_per_step)
        duration_step = max(1, int(end_step - start_step))
        quantized.append(
            QuantizedNoteEvent(
                pitch=int(note.pitch),
                start_step=int(start_step),
                duration_step=int(duration_step),
                velocity=int(note.velocity),
                program=int(note.program),
                channel=int(note.channel),
            )
        )

    quantized.sort(key=lambda n: (n.start_step, n.pitch, n.program, n.channel))
    return quantized


def events_to_dicts(events: Iterable[MidiEvent]) -> list[dict]:
    """Convert events to plain dicts (JSON-friendly), including an explicit type."""

    def _type_name(
        e: MidiEvent,
    ) -> Literal["note", "program_change", "tempo", "time_signature", "key_signature"]:
        if isinstance(e, NoteEvent):
            return "note"
        if isinstance(e, ProgramChangeEvent):
            return "program_change"
        if isinstance(e, TempoEvent):
            return "tempo"
        if isinstance(e, TimeSignatureEvent):
            return "time_signature"
        return "key_signature"

    out: list[dict] = []
    for event in events:
        payload = asdict(event)
        payload["type"] = _type_name(event)
        out.append(payload)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse MIDI into typed tick-timed events.")
    parser.add_argument("midi_path", help="Path to input MIDI file.")
    parser.add_argument(
        "--include-meta",
        action="store_true",
        help="Include tempo/time signature/key signature.",
    )
    parser.add_argument("--keep-drums", action="store_true", help="Do not filter channel 10.")
    parser.add_argument(
        "--format",
        choices=["events", "quantized_notes"],
        default="events",
        help="Output format: full event stream or quantized note events.",
    )
    parser.add_argument(
        "--steps-per-beat",
        type=int,
        default=4,
        help="Quantization grid steps per beat (used with --format=quantized_notes).",
    )
    parser.add_argument("--out", help="Optional output JSON path.")
    parser.add_argument(
        "--print-limit",
        type=int,
        default=20,
        help="How many parsed note events to print when --out is not set.",
    )
    args = parser.parse_args()

    midi = mido.MidiFile(str(args.midi_path)) if mido is not None else None
    events = parse_midi_file(
        args.midi_path,
        ignore_drums=not args.keep_drums,
        include_meta=bool(args.include_meta),
        notes_only=False,
    )

    if args.format == "quantized_notes":
        if midi is None:
            raise RuntimeError("mido MidiFile load failed unexpectedly.")
        notes = [e for e in events if isinstance(e, NoteEvent)]
        quantized = quantize_note_events(
            notes, ticks_per_beat=int(midi.ticks_per_beat), steps_per_beat=int(args.steps_per_beat)
        )
        payload = [asdict(n) for n in quantized]
    else:
        payload = events_to_dicts(events)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved {len(payload)} items to {out_path}")
    else:
        preview = payload[: args.print_limit]
        print(json.dumps(preview, indent=2))
        print(f"\nTotal items: {len(payload)}")


if __name__ == "__main__":
    main()
