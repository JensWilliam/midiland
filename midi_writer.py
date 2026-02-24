#!/usr/bin/env python3
"""
Parse and write MIDI files from/to simple event objects.

CLI:
    python midi_writer.py input.mid output.mid
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable, TypeAlias

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

    channel_program = {channel: 0 for channel in range(16)}
    active_notes: DefaultDict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(
        list
    )

    events: list[MidiEvent] = []
    abs_tick = 0

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
            continue

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
            return (
                int(e.time),
                order[TimeSignatureEvent],
                int(e.numerator),
                int(e.denominator),
            )
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
    ticks_per_step = int(ticks_per_beat) / float(int(steps_per_beat))
    quantized: list[QuantizedNoteEvent] = []
    for note in notes:
        start_step = int(round(int(note.time) / ticks_per_step))
        end_step = int(round(int(note.time + note.duration_ticks) / ticks_per_step))
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


@dataclass(frozen=True, slots=True)
class _ScheduledMessage:
    abs_tick: int
    priority: int
    channel: int
    key: int
    message: "mido.Message | mido.MetaMessage"


def _event_sort_key(event: _ScheduledMessage) -> tuple[int, int, int, int]:
    return (event.abs_tick, event.priority, event.channel, event.key)


def write_midi(path_out: str, *, ticks_per_beat: int, events: list[Event]) -> None:
    """
    Write a type-1 MIDI from absolute-tick events.

    Supported input events:
    - TempoEvent (track 0 conductor)
    - TimeSignatureEvent (track 0 conductor)
    - KeySignatureEvent (track 0 conductor)
    - ProgramChangeEvent
    - NoteEvent (expanded to note_on + note_off, track 1)
    """
    if mido is None:
        raise ModuleNotFoundError(
            "mido is required for MIDI writing. Install it with: pip install mido"
        )

    conductor_scheduled: list[_ScheduledMessage] = []
    performance_scheduled: list[_ScheduledMessage] = []

    for event in events:
        if isinstance(event, TempoEvent):
            conductor_scheduled.append(
                _ScheduledMessage(
                    abs_tick=int(event.time),
                    priority=0,
                    channel=-1,
                    key=int(event.tempo),
                    message=mido.MetaMessage(
                        "set_tempo",
                        tempo=int(event.tempo),
                        time=0,
                    ),
                )
            )
            continue

        if isinstance(event, TimeSignatureEvent):
            conductor_scheduled.append(
                _ScheduledMessage(
                    abs_tick=int(event.time),
                    priority=1,
                    channel=-1,
                    key=int(event.numerator),
                    message=mido.MetaMessage(
                        "time_signature",
                        numerator=int(event.numerator),
                        denominator=int(event.denominator),
                        time=0,
                    ),
                )
            )
            continue

        if isinstance(event, KeySignatureEvent):
            conductor_scheduled.append(
                _ScheduledMessage(
                    abs_tick=int(event.time),
                    priority=2,
                    channel=-1,
                    key=0,
                    message=mido.MetaMessage(
                        "key_signature",
                        key=str(event.key),
                        time=0,
                    ),
                )
            )
            continue

        if isinstance(event, ProgramChangeEvent):
            performance_scheduled.append(
                _ScheduledMessage(
                    abs_tick=int(event.time),
                    priority=0,  # program_change first at same tick
                    channel=int(event.channel),
                    key=int(event.program),
                    message=mido.Message(
                        "program_change",
                        channel=int(event.channel),
                        program=int(event.program),
                        time=0,
                    ),
                )
            )
            continue

        if isinstance(event, NoteEvent):
            start_tick = int(event.time)
            end_tick = int(event.time + max(1, int(event.duration_ticks)))

            performance_scheduled.append(
                _ScheduledMessage(
                    abs_tick=start_tick,
                    priority=1,  # note_on second at same tick
                    channel=int(event.channel),
                    key=int(event.pitch),
                    message=mido.Message(
                        "note_on",
                        channel=int(event.channel),
                        note=int(event.pitch),
                        velocity=int(event.velocity),
                        time=0,
                    ),
                )
            )
            performance_scheduled.append(
                _ScheduledMessage(
                    abs_tick=end_tick,
                    priority=2,  # note_off last at same tick
                    channel=int(event.channel),
                    key=int(event.pitch),
                    message=mido.Message(
                        "note_off",
                        channel=int(event.channel),
                        note=int(event.pitch),
                        velocity=0,
                        time=0,
                    ),
                )
            )

    conductor_scheduled.sort(key=_event_sort_key)
    performance_scheduled.sort(key=_event_sort_key)

    def _append_with_delta_times(
        track: "mido.MidiTrack", scheduled_events: list[_ScheduledMessage]
    ) -> None:
        prev_abs_tick = 0
        for item in scheduled_events:
            delta = int(item.abs_tick - prev_abs_tick)
            if delta < 0:
                raise ValueError("Scheduled events must be non-decreasing in time.")

            msg = item.message.copy(time=delta)
            track.append(msg)
            prev_abs_tick = item.abs_tick

        track.append(mido.MetaMessage("end_of_track", time=0))

    midi = mido.MidiFile(type=1, ticks_per_beat=int(ticks_per_beat))
    conductor_track = mido.MidiTrack()
    performance_track = mido.MidiTrack()
    midi.tracks.append(conductor_track)
    midi.tracks.append(performance_track)

    _append_with_delta_times(conductor_track, conductor_scheduled)
    _append_with_delta_times(performance_track, performance_scheduled)
    midi.save(path_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round-trip MIDI via parsed events.")
    parser.add_argument("input_mid", help="Path to input MIDI file.")
    parser.add_argument("output_mid", help="Path to output MIDI file.")
    args = parser.parse_args()

    if mido is None:
        raise ModuleNotFoundError(
            "mido is required for MIDI writing. Install it with: pip install mido"
        )

    input_path = Path(args.input_mid)
    output_path = Path(args.output_mid)

    midi_in = mido.MidiFile(str(input_path))
    parsed_events = parse_midi_file(
        input_path,
        ignore_drums=True,
        include_meta=True,
        notes_only=False,
    )
    write_midi(
        str(output_path),
        ticks_per_beat=int(midi_in.ticks_per_beat),
        events=[
            event
            for event in parsed_events
            if isinstance(
                event,
                (
                    NoteEvent,
                    ProgramChangeEvent,
                    TempoEvent,
                    TimeSignatureEvent,
                    KeySignatureEvent,
                ),
            )
        ],
    )

    print(f"Wrote MIDI: {output_path}")


if __name__ == "__main__":
    main()
