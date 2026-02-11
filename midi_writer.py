#!/usr/bin/env python3
"""
Write MIDI files from parsed event objects.

CLI:
    python midi_writer.py input.mid output.mid
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from midi_parser import (
    Event,
    KeySignatureEvent,
    NoteEvent,
    ProgramChangeEvent,
    TempoEvent,
    TimeSignatureEvent,
    parse_midi_file,
)

try:
    import mido
except ModuleNotFoundError:
    mido = None


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
