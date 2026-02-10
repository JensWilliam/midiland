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

from midi_parser import Event, NoteEvent, ProgramChangeEvent, parse_midi_file

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
    message: "mido.Message"


def _event_sort_key(event: _ScheduledMessage) -> tuple[int, int, int, int]:
    return (event.abs_tick, event.priority, event.channel, event.key)


def write_midi(path_out: str, *, ticks_per_beat: int, events: list[Event]) -> None:
    """
    Write a single-track MIDI from absolute-tick events.

    Supported input events:
    - ProgramChangeEvent
    - NoteEvent (expanded to note_on + note_off)
    """
    if mido is None:
        raise ModuleNotFoundError(
            "mido is required for MIDI writing. Install it with: pip install mido"
        )

    scheduled: list[_ScheduledMessage] = []

    for event in events:
        if isinstance(event, ProgramChangeEvent):
            scheduled.append(
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

            scheduled.append(
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
            scheduled.append(
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

    scheduled.sort(key=_event_sort_key)

    midi = mido.MidiFile(type=0, ticks_per_beat=int(ticks_per_beat))
    track = mido.MidiTrack()
    midi.tracks.append(track)

    prev_abs_tick = 0
    for item in scheduled:
        delta = int(item.abs_tick - prev_abs_tick)
        if delta < 0:
            raise ValueError("Scheduled events must be non-decreasing in time.")

        msg = item.message.copy(time=delta)
        track.append(msg)
        prev_abs_tick = item.abs_tick

    track.append(mido.MetaMessage("end_of_track", time=0))
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
        include_meta=False,
        notes_only=False,
    )
    write_midi(
        str(output_path),
        ticks_per_beat=int(midi_in.ticks_per_beat),
        events=[event for event in parsed_events if isinstance(event, (NoteEvent, ProgramChangeEvent))],
    )

    print(f"Wrote MIDI: {output_path}")


if __name__ == "__main__":
    main()
