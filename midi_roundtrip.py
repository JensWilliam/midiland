#!/usr/bin/env python3
"""
Round-trip test: MIDI -> (quantized+binned events) -> tokens -> events -> MIDI.

This is meant as a "does my representation actually work?" sanity check.

Important idea: the representation is *lossy* (quantization + velocity bins + tempo bins),
so the right notion of "strict" is:
  canonical(midi_in) == canonical(midi_out)

Where canonical() means "parse MIDI, quantize to steps, and bin values according to tokenizer config".
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterable

from tokenizer import (
    MidiEventTokenizer,
    Note,
    ProgramChange,
    TempoChange,
    TimeSignature,
    TokenizerConfig,
    bpm_to_bin,
    velocity_to_bin,
)

try:
    import mido
except ModuleNotFoundError:
    mido = None


def _tempo_to_bpm(tempo_us_per_beat: int) -> float:
    if mido is not None and hasattr(mido, "tempo2bpm"):
        return float(mido.tempo2bpm(int(tempo_us_per_beat)))
    return 60_000_000.0 / float(int(tempo_us_per_beat))


def _bpm_to_tempo(bpm: float) -> int:
    if mido is not None and hasattr(mido, "bpm2tempo"):
        return int(mido.bpm2tempo(float(bpm)))
    return int(round(60_000_000.0 / float(bpm)))


def midi_to_canonical_events(
    midi_path: str | Path,
    *,
    cfg: TokenizerConfig,
    ignore_drums: bool = True,
) -> tuple[list[object], int]:
    """
    Parse a MIDI file to the tokenizer's lossy canonical event list.

    Returns: (events, ticks_per_beat)
    """
    if mido is None:
        raise ModuleNotFoundError("mido is required. Install with: pip install mido")

    midi = mido.MidiFile(str(midi_path))
    ticks_per_beat = int(midi.ticks_per_beat)
    ticks_per_step = float(ticks_per_beat) / float(int(cfg.steps_per_beat))

    channel_program = {ch: 0 for ch in range(16)}
    active_notes: DefaultDict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    events: list[object] = []
    abs_tick = 0

    for msg in mido.merge_tracks(midi.tracks):
        abs_tick += msg.time

        if msg.is_meta:
            if msg.type == "set_tempo" and hasattr(msg, "tempo"):
                bpm = _tempo_to_bpm(int(msg.tempo))
                bpm_bin = bpm_to_bin(
                    bpm,
                    min_bpm=int(cfg.tempo_min_bpm),
                    max_bpm=int(cfg.tempo_max_bpm),
                    bins=int(cfg.tempo_bins),
                )
                step = int(round(abs_tick / ticks_per_step))
                events.append(TempoChange(step=step, bpm_bin=bpm_bin))
            elif msg.type == "time_signature" and hasattr(msg, "numerator"):
                numerator = int(getattr(msg, "numerator", 4))
                denominator = int(getattr(msg, "denominator", 4))
                if denominator not in cfg.ts_denominators:
                    # Keep it simple: clamp unsupported denominators to 4/4.
                    numerator, denominator = 4, 4
                numerator = max(1, min(int(cfg.max_ts_numerator), numerator))
                step = int(round(abs_tick / ticks_per_step))
                events.append(TimeSignature(step=step, numerator=numerator, denominator=denominator))
            continue

        if not hasattr(msg, "channel"):
            continue
        channel = int(msg.channel)
        if ignore_drums and channel == 9:
            continue

        if msg.type == "program_change":
            channel_program[channel] = int(msg.program)
            step = int(round(abs_tick / ticks_per_step))
            events.append(ProgramChange(step=step, channel=channel, program=int(msg.program)))
            continue

        if msg.type == "note_on" and int(msg.velocity) > 0:
            active_notes[(channel, int(msg.note))].append((abs_tick, int(msg.velocity)))
            continue

        is_note_off = msg.type == "note_off" or (msg.type == "note_on" and int(msg.velocity) == 0)
        if not is_note_off:
            continue

        key = (channel, int(msg.note))
        if not active_notes[key]:
            continue

        start_tick, start_vel = active_notes[key].pop(0)
        end_tick = max(start_tick + 1, abs_tick)
        start_step = int(round(start_tick / ticks_per_step))
        end_step = int(round(end_tick / ticks_per_step))
        duration_steps = max(1, int(end_step - start_step))
        duration_steps = min(int(cfg.max_duration), duration_steps)

        vel_bin = velocity_to_bin(start_vel, bins=int(cfg.velocity_bins))
        events.append(
            Note(
                step=int(start_step),
                channel=int(channel),
                pitch=int(msg.note),
                duration=int(duration_steps),
                velocity_bin=int(vel_bin),
            )
        )

    # Stable ordering, consistent with tokenizer assumptions.
    def _key(e: object) -> tuple[int, int, int, int]:
        step = int(getattr(e, "step"))
        if isinstance(e, TimeSignature):
            return (step, 0, int(e.numerator), int(e.denominator))
        if isinstance(e, TempoChange):
            return (step, 1, int(e.bpm_bin), 0)
        if isinstance(e, ProgramChange):
            return (step, 2, int(e.channel), int(e.program))
        if isinstance(e, Note):
            return (step, 3, int(e.channel), int(e.pitch))
        return (step, 9, 0, 0)

    events.sort(key=_key)
    return events, ticks_per_beat


def canonical_events_to_midi(
    events: Iterable[object],
    *,
    cfg: TokenizerConfig,
    ticks_per_beat: int,
    out_path: str | Path,
) -> None:
    if mido is None:
        raise ModuleNotFoundError("mido is required. Install with: pip install mido")

    ticks_per_step = float(int(ticks_per_beat)) / float(int(cfg.steps_per_beat))

    def _step_to_tick(step: int) -> int:
        return int(round(int(step) * ticks_per_step))

    scheduled: list[tuple[int, int, mido.Message | mido.MetaMessage]] = []
    # priority: TS(0) < TEMPO(1) < PROG(2) < NOTE_ON(3) < NOTE_OFF(4)

    for e in events:
        if isinstance(e, TimeSignature):
            scheduled.append(
                (
                    _step_to_tick(e.step),
                    0,
                    mido.MetaMessage(
                        "time_signature",
                        numerator=int(e.numerator),
                        denominator=int(e.denominator),
                        time=0,
                    ),
                )
            )
        elif isinstance(e, TempoChange):
            bpm = float(
                30
                + (int(e.bpm_bin) * (int(cfg.tempo_max_bpm) - int(cfg.tempo_min_bpm)))
                / float(int(cfg.tempo_bins) - 1)
            )
            scheduled.append(
                (
                    _step_to_tick(e.step),
                    1,
                    mido.MetaMessage("set_tempo", tempo=_bpm_to_tempo(bpm), time=0),
                )
            )
        elif isinstance(e, ProgramChange):
            scheduled.append(
                (
                    _step_to_tick(e.step),
                    2,
                    mido.Message(
                        "program_change",
                        channel=int(e.channel),
                        program=int(e.program),
                        time=0,
                    ),
                )
            )
        elif isinstance(e, Note):
            start_tick = _step_to_tick(e.step)
            end_tick = _step_to_tick(int(e.step) + int(e.duration))
            end_tick = max(start_tick + 1, end_tick)
            velocity = int(round(int(e.velocity_bin) * 127 / max(1, int(cfg.velocity_bins) - 1)))
            scheduled.append(
                (
                    start_tick,
                    3,
                    mido.Message(
                        "note_on",
                        channel=int(e.channel),
                        note=int(e.pitch),
                        velocity=int(velocity),
                        time=0,
                    ),
                )
            )
            scheduled.append(
                (
                    end_tick,
                    4,
                    mido.Message(
                        "note_off",
                        channel=int(e.channel),
                        note=int(e.pitch),
                        velocity=0,
                        time=0,
                    ),
                )
            )

    scheduled.sort(key=lambda x: (x[0], x[1]))

    track = mido.MidiTrack()
    prev = 0
    for abs_tick, _prio, msg in scheduled:
        delta = int(abs_tick - prev)
        if delta < 0:
            delta = 0
        track.append(msg.copy(time=delta))
        prev = abs_tick
    track.append(mido.MetaMessage("end_of_track", time=0))

    midi = mido.MidiFile(type=1, ticks_per_beat=int(ticks_per_beat))
    midi.tracks.append(track)
    midi.save(str(out_path))


def _diff(a: list[object], b: list[object], *, limit: int = 30) -> str:
    if a == b:
        return "OK (canonical event lists match)"
    lines: list[str] = []
    lines.append(f"DIFF: canonical event lists differ (len {len(a)} vs {len(b)})")
    for i in range(min(len(a), len(b), limit)):
        if a[i] != b[i]:
            lines.append(f"- a[{i}] = {a[i]!r}")
            lines.append(f"+ b[{i}] = {b[i]!r}")
            break
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Round-trip test MIDI <-> tokens <-> MIDI.")
    p.add_argument("input_mid")
    p.add_argument(
        "--out",
        help="Optional output .mid path (default: input filename + _out).",
    )
    p.add_argument("--steps-per-beat", type=int, default=4)
    p.add_argument("--ignore-drums", action="store_true", default=True)
    p.add_argument("--print-tokens", type=int, default=80)
    args = p.parse_args()

    cfg = TokenizerConfig(steps_per_beat=int(args.steps_per_beat))
    tok = MidiEventTokenizer(cfg)

    input_path = Path(args.input_mid)
    if args.out:
        output_path = Path(args.out)
    else:
        suffix = input_path.suffix if input_path.suffix else ".mid"
        output_path = input_path.with_name(f"{input_path.stem}_out{suffix}")

    events1, tpb = midi_to_canonical_events(
        input_path, cfg=cfg, ignore_drums=bool(args.ignore_drums)
    )
    tokens = tok.encode(events1)  # type: ignore[arg-type]
    events2 = tok.decode(tokens)

    # tokenizer strictness check (should match exactly because decode(encode(canonical)) is deterministic)
    if list(events2) != list(events1):
        print("Tokenizer mismatch (unexpected):")
        print(_diff(list(events1), list(events2)))
        raise SystemExit(2)

    # write MIDI, then canonicalize again and compare
    canonical_events_to_midi(events2, cfg=cfg, ticks_per_beat=tpb, out_path=output_path)
    events3, _tpb2 = midi_to_canonical_events(
        output_path, cfg=cfg, ignore_drums=bool(args.ignore_drums)
    )
    print(_diff(list(events1), list(events3)))
    print(f"\nWrote: {output_path}")

    if args.print_tokens > 0:
        preview = tokens[: int(args.print_tokens)]
        print("\nToken preview:")
        print(" ".join(tok.token_to_str(t) for t in preview))


if __name__ == "__main__":
    main()
