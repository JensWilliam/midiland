#!/usr/bin/env python3
"""
Step-based MIDI tokenizer (v1, Option A).

Design choices:
- Program changes are standalone tokens.
- Notes are factorized into NOTE + CH + PITCH + DUR + VEL tokens.
- Timing uses STEP_SHIFT tokens.
- Piece and window starts include BOS + metadata/program header.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TypeVar

from midi_parser import (
    KeySignatureEvent,
    MidiEvent,
    NoteEvent,
    ProgramChangeEvent,
    TempoEvent,
    TimeSignatureEvent,
    parse_midi_file,
    quantize_note_events,
    quantize_tick_to_step,
)

try:
    import mido
except ModuleNotFoundError:
    mido = None


@dataclass(slots=True)
class _WindowState:
    time_signature: tuple[int, int]
    tempo_bpm: int
    key: str
    programs: dict[int, int]


class Vocabulary:
    """Deterministic integer mapping for MIDI tokenizer tokens."""

    DEFAULT_TIME_SIGNATURES: tuple[tuple[int, int], ...] = (
        (4, 4),
        (3, 4),
        (2, 4),
        (6, 8),
        (12, 8),
        (5, 4),
        (7, 8),
        (9, 8),
        (3, 8),
        (2, 2),
        (6, 4),
    )
    DEFAULT_TEMPO_BPM_BUCKETS: tuple[int, ...] = (
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        110,
        120,
        130,
        140,
        150,
        160,
        180,
        200,
        220,
    )
    DEFAULT_KEYS: tuple[str, ...] = (
        "C",
        "G",
        "D",
        "A",
        "E",
        "B",
        "F#",
        "C#",
        "F",
        "Bb",
        "Eb",
        "Ab",
        "Db",
        "Gb",
        "Cb",
        "Am",
        "Em",
        "Bm",
        "F#m",
        "C#m",
        "G#m",
        "D#m",
        "A#m",
        "Dm",
        "Gm",
        "Cm",
        "Fm",
        "Bbm",
        "Ebm",
        "Abm",
    )

    def __init__(
        self,
        *,
        max_shift_steps: int = 64,
        max_duration_steps: int = 64,
        velocity_bins: int = 16,
        time_signatures: Sequence[tuple[int, int]] | None = None,
        tempo_bpm_buckets: Sequence[int] | None = None,
        keys: Sequence[str] | None = None,
    ) -> None:
        if max_shift_steps < 1:
            raise ValueError("max_shift_steps must be >= 1")
        if max_duration_steps < 1:
            raise ValueError("max_duration_steps must be >= 1")
        if velocity_bins < 1:
            raise ValueError("velocity_bins must be >= 1")

        self.max_shift_steps = int(max_shift_steps)
        self.max_duration_steps = int(max_duration_steps)
        self.velocity_bins = int(velocity_bins)
        self.time_signatures = tuple(time_signatures or self.DEFAULT_TIME_SIGNATURES)
        self.tempo_bpm_buckets = tuple(tempo_bpm_buckets or self.DEFAULT_TEMPO_BPM_BUCKETS)
        self.keys = tuple(keys or self.DEFAULT_KEYS)

        self._time_signature_to_index = {value: idx for idx, value in enumerate(self.time_signatures)}
        self._key_to_index = {value: idx for idx, value in enumerate(self.keys)}

        self.pad_id = 0
        self.bos_id = 1
        offset = 2

        self.time_signature_offset = offset
        offset += len(self.time_signatures)

        self.tempo_offset = offset
        offset += len(self.tempo_bpm_buckets)

        self.key_offset = offset
        offset += len(self.keys)

        self.shift_offset = offset
        offset += self.max_shift_steps

        self.program_change_offset = offset
        offset += 16 * 128

        self.note_offset = offset
        offset += 1

        self.note_channel_offset = offset
        offset += 16

        self.note_pitch_offset = offset
        offset += 128

        self.note_duration_offset = offset
        offset += self.max_duration_steps

        self.note_velocity_offset = offset
        offset += self.velocity_bins

        self.vocab_size = offset

    @property
    def default_time_signature(self) -> tuple[int, int]:
        return (4, 4)

    @property
    def default_tempo_bpm(self) -> int:
        return 120

    @property
    def default_key(self) -> str:
        return "C"

    @staticmethod
    def _clamp(value: int, lo: int, hi: int) -> int:
        return max(lo, min(hi, int(value)))

    def nearest_time_signature(self, numerator: int, denominator: int) -> tuple[int, int]:
        candidate = (int(numerator), int(denominator))
        if candidate in self._time_signature_to_index:
            return candidate

        den = max(1, int(denominator))
        target_ratio = float(int(numerator)) / float(den)
        return min(
            self.time_signatures,
            key=lambda ts: (
                abs(target_ratio - (float(ts[0]) / float(ts[1]))),
                abs(int(numerator) - ts[0]),
                abs(int(denominator) - ts[1]),
            ),
        )

    def nearest_tempo_bucket(self, bpm: int) -> int:
        bpm_i = int(bpm)
        return min(self.tempo_bpm_buckets, key=lambda value: abs(value - bpm_i))

    def normalize_key(self, key: str | None) -> str:
        if key is None:
            return self.default_key

        text = str(key).strip().replace("♯", "#").replace("♭", "b")
        if not text:
            return self.default_key

        is_minor = text.endswith("m") or text.islower()
        root = text[:-1] if text.endswith("m") else text
        if not root:
            return self.default_key

        normalized_root = root[0].upper() + root[1:]
        normalized = f"{normalized_root}m" if is_minor else normalized_root
        return normalized if normalized in self._key_to_index else self.default_key

    def duration_bucket(self, duration_step: int) -> int:
        return self._clamp(int(duration_step), 1, self.max_duration_steps)

    def velocity_bucket(self, velocity: int) -> int:
        vel = self._clamp(int(velocity), 1, 127)
        return min(self.velocity_bins - 1, ((vel - 1) * self.velocity_bins) // 127)

    def time_signature_token(self, numerator: int, denominator: int) -> int:
        value = self.nearest_time_signature(numerator, denominator)
        return self.time_signature_offset + self._time_signature_to_index[value]

    def tempo_token_from_bpm(self, bpm: int) -> int:
        bucket = self.nearest_tempo_bucket(int(bpm))
        return self.tempo_offset + self.tempo_bpm_buckets.index(bucket)

    def key_signature_token(self, key: str | None) -> int:
        normalized = self.normalize_key(key)
        return self.key_offset + self._key_to_index[normalized]

    def step_shift_token(self, shift_steps: int) -> int:
        shift = self._clamp(int(shift_steps), 1, self.max_shift_steps)
        return self.shift_offset + (shift - 1)

    def program_change_token(self, channel: int, program: int) -> int:
        channel_i = self._clamp(int(channel), 0, 15)
        program_i = self._clamp(int(program), 0, 127)
        return self.program_change_offset + channel_i * 128 + program_i

    def note_prefix_token(self) -> int:
        return self.note_offset

    def note_channel_token(self, channel: int) -> int:
        channel_i = self._clamp(int(channel), 0, 15)
        return self.note_channel_offset + channel_i

    def note_pitch_token(self, pitch: int) -> int:
        pitch_i = self._clamp(int(pitch), 0, 127)
        return self.note_pitch_offset + pitch_i

    def note_duration_token(self, duration_step: int) -> int:
        duration_i = self.duration_bucket(int(duration_step))
        return self.note_duration_offset + (duration_i - 1)

    def note_velocity_token(self, velocity: int) -> int:
        velocity_i = self.velocity_bucket(int(velocity))
        return self.note_velocity_offset + velocity_i

    def note_tokens(
        self,
        *,
        channel: int,
        pitch: int,
        duration_step: int,
        velocity: int,
    ) -> tuple[int, int, int, int, int]:
        return (
            self.note_prefix_token(),
            self.note_channel_token(channel),
            self.note_pitch_token(pitch),
            self.note_duration_token(duration_step),
            self.note_velocity_token(velocity),
        )

    def note_token(
        self,
        *,
        channel: int,
        pitch: int,
        duration_step: int,
        velocity: int,
    ) -> int:
        # Backward-compatible single-token accessor; note payload is now factorized.
        return self.note_tokens(
            channel=channel,
            pitch=pitch,
            duration_step=duration_step,
            velocity=velocity,
        )[0]

    def decode_token(self, token_id: int) -> tuple[str, object]:
        token_i = int(token_id)
        if token_i == self.pad_id:
            return ("pad", None)
        if token_i == self.bos_id:
            return ("bos", None)

        if self.time_signature_offset <= token_i < self.tempo_offset:
            idx = token_i - self.time_signature_offset
            return ("time_signature", self.time_signatures[idx])

        if self.tempo_offset <= token_i < self.key_offset:
            idx = token_i - self.tempo_offset
            return ("tempo", self.tempo_bpm_buckets[idx])

        if self.key_offset <= token_i < self.shift_offset:
            idx = token_i - self.key_offset
            return ("key_signature", self.keys[idx])

        if self.shift_offset <= token_i < self.program_change_offset:
            shift = 1 + (token_i - self.shift_offset)
            return ("step_shift", shift)

        if self.program_change_offset <= token_i < self.note_offset:
            rel = token_i - self.program_change_offset
            channel = rel // 128
            program = rel % 128
            return ("program_change", (channel, program))

        if token_i == self.note_offset:
            return ("note", None)
        if self.note_channel_offset <= token_i < self.note_pitch_offset:
            return ("note_channel", token_i - self.note_channel_offset)
        if self.note_pitch_offset <= token_i < self.note_duration_offset:
            return ("note_pitch", token_i - self.note_pitch_offset)
        if self.note_duration_offset <= token_i < self.note_velocity_offset:
            return ("note_duration", 1 + (token_i - self.note_duration_offset))
        if self.note_velocity_offset <= token_i < self.vocab_size:
            return ("note_velocity", token_i - self.note_velocity_offset)

        return ("unknown", token_i)

    def token_to_string(self, token_id: int) -> str:
        kind, payload = self.decode_token(token_id)
        if kind == "pad":
            return "PAD"
        if kind == "bos":
            return "BOS"
        if kind == "time_signature":
            ts = payload
            return f"TIME_SIGNATURE_{ts[0]}/{ts[1]}"
        if kind == "tempo":
            return f"TEMPO_{payload}BPM"
        if kind == "key_signature":
            return f"KEY_{payload}"
        if kind == "step_shift":
            return f"SHIFT_{payload}"
        if kind == "program_change":
            channel, program = payload
            return f"PROGRAM_CH{channel}_P{program}"
        if kind == "note":
            return "NOTE"
        if kind == "note_channel":
            return f"CH_{payload}"
        if kind == "note_pitch":
            return f"PITCH_{payload}"
        if kind == "note_duration":
            return f"DUR_{payload}"
        if kind == "note_velocity":
            return f"VEL_{payload}"
        return f"UNKNOWN_{token_id}"


VOCAB = Vocabulary()
T = TypeVar("T")


def _tempo_us_per_beat_to_bpm(tempo: int) -> int:
    tempo_i = int(tempo)
    if tempo_i <= 0:
        return 120
    return int(round(60_000_000.0 / float(tempo_i)))


def _collapse_step_values(pairs: Iterable[tuple[int, T]]) -> list[tuple[int, T]]:
    collapsed: list[tuple[int, T]] = []
    for step, value in sorted(pairs, key=lambda item: item[0]):
        step_i = int(step)
        if collapsed and collapsed[-1][0] == step_i:
            collapsed[-1] = (step_i, value)
        else:
            collapsed.append((step_i, value))
    return collapsed


def _initial_value_at_step_zero(events: Sequence[tuple[int, T]], default: T) -> T:
    current = default
    for step, value in events:
        if int(step) <= 0:
            current = value
        else:
            break
    return current


def _build_header_tokens(
    *,
    time_signature: tuple[int, int],
    tempo_bpm: int,
    key: str,
    program_state: dict[int, int],
    active_channels: Iterable[int],
    vocab: Vocabulary,
) -> list[int]:
    header = [
        vocab.bos_id,
        vocab.time_signature_token(int(time_signature[0]), int(time_signature[1])),
        vocab.tempo_token_from_bpm(int(tempo_bpm)),
        vocab.key_signature_token(key),
    ]
    for channel in sorted(set(int(c) for c in active_channels)):
        header.append(
            vocab.program_change_token(int(channel), int(program_state.get(int(channel), 0)))
        )
    return header


def tokenize_piece(
    events: Sequence[MidiEvent],
    ticks_per_beat: int,
    steps_per_beat: int = 4,
) -> list[int]:
    """
    Tokenize one MIDI piece event list to a flat integer token sequence.

    Option A behavior:
    - Program changes are separate tokens.
    - Notes are factorized as NOTE + CH + PITCH + DUR + VEL.
    """
    if int(steps_per_beat) <= 0:
        raise ValueError("steps_per_beat must be >= 1")
    if int(ticks_per_beat) <= 0:
        raise ValueError("ticks_per_beat must be >= 1")

    ticks_per_step = float(ticks_per_beat) / float(steps_per_beat)

    def tick_to_step(tick: int) -> int:
        return max(0, int(quantize_tick_to_step(int(tick), ticks_per_step)))

    note_events = [event for event in events if isinstance(event, NoteEvent)]
    quantized_notes = quantize_note_events(
        note_events,
        ticks_per_beat=int(ticks_per_beat),
        steps_per_beat=int(steps_per_beat),
    )

    time_signature_events = _collapse_step_values(
        (
            tick_to_step(event.time),
            VOCAB.nearest_time_signature(event.numerator, event.denominator),
        )
        for event in events
        if isinstance(event, TimeSignatureEvent)
    )
    tempo_events = _collapse_step_values(
        (
            tick_to_step(event.time),
            VOCAB.nearest_tempo_bucket(_tempo_us_per_beat_to_bpm(event.tempo)),
        )
        for event in events
        if isinstance(event, TempoEvent)
    )
    key_events = _collapse_step_values(
        (
            tick_to_step(event.time),
            VOCAB.normalize_key(event.key),
        )
        for event in events
        if isinstance(event, KeySignatureEvent)
    )

    explicit_program_by_channel: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for event in events:
        if isinstance(event, ProgramChangeEvent):
            explicit_program_by_channel[int(event.channel)].append(
                (tick_to_step(event.time), int(event.program))
            )
    for channel in list(explicit_program_by_channel.keys()):
        explicit_program_by_channel[channel] = _collapse_step_values(
            explicit_program_by_channel[channel]
        )

    notes_by_channel: dict[int, list] = defaultdict(list)
    for note in quantized_notes:
        notes_by_channel[int(note.channel)].append(note)

    inferred_program_by_channel: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for channel, channel_notes in notes_by_channel.items():
        sorted_notes = sorted(
            channel_notes,
            key=lambda note: (int(note.start_step), int(note.pitch), int(note.velocity)),
        )
        previous_program: int | None = None
        for note in sorted_notes:
            program = int(note.program)
            step = max(0, int(note.start_step))
            if previous_program is None or program != previous_program:
                inferred_program_by_channel[channel].append((step, program))
                previous_program = program
        inferred_program_by_channel[channel] = _collapse_step_values(
            inferred_program_by_channel[channel]
        )

    active_channels = set(notes_by_channel.keys())
    active_channels.update(explicit_program_by_channel.keys())
    if not active_channels and quantized_notes:
        active_channels.update(int(note.channel) for note in quantized_notes)

    initial_time_signature = _initial_value_at_step_zero(
        time_signature_events, VOCAB.default_time_signature
    )
    initial_tempo_bpm = _initial_value_at_step_zero(tempo_events, VOCAB.default_tempo_bpm)
    initial_key = _initial_value_at_step_zero(key_events, VOCAB.default_key)

    initial_program_state: dict[int, int] = {}
    for channel in sorted(active_channels):
        explicit_events = explicit_program_by_channel.get(channel, [])
        inferred_events = inferred_program_by_channel.get(channel, [])
        explicit_initial = _initial_value_at_step_zero(explicit_events, None)
        if explicit_initial is not None:
            initial_program_state[channel] = int(explicit_initial)
            continue
        if inferred_events:
            initial_program_state[channel] = int(inferred_events[0][1])
            continue
        initial_program_state[channel] = 0

    timed_tokens: list[tuple[int, int, int, int]] = []
    priority = {
        "time_signature": 0,
        "tempo": 1,
        "key_signature": 2,
        "program_change": 3,
        "note": 4,
    }
    sequence_index = 0

    def append_timed(step: int, kind: str, token_id: int) -> None:
        nonlocal sequence_index
        timed_tokens.append((int(step), priority[kind], sequence_index, int(token_id)))
        sequence_index += 1

    current_time_signature = initial_time_signature
    for step, value in time_signature_events:
        if int(step) <= 0:
            current_time_signature = value
            continue
        if value != current_time_signature:
            append_timed(
                int(step),
                "time_signature",
                VOCAB.time_signature_token(value[0], value[1]),
            )
            current_time_signature = value

    current_tempo_bpm = initial_tempo_bpm
    for step, bpm in tempo_events:
        if int(step) <= 0:
            current_tempo_bpm = int(bpm)
            continue
        if int(bpm) != int(current_tempo_bpm):
            append_timed(
                int(step),
                "tempo",
                VOCAB.tempo_token_from_bpm(int(bpm)),
            )
            current_tempo_bpm = int(bpm)

    current_key = initial_key
    for step, key in key_events:
        if int(step) <= 0:
            current_key = key
            continue
        if key != current_key:
            append_timed(
                int(step),
                "key_signature",
                VOCAB.key_signature_token(key),
            )
            current_key = key

    for channel in sorted(active_channels):
        program_sequence = explicit_program_by_channel.get(channel)
        if program_sequence:
            source_sequence = program_sequence
        else:
            source_sequence = inferred_program_by_channel.get(channel, [])

        current_program = int(initial_program_state.get(channel, 0))
        for step, program in source_sequence:
            if int(step) <= 0:
                current_program = int(program)
                continue
            if int(program) != int(current_program):
                append_timed(
                    int(step),
                    "program_change",
                    VOCAB.program_change_token(channel, program),
                )
                current_program = int(program)

    sorted_quantized_notes = sorted(
        quantized_notes,
        key=lambda note: (
            max(0, int(note.start_step)),
            int(note.channel),
            int(note.pitch),
            int(note.duration_step),
            int(note.velocity),
        ),
    )
    for note in sorted_quantized_notes:
        step = max(0, int(note.start_step))
        for token_id in VOCAB.note_tokens(
            channel=int(note.channel),
            pitch=int(note.pitch),
            duration_step=int(note.duration_step),
            velocity=int(note.velocity),
        ):
            append_timed(step, "note", token_id)

    timed_tokens.sort(key=lambda value: (value[0], value[1], value[2]))

    tokens = _build_header_tokens(
        time_signature=initial_time_signature,
        tempo_bpm=initial_tempo_bpm,
        key=initial_key,
        program_state=initial_program_state,
        active_channels=active_channels,
        vocab=VOCAB,
    )

    current_step = 0
    for step, _, _, token_id in timed_tokens:
        delta = int(step) - int(current_step)
        while delta > 0:
            shift = min(VOCAB.max_shift_steps, delta)
            tokens.append(VOCAB.step_shift_token(shift))
            delta -= shift
        tokens.append(int(token_id))
        current_step = int(step)

    return tokens


def _extract_initial_state(tokens: Sequence[int], vocab: Vocabulary) -> tuple[_WindowState, int]:
    state = _WindowState(
        time_signature=vocab.default_time_signature,
        tempo_bpm=vocab.default_tempo_bpm,
        key=vocab.default_key,
        programs={},
    )
    index = 0
    if tokens and int(tokens[0]) == vocab.bos_id:
        index = 1

    while index < len(tokens):
        kind, payload = vocab.decode_token(int(tokens[index]))
        if kind == "time_signature":
            state.time_signature = (int(payload[0]), int(payload[1]))
            index += 1
            continue
        if kind == "tempo":
            state.tempo_bpm = int(payload)
            index += 1
            continue
        if kind == "key_signature":
            state.key = str(payload)
            index += 1
            continue
        if kind == "program_change":
            channel, program = payload
            state.programs[int(channel)] = int(program)
            index += 1
            continue
        break

    return state, index


def _collect_piece_channels(
    tokens: Sequence[int], vocab: Vocabulary, initial_programs: dict[int, int]
) -> set[int]:
    channels = set(int(channel) for channel in initial_programs.keys())
    for token in tokens:
        kind, payload = vocab.decode_token(int(token))
        if kind == "program_change":
            channels.add(int(payload[0]))
            continue
        if kind == "note_channel":
            channels.add(int(payload))
    return channels


def _update_state_from_token(state: _WindowState, token: int, vocab: Vocabulary) -> None:
    kind, payload = vocab.decode_token(int(token))
    if kind == "time_signature":
        state.time_signature = (int(payload[0]), int(payload[1]))
        return
    if kind == "tempo":
        state.tempo_bpm = int(payload)
        return
    if kind == "key_signature":
        state.key = str(payload)
        return
    if kind == "program_change":
        channel, program = payload
        state.programs[int(channel)] = int(program)


def split_windows(
    tokens: Sequence[int],
    window_size: int = 4096,
    overlap: int = 512,
    min_len: int = 1024,
    pad_id: int | None = None,
) -> list[list[int]]:
    """Split token stream into overlapping windows with BOS + header at each start."""
    if int(window_size) <= 0:
        raise ValueError("window_size must be >= 1")
    if int(overlap) < 0:
        raise ValueError("overlap must be >= 0")
    if int(overlap) >= int(window_size):
        raise ValueError("overlap must be < window_size")

    if not tokens:
        return []

    if pad_id is None:
        pad_id = VOCAB.pad_id

    initial_state, header_end = _extract_initial_state(tokens, VOCAB)
    piece_channels = _collect_piece_channels(tokens, VOCAB, initial_state.programs)
    stride = int(window_size) - int(overlap)

    windows: list[list[int]] = []
    state = _WindowState(
        time_signature=initial_state.time_signature,
        tempo_bpm=initial_state.tempo_bpm,
        key=initial_state.key,
        programs=dict(initial_state.programs),
    )
    cursor = int(header_end)

    for start in range(0, len(tokens), stride):
        while cursor < start:
            _update_state_from_token(state, int(tokens[cursor]), VOCAB)
            cursor += 1

        header = _build_header_tokens(
            time_signature=state.time_signature,
            tempo_bpm=state.tempo_bpm,
            key=state.key,
            program_state=state.programs,
            active_channels=piece_channels,
            vocab=VOCAB,
        )
        content_budget = int(window_size) - len(header)
        if content_budget <= 0:
            raise ValueError(
                "window_size is too small for BOS + metadata/program header size."
            )

        content_start = max(int(start), int(header_end))
        content = list(tokens[content_start : content_start + content_budget])
        window = header + content

        if len(window) < int(min_len):
            continue
        if len(window) < int(window_size):
            window.extend([int(pad_id)] * (int(window_size) - len(window)))
        windows.append(window)

    return windows


def _preview_tokens(tokens: Sequence[int], *, limit: int = 24) -> list[str]:
    preview: list[str] = []
    max_tokens = max(0, int(limit))
    upper_bound = min(len(tokens), max_tokens)
    index = 0

    while index < upper_bound:
        token_id = int(tokens[index])
        kind, _ = VOCAB.decode_token(token_id)
        if kind == "note" and index + 4 < upper_bound:
            ch_kind, channel = VOCAB.decode_token(int(tokens[index + 1]))
            pitch_kind, pitch = VOCAB.decode_token(int(tokens[index + 2]))
            dur_kind, duration = VOCAB.decode_token(int(tokens[index + 3]))
            vel_kind, velocity_bucket = VOCAB.decode_token(int(tokens[index + 4]))
            if (
                ch_kind == "note_channel"
                and pitch_kind == "note_pitch"
                and dur_kind == "note_duration"
                and vel_kind == "note_velocity"
            ):
                preview.append(
                    f"NOTE_CH{int(channel)}_P{int(pitch)}_D{int(duration)}_V{int(velocity_bucket)}"
                )
                index += 5
                continue

        preview.append(VOCAB.token_to_string(token_id))
        index += 1

    return preview


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenize MIDI into integer IDs.")
    parser.add_argument("midi_path", help="Path to input .mid file.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output path for saving token IDs as a .npy file.",
    )
    parser.add_argument("--steps-per-beat", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=4096)
    parser.add_argument("--overlap", type=int, default=512)
    parser.add_argument("--min-len", type=int, default=1024)
    parser.add_argument("--preview", type=int, default=24, help="How many tokens to preview.")
    args = parser.parse_args()

    if mido is None:
        raise ModuleNotFoundError(
            "mido is required for tokenization CLI. Install it with: pip install mido"
        )

    midi_path = Path(args.midi_path)
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
        steps_per_beat=int(args.steps_per_beat),
    )
    windows = split_windows(
        tokens,
        window_size=int(args.window_size),
        overlap=int(args.overlap),
        min_len=int(args.min_len),
        pad_id=VOCAB.pad_id,
    )

    print(f"Token count: {len(tokens)}")
    print(f"Window count: {len(windows)}")
    print(f"Vocabulary size: {VOCAB.vocab_size}")
    print(f"Preview token IDs: {list(tokens[: max(0, int(args.preview))])}")
    print(f"Preview decoded: {_preview_tokens(tokens, limit=int(args.preview))}")

    if windows:
        first_window = windows[0][: max(0, int(args.preview))]
        print(f"First window prefix IDs: {first_window}")
        print(f"First window prefix decoded: {_preview_tokens(first_window, limit=int(args.preview))}")

    if args.out:
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "numpy is required to save .npy output. Install it with: pip install numpy"
            ) from exc
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, np.asarray(tokens, dtype=np.int64))
        print(f"Saved token IDs to: {out_path}")


if __name__ == "__main__":
    main()
