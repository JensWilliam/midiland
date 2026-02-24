#!/usr/bin/env python3
"""
Tokenizer v0 (steps + time-shifts, drums ignored upstream).

Core idea
  - Represent time in quantized "steps" (e.g. 1 step = 1/16 note).
  - Maintain a time cursor during decoding.
  - `TIME_SHIFT(n)` advances the cursor by n steps (n in 1..64).
  - `TIME_SHIFT_COARSE(n)` advances the cursor by n * coarse_unit steps (default coarse_unit=64).
  - Events that appear without an intervening time-shift are simultaneous (a chord).

Event grammar (token stream)
  [BOS]
    (TIME_SHIFT(n))*                     # move forward in time, n in 1..max_time_shift
    (TIME_SHIFT_COARSE(n))*              # move forward faster, n in 1..max_coarse_time_shift
    (EV_TS, TS_NUM(a), TS_DEN(b))*       # time signature changes at current time
    (EV_TEMPO, BPM_BIN(t))*              # tempo changes at current time
    (EV_PROG, CH(c), PROG(p))*           # program changes at current time
    (BAR, POS(s))*                       # optional context tokens (do not affect decoding)
    (EV_NOTE, CH(c), PITCH(k), DUR(d), VEL(v))*  # notes at current time
    ... repeated ...
  [EOS]

Where:
  - c in 0..15
  - p in 0..127
  - k in 0..127
  - d in 1..128 steps
  - v in 0..31 (velocity bin; 32 bins)
  - BPM_BIN in 0..63 (maps to BPM range 30..240)
  - TS_NUM in 1..16, TS_DEN in {2,4,8,16}
  - POS(s) is step-in-bar (0..127) derived from cursor + time signature

This keeps sequences compact (notes carry channel; instrument is controlled by program changes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence, TypeAlias


TokenType = Literal[
    "pad",
    "bos",
    "eos",
    "time_shift",
    "time_shift_coarse",
    "ev_note",
    "ev_prog",
    "ev_tempo",
    "ev_ts",
    "bar",
    "pos",
    "ch",
    "pitch",
    "dur",
    "vel",
    "prog",
    "bpm",
    "ts_num",
    "ts_den",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class TokenizerConfig:
    steps_per_beat: int = 4  # "steps per quarter note" for bar math
    max_time_shift: int = 64
    coarse_time_shift_unit: int = 64
    max_coarse_time_shift: int = 64
    max_duration: int = 128
    velocity_bins: int = 32
    tempo_min_bpm: int = 30
    tempo_max_bpm: int = 240
    tempo_bins: int = 64
    max_ts_numerator: int = 16
    ts_denominators: tuple[int, ...] = (2, 4, 8, 16)
    max_pos: int = 128
    strict: bool = True


@dataclass(frozen=True, slots=True)
class ProgramChange:
    step: int
    channel: int
    program: int


@dataclass(frozen=True, slots=True)
class Note:
    step: int
    channel: int
    pitch: int
    duration: int
    velocity_bin: int


@dataclass(frozen=True, slots=True)
class TempoChange:
    step: int
    bpm_bin: int


@dataclass(frozen=True, slots=True)
class TimeSignature:
    step: int
    numerator: int
    denominator: int


Event: TypeAlias = ProgramChange | TempoChange | TimeSignature | Note


def bpm_to_bin(bpm: float, *, min_bpm: int = 30, max_bpm: int = 240, bins: int = 64) -> int:
    if bins <= 1:
        return 0
    bpm_f = float(bpm)
    if bpm_f < float(min_bpm):
        bpm_f = float(min_bpm)
    if bpm_f > float(max_bpm):
        bpm_f = float(max_bpm)
    return int(round((bpm_f - float(min_bpm)) * (bins - 1) / float(max_bpm - min_bpm)))


def bpm_bin_to_bpm(bpm_bin: int, *, min_bpm: int = 30, max_bpm: int = 240, bins: int = 64) -> float:
    if bins <= 1:
        return float(min_bpm)
    b = int(bpm_bin)
    if b < 0:
        b = 0
    if b >= bins:
        b = bins - 1
    return float(min_bpm) + float(b) * float(max_bpm - min_bpm) / float(bins - 1)


def velocity_to_bin(velocity: int, *, bins: int = 32) -> int:
    velocity = int(velocity)
    if velocity < 0:
        velocity = 0
    if velocity > 127:
        velocity = 127
    if bins <= 1:
        return 0
    return int(round(velocity * (bins - 1) / 127.0))


def velocity_bin_to_velocity(velocity_bin: int, *, bins: int = 32) -> int:
    velocity_bin = int(velocity_bin)
    if velocity_bin < 0:
        velocity_bin = 0
    if velocity_bin >= bins:
        velocity_bin = bins - 1
    if bins <= 1:
        return 0
    return int(round(velocity_bin * 127.0 / (bins - 1)))


class MidiEventTokenizer:
    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self.config = config or TokenizerConfig()

        if self.config.steps_per_beat <= 0:
            raise ValueError("steps_per_beat must be > 0")
        if self.config.max_time_shift <= 0:
            raise ValueError("max_time_shift must be > 0 (and TIME_SHIFT(0) is not used).")
        if self.config.coarse_time_shift_unit <= 0:
            raise ValueError("coarse_time_shift_unit must be > 0")
        if self.config.max_coarse_time_shift <= 0:
            raise ValueError("max_coarse_time_shift must be > 0")
        if self.config.max_duration <= 0:
            raise ValueError("max_duration must be > 0")
        if self.config.velocity_bins <= 0:
            raise ValueError("velocity_bins must be > 0")
        if self.config.tempo_bins <= 0:
            raise ValueError("tempo_bins must be > 0")
        if self.config.tempo_max_bpm <= self.config.tempo_min_bpm:
            raise ValueError("tempo_max_bpm must be > tempo_min_bpm")
        if self.config.max_ts_numerator <= 0:
            raise ValueError("max_ts_numerator must be > 0")
        if not self.config.ts_denominators:
            raise ValueError("ts_denominators must be non-empty")
        if self.config.max_pos <= 0:
            raise ValueError("max_pos must be > 0")

        self._time_shift_base = 3
        self._time_shift_coarse_base = self._time_shift_base + self.config.max_time_shift
        self._ev_note_id = self._time_shift_coarse_base + self.config.max_coarse_time_shift
        self._ev_prog_id = self._ev_note_id + 1
        self._ev_tempo_id = self._ev_prog_id + 1
        self._ev_ts_id = self._ev_tempo_id + 1
        self._bar_id = self._ev_ts_id + 1

        self._ch_base = self._bar_id + 1
        self._pitch_base = self._ch_base + 16
        self._dur_base = self._pitch_base + 128
        self._vel_base = self._dur_base + self.config.max_duration
        self._prog_base = self._vel_base + self.config.velocity_bins
        self._bpm_base = self._prog_base + 128
        self._ts_num_base = self._bpm_base + self.config.tempo_bins
        self._ts_den_base = self._ts_num_base + self.config.max_ts_numerator
        self._pos_base = self._ts_den_base + len(self.config.ts_denominators)

        self.vocab_size = self._pos_base + self.config.max_pos

    def token_type(self, token_id: int) -> TokenType:
        token_id = int(token_id)
        if token_id == self.PAD:
            return "pad"
        if token_id == self.BOS:
            return "bos"
        if token_id == self.EOS:
            return "eos"
        if self._time_shift_base <= token_id < self._time_shift_coarse_base:
            return "time_shift"
        if self._time_shift_coarse_base <= token_id < self._ev_note_id:
            return "time_shift_coarse"
        if token_id == self._ev_note_id:
            return "ev_note"
        if token_id == self._ev_prog_id:
            return "ev_prog"
        if token_id == self._ev_tempo_id:
            return "ev_tempo"
        if token_id == self._ev_ts_id:
            return "ev_ts"
        if token_id == self._bar_id:
            return "bar"
        if self._ch_base <= token_id < self._pitch_base:
            return "ch"
        if self._pitch_base <= token_id < self._dur_base:
            return "pitch"
        if self._dur_base <= token_id < self._vel_base:
            return "dur"
        if self._vel_base <= token_id < self._prog_base:
            return "vel"
        if self._prog_base <= token_id < self._bpm_base:
            return "prog"
        if self._bpm_base <= token_id < self._ts_num_base:
            return "bpm"
        if self._ts_num_base <= token_id < self._ts_den_base:
            return "ts_num"
        if self._ts_den_base <= token_id < self._pos_base:
            return "ts_den"
        if self._pos_base <= token_id < self.vocab_size:
            return "pos"
        return "unknown"

    def time_shift_id(self, n: int) -> int:
        n = int(n)
        if self.config.strict:
            if not (1 <= n <= self.config.max_time_shift):
                raise ValueError(f"TIME_SHIFT out of range: {n}")
        n = max(1, min(self.config.max_time_shift, n))
        return self._time_shift_base + (n - 1)

    def time_shift_from_id(self, token_id: int) -> int:
        return int(token_id - self._time_shift_base) + 1

    def time_shift_coarse_id(self, n: int) -> int:
        n = int(n)
        if self.config.strict:
            if not (1 <= n <= self.config.max_coarse_time_shift):
                raise ValueError(f"TIME_SHIFT_COARSE out of range: {n}")
        n = max(1, min(self.config.max_coarse_time_shift, n))
        return self._time_shift_coarse_base + (n - 1)

    def time_shift_coarse_from_id(self, token_id: int) -> int:
        return int(token_id - self._time_shift_coarse_base) + 1

    def ch_id(self, c: int) -> int:
        c = int(c)
        if self.config.strict and not (0 <= c <= 15):
            raise ValueError(f"channel out of range: {c}")
        c = max(0, min(15, c))
        return self._ch_base + c

    def ch_from_id(self, token_id: int) -> int:
        return int(token_id - self._ch_base)

    def pitch_id(self, k: int) -> int:
        k = int(k)
        if self.config.strict and not (0 <= k <= 127):
            raise ValueError(f"pitch out of range: {k}")
        k = max(0, min(127, k))
        return self._pitch_base + k

    def pitch_from_id(self, token_id: int) -> int:
        return int(token_id - self._pitch_base)

    def dur_id(self, d: int) -> int:
        d = int(d)
        if self.config.strict and not (1 <= d <= self.config.max_duration):
            raise ValueError(f"duration out of range: {d}")
        d = max(1, min(self.config.max_duration, d))
        return self._dur_base + (d - 1)

    def dur_from_id(self, token_id: int) -> int:
        return int(token_id - self._dur_base) + 1

    def vel_id(self, v: int) -> int:
        v = int(v)
        if self.config.strict and not (0 <= v < self.config.velocity_bins):
            raise ValueError(f"velocity_bin out of range: {v}")
        v = max(0, min(self.config.velocity_bins - 1, v))
        return self._vel_base + v

    def vel_from_id(self, token_id: int) -> int:
        return int(token_id - self._vel_base)

    def prog_id(self, p: int) -> int:
        p = int(p)
        if self.config.strict and not (0 <= p <= 127):
            raise ValueError(f"program out of range: {p}")
        p = max(0, min(127, p))
        return self._prog_base + p

    def prog_from_id(self, token_id: int) -> int:
        return int(token_id - self._prog_base)

    def bpm_id(self, bpm_bin: int) -> int:
        b = int(bpm_bin)
        if self.config.strict and not (0 <= b < self.config.tempo_bins):
            raise ValueError(f"bpm_bin out of range: {b}")
        b = max(0, min(self.config.tempo_bins - 1, b))
        return self._bpm_base + b

    def bpm_bin_from_id(self, token_id: int) -> int:
        return int(token_id - self._bpm_base)

    def ts_num_id(self, numerator: int) -> int:
        n = int(numerator)
        if self.config.strict and not (1 <= n <= self.config.max_ts_numerator):
            raise ValueError(f"time_signature numerator out of range: {n}")
        n = max(1, min(self.config.max_ts_numerator, n))
        return self._ts_num_base + (n - 1)

    def ts_num_from_id(self, token_id: int) -> int:
        return int(token_id - self._ts_num_base) + 1

    def ts_den_id(self, denominator: int) -> int:
        d = int(denominator)
        if d not in self.config.ts_denominators:
            if self.config.strict:
                raise ValueError(f"Unsupported time_signature denominator: {d}")
            d = 4
        idx = self.config.ts_denominators.index(d)
        return self._ts_den_base + idx

    def ts_den_from_id(self, token_id: int) -> int:
        idx = int(token_id - self._ts_den_base)
        if idx < 0 or idx >= len(self.config.ts_denominators):
            if self.config.strict:
                raise ValueError("Invalid TS_DEN token.")
            return 4
        return int(self.config.ts_denominators[idx])

    def pos_id(self, pos: int) -> int:
        p = int(pos)
        if self.config.strict and not (0 <= p < self.config.max_pos):
            raise ValueError(f"POS out of range: {p}")
        p = max(0, min(self.config.max_pos - 1, p))
        return self._pos_base + p

    def pos_from_id(self, token_id: int) -> int:
        return int(token_id - self._pos_base)

    def _bar_steps(self, numerator: int, denominator: int) -> int:
        spq = int(self.config.steps_per_beat)
        num = int(numerator)
        den = int(denominator)
        # bar_steps = steps_per_quarter * (numerator * 4 / denominator)
        return int(spq * num * 4 // den)

    def encode(self, events: Iterable[Event]) -> list[int]:
        def _sort_key(e: Event) -> tuple[int, int, int, int]:
            # At the same step:
            #   TS first (affects BAR/POS context),
            #   then TEMPO,
            #   then PROG,
            #   then NOTES.
            if isinstance(e, TimeSignature):
                return (int(e.step), 0, int(e.numerator), int(e.denominator))
            if isinstance(e, TempoChange):
                return (int(e.step), 1, int(e.bpm_bin), 0)
            if isinstance(e, ProgramChange):
                return (int(e.step), 2, int(e.channel), int(e.program))
            return (int(e.step), 3, int(e.channel), int(e.pitch))

        events_sorted = sorted(events, key=_sort_key)

        tokens: list[int] = [self.BOS]
        cursor = 0
        current_ts_num = 4
        current_ts_den = 4
        last_bar_index: int | None = None
        last_step_emitted: int | None = None

        def _emit_bar_pos() -> None:
            nonlocal last_bar_index
            bar_steps = self._bar_steps(current_ts_num, current_ts_den)
            if bar_steps <= 0:
                bar_steps = self._bar_steps(4, 4)
            bar_index = int(cursor // bar_steps)
            pos = int(cursor % bar_steps)
            # Context-only: emit BAR at least once when bar changes, but don't try to count skipped bars.
            if last_bar_index is None or bar_index != last_bar_index:
                tokens.append(self._bar_id)
                last_bar_index = bar_index
            tokens.append(self.pos_id(pos))

        i = 0
        while i < len(events_sorted):
            step = int(events_sorted[i].step)
            if self.config.strict and step < 0:
                raise ValueError("event.step must be >= 0")
            if step < cursor and self.config.strict:
                raise ValueError("events must be non-decreasing in step time")

            step = max(cursor, step)
            delta = step - cursor
            while delta > 0:
                if delta >= int(self.config.coarse_time_shift_unit):
                    coarse_n = min(
                        delta // int(self.config.coarse_time_shift_unit),
                        int(self.config.max_coarse_time_shift),
                    )
                    tokens.append(self.time_shift_coarse_id(coarse_n))
                    cursor += coarse_n * int(self.config.coarse_time_shift_unit)
                    delta -= coarse_n * int(self.config.coarse_time_shift_unit)
                else:
                    shift = min(delta, int(self.config.max_time_shift))
                    tokens.append(self.time_shift_id(shift))
                    cursor += shift
                    delta -= shift

            step_events: list[Event] = []
            while i < len(events_sorted) and int(events_sorted[i].step) == step:
                step_events.append(events_sorted[i])
                i += 1

            # Apply TS events first (so BAR/POS context uses the new TS at this step).
            for e in step_events:
                if not isinstance(e, TimeSignature):
                    continue
                current_ts_num = int(e.numerator)
                current_ts_den = int(e.denominator)
                tokens.append(self._ev_ts_id)
                tokens.append(self.ts_num_id(current_ts_num))
                tokens.append(self.ts_den_id(current_ts_den))

            # Emit BAR/POS once per step with any events.
            if last_step_emitted is None or step != last_step_emitted:
                _emit_bar_pos()
                last_step_emitted = step

            # Tempo changes at this step.
            for e in step_events:
                if not isinstance(e, TempoChange):
                    continue
                tokens.append(self._ev_tempo_id)
                tokens.append(self.bpm_id(e.bpm_bin))

            # Program changes at this step.
            for e in step_events:
                if not isinstance(e, ProgramChange):
                    continue
                tokens.append(self._ev_prog_id)
                tokens.append(self.ch_id(e.channel))
                tokens.append(self.prog_id(e.program))

            # Notes at this step.
            notes_at_step = [e for e in step_events if isinstance(e, Note)]
            notes_at_step.sort(key=lambda n: (int(n.channel), int(n.pitch)))
            for e in notes_at_step:
                tokens.append(self._ev_note_id)
                tokens.append(self.ch_id(e.channel))
                tokens.append(self.pitch_id(e.pitch))
                tokens.append(self.dur_id(e.duration))
                tokens.append(self.vel_id(e.velocity_bin))

        tokens.append(self.EOS)
        return tokens

    def decode(self, tokens: Sequence[int]) -> list[Event]:
        cursor = 0
        out: list[Event] = []

        i = 0
        while i < len(tokens):
            tok = int(tokens[i])
            t = self.token_type(tok)

            if t in ("pad", "bos"):
                i += 1
                continue
            if t == "eos":
                break
            if t == "time_shift":
                cursor += self.time_shift_from_id(tok)
                i += 1
                continue
            if t == "time_shift_coarse":
                cursor += self.time_shift_coarse_from_id(tok) * int(
                    self.config.coarse_time_shift_unit
                )
                i += 1
                continue
            if t in ("bar", "pos"):
                i += 1
                continue

            if t == "ev_ts":
                if i + 2 >= len(tokens):
                    raise ValueError("Unexpected end of tokens while reading EV_TS.")
                num_tok = int(tokens[i + 1])
                den_tok = int(tokens[i + 2])
                if self.token_type(num_tok) != "ts_num":
                    raise ValueError("EV_TS expected TS_NUM token next.")
                if self.token_type(den_tok) != "ts_den":
                    raise ValueError("EV_TS expected TS_DEN token next.")
                out.append(
                    TimeSignature(
                        step=int(cursor),
                        numerator=int(self.ts_num_from_id(num_tok)),
                        denominator=int(self.ts_den_from_id(den_tok)),
                    )
                )
                i += 3
                continue

            if t == "ev_tempo":
                if i + 1 >= len(tokens):
                    raise ValueError("Unexpected end of tokens while reading EV_TEMPO.")
                bpm_tok = int(tokens[i + 1])
                if self.token_type(bpm_tok) != "bpm":
                    raise ValueError("EV_TEMPO expected BPM_BIN token next.")
                out.append(TempoChange(step=int(cursor), bpm_bin=int(self.bpm_bin_from_id(bpm_tok))))
                i += 2
                continue

            if t == "ev_prog":
                if i + 2 >= len(tokens):
                    raise ValueError("Unexpected end of tokens while reading EV_PROG.")
                ch_tok = int(tokens[i + 1])
                prog_tok = int(tokens[i + 2])
                if self.token_type(ch_tok) != "ch":
                    raise ValueError("EV_PROG expected CH token next.")
                if self.token_type(prog_tok) != "prog":
                    raise ValueError("EV_PROG expected PROG token next.")
                out.append(
                    ProgramChange(
                        step=int(cursor),
                        channel=int(self.ch_from_id(ch_tok)),
                        program=int(self.prog_from_id(prog_tok)),
                    )
                )
                i += 3
                continue

            if t == "ev_note":
                if i + 4 >= len(tokens):
                    raise ValueError("Unexpected end of tokens while reading EV_NOTE.")
                ch_tok = int(tokens[i + 1])
                pitch_tok = int(tokens[i + 2])
                dur_tok = int(tokens[i + 3])
                vel_tok = int(tokens[i + 4])
                if self.token_type(ch_tok) != "ch":
                    raise ValueError("EV_NOTE expected CH token next.")
                if self.token_type(pitch_tok) != "pitch":
                    raise ValueError("EV_NOTE expected PITCH token next.")
                if self.token_type(dur_tok) != "dur":
                    raise ValueError("EV_NOTE expected DUR token next.")
                if self.token_type(vel_tok) != "vel":
                    raise ValueError("EV_NOTE expected VEL token next.")
                out.append(
                    Note(
                        step=int(cursor),
                        channel=int(self.ch_from_id(ch_tok)),
                        pitch=int(self.pitch_from_id(pitch_tok)),
                        duration=int(self.dur_from_id(dur_tok)),
                        velocity_bin=int(self.vel_from_id(vel_tok)),
                    )
                )
                i += 5
                continue

            if self.config.strict:
                raise ValueError(f"Unexpected token {tok} ({t}) at index {i}.")
            i += 1

        return out

    def token_to_str(self, token_id: int) -> str:
        token_id = int(token_id)
        t = self.token_type(token_id)
        if t == "pad":
            return "PAD"
        if t == "bos":
            return "BOS"
        if t == "eos":
            return "EOS"
        if t == "time_shift":
            return f"TIME_SHIFT({self.time_shift_from_id(token_id)})"
        if t == "time_shift_coarse":
            n = self.time_shift_coarse_from_id(token_id)
            unit = int(self.config.coarse_time_shift_unit)
            return f"TIME_SHIFT_COARSE({n}*{unit})"
        if t == "ev_note":
            return "EV_NOTE"
        if t == "ev_prog":
            return "EV_PROG"
        if t == "ev_tempo":
            return "EV_TEMPO"
        if t == "ev_ts":
            return "EV_TS"
        if t == "bar":
            return "BAR"
        if t == "pos":
            return f"POS({self.pos_from_id(token_id)})"
        if t == "ch":
            return f"CH({self.ch_from_id(token_id)})"
        if t == "pitch":
            return f"PITCH({self.pitch_from_id(token_id)})"
        if t == "dur":
            return f"DUR({self.dur_from_id(token_id)})"
        if t == "vel":
            return f"VEL({self.vel_from_id(token_id)})"
        if t == "prog":
            return f"PROG({self.prog_from_id(token_id)})"
        if t == "bpm":
            bpm = bpm_bin_to_bpm(
                self.bpm_bin_from_id(token_id),
                min_bpm=self.config.tempo_min_bpm,
                max_bpm=self.config.tempo_max_bpm,
                bins=self.config.tempo_bins,
            )
            return f"BPM_BIN({self.bpm_bin_from_id(token_id)}~{bpm:.1f})"
        if t == "ts_num":
            return f"TS_NUM({self.ts_num_from_id(token_id)})"
        if t == "ts_den":
            return f"TS_DEN({self.ts_den_from_id(token_id)})"
        return f"UNK({token_id})"


def _self_check() -> None:
    tok = MidiEventTokenizer(
        TokenizerConfig(
            max_time_shift=64,
            max_duration=128,
            velocity_bins=32,
            tempo_min_bpm=30,
            tempo_max_bpm=240,
            tempo_bins=64,
        )
    )
    events: list[Event] = [
        TimeSignature(step=0, numerator=4, denominator=4),
        TempoChange(step=0, bpm_bin=bpm_to_bin(120, min_bpm=30, max_bpm=240, bins=64)),
        ProgramChange(step=0, channel=0, program=0),
        ProgramChange(step=0, channel=1, program=40),
        Note(step=0, channel=0, pitch=60, duration=16, velocity_bin=velocity_to_bin(100, bins=32)),
        Note(step=0, channel=0, pitch=64, duration=16, velocity_bin=velocity_to_bin(90, bins=32)),
        Note(step=8, channel=1, pitch=67, duration=32, velocity_bin=velocity_to_bin(80, bins=32)),
        TempoChange(step=8, bpm_bin=bpm_to_bin(140, min_bpm=30, max_bpm=240, bins=64)),
        ProgramChange(step=12, channel=1, program=41),
        Note(step=12, channel=1, pitch=69, duration=8, velocity_bin=velocity_to_bin(70, bins=32)),
    ]

    ids = tok.encode(events)
    back = tok.decode(ids)
    assert back == sorted(
        events,
        key=lambda e: (
            e.step,
            0
            if isinstance(e, TimeSignature)
            else 1
            if isinstance(e, TempoChange)
            else 2
            if isinstance(e, ProgramChange)
            else 3,
            getattr(e, "channel", 0),
            e.program
            if isinstance(e, ProgramChange)
            else e.bpm_bin
            if isinstance(e, TempoChange)
            else e.pitch
            if isinstance(e, Note)
            else e.numerator,
        ),
    )


if __name__ == "__main__":
    _self_check()
