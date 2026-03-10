# Current MIDI Pipeline

This document describes the MIDI AI pipeline that exists in this repository today. It is intentionally descriptive rather than aspirational: it explains the code paths that are present now, the data they pass around, and the places where behavior is implicit or fragile.

## High-level shape

The current pipeline is split across a small set of modules:

- `midiland/midi_io.py`: MIDI file parsing into a canonical event list, and canonical event list back to MIDI.
- `midiland/tokenizer.py`: event dataclasses, tokenizer configuration, event-to-token encoding, and token-to-event decoding.
- `midiland/cli/preprocess_dataset.py`: batch conversion from a MIDI folder to one token array per source file.
- `midiland/cli/make_windows.py`: conversion from full-document token arrays to fixed-length training windows.
- `midiland/window_dataset.py`: PyTorch dataset wrapper for saved window arrays.
- `midiland/lm_model.py`: a minimal decoder-only Transformer.
- `midiland/cli/train_lm.py`: training loop for next-token prediction.
- `midiland/cli/sample_lm.py`: unconstrained autoregressive sampling from a trained checkpoint.

There are also two adjacent utilities worth knowing about:

- `midiland/cli/midi_roundtrip.py`: sanity-check path for `MIDI -> canonical events -> tokens -> events -> MIDI`.
- `midiland/cli/midi_writer.py`: a separate absolute-tick MIDI parser/writer utility with its own event classes. It is not part of the tokenizer/training pipeline described below.

## Pipeline overview

The current end-to-end path is:

1. Raw `.mid` / `.midi` files are discovered from a local folder by `midiland/cli/preprocess_dataset.py`.
2. Each MIDI file is parsed into a lossy canonical event list by `midiland.midi_io.midi_to_canonical_events()`.
3. The canonical event list is converted to integer token IDs by `midiland.tokenizer.MidiEventTokenizer.encode()`.
4. Each MIDI becomes one `.npy` token array plus a manifest entry.
5. Those per-file token arrays are cut into padded fixed-length windows by `midiland/cli/make_windows.py`.
6. `midiland/window_dataset.py` loads the saved window arrays for training.
7. `midiland/cli/train_lm.py` trains `midiland.lm_model.GPT` for next-token prediction.
8. `midiland/cli/sample_lm.py` samples new token IDs from the trained model.
9. Generated tokens are printed as readable token strings, and can optionally be saved as `.npy`. There is no dedicated CLI in the repo today that takes sampled tokens all the way back to MIDI.

If you want a source of raw MIDI files from Hugging Face, `midiland/cli/download_hf_dataset.py` can materialize a dataset repo into a local folder of `.mid` files, but downstream scripts still consume that folder as ordinary MIDI input.

## Stage 1: raw MIDI to canonical event list

### Responsible files

- `midiland/midi_io.py`
- `midiland/tokenizer.py`

### Key classes and functions

- `midiland.midi_io.midi_to_canonical_events()`
- `midiland.tokenizer.TokenizerConfig`
- `midiland.tokenizer.Note`
- `midiland.tokenizer.ProgramChange`
- `midiland.tokenizer.TempoChange`
- `midiland.tokenizer.TimeSignature`
- `midiland.tokenizer.velocity_to_bin()`
- `midiland.tokenizer.bpm_to_bin()`

### Input

- A MIDI file path.
- A `TokenizerConfig`, mainly for quantization and binning parameters such as `steps_per_beat`, `max_duration`, `velocity_bins`, `tempo_*`, and allowed time-signature denominators.
- `ignore_drums`, which defaults to `True`.

### Output

- `tuple[list[Event], int]`
- `Event` is a type alias in `midiland/tokenizer.py`:
  - `ProgramChange(step, channel, program)`
  - `TempoChange(step, bpm_bin)`
  - `TimeSignature(step, numerator, denominator)`
  - `Note(step, channel, pitch, duration, velocity_bin)`
- The second return value is the source file’s `ticks_per_beat`.

### What the code does

`midi_to_canonical_events()` loads the file with `mido.MidiFile`, merges all tracks with `mido.merge_tracks()`, and walks the merged message stream while maintaining an absolute MIDI tick counter.

Important details:

- Time is quantized to integer `step` positions using:
  - `ticks_per_step = ticks_per_beat / cfg.steps_per_beat`
  - `step = round(abs_tick / ticks_per_step)`
- Drum channel events are skipped when `ignore_drums` is true. The check is `channel == 9`, which corresponds to MIDI channel 10.
- Tempo meta events are converted from microseconds-per-beat to BPM, then binned into `TempoChange(step, bpm_bin)`.
- Time signatures are kept only as numerator and denominator. Unsupported denominators are forced to `4/4`, and numerators are clamped to `cfg.max_ts_numerator`.
- Program changes are emitted as `ProgramChange(step, channel, program)`.
- Notes are reconstructed by storing active note-on events in a `defaultdict(list)` keyed by `(channel, pitch)`. On note-off, the oldest unmatched note-on for that `(channel, pitch)` is popped and turned into a `Note`.
- Note durations are quantized independently from note starts:
  - `duration_ticks = end_tick - start_tick`
  - `duration_steps = round(duration_ticks / ticks_per_step)`
  - `duration_steps` is clamped to at least `1` and at most `cfg.max_duration`
- Note velocity is binned with `velocity_to_bin()`.

The final event list is sorted by step and then by event type priority:

1. `TimeSignature`
2. `TempoChange`
3. `ProgramChange`
4. `Note`

That ordering matters later because tokenization expects time-signature changes to be visible before contextual `BAR` / `POS` tokens are emitted.

### Important data structures

- `active_notes: DefaultDict[tuple[int, int], list[tuple[int, int]]]`
  - Maps `(channel, pitch)` to queued `(start_tick, velocity)` pairs.
- `events: list[Event]`
  - The canonical symbolic representation used everywhere else in the tokenizer pipeline.

### What is lost at this stage

The representation is intentionally lossy:

- timing is snapped to steps
- duration is snapped to steps and capped
- velocity is binned
- tempo is binned
- unsupported time-signature denominators collapse to `4/4`
- key signature and most other meta events are dropped entirely
- original track boundaries are lost because parsing uses a merged track stream

## Stage 2: canonical events to token IDs

### Responsible file

- `midiland/tokenizer.py`

### Key classes and functions

- `midiland.tokenizer.TokenizerConfig`
- `midiland.tokenizer.MidiEventTokenizer`
- `MidiEventTokenizer.encode()`
- `MidiEventTokenizer.decode()`
- `MidiEventTokenizer.token_type()`
- `MidiEventTokenizer.token_to_str()`

### Input

- `Iterable[Event]`, usually the canonical event list returned by `midi_to_canonical_events()`.

### Output

- `list[int]` token IDs.

### Vocabulary shape

`MidiEventTokenizer` builds a fixed integer vocabulary from `TokenizerConfig`. The vocabulary includes:

- special tokens: `PAD`, `BOS`, `EOS`
- time movement: `TIME_SHIFT(n)`, `TIME_SHIFT_COARSE(n)`
- event markers: `EV_NOTE`, `EV_PROG`, `EV_TEMPO`, `EV_TS`
- context tokens: `BAR`, `POS(s)`
- event arguments: `CH(c)`, `PITCH(k)`, `DUR(d)`, `VEL(v)`, `PROG(p)`, `BPM_BIN(t)`, `TS_NUM(a)`, `TS_DEN(b)`

The integer ID ranges are computed in the tokenizer constructor and stored as private offsets like `_time_shift_base`, `_ev_note_id`, `_pos_base`, and so on.

### What `encode()` does

`encode()` sorts events again using the same step/type priority and then serializes them into a flat token stream.

The emitted stream always starts with `BOS` and ends with `EOS`.

Time handling:

- The encoder maintains a `cursor` in quantized steps.
- If the next event step is ahead of the cursor, it emits one or more `TIME_SHIFT_COARSE` tokens first, then `TIME_SHIFT` tokens for the remainder.
- Simultaneous events are represented as adjacent event tokens without any intervening time shift.

Per-step handling:

- At each step, `TimeSignature` events are emitted first.
- Then one `BAR` and one `POS(...)` token are emitted for the step as context.
- Then tempo changes for that step.
- Then program changes.
- Then notes, sorted by `(channel, pitch)`.

Concrete note encoding looks like:

`EV_NOTE CH(c) PITCH(k) DUR(d) VEL(v)`

Important nuance:

- `BAR` and `POS` are context-only. The decoder ignores them.
- `_emit_bar_pos()` only emits a single `BAR` token when the computed bar index changes. It does not try to emit one token per skipped bar if a long time shift jumps across several bars.

### What `decode()` does

`decode()` runs the inverse state machine:

- `TIME_SHIFT` and `TIME_SHIFT_COARSE` advance the cursor.
- `BAR` and `POS` are ignored.
- `EV_TS`, `EV_TEMPO`, `EV_PROG`, and `EV_NOTE` consume their expected argument tokens and rebuild canonical event objects at the current cursor step.

In strict mode, malformed token sequences raise `ValueError`. This matters for generation: sampled token streams are not grammar-constrained, so decoding arbitrary model output is not guaranteed to succeed.

### Important data structures

- `TokenizerConfig`
  - Freezes the symbolic representation. The repo treats this config as part of the dataset contract.
- `list[int]`
  - Raw tokenized document representation, one sequence per MIDI before windowing.

## Stage 3: per-MIDI token document export

### Responsible file

- `midiland/cli/preprocess_dataset.py`

### Key functions

- `_iter_mid_files()`
- `_dtype_from_name()`
- `main()`

### Input

- `input_dir`: folder recursively containing `.mid` / `.midi`
- `out_dir`
- preprocessing flags such as `--steps-per-beat`, `--dtype`, `--keep-drums`, `--overwrite`, `--limit`

### Output

The script writes one token array per MIDI file:

```text
out_dir/
  config.json
  manifest.jsonl
  stats.json
  tokens/**/*.npy
```

### What the code does

For each discovered MIDI file:

1. Build a `TokenizerConfig` from CLI args.
2. Instantiate `MidiEventTokenizer`.
3. Call `midi_to_canonical_events()`.
4. Call `tok.encode(events)`.
5. Save the resulting token ID list to `.npy`.
6. Append one JSON line to `manifest.jsonl`.

`config.json` freezes the export configuration:

- `tokenizer_config`
- `vocab_size`
- `dtype`

Each successful manifest record contains:

- `source`
- `rel_source`
- `tokens`
- `length`
- `ticks_per_beat`

If a file fails, the script still writes a manifest line, but it contains `error` instead of token metadata.

### Important data structures

- per-file token arrays: `numpy.ndarray` of shape `(num_tokens,)`
- export manifest records: JSON objects, one per input MIDI

### What goes in and what comes out

- In: one MIDI file
- Out: one `.npy` token document, plus one manifest row

This stage does not create training windows. It preserves one sequence per original source file.

## Stage 4: fixed-length window building

### Responsible files

- `midiland/cli/make_windows.py`
- `midiland/tokenizer.py`

### Key functions

- `_load_cfg()`
- `_iter_docs()`
- `_compute_next_safe()`
- `_compute_prev_safe()`
- `_header_tokens()` inside `main()`
- `main()`

### Input

- A token export folder from `preprocess_dataset.py`
- Specifically:
  - `config.json`
  - `manifest.jsonl`
  - `.npy` token documents

### Output

```text
out_dir/
  config.json
  manifest.jsonl
  stats.json
  windows/**/*.npy
```

Each window file is a fixed-size 1D token array of length `seq_len`, padded with `PAD=0`.

### What the code does

The script loads tokenizer config from the token export, rebuilds the tokenizer, and iterates through each document from the source manifest.

For each document:

1. Load the full token array.
2. Compute safe token boundaries using token types.
3. Choose window starts at safe boundaries, spaced by `stride`.
4. Optionally compute a fixed header that restates symbolic state at the window start.
5. Choose a safe end boundary that keeps the final body inside `seq_len`.
6. Pad the resulting window to exactly `seq_len`.
7. Save it as `.npy` and append a manifest row.

### Safe boundary logic

The windowing code tries not to cut inside the argument tokens of composite events.

Unsafe token types are the argument tokens:

- `ch`
- `pitch`
- `dur`
- `vel`
- `prog`
- `bpm`
- `ts_num`
- `ts_den`

Safe starts are token types like:

- `bos`
- `eos`
- `time_shift`
- `time_shift_coarse`
- `ev_note`
- `ev_prog`
- `ev_tempo`
- `ev_ts`
- `bar`
- `pos`

`_compute_next_safe()` and `_compute_prev_safe()` create lookup tables so start and end positions can be snapped to these safe indices.

### Header logic

By default, each window gets a fixed header. The goal is to restate state at the window start so a model does not have to infer that state purely from earlier truncated context.

The header contains:

1. `BOS`
2. current time signature as `EV_TS TS_NUM TS_DEN`
3. current tempo as `EV_TEMPO BPM_BIN`
4. one `EV_PROG CH PROG` triple for each of the 16 MIDI channels
5. `BAR POS`

The script computes state at each needed body start by scanning through the source token document once and tracking:

- `cursor_steps`
- current `ts_num`
- current `ts_den`
- current `tempo_bin`
- current `programs[16]`

When headers are enabled and a window would otherwise begin on `BOS`, the body start is advanced so the final window does not contain `BOS BOS`.

### Window manifest fields

Each output row includes:

- `source`
- `rel_source`
- `source_tokens`
- `window_tokens`
- `doc_length`
- `start`
- `end`
- `length`
- `header_len`
- `body_len`

### Important data structures

- source document tokens: `numpy.ndarray` shape `(n,)`
- output window tokens: `numpy.ndarray` shape `(seq_len,)`
- `state_at: dict[int, tuple[int, int, int, int, list[int]]]`
  - maps body-start index to `(cursor_steps, ts_num, ts_den, tempo_bin, programs)`

### What goes in and what comes out

- In: one full-document token array
- Out: zero or more fixed-length padded training windows

This is the stage that turns one sequence per MIDI into many overlapping training examples.

## Stage 5: dataset loading for training

### Responsible file

- `midiland/window_dataset.py`

### Key classes and functions

- `WindowRecord`
- `iter_window_records()`
- `NpyWindowDataset`

### Input

- `data_windows/manifest.jsonl`
- saved `.npy` window files

### Output

- A PyTorch `Dataset` that returns one window tensor per item.

### What the code does

`iter_window_records()` parses each JSON line into a `WindowRecord` dataclass.

`NpyWindowDataset` eagerly loads the manifest rows into memory on construction, but it does not preload the token arrays themselves. In `__getitem__()`, it:

1. loads the `.npy` window from `rec.window_tokens`
2. validates that the array is 1D
3. converts it to `int64`
4. returns `torch.Tensor`

### Important data structures

- `WindowRecord`
  - metadata for one saved training window
- returned tensor shape
  - `(seq_len,)`

## Stage 6: language model training

### Responsible files

- `midiland/cli/train_lm.py`
- `midiland/lm_model.py`

### Key classes and functions

- `midiland.lm_model.GPTConfig`
- `midiland.lm_model.GPT`
- `train_lm.load_windows_config()`
- `train_lm.split_indices()`
- `train_lm.evaluate()`
- `train_lm.save_checkpoint()`
- `train_lm.main()`

### Input

- a window dataset directory produced by `make_windows.py`
- training hyperparameters from CLI

### Model structure

`midiland/lm_model.py` defines a small decoder-only model:

- token embedding
- positional embedding
- dropout
- `nn.TransformerEncoder` used with a causal mask
- final layer norm
- linear LM head

Despite using `TransformerEncoder`, the causal mask in `GPT.forward()` makes the network behave as an autoregressive decoder for next-token prediction.

### What `train_lm.py` does

1. Load `data_windows/config.json`.
2. Rebuild `TokenizerConfig` and `MidiEventTokenizer`.
3. Validate that `tokenizer.vocab_size` matches the saved config.
4. Build `NpyWindowDataset(data_windows / "manifest.jsonl")`.
5. Randomly split window indices into train and validation sets with `split_indices()`.
6. Wrap them in `DataLoader`s.
7. Instantiate `GPT`.
8. Train with AdamW on next-token prediction.

Loss computation:

- Input batch shape: `[B, T]`
- Model output logits: `[B, T, vocab_size]`
- Targets are `x[:, 1:]`
- Predictions are `logits[:, :-1, :]`
- PAD targets are replaced with `-100` so `cross_entropy(..., ignore_index=-100)` ignores padding

Checkpoint contents:

- `step`
- `model_state`
- `optim_state`
- `gpt_config`
- `tokenizer_config`
- `best_val_loss`

The script writes:

- `checkpoints/latest.pt`
- `checkpoints/best.pt` when validation improves

### Important data structures

- training batch: `torch.LongTensor [B, T]`
- checkpoints: `torch.save()` payloads with model and config state

### What goes in and what comes out

- In: padded token windows
- Out: trained model checkpoints

## Stage 7: generation and output

### Responsible files

- `midiland/cli/sample_lm.py`
- `midiland/tokenizer.py`
- `midiland/midi_io.py`
- `midiland/cli/midi_roundtrip.py`

### Key functions

- `sample_lm._load_checkpoint()`
- `sample_lm.top_p_sample()`
- `sample_lm.generate()`
- `MidiEventTokenizer.token_to_str()`
- `MidiEventTokenizer.decode()`
- `midiland.midi_io.canonical_events_to_midi()`

### What exists today

The trained-model generation path currently stops at token generation:

1. `sample_lm.py` loads a checkpoint.
2. It rebuilds `GPT` and `MidiEventTokenizer` from the checkpoint configs.
3. It starts from a prompt of `[BOS]`.
4. It repeatedly samples one next token using temperature scaling and top-p sampling.
5. It stops at `EOS` or after `max_new_tokens`.
6. It prints the first 300 tokens using `tokenizer.token_to_str()`.
7. It can optionally save the raw generated IDs as `.npy`.

### What does not exist as a CLI

There is no dedicated command in the repo today that does:

`checkpoint -> sampled token IDs -> decode() -> canonical_events_to_midi() -> .mid`

The lower-level pieces do exist:

- `MidiEventTokenizer.decode()` can turn a valid token stream back into canonical events.
- `canonical_events_to_midi()` can write canonical events back to a MIDI file.

But `sample_lm.py` does not currently call either of those functions.

### Existing MIDI output path

The only fully wired MIDI output path in the current tokenizer pipeline is `midiland/cli/midi_roundtrip.py`, which does:

`input MIDI -> canonical events -> tokens -> decode back to events -> write MIDI`

That script is a representation sanity check, not a trained-model generation script.

## Related but separate utility: `midi_writer.py`

`midiland/cli/midi_writer.py` defines another MIDI parser/writer abstraction with absolute-tick dataclasses such as `NoteEvent`, `ProgramChangeEvent`, and `TempoEvent`.

It is not used by:

- `preprocess_dataset.py`
- `make_windows.py`
- `train_lm.py`
- `sample_lm.py`
- `midiland/midi_io.py`

For a refactor, it should be treated as adjacent utility code rather than part of the core tokenizer/LM path unless you explicitly decide to merge the two representations.

## Current limitations / unclear areas

- The symbolic representation is intentionally lossy. Timing, tempo, velocity, and some metadata are irreversibly simplified in `midi_to_canonical_events()`.
- Track structure is flattened with `mido.merge_tracks()`. The canonical representation does not preserve original track grouping.
- Time-signature handling is partial. Unsupported denominators are silently replaced with `4/4`, and numerators are clamped.
- Active note matching is FIFO per `(channel, pitch)`. That is simple and workable, but it bakes in assumptions for overlapping same-pitch notes on the same channel.
- `canonical_events_to_midi()` writes everything into a single track of a type-1 MIDI file. That is good enough for roundtripping canonical events, but it does not reconstruct original multitrack layout.
- `BAR` and `POS` are one-way context tokens. They are emitted during encoding and ignored during decoding.
- The window header is built with private tokenizer members such as `_ev_ts_id`, `_ev_prog_id`, `_bar_id`, and `_bar_steps()`. That creates tight coupling between `make_windows.py` and tokenizer internals.
- The safe-boundary logic is token-type based rather than grammar-validated. It avoids splitting inside argument tokens, but it does not prove that every possible sampled or hand-edited sequence is structurally valid.
- `make_windows.py` silently skips several bad inputs, including missing token files, non-1D arrays, and empty arrays. That keeps the run going but makes omissions easy to miss unless you inspect outputs.
- Validation split happens at the window level, not the document level. Windows from the same source MIDI can land in both train and validation, which makes validation loss optimistic.
- `sample_lm.py` uses unconstrained sampling from raw logits. It can generate invalid event grammar, including tokens that `decode()` would reject in strict mode.
- The sampling script starts from only `[BOS]`, so generation is unconditional. There is no built-in prompting from an existing prefix, metadata, or partial score.
- Generated-token-to-MIDI export is not wired into a CLI, so the current “generation” stage is only partially operational from a user workflow point of view.
- Checkpoints save tokenizer config but not an explicit output `ticks_per_beat` for future MIDI writing. That value exists during preprocessing and roundtrip, but there is no canonical choice attached to a trained model alone.
- `midiland/cli/midi_writer.py` overlaps conceptually with `midiland/midi_io.py` but uses a different event model, which makes the repository’s MIDI abstractions harder to reason about at a glance.

## Questions to resolve before refactoring

- Should the repository keep one canonical symbolic event model, or continue to maintain both the tokenizer event model and the separate `midi_writer.py` event model?
- Is the intended primary output of the system token sequences, canonical events, or finished MIDI files?
- Should generated-token decoding be grammar-constrained so invalid token sequences are impossible or at least rarer?
- Should validation split by source document instead of by window to avoid train/validation leakage?
- Are `BAR` / `POS` meant to remain auxiliary context tokens, or should they become part of a more explicit structural representation?
- Should the state header remain a preprocessing-time concern, or should window conditioning be handled differently at model time?
- Which parts of the tokenizer are stable public API, and which parts are allowed to stay private? `make_windows.py` currently depends on private tokenizer internals.
- What should happen with unsupported or out-of-range musical structure, especially uncommon time signatures and metadata that is currently dropped?
- Should the canonical representation preserve more information, such as track grouping, key signatures, or richer control events?
- If the project should support end-to-end generation to MIDI, what `ticks_per_beat`, prompting scheme, and post-processing rules should define that export path?
