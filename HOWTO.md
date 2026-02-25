# Midiland HOWTO

This repo is built around one idea:

1. Convert MIDI into a **canonical, lossy event list** (quantized to steps + binned values).
2. Convert events into **token IDs** (ints) with `tokenizer.py`.
3. Use those token IDs for training / generation.

Lossy is intentional: you choose the resolution/bins up front so the model has a simpler job.

## Setup

Create/activate your venv however you like, then install deps:

```bash
pip install -r requirements.txt
```

Notes:
- `mido` is required for MIDI parsing/writing used by the scripts below.
- Your `.env` can contain a Hugging Face token, but preprocessing local `.mid` files does not require it.

## Important Rule: Freeze the “Representation Knobs”

Once you start exporting tokens for a dataset, do **not** change these without re-exporting:
- `steps_per_beat` (default is `8`)
- `velocity_bins`, `tempo_bins` + tempo range
- time shift scheme (`TIME_SHIFT` / `TIME_SHIFT_COARSE`)
- any token types included/excluded (e.g. `BAR/POS`, tempo, time signature)

If you change these later, previously exported token files won’t match the new tokenizer.

## Files and What They Do

- `tokenizer.py` — defines the token vocabulary and `encode()/decode()` between events ↔ token IDs.
- `midi_io.py` — MIDI ↔ canonical event list (quantize + bin; meant to be reused by other scripts).
- `midi_roundtrip.py` — CLI sanity check: MIDI → canonical events → tokens → events → MIDI.
- `preprocess_dataset.py` — converts a folder of MIDIs into per-file `.npy` token arrays + manifest/stats.

## Round-trip Test (Recommended First Check)

Runs a “lossy but strict” roundtrip:
- strict for the canonical representation (quantized+binned)
- not strict for raw MIDI values (because we intentionally quantize/bin)

```bash
python midi_roundtrip.py path/to/song.mid
```

Outputs:
- `path/to/song_out.mid` (default naming: input name + `_out`)
- prints whether canonical event lists match

Useful flags:
```bash
python midi_roundtrip.py song.mid --steps-per-beat 8
python midi_roundtrip.py song.mid --keep-drums
python midi_roundtrip.py song.mid --print-tokens 120
python midi_roundtrip.py song.mid --out /tmp/custom_name.mid
```

## Preprocess a Dataset Folder into `.npy` Tokens

This creates **one token file per MIDI** (keeps song boundaries).

```bash
python preprocess_dataset.py /path/to/midi_folder data_tokens
```

It writes:
- `data_tokens/config.json` — frozen tokenizer config used for export
- `data_tokens/manifest.jsonl` — one JSON line per MIDI (paths + token length + ticks_per_beat)
- `data_tokens/stats.json` — summary stats (min/max/mean length, failures, etc.)
- `data_tokens/tokens/**/*.npy` — token arrays mirroring the input folder layout

Useful flags:
```bash
python preprocess_dataset.py midis data_tokens --steps-per-beat 8
python preprocess_dataset.py midis data_tokens --dtype uint16
python preprocess_dataset.py midis data_tokens --dtype int32
python preprocess_dataset.py midis data_tokens --keep-drums
python preprocess_dataset.py midis data_tokens --overwrite
python preprocess_dataset.py midis data_tokens --limit 100
python preprocess_dataset.py midis data_tokens --print-every 50
```

Notes:
- Default `--dtype` is `uint16`. If your tokenizer `vocab_size` ever exceeds 65535, use `--dtype int32`.
- `manifest.jsonl` includes failures too (with an `"error": ...` field), so you can inspect what broke.

## Debugging Tips

### Print tokens as readable “words”

When a script prints tokens, it uses `tokenizer.MidiEventTokenizer.token_to_str()` to show values like:
- `TIME_SHIFT(8)`
- `TIME_SHIFT_COARSE(2*64)`
- `EV_NOTE CH(0) PITCH(60) DUR(16) VEL(10)`

If something sounds wrong in the `_out.mid`:
- try a higher `--steps-per-beat` (more timing resolution)
- compare canonical event lists (the roundtrip script prints `OK` vs `DIFF`)

### Common “Why is X different?” answers

- **Tempo is slightly off**: expected (tempo is binned).
- **Very short notes become longer**: expected if they’re shorter than 1 step at your `steps_per_beat`.
- **Key differs**: we currently don’t preserve key signature tokens in the canonical event list.

## Quick Reference

| Task | Command |
| --- | --- |
| Roundtrip one file | `python midi_roundtrip.py song.mid` |
| Roundtrip w/ more resolution | `python midi_roundtrip.py song.mid --steps-per-beat 8` |
| Preprocess a dataset folder | `python preprocess_dataset.py midis data_tokens` |
| Make training windows | `python make_windows.py data_tokens data_windows` |

## Train (Toy Baseline)

These scripts are a minimal baseline for next-token prediction.

Train:
```bash
python train_lm.py data_windows --steps 10000 --batch-size 16
```

Checkpoint outputs:
- `checkpoints/latest.pt` — latest step (good for resuming)
- `checkpoints/best.pt` — best validation loss (often best for sampling)

Sample (prints readable token strings):
```bash
python sample_lm.py checkpoints/best.pt --max-new 512
```

## Make Fixed-Length Training Windows

Turns per-MIDI token arrays into fixed-length windows (default `seq_len=2048`, `stride=1024`).
Windows never cross song boundaries, and window boundaries are aligned so they don’t cut inside EV_* arguments.

```bash
python make_windows.py data_tokens data_windows
```

Useful flags:
```bash
python make_windows.py data_tokens data_windows --seq-len 2048 --stride 1024
python make_windows.py data_tokens data_windows --min-len 512
python make_windows.py data_tokens data_windows --limit-docs 50
python make_windows.py data_tokens data_windows --overwrite
```

## Download a HF Dataset Repo (MIDI Files)

If your dataset lives on Hugging Face as a dataset repo with `.mid` files (like `drengskapur/midi-classical-music`),
you can download the MIDI files into a local folder:

```bash
python download_hf_dataset.py drengskapur/midi-classical-music data_midi
```

Outputs:
- `data_midi/midis/` — the MIDI files (hardlinked by default; falls back to copy)
- `data_midi/manifest.jsonl` + `data_midi/stats.json`

Useful flags:
```bash
python download_hf_dataset.py drengskapur/midi-classical-music data_midi --mode copy
python download_hf_dataset.py drengskapur/midi-classical-music data_midi --limit 50
python download_hf_dataset.py drengskapur/midi-classical-music data_midi --revision main
python download_hf_dataset.py drengskapur/midi-classical-music data_midi --overwrite
```
