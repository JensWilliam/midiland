# Midiland

Midiland is a small MIDI tokenization and language-model training sandbox. It turns MIDI files into a lossy but stable event representation, tokenizes those events, and provides a minimal pipeline for preprocessing, windowing, training, and sampling.

## Project Layout

```text
midiland/
  tokenizer.py        Core vocabulary and event <-> token conversion
  midi_io.py          MIDI <-> canonical event conversion
  lm_model.py         Small decoder-only Transformer
  window_dataset.py   Dataset loader for token windows
  cli/                Command-line entry points
docs/
  HOWTO.md            Longer workflow notes and command reference
requirements.txt      Direct dependency list
pyproject.toml        Optional editable install + console scripts
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional, but recommended if you want named CLI commands:

```bash
pip install -e .
```

Without installation, run commands as Python modules. With an editable install, the matching `midiland-*` console scripts are also available.

## Quick Start

Roundtrip a MIDI file through the canonical representation:

```bash
python -m midiland.cli.midi_roundtrip "test_midi's/chopin-etude_op25_n09.mid"
```

Tokenize a folder of MIDI files:

```bash
python -m midiland.cli.preprocess_dataset /path/to/midis data_tokens
```

Create fixed-length training windows:

```bash
python -m midiland.cli.make_windows data_tokens data_windows
```

Train the baseline language model:

```bash
python -m midiland.cli.train_lm data_windows --steps 10000 --batch-size 16
```

Sample from a trained checkpoint:

```bash
python -m midiland.cli.sample_lm checkpoints/best.pt --max-new 512
```

Download MIDI files from a Hugging Face dataset repo:

```bash
python -m midiland.cli.download_hf_dataset drengskapur/midi-classical-music data_midi
```

## Representation Contract

Once you export a dataset, keep these tokenizer settings fixed until you regenerate everything:

- `steps_per_beat`
- velocity and tempo binning
- time-shift scheme
- included token types such as tempo, time signature, and bar/position markers

Changing them midstream will make old token files incompatible with the current tokenizer.

## CLI Entry Points

If you ran `pip install -e .`, these commands are available:

- `midiland-roundtrip`
- `midiland-preprocess`
- `midiland-make-windows`
- `midiland-train`
- `midiland-sample`
- `midiland-download-hf`
- `midiland-midi-writer`

## Notes

- The model code is intentionally simple and readable, not optimized.
- `docs/HOWTO.md` keeps the longer command reference and workflow details.
- Existing generated artifacts such as `data_tokens/`, `data_windows/`, and `checkpoints/` are treated as local outputs and ignored by git.
