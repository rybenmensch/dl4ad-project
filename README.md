# nbform

`nbform` provides three workflows for bending pretrained neural audio models:

- `generate`: interactively modify layers, listen to the current output, and export audio.
- `analyze`: sweep configured interventions for one audio file.
- `evaluate`: run the same sweep over a WAV corpus and export per-trial and aggregate CSVs.

## Install

The project supports Python 3.11 and 3.12. On macOS, a Conda environment is a
convenient way to keep the ML dependencies isolated:

```sh
conda create -n nbform python=3.12 -y
conda activate nbform
conda install -c conda-forge "numpy<2" numba llvmlite -y
python -m pip install -e .
```

For RAVE support, install the optional dependencies:

```sh
python -m pip install -e ".[rave]"
```

RAVE commands require a training run directory containing its `config.gin` and
checkpoint. EnCodec commands use the selected pretrained model and do not
require a local checkpoint.

## Commands

```sh
nbform --help
nbform generate --model encodec --input audio/source --output generated
nbform analyze --model encodec --input audio/source/GLM.wav --output analysis
nbform evaluate --model encodec --corpus-dir audio/source --output results
nbform evaluate --model rave --rave-path models/rave-run --corpus-dir audio/source --output results-rave
```

`generate` is interactive; type `help` at its prompt to list actions. `analyze`
and `evaluate` accept `--nets`, `--interventions`, and `--add-offset` options.
