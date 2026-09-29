# nbform

`nbform` provides interactive bending and analysis workflows for pretrained neural audio models:

- `bend`: interactively modify layers, swap components, listen to model variants, and export audio or models.
- `analyze`: run the interactive impact analysis and select which artifacts to save.
- `sweep`: run the reproducible `analysis_kit` sweep for one audio file.
- `evaluate`: run the corpus sweep and export per-trial and aggregate CSVs.

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
nbform bend --type Encodec --input audio/source --output generated
nbform analyze --type Encodec --input audio/source/GLM.wav --output analysis
nbform sweep --model encodec --input audio/source/GLM.wav --output sweep
nbform evaluate --model encodec --corpus-dir audio/source --output results
nbform evaluate --rave-path models/rave-run --corpus-dir audio/source --output results-rave
```

`bend` is interactive; type `help` at its prompt to list actions. `generate` remains an alias for `bend`. The legacy `sweep` and `evaluate` commands retain the `--nets`, `--interventions`, and `--add-offset` options.
