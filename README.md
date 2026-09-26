# dl4ad-project

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12. The pinned PyTorch 2.2 dependencies do not provide wheels for Python 3.13 or newer. From the project directory, install the project dependencies and development tools:

```sh
uv sync --group dev
```

uv creates a local `.venv` and installs the versions recorded in `uv.lock`. Run the application with its command name:

```sh
uv run nbform --help
```

## Lint and format

Ruff checks the code and sorts imports; Black formats the code:

```sh
uv run ruff check --select I --fix .
uv run ruff check .
uv run black .
```

## Project layout

- `nbform` is the application command, implemented in `cli/app.py`.
- `cli/` contains argument handling and the generate and analyze flows.
- `library/` contains reusable audio, model, plotting, and RAVE components.
