# dl4ad-project

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12 or newer. From the project directory, install the project dependencies and development tools:

```sh
uv sync --group dev
```

uv creates a local `.venv` and installs the versions recorded in `uv.lock`. To run Python commands in that environment, prefix them with `uv run`, for example:

```sh
uv run python app.py
```

## Lint and format

Use Ruff for linting and Black for formatting:

```sh
uv run ruff check .
uv run black .
```

## Project layout

- `app.py` is the application entry point.
- `cli/` contains argument handling and the generate and analyze flows.
- `library/` contains reusable audio, model, plotting, and RAVE components.
- `experiments/` contains standalone research scripts.
- `tests/` contains project tests.
