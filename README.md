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

Check the code with Ruff and apply formatting with:

```sh
uv run ruff check .
uv run ruff format .
```
