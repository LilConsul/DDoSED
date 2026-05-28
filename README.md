# DDoSED

Development dependencies (for pre-commit hooks, linters, etc.)

```bash
uv sync --extra dev
```

## Training Commands

### Install dependencies

```bash
uv sync --extra dev
```

### Inspect available models

```bash
uv run python -m src.main
```

### Run training module

```bash
uv run python -m src.train
```

### Lint source files

```bash
uv run ruff check src
```
