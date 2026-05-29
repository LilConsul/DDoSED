# DDoSED

Development dependencies (for pre-commit hooks, linters, etc.)

```powershell
uv sync --extra dev
```

Train an Attack Type classifier

```powershell
uv run python -m src.train_attack_type
```

Try all feature/target approaches (baseline, engineered, ports + timestamp, binary)

```powershell
uv run python -m src.train_attack_type --approach all --model all
```

Aggregate rows into time windows (batching by source IP)

```powershell
uv run python -m src.train_attack_type --approach all --model all --aggregate-window 60 --aggregate-group src
```

Enable diagnostics for feature signal checks

```powershell
uv run python -m src.train_attack_type --diagnostics --model hist_gb
```

