import json
from pathlib import Path

import joblib


def save_json_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_model_artifact(path: Path, model: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model_artifact(path: Path) -> object:
    return joblib.load(path)
