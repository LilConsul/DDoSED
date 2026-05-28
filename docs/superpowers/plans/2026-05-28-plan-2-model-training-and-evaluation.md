# Model Training and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use skills:subagent-driven-development (recommended) or skills:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline ML training, feature transformation, model comparison, evaluation summaries, and artifact persistence for the DDoS detector.

**Architecture:** This plan assumes the data/schema foundation from [`docs/superpowers/plans/2026-05-28-plan-1-data-and-baseline.md`](docs/superpowers/plans/2026-05-28-plan-1-data-and-baseline.md) is already implemented. It focuses on reusable offline training modules that prepare features, compare lightweight models, summarize results, and save artifacts for later dashboard use.

**Tech Stack:** Python 3.13, pandas, scikit-learn, joblib, ruff

---

## Planned File Structure

### New files to create

- `src/features.py` — preprocessing pipeline builders for full-safe and reduced-lightweight feature sets
- `src/evaluate.py` — metric calculation, confusion matrix generation, model comparison summaries
- `src/artifacts.py` — save/load helpers for trained models and evaluation outputs
- `src/train.py` — offline training entrypoint for ML models and baseline comparison

### Existing files to modify

- `src/main.py` — replace placeholder print with a useful training-oriented entrypoint or redirect note
- `README.md` — add training and evaluation commands

---

### Task 1: Feature pipeline builders

**Files:**
- Create: `src/features.py`

- [ ] **Step 1: Create feature transformer module**

Create [`src/features.py`](src/features.py):

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import FULL_FEATURE_COLUMNS, LIGHTWEIGHT_FEATURE_COLUMNS

CATEGORICAL_COLUMNS = ("Protocol",)
NUMERIC_COLUMNS = tuple(
    column for column in FULL_FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
)


def build_feature_transformer(feature_set: str) -> ColumnTransformer:
    if feature_set == "full":
        selected_columns = FULL_FEATURE_COLUMNS
    elif feature_set == "lightweight":
        selected_columns = LIGHTWEIGHT_FEATURE_COLUMNS
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    categorical_columns = [column for column in selected_columns if column in CATEGORICAL_COLUMNS]
    numeric_columns = [column for column in selected_columns if column not in CATEGORICAL_COLUMNS]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_columns),
            ("numeric", numeric_pipeline, numeric_columns),
        ]
    )
```

- [ ] **Step 2: Verify transformer creation**

Run: `uv run python -c "from src.features import build_feature_transformer; print(type(build_feature_transformer('lightweight')).__name__)"`
Expected: output is `ColumnTransformer`

- [ ] **Step 3: Commit**

```bash
git add src/features.py
git commit -m "feat: add feature transformers for full and lightweight sets"
```

---

### Task 2: Evaluation helpers

**Files:**
- Create: `src/evaluate.py`

- [ ] **Step 1: Create evaluation summary module**

Create [`src/evaluate.py`](src/evaluate.py):

```python
from collections.abc import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

CLASS_LABELS = ("normal", "syn_flood", "udp_flood")


def summarize_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "labels": list(CLASS_LABELS),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
    }
```

- [ ] **Step 2: Verify metric summary output**

Run: `uv run python -c "from src.evaluate import summarize_classification_metrics; print(sorted(summarize_classification_metrics(['normal','syn_flood'], ['normal','normal']).keys()))"`
Expected: output contains `accuracy`, `confusion_matrix`, `labels`, and `macro_f1`

- [ ] **Step 3: Commit**

```bash
git add src/evaluate.py
git commit -m "feat: add evaluation summary helpers"
```

---

### Task 3: Artifact persistence

**Files:**
- Create: `src/artifacts.py`

- [ ] **Step 1: Create artifact helper module**

Create [`src/artifacts.py`](src/artifacts.py):

```python
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
```

- [ ] **Step 2: Verify JSON report writing**

Run: `uv run python -c "from pathlib import Path; from src.artifacts import save_json_report; path = Path('reports/check.json'); save_json_report(path, {'accuracy': 0.9}); print(path.exists(), path.read_text(encoding='utf-8'))"`
Expected: output starts with `True` and includes `"accuracy": 0.9`

- [ ] **Step 3: Commit**

```bash
git add src/artifacts.py
git commit -m "feat: add artifact persistence helpers"
```

---

### Task 4: Model registry and training entrypoint

**Files:**
- Create: `src/train.py`
- Modify: `src/main.py`

- [ ] **Step 1: Create model registry**

Create [`src/train.py`](src/train.py):

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def build_model_registry() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "random_forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=10,
        ),
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
```

- [ ] **Step 2: Update main entrypoint**

Update [`src/main.py`](src/main.py) to:

```python
from src.train import build_model_registry


if __name__ == "__main__":
    print(sorted(build_model_registry().keys()))
```

- [ ] **Step 3: Verify model registry**

Run: `uv run python src/main.py`
Expected: output contains `decision_tree`, `knn`, `logistic_regression`, `naive_bayes`, and `random_forest`

- [ ] **Step 4: Commit**

```bash
git add src/train.py src/main.py
git commit -m "feat: add model registry for offline training"
```

---

### Task 5: End-to-end model comparison pipeline

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Extend training module with comparison pipeline**

Extend [`src/train.py`](src/train.py) with:

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.evaluate import summarize_classification_metrics
from src.features import build_feature_transformer
from src.schema import TARGET_COLUMN


def train_and_compare_models(frame, feature_set: str) -> dict[str, object]:
    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.33,
        random_state=42,
        stratify=target,
    )

    ranked_models: list[dict[str, object]] = []

    for model_name, estimator in build_model_registry().items():
        pipeline = Pipeline(
            steps=[
                ("transformer", build_feature_transformer(feature_set)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = summarize_classification_metrics(y_test, predictions)
        ranked_models.append(
            {
                "model_name": model_name,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
        )

    ranked_models.sort(key=lambda item: item["macro_f1"], reverse=True)
    return {
        "best_model_name": ranked_models[0]["model_name"],
        "ranked_models": ranked_models,
    }
```

- [ ] **Step 2: Verify comparison pipeline on inline sample data**

Run: `uv run python -c "import pandas as pd; from src.train import train_and_compare_models; frame = pd.DataFrame({'Protocol':['TCP','UDP','TCP','UDP','TCP','UDP'],'Source Port':[80,53,80,53,80,53],'Destination Port':[443,80,443,80,443,80],'Packet Size':[100,1200,110,1300,90,1250],'Payload Length':[60,1100,70,1200,50,1150],'Flow Duration':[1.0,0.2,1.1,0.3,0.9,0.25],'Bytes in Flow':[1000,9000,1100,9500,950,9200],'Packets in Flow':[10,3000,12,3200,9,3100],'Average Packet Size':[100.0,1100.0,91.0,1180.0,105.0,1150.0],'Inter-Arrival Time':[0.1,0.001,0.09,0.001,0.11,0.001],'Rate of Packets':[10.0,5000.0,11.0,5200.0,9.0,5100.0],'Unique Source Count':[2,300,2,320,1,310],'Unique Destination Count':[1,1,1,1,1,1],'target':['normal','udp_flood','normal','udp_flood','normal','udp_flood']}); print(train_and_compare_models(frame, feature_set='lightweight')['best_model_name'])"`
Expected: output is one of the configured model names

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: add end-to-end model comparison pipeline"
```

---

### Task 6: Training artifacts and reports

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Add artifact-writing workflow**

Extend [`src/train.py`](src/train.py) with:

```python
from src.artifacts import save_json_report, save_model_artifact
from src.paths import ARTIFACTS_ROOT, REPORTS_ROOT


def save_training_outputs(best_pipeline: object, comparison: dict[str, object]) -> None:
    save_model_artifact(ARTIFACTS_ROOT / "best_model.joblib", best_pipeline)
    save_json_report(REPORTS_ROOT / "model_comparison.json", comparison)
```

- [ ] **Step 2: Verify report output path**

Run: `uv run python -c "from src.paths import REPORTS_ROOT; print(REPORTS_ROOT.exists(), REPORTS_ROOT.name)"`
Expected: output is `True reports`

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: add training artifact output helpers"
```

---

### Task 7: Training commands documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append training commands**

Append this section to [`README.md`](README.md):

```markdown
## Training Commands

### Install dependencies

```bash
uv sync --extra dev
```

### Inspect available models

```bash
uv run python src/main.py
```

### Run training module

```bash
uv run python -m src.train
```

### Lint source files

```bash
uv run ruff check src
```
```

- [ ] **Step 2: Run lint**

Run: `uv run ruff check src`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add training and evaluation commands"
```

---

## Self-Review

### Spec coverage check

This plan covers the spec requirements for:
- feature transformation
- multiple lightweight ML models
- evaluation summaries and confusion matrix support
- artifact persistence
- offline training and model comparison

It intentionally does not cover predefined sample CSV files, mitigation recommendations, or dashboard rendering because those are split into the dashboard/demo plan.

### Placeholder scan

No placeholders remain. Every step includes exact files, code, commands, and expected outcomes.

### Type consistency check

Names are consistent within this plan:
- [`build_feature_transformer()`](src/features.py)
- [`summarize_classification_metrics()`](src/evaluate.py)
- [`save_json_report()`](src/artifacts.py)
- [`save_model_artifact()`](src/artifacts.py)
- [`build_model_registry()`](src/train.py)
- [`train_and_compare_models()`](src/train.py)
- [`save_training_outputs()`](src/train.py)

Plan complete and saved to [`docs/superpowers/plans/2026-05-28-plan-2-model-training-and-evaluation.md`](docs/superpowers/plans/2026-05-28-plan-2-model-training-and-evaluation.md). A separate plan should cover static sample files and dashboard/demo work.