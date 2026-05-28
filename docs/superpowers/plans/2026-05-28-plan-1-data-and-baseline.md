# Data and Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use skills:subagent-driven-development (recommended) or skills:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the dataset handling, label normalization, feature policy, and rule-based baseline needed for the DDoS detector.

**Architecture:** This plan covers the lowest-level project foundations: paths, schema definitions, dataset preparation, and the non-ML baseline. It intentionally excludes model training and dashboard work so this session stays focused and produces reusable core modules for later plans.

**Tech Stack:** Python 3.13, pandas, scikit-learn, ruff

---

## Planned File Structure

### New files to create

- `src/schema.py` — dataset column names, label mapping, feature lists, schema validation helpers
- `src/preprocessing.py` — column filtering and target normalization helpers
- `src/baseline.py` — rule-based baseline classifier and prediction helpers

### Existing files to modify

- `pyproject.toml` — add runtime dependencies needed for the project
- `src/load_dataset.py` — align imports and reuse schema-aware loading helpers
- `src/paths.py` — add artifact, sample, and report directories

---

### Task 1: Project paths and dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `src/paths.py`

- [ ] **Step 1: Update project dependencies**

Update [`pyproject.toml`](pyproject.toml) dependencies to:

```toml
dependencies = [
    "pandas>=3.0.3",
    "scikit-learn>=1.5.1",
    "joblib>=1.4.2",
    "streamlit>=1.37.0",
]
```

Update the dev dependencies block to:

```toml
[project.optional-dependencies]
dev = [
    "ruff>=0.5.5",
    "mdformat>=0.7.22",
    "pre-commit>=3.7.0",
]
```

- [ ] **Step 2: Add project directories**

Update [`src/paths.py`](src/paths.py) to:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
REPORTS_ROOT = PROJECT_ROOT / "reports"
SAMPLE_DATA_ROOT = DATA_ROOT / "samples"

DATA_ROOT.mkdir(exist_ok=True, parents=True)
ARTIFACTS_ROOT.mkdir(exist_ok=True, parents=True)
REPORTS_ROOT.mkdir(exist_ok=True, parents=True)
SAMPLE_DATA_ROOT.mkdir(exist_ok=True, parents=True)

DATASET_PATH = DATA_ROOT / "inddos24-dataset.zip"
```

- [ ] **Step 3: Verify imports and formatting**

Run: `uv run python -c "from src.paths import ARTIFACTS_ROOT, REPORTS_ROOT, SAMPLE_DATA_ROOT; print(ARTIFACTS_ROOT.name, REPORTS_ROOT.name, SAMPLE_DATA_ROOT.name)"`
Expected: output contains `artifacts reports samples`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml src/paths.py
git commit -m "build: add project paths and runtime dependencies"
```

---

### Task 2: Dataset schema and label normalization

**Files:**
- Create: `src/schema.py`

- [ ] **Step 1: Create schema module**

Create [`src/schema.py`](src/schema.py):

```python
from collections.abc import Sequence

import pandas as pd

ATTACK_TYPE_COLUMN = "Attack Type"
TARGET_COLUMN = "target"

LABEL_MAPPING = {
    "No Attack": "normal",
    "SYN Flood": "syn_flood",
    "UDP Flood": "udp_flood",
}

DROP_COLUMNS: tuple[str, ...] = (
    "Timestamp",
    "Source IP",
    "Destination IP",
    "Device Type",
    "Operating System",
    "Firmware Version",
    "Anomaly Score",
    "Labels",
)

FULL_FEATURE_COLUMNS: tuple[str, ...] = (
    "Protocol",
    "Source Port",
    "Destination Port",
    "Packet Size",
    "Payload Length",
    "Flow Duration",
    "Bytes in Flow",
    "Packets in Flow",
    "Average Packet Size",
    "Inter-Arrival Time",
    "Rate of Packets",
    "Unique Source Count",
    "Unique Destination Count",
)

LIGHTWEIGHT_FEATURE_COLUMNS: tuple[str, ...] = (
    "Protocol",
    "Source Port",
    "Destination Port",
    "Packet Size",
    "Payload Length",
    "Flow Duration",
    "Bytes in Flow",
    "Packets in Flow",
    "Inter-Arrival Time",
    "Rate of Packets",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    ATTACK_TYPE_COLUMN,
    *FULL_FEATURE_COLUMNS,
)


def validate_required_columns(columns: Sequence[str]) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")


def normalize_attack_labels(frame: pd.DataFrame) -> pd.Series:
    labels = frame[ATTACK_TYPE_COLUMN].map(LABEL_MAPPING)
    if labels.isna().any():
        unknown_labels = sorted(frame.loc[labels.isna(), ATTACK_TYPE_COLUMN].unique())
        raise ValueError(f"Unsupported attack labels: {unknown_labels}")
    return labels
```

- [ ] **Step 2: Verify schema module loads**

Run: `uv run python -c "from src.schema import LABEL_MAPPING, DROP_COLUMNS; print(LABEL_MAPPING['No Attack'], 'Source IP' in DROP_COLUMNS)"`
Expected: output contains `normal True`

- [ ] **Step 3: Commit**

```bash
git add src/schema.py
git commit -m "feat: add dataset schema and label normalization"
```

---

### Task 3: Dataset loading and preprocessing

**Files:**
- Modify: `src/load_dataset.py`
- Create: `src/preprocessing.py`

- [ ] **Step 1: Create preprocessing helper**

Create [`src/preprocessing.py`](src/preprocessing.py):

```python
import pandas as pd

from src.schema import DROP_COLUMNS, TARGET_COLUMN, normalize_attack_labels


def prepare_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.drop(columns=list(DROP_COLUMNS), errors="ignore").copy()
    prepared[TARGET_COLUMN] = normalize_attack_labels(frame)
    return prepared
```

- [ ] **Step 2: Update dataset loader**

Update [`src/load_dataset.py`](src/load_dataset.py) to:

```python
from pathlib import Path

import pandas as pd

from src.paths import DATASET_PATH
from src.schema import validate_required_columns


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_required_columns(frame.columns)
    return frame


if __name__ == "__main__":
    dataset = load_dataset(DATASET_PATH)
    print(dataset.head())
    print(dataset.shape)
    print(dataset.dtypes)
    print(dataset["Attack Type"].value_counts())
```

- [ ] **Step 3: Verify preprocessing on a small inline frame**

Run: `uv run python -c "import pandas as pd; from src.preprocessing import prepare_model_frame; frame = pd.DataFrame({'Attack Type':['No Attack'],'Protocol':['TCP'],'Source Port':[80],'Destination Port':[443],'Packet Size':[100],'Payload Length':[60],'Flow Duration':[1.0],'Bytes in Flow':[1000],'Packets in Flow':[10],'Average Packet Size':[100.0],'Inter-Arrival Time':[0.1],'Rate of Packets':[10.0],'Unique Source Count':[5],'Unique Destination Count':[1],'Source IP':['10.0.0.1']}); prepared = prepare_model_frame(frame); print(prepared.columns.tolist(), prepared['target'].tolist())"`
Expected: output includes `target` and excludes `Source IP`

- [ ] **Step 4: Commit**

```bash
git add src/load_dataset.py src/preprocessing.py
git commit -m "feat: add preprocessing for model-ready dataset frames"
```

---

### Task 4: Rule-based baseline

**Files:**
- Create: `src/baseline.py`

- [ ] **Step 1: Create baseline classifier**

Create [`src/baseline.py`](src/baseline.py):

```python
import pandas as pd


def predict_rule_based(frame: pd.DataFrame) -> pd.Series:
    predictions: list[str] = []

    for _, row in frame.iterrows():
        protocol = row["Protocol"]
        packet_rate = row["Rate of Packets"]
        packets_in_flow = row["Packets in Flow"]
        unique_source_count = row["Unique Source Count"]

        if (
            protocol == "TCP"
            and packet_rate >= 3000
            and packets_in_flow >= 2000
            and unique_source_count >= 100
        ):
            predictions.append("syn_flood")
        elif (
            protocol == "UDP"
            and packet_rate >= 3000
            and packets_in_flow >= 2000
            and unique_source_count >= 100
        ):
            predictions.append("udp_flood")
        else:
            predictions.append("normal")

    return pd.Series(predictions)
```

- [ ] **Step 2: Verify baseline predictions manually**

Run: `uv run python -c "import pandas as pd; from src.baseline import predict_rule_based; frame = pd.DataFrame({'Protocol':['TCP','UDP','TCP'],'Rate of Packets':[5000.0,4500.0,10.0],'Packets in Flow':[4000,3500,10],'Unique Source Count':[300,250,2]}); print(predict_rule_based(frame).tolist())"`
Expected: output is `['syn_flood', 'udp_flood', 'normal']`

- [ ] **Step 3: Commit**

```bash
git add src/baseline.py
git commit -m "feat: add rule-based ddos baseline"
```

---

### Task 5: Foundation verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add foundation commands to documentation**

Append this section to [`README.md`](README.md):

```markdown
## Foundation Commands

### Inspect dataset

```bash
uv run python src/load_dataset.py
```

### Verify preprocessing helper

```bash
uv run python -c "import pandas as pd; from src.preprocessing import prepare_model_frame; frame = pd.DataFrame({'Attack Type':['No Attack'],'Protocol':['TCP'],'Source Port':[80],'Destination Port':[443],'Packet Size':[100],'Payload Length':[60],'Flow Duration':[1.0],'Bytes in Flow':[1000],'Packets in Flow':[10],'Average Packet Size':[100.0],'Inter-Arrival Time':[0.1],'Rate of Packets':[10.0],'Unique Source Count':[5],'Unique Destination Count':[1]}); print(prepare_model_frame(frame).head())"
```

### Verify rule baseline

```bash
uv run python -c "import pandas as pd; from src.baseline import predict_rule_based; frame = pd.DataFrame({'Protocol':['TCP','UDP'],'Rate of Packets':[5000.0,4500.0],'Packets in Flow':[4000,3500],'Unique Source Count':[300,250]}); print(predict_rule_based(frame).tolist())"
```
```

- [ ] **Step 2: Run lint**

Run: `uv run ruff check src`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add foundation verification commands"
```

---

## Self-Review

### Spec coverage check

This plan covers the spec requirements for:
- dataset loading and inspection
- label normalization
- feature dropping policy
- lightweight feature definitions
- rule-based baseline
- foundational project directories and dependencies

It intentionally does not cover ML training, artifact persistence, static sample files, or dashboard work because those are split into later plans.

### Placeholder scan

No placeholders remain. Every step includes exact files, code, commands, and expected outcomes.

### Type consistency check

Names are consistent within this plan:
- [`ATTACK_TYPE_COLUMN`](src/schema.py)
- [`TARGET_COLUMN`](src/schema.py)
- [`validate_required_columns()`](src/schema.py)
- [`normalize_attack_labels()`](src/schema.py)
- [`prepare_model_frame()`](src/preprocessing.py)
- [`predict_rule_based()`](src/baseline.py)

Plan complete and saved to [`docs/superpowers/plans/2026-05-28-plan-1-data-and-baseline.md`](docs/superpowers/plans/2026-05-28-plan-1-data-and-baseline.md). Additional plans should cover model training/evaluation and dashboard/demo separately.