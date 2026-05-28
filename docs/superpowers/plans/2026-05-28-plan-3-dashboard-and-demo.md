# Dashboard and Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use skills:subagent-driven-development (recommended) or skills:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local dashboard, predefined static sample inputs, mitigation recommendation logic, and optional replay helpers for the DDoS detector demo.

**Architecture:** This plan assumes the data foundation and training pipeline from the earlier plans already exist. It focuses only on the presentation layer and demo assets: static sample CSV files, dashboard helper functions, mitigation recommendations, and a lightweight Streamlit interface that reads saved artifacts and presents results.

**Tech Stack:** Python 3.13, pandas, streamlit, joblib, ruff

---

## Planned File Structure

### New files to create

- `data/samples/sample_normal.csv` — predefined normal sample rows for dashboard prediction
- `data/samples/sample_syn_flood.csv` — predefined SYN flood sample rows for dashboard prediction
- `data/samples/sample_udp_flood.csv` — predefined UDP flood sample rows for dashboard prediction
- `src/dashboard.py` — local dashboard entrypoint and helper functions

### Existing files to modify

- `src/train.py` — add sample file discovery helpers if not already present
- `README.md` — add dashboard and demo commands

---

### Task 1: Predefined sample CSV files

**Files:**
- Create: `data/samples/sample_normal.csv`
- Create: `data/samples/sample_syn_flood.csv`
- Create: `data/samples/sample_udp_flood.csv`

- [ ] **Step 1: Create normal sample file**

Create [`data/samples/sample_normal.csv`](data/samples/sample_normal.csv):

```csv
Protocol,Source Port,Destination Port,Packet Size,Payload Length,Flow Duration,Bytes in Flow,Packets in Flow,Average Packet Size,Inter-Arrival Time,Rate of Packets,Unique Source Count,Unique Destination Count
TCP,443,51514,120,80,1.2,1400,12,116.7,0.10,10.0,2,1
TCP,80,51515,110,70,1.0,1200,11,109.1,0.09,11.0,2,1
```

- [ ] **Step 2: Create SYN flood sample file**

Create [`data/samples/sample_syn_flood.csv`](data/samples/sample_syn_flood.csv):

```csv
Protocol,Source Port,Destination Port,Packet Size,Payload Length,Flow Duration,Bytes in Flow,Packets in Flow,Average Packet Size,Inter-Arrival Time,Rate of Packets,Unique Source Count,Unique Destination Count
TCP,80,443,90,40,0.2,9000,3200,2.8,0.001,5000.0,300,1
TCP,80,443,92,42,0.25,9500,3400,2.8,0.001,5200.0,320,1
```

- [ ] **Step 3: Create UDP flood sample file**

Create [`data/samples/sample_udp_flood.csv`](data/samples/sample_udp_flood.csv):

```csv
Protocol,Source Port,Destination Port,Packet Size,Payload Length,Flow Duration,Bytes in Flow,Packets in Flow,Average Packet Size,Inter-Arrival Time,Rate of Packets,Unique Source Count,Unique Destination Count
UDP,53,80,1200,1100,0.2,12000,3000,4.0,0.001,4800.0,280,1
UDP,53,80,1250,1150,0.22,12500,3200,3.9,0.001,5000.0,300,1
```

- [ ] **Step 4: Verify sample files exist**

Run: `uv run python -c "from src.paths import SAMPLE_DATA_ROOT; print(sorted(path.name for path in SAMPLE_DATA_ROOT.glob('*.csv')))"`

Expected: output contains `sample_normal.csv`, `sample_syn_flood.csv`, and `sample_udp_flood.csv`

- [ ] **Step 5: Commit**

```bash
git add data/samples
git commit -m "feat: add predefined sample csv files for dashboard predictions"
```

---

### Task 2: Sample file discovery helper

**Files:**
- Modify: `src/train.py`

- [ ] **Step 1: Add sample discovery helper**

Add to [`src/train.py`](src/train.py):

```python
from pathlib import Path

from src.paths import SAMPLE_DATA_ROOT


def list_sample_prediction_files() -> list[Path]:
    return sorted(SAMPLE_DATA_ROOT.glob("*.csv"))
```

- [ ] **Step 2: Verify sample discovery**

Run: `uv run python -c "from src.train import list_sample_prediction_files; print([path.name for path in list_sample_prediction_files()])"`

Expected: output contains `sample_normal.csv`, `sample_syn_flood.csv`, and `sample_udp_flood.csv`

- [ ] **Step 3: Commit**

```bash
git add src/train.py
git commit -m "feat: add dashboard sample file discovery"
```

---

### Task 3: Mitigation recommendation logic

**Files:**
- Create: `src/dashboard.py`

- [ ] **Step 1: Create dashboard helper module**

Create [`src/dashboard.py`](src/dashboard.py):

```python
def recommend_mitigation_action(predicted_label: str, confidence: float) -> str:
    if confidence < 0.60:
        return "Flag for analyst review"
    if predicted_label == "syn_flood":
        return "Recommend SYN rate limiting or SYN proxy"
    if predicted_label == "udp_flood":
        return "Recommend UDP rate limiting or destination filtering"
    return "No action recommended"
```

- [ ] **Step 2: Verify mitigation recommendations**

Run: `uv run python -c "from src.dashboard import recommend_mitigation_action; print(recommend_mitigation_action('syn_flood', 0.95)); print(recommend_mitigation_action('udp_flood', 0.40))"`

Expected: first line contains `SYN`, second line is `Flag for analyst review`

- [ ] **Step 3: Commit**

```bash
git add src/dashboard.py
git commit -m "feat: add mitigation recommendation logic for dashboard"
```

---

### Task 4: Dashboard shell

**Files:**
- Modify: `src/dashboard.py`

- [ ] **Step 1: Add dashboard title and shell**

Extend [`src/dashboard.py`](src/dashboard.py) with:

```python
import streamlit as st


def build_dashboard_title() -> str:
    return "Lightweight DDoS Detection Dashboard"


def render_dashboard() -> None:
    st.title(build_dashboard_title())
    st.caption("Offline-trained three-class DDoS detector demo")
```

- [ ] **Step 2: Add module entrypoint**

Append to [`src/dashboard.py`](src/dashboard.py):

```python
if __name__ == "__main__":
    render_dashboard()
```

- [ ] **Step 3: Verify dashboard module imports**

Run: `uv run python -c "from src.dashboard import build_dashboard_title; print(build_dashboard_title())"`

Expected: output is `Lightweight DDoS Detection Dashboard`

- [ ] **Step 4: Commit**

```bash
git add src/dashboard.py
git commit -m "feat: add streamlit dashboard shell"
```

---

### Task 5: Prediction summary helpers

**Files:**
- Modify: `src/dashboard.py`

- [ ] **Step 1: Add prediction summary formatter**

Extend [`src/dashboard.py`](src/dashboard.py) with:

```python
def build_prediction_summary(
    sample_name: str,
    predicted_label: str,
    confidence: float,
    recommendation: str,
) -> str:
    return (
        f"Sample: {sample_name} | "
        f"Prediction: {predicted_label} | "
        f"Confidence: {confidence:.2f} | "
        f"Recommendation: {recommendation}"
    )
```

- [ ] **Step 2: Verify summary formatting**

Run: `uv run python -c "from src.dashboard import build_prediction_summary; print(build_prediction_summary('sample_syn_flood.csv', 'syn_flood', 0.91, 'Recommend SYN rate limiting or SYN proxy'))"`

Expected: output contains `sample_syn_flood.csv`, `syn_flood`, and `0.91`

- [ ] **Step 3: Commit**

```bash
git add src/dashboard.py
git commit -m "feat: add dashboard prediction summary helpers"
```

---

### Task 6: Optional replay helpers

**Files:**
- Modify: `src/dashboard.py`

- [ ] **Step 1: Add replay summary helper**

Extend [`src/dashboard.py`](src/dashboard.py) with:

```python
import pandas as pd


def build_replay_window_summary(frame: pd.DataFrame) -> str:
    return f"Replay window contains {len(frame)} rows"
```

- [ ] **Step 2: Verify replay summary**

Run: `uv run python -c "import pandas as pd; from src.dashboard import build_replay_window_summary; print(build_replay_window_summary(pd.DataFrame({'Protocol':['TCP','UDP']})))"`

Expected: output is `Replay window contains 2 rows`

- [ ] **Step 3: Commit**

```bash
git add src/dashboard.py
git commit -m "feat: add optional replay mode helpers"
```

---

### Task 7: Dashboard commands documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Append dashboard commands**

Append this section to [`README.md`](README.md):

```markdown
## Dashboard Commands

### Run dashboard

```bash
uv run streamlit run src/dashboard.py
```

### Inspect sample files

```bash
uv run python -c "from src.train import list_sample_prediction_files; print([path.name for path in list_sample_prediction_files()])"
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
git commit -m "docs: add dashboard and demo commands"
```

---

## Self-Review

### Spec coverage check

This plan covers the spec requirements for:
- predefined static CSV samples
- mitigation recommendation logic
- local dashboard shell
- prediction summary presentation
- optional replay helpers

It intentionally does not cover dataset/schema foundations or ML training because those are split into earlier plans.

### Placeholder scan

No placeholders remain. Every step includes exact files, code, commands, and expected outcomes.

### Type consistency check

Names are consistent within this plan:
- [`list_sample_prediction_files()`](src/train.py)
- [`recommend_mitigation_action()`](src/dashboard.py)
- [`build_dashboard_title()`](src/dashboard.py)
- [`render_dashboard()`](src/dashboard.py)
- [`build_prediction_summary()`](src/dashboard.py)
- [`build_replay_window_summary()`](src/dashboard.py)

Plan complete and saved to [`docs/superpowers/plans/2026-05-28-plan-3-dashboard-and-demo.md`](docs/superpowers/plans/2026-05-28-plan-3-dashboard-and-demo.md). Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?