from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
SRC_ROOT = PROJECT_ROOT / "src"
# ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
REPORTS_ROOT = PROJECT_ROOT / "reports"
MODELS_ROOT = PROJECT_ROOT / "models"
# SAMPLE_DATA_ROOT = DATA_ROOT / "samples"

DATA_ROOT.mkdir(exist_ok=True, parents=True)
# ARTIFACTS_ROOT.mkdir(exist_ok=True, parents=True)
REPORTS_ROOT.mkdir(exist_ok=True, parents=True)
MODELS_ROOT.mkdir(exist_ok=True, parents=True)
# SAMPLE_DATA_ROOT.mkdir(exist_ok=True, parents=True)

DATASET_PATH = DATA_ROOT / "inddos24-dataset.zip"
