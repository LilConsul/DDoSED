from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
SRC_ROOT = PROJECT_ROOT / "src"

DATA_ROOT.mkdir(exist_ok=True, parents=True)

DATASET_PATH = DATA_ROOT / "inddos24-dataset.zip"
