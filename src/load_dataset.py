import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.paths import DATASET_PATH
from src.schema import validate_required_columns

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/datasetengineer/inddos24-dataset"
)


def download_dataset(path: Path) -> None:
    print(f"Dataset not found at {path}")
    print("Downloading from Kaggle...")

    try:
        subprocess.run(
            ["curl", "-L", "-o", str(path), KAGGLE_DATASET_URL],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Dataset downloaded successfully to {path}")
    except subprocess.CalledProcessError as e:
        print(f"Error downloading dataset: {e.stderr}", file=sys.stderr)
        raise
    except FileNotFoundError:
        print("Error: curl command not found. Please install curl.", file=sys.stderr)
        raise


def load_dataset(path: Path, auto_download: bool = True) -> pd.DataFrame:
    if not path.exists():
        if auto_download:
            download_dataset(path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Set auto_download=True to download automatically."
            )

    frame = pd.read_csv(path)
    validate_required_columns(frame.columns)
    return frame


if __name__ == "__main__":
    dataset = load_dataset(DATASET_PATH)
    print(dataset.head())
    print(dataset.shape)
    print(dataset.dtypes)
    print(dataset["Attack Type"].value_counts())
