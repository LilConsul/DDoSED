import logging
import subprocess
from pathlib import Path

import pandas as pd

from src.paths import DATASET_PATH
from src.schema import validate_required_columns

logger = logging.getLogger(__name__)

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/datasetengineer/inddos24-dataset"
)


def download_dataset(path: Path) -> None:
    logger.info("Dataset not found at %s", path)
    logger.info("Downloading from Kaggle...")

    try:
        result = subprocess.run(
            ["curl", "-L", "-o", str(path), KAGGLE_DATASET_URL],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Dataset downloaded successfully to %s", path)
        else:
            logger.error("Download failed with return code %d", result.returncode)
            raise RuntimeError(f"Download failed: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logger.error("Error downloading dataset: %s", e.stderr)
        raise RuntimeError(f"Failed to download dataset: {e.stderr}") from e
    except FileNotFoundError as e:
        logger.error("curl command not found. Please install curl.")
        raise RuntimeError(
            "curl is required for downloading. Please install curl."
        ) from e


def load_dataset(path: Path, auto_download: bool = True) -> pd.DataFrame:
    if not path.exists():
        if auto_download:
            download_dataset(path)
        else:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Set auto_download=True to download automatically."
            )

    logger.info("Loading dataset from %s", path)
    frame = pd.read_csv(path)
    validate_required_columns(frame.columns)
    logger.info("Dataset loaded successfully: %d rows, %d columns", len(frame), len(frame.columns))
    return frame


if __name__ == "__main__":
    dataset = load_dataset(DATASET_PATH)
    print(dataset.head())
    print(dataset.shape)
    print(dataset.dtypes)
    print(dataset["Attack Type"].value_counts())
