import logging
import subprocess
from pathlib import Path

import pandas as pd

from src.paths import DATASET_PATH, PROJECT_ROOT

logger = logging.getLogger(__name__)

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/datasetengineer/inddos24-dataset"
)

EXCLUDE_COLUMNS = [
    "Labels",
    "Firmware Version",
    "Anomaly Score",
    "Target Device",
    "Operating System",
    "Device Type",
    "Timestamp",
    "Source IP",
    "Destination IP",
    "Source Port",
    "Destination Port",
]


def download_dataset(path: Path) -> None:
    relative_path = path.relative_to(PROJECT_ROOT)
    logger.info("Dataset not found at %s", relative_path)
    logger.info("Downloading from Kaggle...")

    try:
        result = subprocess.run(
            ["curl", "-L", "-o", str(path), KAGGLE_DATASET_URL],
            check=True,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Dataset downloaded successfully to %s", relative_path)
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
            relative_path = path.relative_to(PROJECT_ROOT)
            raise FileNotFoundError(
                f"Dataset not found at {relative_path}. Set auto_download=True to download automatically."
            )

    relative_path = path.relative_to(PROJECT_ROOT)
    logger.info("Loading dataset from %s", relative_path)
    frame = pd.read_csv(path)
    logger.info(
        "Dataset loaded successfully: %d rows, %d columns",
        len(frame),
        len(frame.columns),
    )
    return frame


def preprocess_dataset(
    dataset: pd.DataFrame, exclude_columns: list[str] | None = None
) -> pd.DataFrame:
    columns_to_drop = EXCLUDE_COLUMNS if exclude_columns is None else exclude_columns
    dataset = dataset.drop(columns=columns_to_drop, errors="ignore")

    return dataset


if __name__ == "__main__":
    dataset = load_dataset(DATASET_PATH)
    processed_dataset = preprocess_dataset(dataset)
