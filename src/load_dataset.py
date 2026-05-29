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

PROTOCOL_ENCODING = {
    "ICMP": 1,
    "TCP": 2,
    "UDP": 3,
}

ATTACK_TYPE_ENCODING = {
    "No Attack": 1,
    "UDP Flood": 2,
    "SYN Flood": 3,
}


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
    logger.info("Dataset loaded successfully: %d rows, %d columns", len(frame), len(frame.columns))
    return frame


def preprocess_dataset(dataset: pd.DataFrame, exclude_columns: list[str] | None = None) -> pd.DataFrame:
    columns_to_drop = EXCLUDE_COLUMNS if exclude_columns is None else exclude_columns
    dataset = dataset.drop(columns=columns_to_drop, errors="ignore")

    if "Protocol" in dataset.columns:
        unknown_protocols = sorted(set(dataset["Protocol"].dropna()) - set(PROTOCOL_ENCODING))
        if unknown_protocols:
            raise ValueError(f"Unknown Protocol values: {unknown_protocols}")
        dataset["Protocol"] = dataset["Protocol"].map(PROTOCOL_ENCODING)

    if "Attack Type" in dataset.columns:
        unknown_attack_types = sorted(set(dataset["Attack Type"].dropna()) - set(ATTACK_TYPE_ENCODING))
        if unknown_attack_types:
            raise ValueError(f"Unknown Attack Type values: {unknown_attack_types}")
        dataset["Attack Type"] = dataset["Attack Type"].map(ATTACK_TYPE_ENCODING)

    return dataset


def build_exclude_columns(
    include_ports: bool,
    include_timestamp: bool,
    include_ips: bool,
) -> list[str]:
    columns_to_drop = EXCLUDE_COLUMNS.copy()
    if include_ports:
        columns_to_drop = [
            column for column in columns_to_drop if column not in {"Source Port", "Destination Port"}
        ]
    if include_timestamp:
        columns_to_drop = [column for column in columns_to_drop if column != "Timestamp"]
    if include_ips:
        columns_to_drop = [
            column for column in columns_to_drop if column not in {"Source IP", "Destination IP"}
        ]
    return columns_to_drop


if __name__ == "__main__":
    dataset = load_dataset(DATASET_PATH)
    processed_dataset = preprocess_dataset(dataset)
