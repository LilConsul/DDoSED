"""
load_dataset.py — Handles downloading and extracting the CICDDoS2019 dataset.
"""

import logging
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path
import pandas as pd
from paths import DATASET_PATH, PROJECT_ROOT

logger = logging.getLogger(__name__)

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/dhoogla/cicddos2019"
)


def download_dataset(path: Path) -> None:
    relative_path = (
        path.relative_to(PROJECT_ROOT) if PROJECT_ROOT in path.parents else path
    )
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
            logger.info("Dataset downloaded successfully.")
        else:
            raise RuntimeError(f"Download failed: {result.stderr}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download dataset: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("curl is required for downloading. Please install curl.")


def load_syn_udp_split(
    path: Path = DATASET_PATH, auto_download: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the specific SYN/UDP train and test parquet files from the ZIP.
    Returns: (train_df, test_df)
    """
    if not path.exists():
        if auto_download:
            download_dataset(path)
        else:
            raise FileNotFoundError(f"Dataset not found at {path}")

    train_members = ["Syn-training.parquet", "UDP-training.parquet"]
    test_members = ["Syn-testing.parquet", "UDP-testing.parquet"]

    train_frames, test_frames = [], []

    logger.info("Extracting train/test split from %s", path.name)
    with zipfile.ZipFile(path) as archive:
        available = archive.namelist()
        for m in train_members:
            if m not in available:
                raise FileNotFoundError(f"{m} missing from ZIP")
            with archive.open(m) as f:
                train_frames.append(pd.read_parquet(BytesIO(f.read())))

        for m in test_members:
            if m not in available:
                raise FileNotFoundError(f"{m} missing from ZIP")
            with archive.open(m) as f:
                test_frames.append(pd.read_parquet(BytesIO(f.read())))

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    logger.info(
        "Loaded native split: Train=%d rows, Test=%d rows", len(train_df), len(test_df)
    )
    return train_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tr, te = load_syn_udp_split()
    print(f"Train labels: {tr['Label'].unique()}")
    print(f"Test labels: {te['Label'].unique()}")
