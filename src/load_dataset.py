import logging
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from src.paths import DATASET_PATH, PROJECT_ROOT

logger = logging.getLogger(__name__)

KAGGLE_DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/dhoogla/cicddos2019"
)


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


def _load_member_from_zip(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member) as handle:
        data = BytesIO(handle.read())

    if member.lower().endswith(".parquet"):
        return pd.read_parquet(data)
    if member.lower().endswith(".csv"):
        return pd.read_csv(data)

    raise ValueError(f"Unsupported file in ZIP: {member}")


def _select_zip_members(archive: zipfile.ZipFile, requested: list[str] | None) -> list[str]:
    members = [name for name in archive.namelist() if not name.endswith("/")]
    if requested:
        missing = [name for name in requested if name not in members]
        if missing:
            raise FileNotFoundError(
                f"Requested files not found in ZIP: {missing}. Available: {members}"
            )
        return requested

    if len(members) == 1:
        return members

    data_members = [
        name
        for name in members
        if name.lower().endswith((".parquet", ".csv"))
    ]
    if not data_members:
        raise ValueError(f"No data files found in ZIP. Available: {members}")

    return data_members


def load_dataset(
    path: Path,
    auto_download: bool = True,
    zip_members: list[str] | None = None,
) -> pd.DataFrame:
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

    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = _select_zip_members(archive, zip_members)
            logger.info("Loading %d file(s) from ZIP", len(members))
            frames = [_load_member_from_zip(archive, member) for member in members]
        frame = pd.concat(frames, ignore_index=True)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)

    logger.info(
        "Dataset loaded successfully: %d rows, %d columns",
        len(frame),
        len(frame.columns),
    )
    return frame


if __name__ == "__main__":
    dataset = load_dataset(DATASET_PATH)
