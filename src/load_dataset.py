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

# Made with Bob
