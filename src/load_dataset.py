from pathlib import Path
import pandas as pd
from paths import DATASET_PATH


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_dataset(DATASET_PATH)
    print(df.head())
    print(df.shape)
    print(df.dtypes)

    print(df["Attack Type"].value_counts())
