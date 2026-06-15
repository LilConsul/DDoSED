from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def split_dataset(
    frame: pd.DataFrame,
    *,
    target_column: str,
    train_size: float = 0.7,
    test_size: float = 0.3,
    random_state: int = 42,
    shuffle: bool = True,
    stratify: bool = True,
) -> DatasetSplit:
    total = train_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_size + test_size must equal 1.0")

    if shuffle:
        frame = frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    X = frame.drop(columns=[target_column])
    y = frame[target_column]

    stratify_values = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=stratify_values,
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
