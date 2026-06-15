from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.schema import FeatureSchema, DEFAULT_SCHEMA, validate_schema


@dataclass
class WindowedDataset:
    data: pd.DataFrame
    numeric_features: list[str]
    categorical_features: list[str]


def _mode_or_first(series: pd.Series) -> str:
    if series.empty:
        return ""
    mode_values = series.mode()
    if not mode_values.empty:
        return str(mode_values.iloc[0])
    return str(series.iloc[0])


def build_windowed_dataset(
    frame: pd.DataFrame,
    window_size: int,
    *,
    schema: FeatureSchema = DEFAULT_SCHEMA,
) -> WindowedDataset:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")

    validate_schema(frame, schema)

    numeric_features: list[str] = []
    for column in schema.numerical:
        numeric_features.extend(
            [
                f"{column}__mean",
                f"{column}__std",
                f"{column}__min",
                f"{column}__max",
            ]
        )
    categorical_features = [f"{column}__mode" for column in schema.categorical]

    rows: list[dict[str, object]] = []
    for start in range(0, len(frame), window_size):
        chunk = frame.iloc[start : start + window_size]
        if chunk.empty:
            continue

        row: dict[str, object] = {}
        for column in schema.numerical:
            series = chunk[column]
            row[f"{column}__mean"] = float(series.mean())
            row[f"{column}__std"] = float(series.std(ddof=0))
            row[f"{column}__min"] = float(series.min())
            row[f"{column}__max"] = float(series.max())

        for column in schema.categorical:
            row[f"{column}__mode"] = _mode_or_first(chunk[column])

        labels = chunk[schema.target]
        attack_ratio = float((labels != "No Attack").mean())
        score = int(round(attack_ratio * 10))
        row[schema.target] = max(1, min(10, score))

        rows.append(row)

    windowed = pd.DataFrame(rows)
    if not windowed.empty:
        std_columns = [col for col in windowed.columns if col.endswith("__std")]
        windowed[std_columns] = windowed[std_columns].fillna(0.0)

    return WindowedDataset(
        data=windowed,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )
