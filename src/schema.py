from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET_COLUMN = "Attack Type"

FEATURE_CATEGORICAL_COLUMNS = [
    "Protocol",
]

NUMERICAL_COLUMNS = [
    "Packet Size",
    "Payload Length",
    "Flow Duration",
    "Bytes in Flow",
    "Packets in Flow",
    "Average Packet Size",
    "Inter-Arrival Time",
    "Rate of Packets",
    "Unique Source Count",
    "Unique Destination Count",
    "Attack Duration",
]

ATTACK_LABELS = ["No Attack", "UDP Flood", "SYN Flood"]


@dataclass(frozen=True)
class FeatureSchema:
    categorical: list[str]
    numerical: list[str]
    target: str


DEFAULT_SCHEMA = FeatureSchema(
    categorical=FEATURE_CATEGORICAL_COLUMNS,
    numerical=NUMERICAL_COLUMNS,
    target=TARGET_COLUMN,
)


def validate_schema(frame: pd.DataFrame, schema: FeatureSchema = DEFAULT_SCHEMA) -> None:
    missing = [
        column
        for column in schema.categorical + schema.numerical + [schema.target]
        if column not in frame.columns
    ]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing expected columns: {missing_text}")


def to_binary_target(labels: pd.Series) -> pd.Series:
    return labels.apply(lambda value: "No Attack" if value == "No Attack" else "DDoS")

