from collections.abc import Sequence

import pandas as pd

ATTACK_TYPE_COLUMN = "Attack Type"
TARGET_COLUMN = "target"

LABEL_MAPPING = {
    "No Attack": "normal",
    "SYN Flood": "syn_flood",
    "UDP Flood": "udp_flood",
}

DROP_COLUMNS: tuple[str, ...] = (
    "Timestamp",
    "Source IP",
    "Destination IP",
    "Device Type",
    "Operating System",
    "Firmware Version",
    "Anomaly Score",
    "Labels",
)

FULL_FEATURE_COLUMNS: tuple[str, ...] = (
    "Protocol",
    "Source Port",
    "Destination Port",
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
)

LIGHTWEIGHT_FEATURE_COLUMNS: tuple[str, ...] = (
    "Protocol",
    "Source Port",
    "Destination Port",
    "Packet Size",
    "Payload Length",
    "Flow Duration",
    "Bytes in Flow",
    "Packets in Flow",
    "Inter-Arrival Time",
    "Rate of Packets",
)

REQUIRED_COLUMNS: tuple[str, ...] = (
    ATTACK_TYPE_COLUMN,
    *FULL_FEATURE_COLUMNS,
)


def validate_required_columns(columns: Sequence[str]) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Dataset is missing required columns: {missing}")


def normalize_attack_labels(frame: pd.DataFrame) -> pd.Series:
    labels = frame[ATTACK_TYPE_COLUMN].map(LABEL_MAPPING)
    if labels.isna().any():
        unknown_labels = sorted(frame.loc[labels.isna(), ATTACK_TYPE_COLUMN].unique())
        raise ValueError(f"Unsupported attack labels: {unknown_labels}")
    return labels
