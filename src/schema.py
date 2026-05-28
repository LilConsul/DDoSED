from collections.abc import Sequence

import pandas as pd

ATTACK_TYPE_COLUMN = "Attack Type"
TARGET_COLUMN = "target"

PROTOCOL_COLUMN = "Protocol"
SOURCE_PORT_COLUMN = "Source Port"
DESTINATION_PORT_COLUMN = "Destination Port"
PACKET_SIZE_COLUMN = "Packet Size"
PAYLOAD_LENGTH_COLUMN = "Payload Length"
FLOW_DURATION_COLUMN = "Flow Duration"
BYTES_IN_FLOW_COLUMN = "Bytes in Flow"
PACKETS_IN_FLOW_COLUMN = "Packets in Flow"
AVERAGE_PACKET_SIZE_COLUMN = "Average Packet Size"
INTER_ARRIVAL_TIME_COLUMN = "Inter-Arrival Time"
RATE_OF_PACKETS_COLUMN = "Rate of Packets"
UNIQUE_SOURCE_COUNT_COLUMN = "Unique Source Count"
UNIQUE_DESTINATION_COUNT_COLUMN = "Unique Destination Count"

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
    PROTOCOL_COLUMN,
    SOURCE_PORT_COLUMN,
    DESTINATION_PORT_COLUMN,
    PACKET_SIZE_COLUMN,
    PAYLOAD_LENGTH_COLUMN,
    FLOW_DURATION_COLUMN,
    BYTES_IN_FLOW_COLUMN,
    PACKETS_IN_FLOW_COLUMN,
    AVERAGE_PACKET_SIZE_COLUMN,
    INTER_ARRIVAL_TIME_COLUMN,
    RATE_OF_PACKETS_COLUMN,
    UNIQUE_SOURCE_COUNT_COLUMN,
    UNIQUE_DESTINATION_COUNT_COLUMN,
)

LIGHTWEIGHT_FEATURE_COLUMNS: tuple[str, ...] = (
    PROTOCOL_COLUMN,
    SOURCE_PORT_COLUMN,
    DESTINATION_PORT_COLUMN,
    PACKET_SIZE_COLUMN,
    PAYLOAD_LENGTH_COLUMN,
    FLOW_DURATION_COLUMN,
    BYTES_IN_FLOW_COLUMN,
    PACKETS_IN_FLOW_COLUMN,
    INTER_ARRIVAL_TIME_COLUMN,
    RATE_OF_PACKETS_COLUMN,
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
