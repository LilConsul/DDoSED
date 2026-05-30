from __future__ import annotations

import logging

import pandas as pd

from src.dataset_split import split_dataset
from src.schema import DEFAULT_SCHEMA
from src.windowing import build_windowed_dataset

logger = logging.getLogger(__name__)


def build_fake_data(rows: int = 120) -> pd.DataFrame:
    protocols = ["TCP", "UDP", "ICMP"]
    attacks = ["No Attack", "UDP Flood", "SYN Flood"]
    return pd.DataFrame(
        {
            "Protocol": [protocols[i % len(protocols)] for i in range(rows)],
            "Attack Type": [attacks[i % len(attacks)] for i in range(rows)],
            "Packet Size": list(range(100, 100 + rows)),
            "Payload Length": list(range(200, 200 + rows)),
            "Flow Duration": [1.5] * rows,
            "Bytes in Flow": list(range(300, 300 + rows)),
            "Packets in Flow": list(range(10, 10 + rows)),
            "Average Packet Size": [64.0] * rows,
            "Inter-Arrival Time": [0.2] * rows,
            "Rate of Packets": [3.0] * rows,
            "Unique Source Count": [2] * rows,
            "Unique Destination Count": [5] * rows,
            "Attack Duration": [0.5] * rows,
        }
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    frame = build_fake_data()

    windowed = build_windowed_dataset(
        frame,
        window_size=4,
        schema=DEFAULT_SCHEMA,
    )

    split = split_dataset(
        windowed.data,
        target_column=DEFAULT_SCHEMA.target,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
        shuffle=True,
        stratify=False,
    )

    logger.info("Windowed rows: %d", len(windowed.data))
    logger.info("Train rows: %d", len(split.X_train))
    logger.info("Validation rows: %d", len(split.X_val))
    logger.info("Test rows: %d", len(split.X_test))


if __name__ == "__main__":
    main()
