import pandas as pd

THRESHOLD_PACKET_RATE = 3000
THRESHOLD_PACKETS_IN_FLOW = 2000
THRESHOLD_UNIQUE_SOURCE_COUNT = 100


def predict_rule_based(frame: pd.DataFrame) -> pd.Series:
    predictions: list[str] = []

    for _, row in frame.iterrows():
        protocol = row["Protocol"]
        packet_rate = row["Rate of Packets"]
        packets_in_flow = row["Packets in Flow"]
        unique_source_count = row["Unique Source Count"]

        if (
            protocol == "TCP"
            and packet_rate >= THRESHOLD_PACKET_RATE
            and packets_in_flow >= THRESHOLD_PACKETS_IN_FLOW
            and unique_source_count >= THRESHOLD_UNIQUE_SOURCE_COUNT
        ):
            predictions.append("syn_flood")
        elif (
            protocol == "UDP"
            and packet_rate >= THRESHOLD_PACKET_RATE
            and packets_in_flow >= THRESHOLD_PACKETS_IN_FLOW
            and unique_source_count >= THRESHOLD_UNIQUE_SOURCE_COUNT
        ):
            predictions.append("udp_flood")
        else:
            predictions.append("normal")

    return pd.Series(predictions)
