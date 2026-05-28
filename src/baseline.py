import pandas as pd


def predict_rule_based(frame: pd.DataFrame) -> pd.Series:
    predictions: list[str] = []

    for _, row in frame.iterrows():
        protocol = row["Protocol"]
        packet_rate = row["Rate of Packets"]
        packets_in_flow = row["Packets in Flow"]
        unique_source_count = row["Unique Source Count"]

        if (
            protocol == "TCP"
            and packet_rate >= 3000
            and packets_in_flow >= 2000
            and unique_source_count >= 100
        ):
            predictions.append("syn_flood")
        elif (
            protocol == "UDP"
            and packet_rate >= 3000
            and packets_in_flow >= 2000
            and unique_source_count >= 100
        ):
            predictions.append("udp_flood")
        else:
            predictions.append("normal")

    return pd.Series(predictions)
