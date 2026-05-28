import pandas as pd

THRESHOLD_PACKET_RATE = 3000
THRESHOLD_PACKETS_IN_FLOW = 2000
THRESHOLD_UNIQUE_SOURCE_COUNT = 100

PROTOCOL_COLUMN = "Protocol"
RATE_OF_PACKETS_COLUMN = "Rate of Packets"
PACKETS_IN_FLOW_COLUMN = "Packets in Flow"
UNIQUE_SOURCE_COUNT_COLUMN = "Unique Source Count"


def predict_rule_based(frame: pd.DataFrame) -> pd.Series:
    predictions = pd.Series("normal", index=frame.index)

    tcp_attack_mask = (
        (frame[PROTOCOL_COLUMN] == "TCP")
        & (frame[RATE_OF_PACKETS_COLUMN] >= THRESHOLD_PACKET_RATE)
        & (frame[PACKETS_IN_FLOW_COLUMN] >= THRESHOLD_PACKETS_IN_FLOW)
        & (frame[UNIQUE_SOURCE_COUNT_COLUMN] >= THRESHOLD_UNIQUE_SOURCE_COUNT)
    )

    udp_attack_mask = (
        (frame[PROTOCOL_COLUMN] == "UDP")
        & (frame[RATE_OF_PACKETS_COLUMN] >= THRESHOLD_PACKET_RATE)
        & (frame[PACKETS_IN_FLOW_COLUMN] >= THRESHOLD_PACKETS_IN_FLOW)
        & (frame[UNIQUE_SOURCE_COUNT_COLUMN] >= THRESHOLD_UNIQUE_SOURCE_COUNT)
    )

    predictions[tcp_attack_mask] = "syn_flood"
    predictions[udp_attack_mask] = "udp_flood"

    return predictions
