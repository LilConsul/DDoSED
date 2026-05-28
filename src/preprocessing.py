import pandas as pd

from src.schema import DROP_COLUMNS, TARGET_COLUMN, normalize_attack_labels


def prepare_model_frame(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.drop(columns=list(DROP_COLUMNS), errors="ignore").copy()
    prepared[TARGET_COLUMN] = normalize_attack_labels(frame)
    return prepared
