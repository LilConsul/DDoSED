import logging
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from paths import DATASET_PATH, REPORTS_ROOT
from load_dataset import load_dataset, SYN_UDP_MEMBERS
from evaluation import build_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KEEP_LABELS = {"Syn", "UDP", "Benign"}


def prepare_data():
    """Load, filter, clean, and scale the CIC-DDoS2019 dataset."""
    # 1. Load specific parquet members from the ZIP
    logger.info("Loading dataset members: %s", SYN_UDP_MEMBERS)
    df = load_dataset(DATASET_PATH, auto_download=True, zip_members=SYN_UDP_MEMBERS)

    # 2. Filter labels
    df = df[df["Label"].isin(KEEP_LABELS)].reset_index(drop=True)
    logger.info("Filtered to %d rows with labels: %s", len(df), KEEP_LABELS)

    # 3. Drop single-value columns
    single_val_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df.drop(columns=single_val_cols, inplace=True)

    # 4. Drop highly correlated features (>0.8)
    num_cols = [c for c in df.columns if c != "Label"]
    corr = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [c for c in upper.columns if any(upper[c] > 0.8)]
    df.drop(columns=drop_corr, inplace=True)

    # 5. Remove duplicates
    df.drop_duplicates(inplace=True)
    logger.info("After cleaning: %d rows, %d features", len(df), len(df.columns) - 1)

    # 6. Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])
    label_names = list(le.classes_)
    X = df.drop(columns=["Label"]).values

    # 7. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 8. Scale features
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, label_names


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, label_names = prepare_data()

    scores = build_report(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        label_names=label_names,
        output_dir=REPORTS_ROOT,
        cv_folds=5,
    )
