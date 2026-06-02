import zipfile
from io import BytesIO
import pandas as pd
from src.paths import DATASET_PATH

SYN_UDP_MEMBERS = [
    "Syn-training.parquet",
    "Syn-testing.parquet",
    "UDP-training.parquet",
    "UDP-testing.parquet",
]

# --- Load ---
with zipfile.ZipFile(DATASET_PATH) as archive:
    train_frames = []
    test_frames = []
    for member in SYN_UDP_MEMBERS:
        with archive.open(member) as f:
            df = pd.read_parquet(BytesIO(f.read()))
        if "training" in member:
            train_frames.append(df)
        else:
            test_frames.append(df)

train_df = pd.concat(train_frames, ignore_index=True)
test_df = pd.concat(test_frames, ignore_index=True)

# --- Add this BEFORE the KEEP_LABELS filtering step ---
LABEL_MAP = {
    "DrDoS_UDP": "UDP",
    "UDP-lag": "UDP",
    "UDPLag": "UDP",
    "BENIGN": "Benign",
    "DrDoS_Syn": "Syn",
}
train_df["Label"] = train_df["Label"].replace(LABEL_MAP)
test_df["Label"] = test_df["Label"].replace(LABEL_MAP)

# Now the filter will work correctly:
KEEP_LABELS = {"Syn", "UDP", "Benign"}
train_df = train_df[train_df["Label"].isin(KEEP_LABELS)].reset_index(drop=True)
test_df = test_df[test_df["Label"].isin(KEEP_LABELS)].reset_index(drop=True)

# --- Preprocessing (mirrors original notebook) ---
# Drop single-unique-value columns
single_val_cols = [c for c in train_df.columns if train_df[c].nunique() <= 1]
train_df.drop(columns=single_val_cols, inplace=True)
test_df.drop(columns=single_val_cols, errors="ignore", inplace=True)

# Drop highly correlated columns (threshold 0.8)
corr_matrix = train_df.drop(columns=["Label"]).corr().abs()
upper = corr_matrix.where(
    pd.np.triu(pd.np.ones(corr_matrix.shape), k=1).astype(bool)
)  # upper triangle
drop_corr = [col for col in upper.columns if any(upper[col] > 0.8)]
train_df.drop(columns=drop_corr, inplace=True)
test_df.drop(columns=drop_corr, errors="ignore", inplace=True)

# Remove duplicates
train_df.drop_duplicates(inplace=True)

# --- Encode labels ---
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

le = LabelEncoder()
y_train = le.fit_transform(train_df["Label"])
y_test = le.transform(test_df["Label"])

label_map = dict(enumerate(le.classes_))
print("Label map:", label_map)
# e.g. {0: 'Benign', 1: 'Syn', 2: 'UDP'}

X_train = train_df.drop(columns=["Label"])
X_test = test_df.drop(columns=["Label"])

# Train/val split from training set
X_train = train_df.drop(columns=["Label"])
y_train = le.transform(train_df["Label"])
X_val = test_df.drop(columns=["Label"])
y_val = le.transform(test_df["Label"])
# --- Scale ---
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
