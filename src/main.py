"""
main.py — Test driver for evaluation.py using synthetic data.
Run: python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from evaluation import build_report
from paths import REPORTS_ROOT

# ---------------------------------------------------------------------------
# Generate synthetic 3-class data mimicking Benign / Syn / UDP
# ---------------------------------------------------------------------------

print("Generating synthetic dataset...")

X, y = make_classification(
    n_samples=6000,
    n_features=30,
    n_classes=3,
    n_informative=20,
    n_redundant=5,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Classes: Benign={np.sum(y_test==0)}  Syn={np.sum(y_test==1)}  UDP={np.sum(y_test==2)}\n")

# ---------------------------------------------------------------------------
# Run evaluation
# ---------------------------------------------------------------------------

scores = build_report(
    X_train, X_test, y_train, y_test,
    label_names=["Benign", "Syn", "UDP"],
    output_dir=REPORTS_ROOT,
    cv_folds=3,  # keep low for quick test
)

print(f"\nDone. Check {REPORTS_ROOT} for plots and classification_reports.txt")
