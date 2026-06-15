"""
main.py — Test driver for evaluation.py.
Run: python main.py
"""

import warnings

warnings.filterwarnings("ignore")
from evaluation import build_report
from paths import REPORTS_ROOT
from preprocess import preprocess_data


def main():

    X_train, X_test, y_train, y_test, label_names, _ = preprocess_data()

    scores = build_report(
        X_train,
        X_test,
        y_train,
        y_test,
        label_names=label_names,
        output_dir=REPORTS_ROOT,
        cv_folds=3,  # keep low for quick test
    )
    print(f"\nDone. Check {REPORTS_ROOT} for plots and classification_reports.txt")


if __name__ == "__main__":
    main()
