"""
evaluation.py — Model training & evaluation report for SYN/UDP DDoS detection.

Usage:
    from evaluation import build_report
    build_report(X_train, X_test, y_train, y_test, label_names, output_dir="reports")

Or run directly:
    python evaluation.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier

from paths import REPORTS_ROOT

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")

# ---------------------------------------------------------------------------
# Default model catalogue
# ---------------------------------------------------------------------------

DEFAULT_MODELS: dict = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "KNN":           KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
    "Extra Trees":   ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "MLP":           MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "XGBoost":       XGBClassifier(n_estimators=100, random_state=42, eval_metric="mlogloss", verbosity=0),
}


# ---------------------------------------------------------------------------
# Core training + metrics
# ---------------------------------------------------------------------------

def train_and_evaluate(
    models: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """
    Train every model and collect metrics.

    Returns
    -------
    scores_df : DataFrame with one row per model
    trained   : dict mapping model name → fitted estimator
    """
    rows = []
    trained = {}

    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        roc_auc   = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr")
        cv_score  = float(np.mean(cross_val_score(model, X_train, y_train, cv=cv_folds, n_jobs=-1)))

        rows.append({
            "Model":     name,
            "Accuracy":  round(accuracy,  4),
            "Precision": round(precision, 4),
            "Recall":    round(recall,    4),
            "F1 Score":  round(f1,        4),
            "ROC AUC":   round(roc_auc,   4),
            f"CV ({cv_folds}-fold)": round(cv_score, 4),
        })
        trained[name] = model
        logger.info(
            "  %-16s  acc=%.4f  f1=%.4f  roc_auc=%.4f",
            name, accuracy, f1, roc_auc,
        )

    scores_df = pd.DataFrame(rows).set_index("Model")
    return scores_df, trained


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_metrics_comparison(scores_df: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar chart comparing all metrics side-by-side for each model."""
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    data = scores_df[metrics]

    x      = np.arange(len(data))
    width  = 0.15
    colors = cm.tab10(np.linspace(0, 0.6, len(metrics)))

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics) / 2) * width + width / 2
        bars = ax.bar(x + offset, data[metric], width, label=metric, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(data.index, rotation=20, ha="right")
    ax.set_ylim(0.85, 1.03)
    ax.set_ylabel("Score")
    ax.set_title("Model Metrics Comparison — SYN / UDP DDoS Detection")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / "metrics_comparison.png")


def plot_roc_curves(
    trained: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Sequence[str],
    output_dir: Path,
) -> None:
    """One ROC subplot per class with all models overlaid."""
    n_classes = len(label_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5), sharey=True)
    if n_classes == 1:
        axes = [axes]

    colors = cm.tab10(np.linspace(0, 0.9, len(trained)))

    for ax, class_idx, class_name in zip(axes, range(n_classes), label_names):
        for (name, model), color in zip(trained.items(), colors):
            proba = model.predict_proba(X_test)[:, class_idx]
            fpr, tpr, _ = roc_curve((y_test == class_idx).astype(int), proba)
            auc = roc_auc_score((y_test == class_idx).astype(int), proba)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)

        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"ROC — class: {class_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    fig.suptitle("ROC Curves per Class", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "roc_curves.png")


def plot_confusion_matrices(
    trained: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Sequence[str],
    output_dir: Path,
) -> None:
    """Grid of normalised confusion matrices, one per model."""
    n     = len(trained)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()

    for ax, (name, model) in zip(axes, trained.items()):
        cm_arr = confusion_matrix(y_test, model.predict(X_test), normalize="true")
        disp   = ConfusionMatrixDisplay(cm_arr, display_labels=label_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
        ax.set_title(name, fontsize=10)

    for ax in axes[len(trained):]:
        ax.set_visible(False)

    fig.suptitle("Normalised Confusion Matrices", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir / "confusion_matrices.png")


def plot_cv_scores(scores_df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of cross-validation scores."""
    cv_col = [c for c in scores_df.columns if c.startswith("CV")][0]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = cm.viridis(np.linspace(0.2, 0.8, len(scores_df)))
    bars   = ax.bar(scores_df.index, scores_df[cv_col], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(scores_df[cv_col].min() - 0.02, 1.01)
    ax.set_ylabel("CV Score")
    ax.set_title(cv_col)
    ax.set_xticklabels(scores_df.index, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / "cv_scores.png")


# ---------------------------------------------------------------------------
# Text classification reports
# ---------------------------------------------------------------------------

def save_classification_reports(
    trained: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Sequence[str],
    output_dir: Path,
) -> None:
    """Write per-class precision/recall/f1 for every model to a .txt file."""
    output_path = output_dir / "classification_reports.txt"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("=" * 70 + "\n")
        fh.write("  CLASSIFICATION REPORTS — SYN / UDP DDoS Detection\n")
        fh.write("=" * 70 + "\n\n")
        for name, model in trained.items():
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=label_names, digits=4)
            fh.write(f"{'─' * 70}\n")
            fh.write(f"  {name}\n")
            fh.write(f"{'─' * 70}\n")
            fh.write(report + "\n\n")
    logger.info("Saved → %s", output_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_report(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    label_names: Sequence[str],
    models: dict | None = None,
    output_dir: str | Path = REPORTS_ROOT,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Train models, evaluate them, and save all plots + reports.

    Parameters
    ----------
    X_train, X_test : already-scaled feature arrays (numpy or compatible)
    y_train, y_test : integer-encoded label arrays
    label_names     : class names in label-encoder order, e.g. ['Benign', 'Syn', 'UDP']
    models          : dict {name: sklearn-compatible estimator}; None → DEFAULT_MODELS
    output_dir      : directory where all outputs are saved
    cv_folds        : number of cross-validation folds

    Returns
    -------
    scores_df : DataFrame with one row per model (also printed + saved as CSV)

    Output files
    ------------
    reports/
        scores_summary.csv
        metrics_comparison.png
        roc_curves.png
        confusion_matrices.png
        cv_scores.png
        classification_reports.txt
    """
    if models is None:
        models = DEFAULT_MODELS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_df, trained = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, cv_folds=cv_folds
    )

    csv_path = output_dir / "scores_summary.csv"
    scores_df.to_csv(csv_path)
    logger.info("Saved → %s", csv_path)

    print("\n" + "=" * 60)
    print("  MODEL SCORES SUMMARY")
    print("=" * 60)
    print(scores_df.to_string())
    print("=" * 60 + "\n")

    plot_metrics_comparison(scores_df, output_dir)
    plot_roc_curves(trained, X_test, y_test, label_names, output_dir)
    plot_confusion_matrices(trained, X_test, y_test, label_names, output_dir)
    plot_cv_scores(scores_df, output_dir)
    save_classification_reports(trained, X_test, y_test, label_names, output_dir)

    logger.info("All outputs saved to: %s/", output_dir)
    return scores_df


# ---------------------------------------------------------------------------
# Standalone runner — falls back to synthetic data if dataset is missing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    try:
        import zipfile
        from io import BytesIO

        DATASET_ZIP = Path("data/cicddos2019-dataset.zip")
        KEEP_LABELS = {"Syn", "UDP", "Benign"}
        MEMBERS     = [
            "Syn-training.parquet", "Syn-testing.parquet",
            "UDP-training.parquet", "UDP-testing.parquet",
        ]

        logger.info("Loading dataset from %s", DATASET_ZIP)
        with zipfile.ZipFile(DATASET_ZIP) as arc:
            train_frames, test_frames = [], []
            for m in MEMBERS:
                with arc.open(m) as f:
                    df = pd.read_parquet(BytesIO(f.read()))
                (train_frames if "training" in m else test_frames).append(df)

        train_df = pd.concat(train_frames, ignore_index=True)
        test_df  = pd.concat(test_frames,  ignore_index=True)

        for df_ in (train_df, test_df):
            df_.drop(df_[~df_["Label"].isin(KEEP_LABELS)].index, inplace=True)

        # Drop low-info columns
        single = [c for c in train_df.columns if train_df[c].nunique() <= 1]
        train_df.drop(columns=single, inplace=True)
        test_df.drop(columns=single, errors="ignore", inplace=True)

        # Drop high-correlation columns
        num_cols = [c for c in train_df.columns if c != "Label"]
        corr  = train_df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_corr = [c for c in upper.columns if any(upper[c] > 0.8)]
        train_df.drop(columns=drop_corr, inplace=True)
        test_df.drop(columns=drop_corr, errors="ignore", inplace=True)
        train_df.drop_duplicates(inplace=True)

        le      = LabelEncoder()
        y_train = le.fit_transform(train_df["Label"])
        y_test  = le.transform(test_df["Label"])
        label_names = list(le.classes_)

        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            train_df.drop(columns=["Label"]), y_train,
            test_size=0.2, random_state=42, stratify=y_train,
        )

        scaler    = MinMaxScaler()
        X_train_s = scaler.fit_transform(X_train_raw)
        X_test_s  = scaler.transform(test_df.drop(columns=["Label"]))

        logger.info("Ready — train=%d  test=%d  classes=%s",
                    len(y_train), len(y_test), label_names)

    except (FileNotFoundError, KeyError) as exc:
        logger.warning("Dataset not available (%s) — running on synthetic data.", exc)
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=6000, n_features=30, n_classes=3,
            n_informative=20, n_redundant=5, random_state=42,
        )
        X_train_s, X_test_s, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42,
        )
        label_names = ["Benign", "Syn", "UDP"]

    build_report(
        X_train_s, X_test_s, y_train, y_test,
        label_names=label_names,
        output_dir=REPORTS_ROOT,
        cv_folds=5,
    )
