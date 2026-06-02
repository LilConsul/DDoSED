"""
evaluation.py — Model training & evaluation report for SYN/UDP DDoS detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

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
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from paths import REPORTS_ROOT
from preprocess import preprocess_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_MODELS: Dict[str, object] = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "KNN": KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
    "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
        use_label_encoder=False,
    ),
}


def train_and_evaluate(
    models: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int = 5,
):
    rows, trained = [], {}
    for name, model in models.items():
        logger.info("Training %s ...", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        try:
            proba = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, proba, multi_class="ovr")
        except Exception:
            roc_auc = 0.0

        cv_score = float(
            np.mean(cross_val_score(model, X_train, y_train, cv=cv_folds, n_jobs=-1))
        )

        rows.append(
            {
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4),
                "ROC AUC": round(roc_auc, 4),
                f"CV ({cv_folds}-fold)": round(cv_score, 4),
            }
        )
        trained[name] = model

    return pd.DataFrame(rows).set_index("Model"), trained


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_metrics_comparison(scores_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    data = scores_df[metrics]
    x = np.arange(len(data))
    width = 0.15
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
    n_classes = len(label_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 5), sharey=True)
    if n_classes == 1:
        axes = [axes]
    colors = cm.tab10(np.linspace(0, 0.9, len(trained)))
    for ax, class_idx, class_name in zip(axes, range(n_classes), label_names):
        for (name, model), color in zip(trained.items(), colors):
            proba = model.predict_proba(X_test)[:, class_idx]
            y_bin = (y_test == class_idx).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_bin, proba)
            auc = roc_auc_score(y_bin, proba)
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
    n = len(trained)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).flatten()
    for ax, (name, model) in zip(axes, trained.items()):
        cm_arr = confusion_matrix(y_test, model.predict(X_test), normalize="true")
        disp = ConfusionMatrixDisplay(cm_arr, display_labels=label_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format=".2f")
        ax.set_title(name, fontsize=10)
    for ax in axes[len(trained) :]:
        ax.set_visible(False)
    fig.suptitle("Normalised Confusion Matrices", fontsize=13)
    fig.tight_layout()
    _save(fig, output_dir / "confusion_matrices.png")


def plot_cv_scores(scores_df: pd.DataFrame, output_dir: Path) -> None:
    cv_col = [c for c in scores_df.columns if c.startswith("CV")][0]
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = cm.viridis(np.linspace(0.2, 0.8, len(scores_df)))
    bars = ax.bar(scores_df.index, scores_df[cv_col], color=colors, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(scores_df[cv_col].min() - 0.02, 1.01)
    ax.set_ylabel("CV Score")
    ax.set_title(cv_col)
    ax.set_xticks(range(len(scores_df)))
    ax.set_xticklabels(scores_df.index, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, output_dir / "cv_scores.png")


def save_classification_reports(
    trained: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: Sequence[str],
    output_dir: Path,
) -> None:
    output_path = output_dir / "classification_reports.txt"
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(
            "=" * 70
            + "\n  CLASSIFICATION REPORTS — SYN / UDP DDoS Detection\n"
            + "=" * 70
            + "\n\n"
        )
        for name, model in trained.items():
            y_pred = model.predict(X_test)
            report = classification_report(
                y_test, y_pred, target_names=label_names, digits=4
            )
            fh.write(f"{'─' * 70}\n  {name}\n{'─' * 70}\n{report}\n\n")


def build_report(
    X_train,
    X_test,
    y_train,
    y_test,
    label_names,
    models=None,
    output_dir=REPORTS_ROOT,
    cv_folds=5,
):
    if models is None:
        models = DEFAULT_MODELS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_df, trained = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, cv_folds
    )
    scores_df.to_csv(output_dir / "scores_summary.csv")

    print("\n" + "=" * 60 + "\n  MODEL SCORES SUMMARY\n" + "=" * 60)
    print(scores_df.to_string())

    plot_metrics_comparison(scores_df, output_dir)
    plot_roc_curves(trained, X_test, y_test, label_names, output_dir)
    plot_confusion_matrices(trained, X_test, y_test, label_names, output_dir)
    plot_cv_scores(scores_df, output_dir)
    save_classification_reports(trained, X_test, y_test, label_names, output_dir)

    return scores_df


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, labels, _ = preprocess_data()

    build_report(X_tr, X_te, y_tr, y_te, label_names=labels, cv_folds=5)
