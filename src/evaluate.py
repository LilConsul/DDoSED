from collections.abc import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

CLASS_LABELS = ("normal", "syn_flood", "udp_flood")


def summarize_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "labels": list(CLASS_LABELS),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=CLASS_LABELS).tolist(),
    }
