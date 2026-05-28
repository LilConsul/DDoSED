import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.artifacts import save_json_report, save_model_artifact
from src.evaluate import summarize_classification_metrics
from src.features import build_feature_transformer
from src.paths import ARTIFACTS_ROOT, REPORTS_ROOT
from src.schema import TARGET_COLUMN


def build_model_registry() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "random_forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=10,
        ),
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=3),
    }


def train_and_compare_models(frame: pd.DataFrame, feature_set: str) -> dict[str, object]:
    features = frame.drop(columns=[TARGET_COLUMN])
    target = frame[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.33,
        random_state=42,
        stratify=target,
    )

    ranked_models: list[dict[str, object]] = []

    for model_name, estimator in build_model_registry().items():
        pipeline = Pipeline(
            steps=[
                ("transformer", build_feature_transformer(feature_set)),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics = summarize_classification_metrics(y_test, predictions)
        ranked_models.append(
            {
                "model_name": model_name,
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
            }
        )

    ranked_models.sort(key=lambda item: item["macro_f1"], reverse=True)
    return {
        "best_model_name": ranked_models[0]["model_name"],
        "ranked_models": ranked_models,
    }


def save_training_outputs(best_pipeline: object, comparison: dict[str, object]) -> None:
    save_model_artifact(ARTIFACTS_ROOT / "best_model.joblib", best_pipeline)
    save_json_report(REPORTS_ROOT / "model_comparison.json", comparison)
