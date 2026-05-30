from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.dataset_split import DatasetSplit, split_dataset
from src.load_dataset import load_dataset, preprocess_dataset
from src.paths import DATASET_PATH, MODELS_ROOT, REPORTS_ROOT
from src.schema import DEFAULT_SCHEMA, FeatureSchema
from src.windowing import WindowedDataset, build_windowed_dataset

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train lightweight models for DDoS detection.")
    parser.add_argument("--max-window-size", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--binary-target", action="store_true")
    parser.add_argument("--target-mode", type=str, default="majority")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.7)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--no-shuffle-before-windowing",
        action="store_false",
        dest="shuffle_before_windowing",
        help="Keep original ordering before window aggregation.",
    )
    parser.set_defaults(shuffle_before_windowing=True)
    return parser.parse_args()


def build_models(random_state: int) -> dict[str, object]:
    return {
        "log_reg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def make_pipeline(
    windowed: WindowedDataset,
    estimator: object,
) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                StandardScaler(),
                windowed.numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                windowed.categorical_features,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )


def evaluate_split(
    split: DatasetSplit,
    pipeline: Pipeline,
) -> dict[str, float]:
    pipeline.fit(split.X_train, split.y_train)
    predictions = pipeline.predict(split.X_test)

    return {
        "accuracy": accuracy_score(split.y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(split.y_test, predictions),
        "f1_macro": f1_score(split.y_test, predictions, average="macro"),
        "f1_weighted": f1_score(split.y_test, predictions, average="weighted"),
        "precision_macro": precision_score(
            split.y_test, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            split.y_test, predictions, average="macro", zero_division=0
        ),
    }


def train_for_window(
    frame: pd.DataFrame,
    window_size: int,
    *,
    schema: FeatureSchema,
    target_mode: str,
    binary_target: bool,
    random_state: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> list[dict[str, object]]:
    windowed = build_windowed_dataset(
        frame,
        window_size,
        schema=schema,
        target_mode=target_mode,
        binary_target=binary_target,
    )

    if windowed.data.empty:
        logger.warning("Window size %d produced no data", window_size)
        return []

    split = split_dataset(
        windowed.data,
        target_column=schema.target,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=True,
    )

    results: list[dict[str, object]] = []
    for model_name, estimator in build_models(random_state).items():
        pipeline = make_pipeline(windowed, estimator)
        metrics = evaluate_split(split, pipeline)

        model_path = MODELS_ROOT / f"{model_name}_w{window_size}.joblib"
        joblib.dump(pipeline, model_path)

        result = {
            "window_size": window_size,
            "model": model_name,
            "binary_target": binary_target,
            **metrics,
        }
        results.append(result)

    return results


def write_reports(results: list[dict[str, object]], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(report_path, index=False)
    json_path = report_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    dataset = load_dataset(DATASET_PATH)
    processed = preprocess_dataset(dataset)

    if args.sample_size:
        processed = processed.sample(n=args.sample_size, random_state=args.random_state)

    if args.shuffle_before_windowing:
        processed = processed.sample(frac=1.0, random_state=args.random_state).reset_index(
            drop=True
        )

    schema = FeatureSchema(
        categorical=DEFAULT_SCHEMA.categorical,
        numerical=DEFAULT_SCHEMA.numerical,
        target=DEFAULT_SCHEMA.target,
    )

    results: list[dict[str, object]] = []
    for window_size in range(1, args.max_window_size + 1, args.step):
        logger.info("Training for window size %d", window_size)
        results.extend(
            train_for_window(
                processed,
                window_size,
                schema=schema,
                target_mode=args.target_mode,
                binary_target=args.binary_target,
                random_state=args.random_state,
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
            )
        )

    if results:
        target_suffix = "binary" if args.binary_target else "multiclass"
        report_path = REPORTS_ROOT / f"metrics_{target_suffix}.csv"
        write_reports(results, report_path)
        logger.info("Saved reports to %s", report_path)


if __name__ == "__main__":
    main()
