import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from src.load_dataset import ATTACK_TYPE_ENCODING, build_exclude_columns, load_dataset, preprocess_dataset
from src.paths import ARTIFACTS_ROOT, DATASET_PATH, PROJECT_ROOT

logger = logging.getLogger(__name__)

TARGET_COLUMN = "Attack Type"
DEFAULT_CATEGORICAL_FEATURES = ["Protocol"]
DEFAULT_NUMERICAL_FEATURES = [
    "Packet Size",
    "Payload Length",
    "Flow Duration",
    "Bytes in Flow",
    "Packets in Flow",
    "Average Packet Size",
    "Inter-Arrival Time",
    "Rate of Packets",
    "Unique Source Count",
    "Unique Destination Count",
    "Attack Duration",
]

SUPPORTED_MODELS = ["logreg", "random_forest", "hist_gb"]
PORT_COLUMNS = ["Source Port", "Destination Port"]
TIMESTAMP_COLUMN = "Timestamp"
SOURCE_IP_COLUMN = "Source IP"
DESTINATION_IP_COLUMN = "Destination IP"
AGG_WINDOW_COLUMN = "Window Start"


def _validate_columns(dataset: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def _prepare_features_and_target(
    dataset: pd.DataFrame,
    categorical_features: list[str],
    numerical_features: list[str],
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column in categorical_features or target_column in numerical_features:
        raise ValueError(f"Target column '{target_column}' must not be in feature lists.")

    _validate_columns(dataset, [target_column], "target")
    _validate_columns(dataset, categorical_features, "categorical")
    _validate_columns(dataset, numerical_features, "numerical")

    features = categorical_features + numerical_features
    X = dataset[features].copy()
    y = dataset[target_column].copy()

    if y.isna().any():
        raise ValueError("Target column contains missing values.")

    return X, y


def _build_preprocessor(
    categorical_features: list[str],
    numerical_features: list[str],
    normalize: str,
) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    if normalize == "standard":
        scaler = StandardScaler()
    elif normalize == "minmax":
        scaler = MinMaxScaler()
    elif normalize == "none":
        scaler = "passthrough"
    else:
        raise ValueError("Normalize must be one of: standard, minmax, none.")

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_features),
            ("numerical", numerical_pipeline, numerical_features),
        ]
    )


def _build_estimator(model_name: str, class_weight: str | None) -> object:
    if model_name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=class_weight)
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight=class_weight,
        )
    if model_name == "hist_gb":
        return HistGradientBoostingClassifier(random_state=42)

    raise ValueError(f"Unsupported model. Choose one of: {', '.join(SUPPORTED_MODELS)}.")


def _summarize_class_balance(target: pd.Series) -> dict[str, object]:
    counts = Counter(target)
    total = len(target)
    majority_count = max(counts.values()) if counts else 0
    baseline_accuracy = (majority_count / total) if total else 0.0
    return {
        "counts": dict(sorted(counts.items())),
        "baseline_accuracy": baseline_accuracy,
    }


def _compute_cv_accuracy(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    folds: int,
    random_state: int,
) -> tuple[float, float]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, features, target, cv=splitter, scoring="accuracy", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def _run_diagnostics(
    dataset: pd.DataFrame,
    categorical_features: list[str],
    numerical_features: list[str],
    target_column: str,
) -> None:
    features = categorical_features + numerical_features
    X, y = _prepare_features_and_target(dataset, categorical_features, numerical_features, target_column)

    missing_rate = X.isna().mean().sort_values(ascending=False)
    if missing_rate.gt(0).any():
        logger.info("Missing rate by feature:\n%s", missing_rate[missing_rate.gt(0)].to_string())
    else:
        logger.info("Missing rate by feature: none")

    constant_features = [column for column in features if X[column].nunique(dropna=True) <= 1]
    if constant_features:
        logger.info("Constant features: %s", constant_features)
    else:
        logger.info("Constant features: none")

    if "Protocol" in categorical_features and "Protocol" in dataset.columns:
        protocol_table = pd.crosstab(dataset["Protocol"], y)
        logger.info("Protocol by class counts:\n%s", protocol_table)

    mi_scores: dict[str, float] = {}
    if numerical_features:
        numeric_frame = X[numerical_features].copy()
        numeric_frame = numeric_frame.fillna(numeric_frame.median())
        numeric_mi = mutual_info_classif(
            numeric_frame,
            y,
            discrete_features=False,
            random_state=42,
        )
        mi_scores.update(dict(zip(numerical_features, numeric_mi, strict=False)))

    if categorical_features:
        categorical_frame = X[categorical_features].copy()
        categorical_frame = categorical_frame.fillna(-1)
        categorical_mi = mutual_info_classif(
            categorical_frame,
            y,
            discrete_features=True,
            random_state=42,
        )
        mi_scores.update(dict(zip(categorical_features, categorical_mi, strict=False)))

    if mi_scores:
        mi_series = pd.Series(mi_scores).sort_values(ascending=False)
        logger.info("Mutual information (top 10):\n%s", mi_series.head(10).to_string())


def _add_timestamp_features(dataset: pd.DataFrame) -> list[str]:
    if TIMESTAMP_COLUMN not in dataset.columns:
        return []

    timestamp = pd.to_datetime(dataset[TIMESTAMP_COLUMN], errors="coerce", utc=True)
    dataset["Timestamp Hour"] = timestamp.dt.hour
    dataset["Timestamp DayOfWeek"] = timestamp.dt.dayofweek
    return ["Timestamp Hour", "Timestamp DayOfWeek"]


def _add_engineered_features(dataset: pd.DataFrame) -> list[str]:
    created: list[str] = []

    def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        denominator = denominator.replace(0, pd.NA)
        return numerator / denominator

    if {"Bytes in Flow", "Flow Duration"}.issubset(dataset.columns):
        dataset["Bytes per Second"] = _safe_divide(dataset["Bytes in Flow"], dataset["Flow Duration"])
        created.append("Bytes per Second")
    if {"Packets in Flow", "Flow Duration"}.issubset(dataset.columns):
        dataset["Packets per Second"] = _safe_divide(dataset["Packets in Flow"], dataset["Flow Duration"])
        created.append("Packets per Second")
    if {"Payload Length", "Packet Size"}.issubset(dataset.columns):
        dataset["Payload Ratio"] = _safe_divide(dataset["Payload Length"], dataset["Packet Size"])
        created.append("Payload Ratio")
    if {"Bytes in Flow", "Packets in Flow"}.issubset(dataset.columns):
        dataset["Bytes per Packet"] = _safe_divide(dataset["Bytes in Flow"], dataset["Packets in Flow"])
        created.append("Bytes per Packet")

    for column, feature_name in {
        "Packet Size": "Packet Size Log",
        "Payload Length": "Payload Length Log",
        "Bytes in Flow": "Bytes in Flow Log",
        "Packets in Flow": "Packets in Flow Log",
        "Flow Duration": "Flow Duration Log",
    }.items():
        if column in dataset.columns:
            dataset[feature_name] = np.log1p(dataset[column])
            created.append(feature_name)

    return created


def _build_feature_lists(
    dataset: pd.DataFrame,
    target_column: str,
    include_ports: bool,
    include_timestamp: bool,
    engineer_features: bool,
    use_all_numeric: bool,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    categorical_features = DEFAULT_CATEGORICAL_FEATURES.copy()
    numerical_features = DEFAULT_NUMERICAL_FEATURES.copy()

    if include_ports:
        for column in PORT_COLUMNS:
            if column in dataset.columns and column not in numerical_features:
                numerical_features.append(column)

    if include_timestamp:
        timestamp_features = _add_timestamp_features(dataset)
        for column in timestamp_features:
            if column not in numerical_features:
                numerical_features.append(column)

    if engineer_features:
        engineered = _add_engineered_features(dataset)
        for column in engineered:
            if column not in numerical_features:
                numerical_features.append(column)

    if use_all_numeric:
        numeric_candidates = [
            column
            for column in dataset.columns
            if column not in categorical_features and column != target_column
        ]
        numerical_features = [
            column
            for column in numeric_candidates
            if pd.api.types.is_numeric_dtype(dataset[column])
        ]

    return dataset, categorical_features, numerical_features


def _prepare_target(
    dataset: pd.DataFrame,
    target_mode: str,
) -> tuple[pd.DataFrame, str, list[int], list[str]]:
    if target_mode == "binary":
        target_column = "Is Attack"
        no_attack_value = ATTACK_TYPE_ENCODING.get("No Attack")
        if no_attack_value is None:
            raise ValueError("No Attack label missing from ATTACK_TYPE_ENCODING.")
        dataset[target_column] = (dataset[TARGET_COLUMN] != no_attack_value).astype(int)
        return dataset, target_column, [0, 1], ["No Attack", "Attack"]

    label_order = [value for _, value in sorted(ATTACK_TYPE_ENCODING.items(), key=lambda item: item[1])]
    target_names = [key for key, _ in sorted(ATTACK_TYPE_ENCODING.items(), key=lambda item: item[1])]
    return dataset, TARGET_COLUMN, label_order, target_names


def _aggregate_dataset(
    dataset: pd.DataFrame,
    window_seconds: int,
    group_by: str,
    label_strategy: str,
) -> pd.DataFrame:
    if TIMESTAMP_COLUMN not in dataset.columns:
        raise ValueError("Timestamp column is required for aggregation.")

    if group_by == "src":
        group_columns = [SOURCE_IP_COLUMN]
    elif group_by == "dst":
        group_columns = [DESTINATION_IP_COLUMN]
    else:
        group_columns = [SOURCE_IP_COLUMN, DESTINATION_IP_COLUMN]

    for column in group_columns:
        if column not in dataset.columns:
            raise ValueError(f"Missing required grouping column: {column}")

    timestamp = pd.to_datetime(dataset[TIMESTAMP_COLUMN], errors="coerce", utc=True)
    working = dataset.copy()
    working[AGG_WINDOW_COLUMN] = timestamp.dt.floor(f"{window_seconds}s")
    working = working.dropna(subset=[AGG_WINDOW_COLUMN])

    def _mode(series: pd.Series) -> object:
        if series.empty:
            return pd.NA
        mode_values = series.mode()
        return mode_values.iloc[0] if not mode_values.empty else pd.NA

    numeric_columns = [column for column in DEFAULT_NUMERICAL_FEATURES if column in working.columns]
    aggregation: dict[str, list[str] | str] = {column: ["mean", "sum", "std"] for column in numeric_columns}
    if "Protocol" in working.columns:
        aggregation["Protocol"] = _mode

    for column in PORT_COLUMNS:
        if column in working.columns:
            aggregation[column] = "nunique"

    grouped = working.groupby([AGG_WINDOW_COLUMN, *group_columns], dropna=False)
    aggregated = grouped.agg(aggregation)

    aggregated.columns = [
        " ".join(filter(None, map(str, column))).strip()
        if isinstance(column, tuple)
        else str(column)
        for column in aggregated.columns
    ]
    aggregated = aggregated.reset_index()

    if "Protocol _mode" in aggregated.columns:
        aggregated = aggregated.rename(columns={"Protocol _mode": "Protocol"})

    if label_strategy == "any_attack":
        no_attack_label = "No Attack"
        attack_label = next(
            (label for label in ATTACK_TYPE_ENCODING.keys() if label != no_attack_label),
            no_attack_label,
        )
        grouped_target = grouped[TARGET_COLUMN].apply(
            lambda series: attack_label if (series != no_attack_label).any() else no_attack_label
        )
        aggregated[TARGET_COLUMN] = grouped_target.values
    else:
        aggregated[TARGET_COLUMN] = grouped[TARGET_COLUMN].apply(_mode).values

    aggregated[TIMESTAMP_COLUMN] = aggregated[AGG_WINDOW_COLUMN]
    aggregated = aggregated.drop(columns=[*group_columns, AGG_WINDOW_COLUMN], errors="ignore")
    return aggregated


def train_attack_type_model(
    dataset: pd.DataFrame,
    categorical_features: list[str],
    numerical_features: list[str],
    target_column: str,
    label_order: list[int],
    target_names: list[str],
    model_name: str,
    test_size: float,
    random_state: int,
    normalize: str,
    class_weight: str | None,
    cv_folds: int,
) -> tuple[Pipeline, dict[str, object]]:
    X, y = _prepare_features_and_target(
        dataset,
        categorical_features,
        numerical_features,
        target_column,
    )

    class_count = y.nunique()
    total_samples = len(y)
    min_test_size = class_count / total_samples
    if test_size < min_test_size:
        logger.warning(
            "test_size %.3f is too small for %d classes; using %.3f instead.",
            test_size,
            class_count,
            min_test_size,
        )
        test_size = min_test_size

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = _build_preprocessor(categorical_features, numerical_features, normalize)
    estimator = _build_estimator(model_name, class_weight)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=label_order),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=label_order,
            target_names=target_names,
            zero_division=0,
        ),
    }

    if cv_folds > 1:
        cv_mean, cv_std = _compute_cv_accuracy(pipeline, X, y, cv_folds, random_state)
        metrics["cv_accuracy_mean"] = cv_mean
        metrics["cv_accuracy_std"] = cv_std

    return pipeline, metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple model to classify attack types.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the dataset CSV or zipped CSV.",
    )
    parser.add_argument(
        "--model",
        choices=[*SUPPORTED_MODELS, "all"],
        default="logreg",
        help="Model type to train.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for testing.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save-model",
        type=Path,
        default=ARTIFACTS_ROOT / "attack_type_model.joblib",
        help="Path to save the trained pipeline.",
    )
    parser.add_argument(
        "--no-auto-download",
        action="store_true",
        help="Disable automatic dataset download.",
    )
    parser.add_argument(
        "--normalize",
        choices=["standard", "minmax", "none"],
        default="standard",
        help="Normalization for numerical features.",
    )
    parser.add_argument(
        "--target",
        choices=["attack_type", "binary"],
        default="attack_type",
        help="Target label to predict.",
    )
    parser.add_argument(
        "--include-ports",
        action="store_true",
        help="Include source/destination ports as features.",
    )
    parser.add_argument(
        "--include-timestamp",
        action="store_true",
        help="Derive features from timestamp and include them.",
    )
    parser.add_argument(
        "--engineer-features",
        action="store_true",
        help="Add engineered rate/ratio/log features.",
    )
    parser.add_argument(
        "--approach",
        choices=["custom", "base", "engineered", "ports_timestamp", "binary", "all"],
        default="custom",
        help="Preset feature/target configurations to run.",
    )
    parser.add_argument(
        "--aggregate-window",
        type=int,
        default=0,
        help="Aggregate rows into time windows (seconds); 0 disables aggregation.",
    )
    parser.add_argument(
        "--aggregate-group",
        choices=["src", "dst", "src_dst"],
        default="src",
        help="Group key for aggregation: source, destination, or both.",
    )
    parser.add_argument(
        "--aggregate-label",
        choices=["mode", "any_attack"],
        default="mode",
        help="Label aggregation strategy for attack type within each window.",
    )
    parser.add_argument(
        "--class-weight",
        choices=["balanced", "none"],
        default="balanced",
        help="Class weighting strategy for supported models.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="Number of cross-validation folds to estimate generalization (0 to skip).",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Log feature diagnostics before training.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    dataset = load_dataset(args.dataset, auto_download=not args.no_auto_download)

    if args.aggregate_window > 0:
        dataset = _aggregate_dataset(
            dataset,
            window_seconds=args.aggregate_window,
            group_by=args.aggregate_group,
            label_strategy=args.aggregate_label,
        )

    if args.approach == "all":
        approach_configs = [
            {
                "name": "base-multiclass",
                "target": "attack_type",
                "include_ports": False,
                "include_timestamp": False,
                "engineer_features": False,
            },
            {
                "name": "engineered-multiclass",
                "target": "attack_type",
                "include_ports": False,
                "include_timestamp": False,
                "engineer_features": True,
            },
            {
                "name": "ports-timestamp-multiclass",
                "target": "attack_type",
                "include_ports": True,
                "include_timestamp": True,
                "engineer_features": True,
            },
            {
                "name": "base-binary",
                "target": "binary",
                "include_ports": False,
                "include_timestamp": False,
                "engineer_features": False,
            },
            {
                "name": "engineered-binary",
                "target": "binary",
                "include_ports": False,
                "include_timestamp": False,
                "engineer_features": True,
            },
            {
                "name": "ports-timestamp-binary",
                "target": "binary",
                "include_ports": True,
                "include_timestamp": True,
                "engineer_features": True,
            },
        ]
    elif args.approach == "base":
        approach_configs = [
            {
                "name": "base-multiclass",
                "target": "attack_type",
                "include_ports": False,
                "include_timestamp": False,
                "engineer_features": False,
            }
        ]
    elif args.approach == "engineered":
        approach_configs = [
            {
                "name": "engineered-multiclass",
                "target": "attack_type",
                "include_ports": False,
                "include_timestamp": False,
                "engineer_features": True,
            }
        ]
    elif args.approach == "ports_timestamp":
        approach_configs = [
            {
                "name": "ports-timestamp-multiclass",
                "target": "attack_type",
                "include_ports": True,
                "include_timestamp": True,
                "engineer_features": True,
            }
        ]
    elif args.approach == "binary":
        approach_configs = [
            {
                "name": "base-binary",
                "target": "binary",
                "include_ports": False,
                "include_timestamp": False,
                "engineer_features": False,
            }
        ]
    else:
        approach_configs = [
            {
                "name": "custom",
                "target": args.target,
                "include_ports": args.include_ports,
                "include_timestamp": args.include_timestamp,
                "engineer_features": args.engineer_features,
            }
        ]

    include_ports_any = any(config["include_ports"] for config in approach_configs)
    include_timestamp_any = any(config["include_timestamp"] for config in approach_configs)
    include_ips_any = args.aggregate_window > 0
    preprocess_exclude = build_exclude_columns(
        include_ports=include_ports_any,
        include_timestamp=include_timestamp_any,
        include_ips=include_ips_any,
    )
    dataset = preprocess_dataset(dataset, exclude_columns=preprocess_exclude)

    class_weight = None if args.class_weight == "none" else "balanced"
    models_to_run = SUPPORTED_MODELS if args.model == "all" else [args.model]
    best_pipeline: Pipeline | None = None
    best_metrics: dict[str, object] | None = None
    best_model_name = None
    best_approach = None

    use_all_numeric = args.aggregate_window > 0
    for approach in approach_configs:
        logger.info("Approach: %s", approach["name"])
        working = dataset.copy()
        working, target_column, label_order, target_names = _prepare_target(
            working,
            approach["target"],
        )
        working, categorical_features, numerical_features = _build_feature_lists(
            working,
            target_column=target_column,
            include_ports=approach["include_ports"],
            include_timestamp=approach["include_timestamp"],
            engineer_features=approach["engineer_features"],
            use_all_numeric=use_all_numeric,
        )
        X, y = _prepare_features_and_target(
            working,
            categorical_features,
            numerical_features,
            target_column,
        )
        working = pd.concat([X, y], axis=1)

        if args.diagnostics:
            _run_diagnostics(
                working,
                categorical_features,
                numerical_features,
                target_column,
            )

        class_summary = _summarize_class_balance(y)
        logger.info("Class counts: %s", class_summary["counts"])
        logger.info("Baseline (majority-class) accuracy: %.4f", class_summary["baseline_accuracy"])

        for model_name in models_to_run:
            pipeline, metrics = train_attack_type_model(
                working,
                categorical_features,
                numerical_features,
                target_column,
                label_order,
                target_names,
                model_name,
                args.test_size,
                args.random_state,
                args.normalize,
                class_weight,
                args.cv_folds,
            )

            logger.info("Model: %s", model_name)
            logger.info("Accuracy: %.4f", metrics["accuracy"])
            logger.info("Balanced Accuracy: %.4f", metrics["balanced_accuracy"])
            if "cv_accuracy_mean" in metrics:
                logger.info(
                    "CV Accuracy: %.4f +/- %.4f",
                    metrics["cv_accuracy_mean"],
                    metrics["cv_accuracy_std"],
                )
            logger.info("Confusion Matrix:\n%s", metrics["confusion_matrix"])
            logger.info("Classification Report:\n%s", metrics["classification_report"])

            if best_metrics is None or metrics["accuracy"] > best_metrics["accuracy"]:
                best_pipeline = pipeline
                best_metrics = metrics
                best_model_name = model_name
                best_approach = approach["name"]

    if best_pipeline is None:
        raise RuntimeError("No model was trained. Check model selection and inputs.")

    args.save_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, args.save_model)
    relative_save_path = args.save_model.resolve().relative_to(PROJECT_ROOT)
    logger.info("Saved best model (%s, %s) to %s", best_model_name, best_approach, relative_save_path)


if __name__ == "__main__":
    main()

