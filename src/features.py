from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.schema import FULL_FEATURE_COLUMNS, LIGHTWEIGHT_FEATURE_COLUMNS

CATEGORICAL_COLUMNS = ("Protocol",)
NUMERIC_COLUMNS = tuple(
    column for column in FULL_FEATURE_COLUMNS if column not in CATEGORICAL_COLUMNS
)


def build_feature_transformer(feature_set: str) -> ColumnTransformer:
    if feature_set == "full":
        selected_columns = FULL_FEATURE_COLUMNS
    elif feature_set == "lightweight":
        selected_columns = LIGHTWEIGHT_FEATURE_COLUMNS
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")

    categorical_columns = [column for column in selected_columns if column in CATEGORICAL_COLUMNS]
    numeric_columns = [column for column in selected_columns if column not in CATEGORICAL_COLUMNS]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_columns),
            ("numeric", numeric_pipeline, numeric_columns),
        ]
    )
