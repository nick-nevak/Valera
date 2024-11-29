import numpy as np

from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
)
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from category_encoders import TargetEncoder


from utils.classes import DataFrameConverter, KNNModeImputer
from utils.constants import (
    ecology_column,
    sub_area_column,
    ecology_column,
)
from utils.constants import (
    ecology_column,
    sub_area_column,
    ecology_column,
    binary_categories_columns,
)


def create_pipeline(
    all_columns,
    numerical_columns,
    include_scaling=False,
    include_knn_imputation=False,
    include_ecology_imputation=False,
):
    steps = [
        (
            "encoding",
            ColumnTransformer(
                [
                    (
                        "ecology_ordinal",
                        OrdinalEncoder(
                            categories=[["poor", "satisfactory", "good", "excellent"]],
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),
                        [ecology_column],
                    ),
                    (
                        "binary_ohe",
                        OneHotEncoder(sparse_output=False, drop="if_binary"),
                        binary_categories_columns,
                    ),
                    ("sub_area_target", TargetEncoder(), [sub_area_column]),
                    ("numerical", "passthrough", numerical_columns),
                ]
            ),
        )
    ]

    # Conditionally add steps
    if include_scaling:
        steps.extend(
            [
                ("scaling", StandardScaler()),
                ("to_dataframe_after_scaling", DataFrameConverter(columns=all_columns)),
            ]
        )

    if include_knn_imputation:
        steps.extend(
            [
                (
                    "knn_imputation",
                    ColumnTransformer(
                        [
                            ("ecology", "passthrough", [ecology_column]),
                            (
                                "knn_imputer",
                                KNNImputer(n_neighbors=3),
                                [col for col in all_columns if col != ecology_column],
                            ),
                        ]
                    ),
                ),
                (
                    "to_dataframe_after_knn_imputation",
                    DataFrameConverter(columns=all_columns),
                ),
            ]
        )

    if include_ecology_imputation:
        steps.append(
            (
                "ecology_imputation",
                ColumnTransformer(
                    transformers=[
                        (
                            "knn_mode_imputer",
                            KNNModeImputer(n_neighbors=3),
                            [ecology_column],
                        )
                    ],
                    remainder="passthrough",
                ),
            )
        )

    # Return the pipeline
    return Pipeline(steps)
