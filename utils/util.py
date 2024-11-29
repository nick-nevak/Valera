import os

import pandas as pd
import numpy as np

import joblib

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score


# Кластеры для районов
# Таргет трайнуть


def check():
    print("ZVC2345")


def color_cells(val):
    """val: значение признака"""

    if val == "float64":
        color = "red"
    elif val == "int64":
        color = "red"
    else:
        color = "blue"
    return f"color: {color}"


def valeraInfo(df):
    info = pd.DataFrame()
    info.index = df.columns
    info["Тип данных"] = df.dtypes
    info["Количесвто уникальных"] = df.nunique()
    info["Количество пропусков"] = df.isna().sum()
    info["Количество значений"] = df.count()
    info["%значений"] = round((df.count() / df.shape[0]) * 100, 2)
    info = info.style.applymap(color_cells, subset=["Тип данных"])
    return info


def load_data(path, files_names):
    return [
        pd.read_csv(os.path.join(path, file), parse_dates=["timestamp"])
        for file in files_names
    ]


def dump_ZV(content, file_name, folder_path="dumps"):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f"{file_name}.joblib")
    joblib.dump(content, file_path)
    print(f"Saved {file_name} to {file_path}")


def undump_ZV(file_name, folder_path="dumps"):
    file_path = os.path.join(folder_path, f"{file_name}.joblib")
    return joblib.load(file_path)


def get_cols_containing(df, features):
    # Initialize an empty list to store columns
    count_columns = []
    # Iterate over the features to process and collect matching columns
    for feature in features:
        count_columns.extend([col for col in df.columns if feature in col])
    return df[count_columns]


def get_columns_by_type(df):
    return (
        df.select_dtypes(include=["number"]).columns.tolist(),
        df.select_dtypes(include=["object", "category"]).columns.tolist(),
    )


def measure_performace(model, X_train, y_train, X_test, y_test):
    # Cross-validated predictions
    cv_predictions = cross_val_predict(model, X_train, y_train, cv=5)

    # Calculate cross-validated metrics
    cv_mse = mean_squared_error(y_train, cv_predictions)
    cv_rmse = np.sqrt(cv_mse)
    cv_r2 = r2_score(y_train, cv_predictions)

    # Calculate LRMSE for cross-validation
    cv_lrmse = np.sqrt(mean_squared_error(np.log1p(y_train), np.log1p(cv_predictions)))

    # Train the model on the full training data
    model.fit(X_train, y_train)

    # Make predictions on the test data
    test_predictions = model.predict(X_test)

    # Calculate test metrics
    test_mse = mean_squared_error(y_test, test_predictions)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, test_predictions)

    # Calculate LRMSE for the test set
    test_lrmse = np.sqrt(
        mean_squared_error(np.log1p(y_test), np.log1p(test_predictions))
    )

    # Calculate Mean Squared Logarithmic Error
    test_msle = mean_squared_log_error(y_test, test_predictions)

    print(f"Cross-Validated RMSE: {cv_rmse}")
    print(f"Cross-Validated R^2 Score: {cv_r2}")
    print(f"Cross-Validated LRMSE: {cv_lrmse}\n")

    print(f"Test RMSE: {test_rmse}")
    print(f"Test R^2 Score: {test_r2}")
    print(f"Test LRMSE: {test_lrmse}\n")

    print(f"Test MSLE: {test_msle}\n")


def feature_importances_to_df(feature_columns, feature_importances):
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_columns, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)

    feature_importance_df.index = range(1, len(feature_importance_df) + 1)
    feature_importance_df.index.name = "Rank"

    print(feature_importance_df)

    return feature_importance_df


def get_feature_to_keep(fi_df, threshold):
    features_all = fi_df["Feature"].tolist()
    features_to_keep = fi_df[fi_df["Importance"] >= threshold]["Feature"].tolist()

    features_below_threshold = fi_df[fi_df["Importance"] < threshold]
    features_above_threshold = fi_df[fi_df["Importance"] >= threshold]

    features_to_keep = features_above_threshold["Feature"].tolist()
    features_to_drop = features_below_threshold["Feature"].tolist()

    print(f"Keep: {len(features_to_keep)}/{len(features_all)} features")
    print(f"Drop: {len(features_to_drop)}/{len(features_all)} features")
    print(f"Keep list:\n {features_to_keep}")

    return features_to_keep


def reduce_features(X, all_features, reduced_features):
    reduced_features_ordered = [col for col in all_features if col in reduced_features]
    X_df = pd.DataFrame(X, columns=all_features)
    X_reduced_df = X_df[reduced_features_ordered]
    X_reduced = X_reduced_df.to_numpy()

    # Report the dimensionality reduction
    original_shape = X.shape
    reduced_shape = X_reduced.shape
    print(
        f"Reduced dimensionality: {original_shape[1]} → {reduced_shape[1]} (Rows: {reduced_shape[0]})"
    )
    return (X_reduced, reduced_features_ordered)
