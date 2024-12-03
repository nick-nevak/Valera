import os

import pandas as pd
import numpy as np

import joblib

from sklearn.cluster import DBSCAN
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    davies_bouldin_score,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
    silhouette_score,
)


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


def feature_scores_to_df(feature_columns, model, importance_type="gain"):
    feature_scores = model.get_booster().get_score(importance_type=importance_type)
    feature_scores = {
        k: v / sum(feature_scores.values()) for k, v in feature_scores.items()
    }
    # Map 'f0', 'f1', ... keys to actual feature names
    mapped_scores = {
        feature_columns[int(key[1:])]: value for key, value in feature_scores.items()
    }

    # Convert to DataFrame, sort by score, and add rank
    feature_score_df = pd.DataFrame(
        list(mapped_scores.items()), columns=["Feature", importance_type]
    ).sort_values(by=importance_type, ascending=False)
    feature_score_df.index = range(1, len(feature_score_df) + 1)
    feature_score_df.index.name = "Rank"

    return feature_score_df


def feature_importances_to_df(feature_columns, feature_importances):
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_columns, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)

    feature_importance_df.index = range(1, len(feature_importance_df) + 1)
    feature_importance_df.index.name = "Rank"

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


def get_outlies_DBSCAN(X, eps, min_samples, measure_scores=False):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = dbscan.fit_predict(X)
    outlier_indices = np.where(labels == -1)[0]
    print(f"Outliers detected: {len(outlier_indices)}")

    if measure_scores:
        non_outlier_mask = labels != -1
        silhouette = silhouette_score(X[non_outlier_mask], labels[non_outlier_mask])
        davies_bouldin = davies_bouldin_score(
            X[non_outlier_mask], labels[non_outlier_mask]
        )
        print(f"Silhouette Score: {silhouette:.2f}")
        print(f"Davies-Bouldin Index: {davies_bouldin:.2f}")

    return labels, outlier_indices


# silhouette_score
# 1.0 (Perfect Score): Indicates clusters are perfectly separated and all points are close to their cluster center. This is rare in real-world datasets.
# 0.7–1.0 (Excellent): Very well-defined clusters with strong separation.
# 0.5–0.7 (Good): Reasonably well-defined clusters with some overlap or noise.
# 0.25–0.5 (Moderate): Clusters may overlap, or the dataset has noise or complex shapes.
# 0.0–0.25 (Weak): Poor clustering; clusters are indistinct or overlap heavily.
# < 0.0 (Negative): Indicates that many points are closer to a different cluster than their assigned cluster. This suggests poor clustering or incorrect parameter choices.


# davies_bouldin_score
# < 0.5: Excellent clustering with compact and well-separated clusters.
# 0.5–1.0: Decent clustering; may include some overlap or less compact clusters.
# > 1.0: Poor clustering, often indicating significant overlap between clusters or very dispersed clusters.
def search_params_DBSCAN(X, eps_values, min_samples_values):
    # Track the best parameters and scores
    best_eps = None
    best_min_samples = None
    best_silhouette_score = -1
    best_davies_bouldin_score = float("inf")  # Lower is better for Davies-Bouldin
    results = []  # Store results for analysis

    # Grid search for the best combination of eps and min_samples
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            labels = dbscan.fit_predict(X)

            # Count outliers
            n_outliers = np.sum(labels == -1)

            # Exclude outliers for Silhouette Score and Davies-Bouldin Index
            if len(set(labels[labels != -1])) > 1:  # Avoid invalid clustering scores
                silhouette = silhouette_score(X[labels != -1], labels[labels != -1])
                davies_bouldin = davies_bouldin_score(
                    X[labels != -1], labels[labels != -1]
                )
            else:
                silhouette = -1  # Invalid clustering
                davies_bouldin = float("inf")  # Invalid clustering

            # Track results
            results.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "silhouette_score": silhouette,
                    "davies_bouldin_score": davies_bouldin,
                    "n_outliers": n_outliers,
                }
            )

            # Update best parameters based on Silhouette Score
            if silhouette > best_silhouette_score:
                best_silhouette_score = silhouette
                best_eps = eps
                best_min_samples = min_samples
                best_davies_bouldin_score = davies_bouldin

    # Print the best combination
    print(f"Best Silhouette Score: {best_silhouette_score:.2f}")
    print(f"Best Davies-Bouldin Index: {best_davies_bouldin_score:.2f}")
    print(f"Best eps: {best_eps}, Best min_samples: {best_min_samples}")

    return pd.DataFrame(results)
