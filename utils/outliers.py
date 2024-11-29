import numpy as np
from utils.constants import (
    features_to_remove_outliers_dynamically,
    target_column,
)


def remove_outliers(df):
    outliers_removed_df = (
        df.pipe(remove_outliers_explicit, target_column, lower=2e6, upper=14e6)
        .pipe(remove_outliers_explicit, "num_room", upper=5)
        .pipe(remove_outliers_explicit, "full_sq", lower=25, upper=100)
        .pipe(remove_outliers_iqr, "life_sq")
        .pipe(remove_outliers_iqr, "kitch_sq")
        .pipe(remove_outliers_iqr, "build_year")
        .pipe(remove_outliers_explicit, "kremlin_km", upper=30)
        .pipe(remove_outliers_explicit, "area_m", upper=0.8e8)
        .pipe(remove_outliers_iqr, "max_floor")
        .pipe(remove_outliers_iqr, "floor")
        .pipe(remove_outliers_iqr, "full_all")
        .pipe(remove_outliers_percentile, features_to_remove_outliers_dynamically)
    )

    print(f"Original data size: {df.shape[0]} rows")
    print(f"Filtered data size: {outliers_removed_df.shape[0]} rows")

    return outliers_removed_df


def count_values_below_threshold(data, col_name, threshold):
    count_below_threshold = (data[col_name] < threshold).sum()

    print(f"Col_name: {col_name}")
    print(f"Number of rows < {threshold}: {count_below_threshold}")


def count_values_above_threshold(data, col_name, percentile=99):
    skewness = data[col_name].skew()
    threshold = np.percentile(data[col_name].dropna(), percentile)
    count_above_threshold = (data[col_name] > threshold).sum()
    na_count = data[col_name].isna().sum()

    print(f"Col_name: {col_name}")
    print(f"Skewness: {skewness}")
    print(f"Upper_bound: {threshold}")
    print(f"Dataset total rows: {len(data)}")
    print(f"Number of rows > {threshold}: {count_above_threshold}")
    print(f"Number of NaN: {na_count}")


def remove_outliers_explicit(data, col_name, lower=None, upper=None):
    data_copy = data.copy()

    # Retain NaN values
    mask = data_copy[col_name].notna()

    # Apply the lower bound if specified
    if lower is not None:
        mask &= data_copy[col_name] >= lower

    # Apply the upper bound if specified
    if upper is not None:
        mask &= data_copy[col_name] <= upper

    # Apply the mask, but retain rows with NaN values
    filtered_data = data_copy[mask | data_copy[col_name].isna()]

    # Print summary
    removed_rows = data.shape[0] - filtered_data.shape[0]
    print(f"{col_name}: {removed_rows} rows")

    return filtered_data


def remove_outliers_iqr(data, column, custom_upper_bound=None):
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR if custom_upper_bound is None else custom_upper_bound

    # Filter rows: Keep rows within bounds or where the value is NaN
    filtered_data = data[
        (data[column].isna())
        | ((data[column] >= lower_bound) & (data[column] <= upper_bound))
    ]

    # Print summary
    print(f"{column}: {data.shape[0] - filtered_data.shape[0]} rows")

    return filtered_data


def remove_outliers_percentile(
    data, features, skewness_threshold=2, upper_percentile=99
):
    data_copy = data.copy()

    for col in features:
        skewness = data_copy[col].skew()

        if skewness >= skewness_threshold:
            lower_bound = np.percentile(data_copy[col].dropna(), 0)
            upper_bound = np.percentile(data_copy[col].dropna(), upper_percentile)

            # Apply filtering
            data_copy = data_copy[
                ((data_copy[col] >= lower_bound) & (data_copy[col] <= upper_bound))
                | (data_copy[col].isna())
            ]

            # Print summary
            removed_rows = data.shape[0] - data_copy.shape[0]
            print(
                f"Processing '{col}': Skewness = {skewness:.2f}, Lower Bound: {
                  lower_bound}, Upper Bound: {upper_bound}"
            )
            print(f"Number of rows removed: {removed_rows} rows")
            print("")
        else:
            print(f"Processing '{col}': Skewness = {skewness:.2f}")
            print(f"Skip")
            print("")

    return data_copy
