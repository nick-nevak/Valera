import numpy as np
from utils.constants import (
    features_to_remove_outliers_dynamically,
    target_column,
)


def fix_logical_errors_macro(df):
    df = df.copy()
    df.loc[df["rent_price_2room_eco"] < 37.5, "rent_price_2room_eco"] = np.nan
    df.loc[df["rent_price_1room_eco"] < 30, "rent_price_1room_eco"] = np.nan
    df.loc[df["rent_price_3room_bus"] > 115, "rent_price_3room_bus"] = np.nan
    df.loc[df["rent_price_4+room_bus"] < 130, "rent_price_4+room_bus"] = np.nan
    df.loc[
        (df["income_per_cap"] < 35e3) | (df["income_per_cap"] > 65e3), "income_per_cap"
    ] = np.nan
    return df


def fix_logical_errors(df):
    df = df.copy()
    df.loc[df["life_sq"] >= df["full_sq"], "life_sq"] = np.nan
    df.loc[df["kitch_sq"] >= df["full_sq"], "kitch_sq"] = np.nan
    df.loc[df["floor"] > df["max_floor"], "floor"] = np.nan
    df.loc[df["num_room"] > 10, "num_room"] = np.nan
    df.loc[df["build_year"] < 1900, "build_year"] = np.nan

    df.loc[df["build_count_block"] > 100, "build_count_block"] = np.nan
    df.loc[df["build_count_wood"] > 100, "build_count_wood"] = np.nan
    df.loc[df["build_count_frame"] > 100, "build_count_frame"] = np.nan
    df.loc[df["build_count_brick"] > 100, "build_count_brick"] = np.nan
    df.loc[df["build_count_monolith"] > 100, "build_count_monolith"] = np.nan
    df.loc[df["build_count_panel"] > 100, "build_count_panel"] = np.nan
    df.loc[df["build_count_foam"] > 100, "build_count_foam"] = np.nan
    df.loc[df["build_count_slag"] > 100, "build_count_slag"] = np.nan
    df.loc[df["build_count_mix"] > 100, "build_count_mix"] = np.nan
    df.loc[df["build_count_before_1920"] > 100, "build_count_before_1920"] = np.nan
    df.loc[df["build_count_1921-1945"] > 100, "build_count_1921-1945"] = np.nan
    df.loc[df["build_count_1946-1970"] > 100, "build_count_1946-1970"] = np.nan
    df.loc[df["build_count_1971-1995"] > 100, "build_count_1971-1995"] = np.nan
    df.loc[df["build_count_after_1995"] > 100, "build_count_after_1995"] = np.nan

    df.loc[df["young_all"] > df["full_all"], "young_all"] = np.nan
    df.loc[df["young_male"] > df["full_all"], "young_male"] = np.nan
    df.loc[df["young_female"] > df["full_all"], "young_female"] = np.nan
    df.loc[df["work_all"] > df["full_all"], "work_all"] = np.nan
    df.loc[df["work_male"] > df["full_all"], "work_male"] = np.nan
    df.loc[df["work_female"] > df["full_all"], "work_female"] = np.nan
    df.loc[df["ekder_all"] > df["full_all"], "ekder_all"] = np.nan
    df.loc[df["ekder_male"] > df["full_all"], "ekder_male"] = np.nan
    df.loc[df["ekder_female"] > df["full_all"], "ekder_female"] = np.nan
    df.loc[df["0_6_all"] > df["full_all"], "0_6_all"] = np.nan
    df.loc[df["7_14_all"] > df["full_all"], "7_14_all"] = np.nan
    df.loc[df["0_17_all"] > df["full_all"], "0_17_all"] = np.nan
    df.loc[df["0_17_male"] > df["full_all"], "0_17_male"] = np.nan
    df.loc[df["0_17_female"] > df["full_all"], "0_17_female"] = np.nan
    df.loc[df["0_13_all"] > df["full_all"], "0_13_all"] = np.nan
    df.loc[df["0_13_male"] > df["full_all"], "0_13_male"] = np.nan
    df.loc[df["0_13_female"] > df["full_all"], "0_13_female"] = np.nan

    df.loc[df["children_preschool"] > df["full_all"], "children_preschool"] = np.nan
    df.loc[df["children_school"] > df["full_all"], "children_school"] = np.nan
    df.loc[df["0_6_male"] > df["male_f"], "0_6_male"] = np.nan
    df.loc[df["0_6_female"] > df["female_f"], "0_6_female"] = np.nan
    df.loc[df["7_14_male"] > df["male_f"], "7_14_male"] = np.nan
    df.loc[df["7_14_female"] > df["female_f"], "7_14_female"] = np.nan
    df.loc[df["0_17_male"] > df["male_f"], "0_17_male"] = np.nan
    df.loc[df["0_17_female"] > df["female_f"], "0_17_female"] = np.nan
    df.loc[df["0_13_male"] > df["male_f"], "0_13_male"] = np.nan
    df.loc[df["0_13_female"] > df["female_f"], "0_13_female"] = np.nan
    df.loc[df["ekder_female"] > df["female_f"], "ekder_female"] = np.nan
    df.loc[df["work_male"] > df["male_f"], "work_male"] = np.nan
    df.loc[df["work_female"] > df["female_f"], "work_female"] = np.nan
    df.loc[df["young_male"] > df["male_f"], "young_male"] = np.nan
    df.loc[df["young_female"] > df["female_f"], "young_female"] = np.nan

    df.loc[df["raion_popul"] > df["full_all"], "raion_popul"] = np.nan

    df.loc[df["office_count_500"] > df["office_sqm_500"], "office_sqm_500"] = np.nan
    df.loc[df["trc_count_500"] > df["trc_sqm_500"], "trc_sqm_500"] = np.nan
    df.loc[df["office_count_1000"] > df["office_sqm_1000"], "office_sqm_1000"] = np.nan
    df.loc[df["trc_count_1000"] > df["trc_sqm_1000"], "trc_sqm_1000"] = np.nan
    df.loc[df["office_count_1500"] > df["office_sqm_1500"], "office_sqm_1500"] = np.nan
    df.loc[df["trc_count_1500"] > df["trc_sqm_1500"], "trc_sqm_1500"] = np.nan
    df.loc[df["office_count_2000"] > df["office_sqm_2000"], "office_sqm_2000"] = np.nan
    df.loc[df["trc_count_2000"] > df["trc_sqm_2000"], "trc_sqm_2000"] = np.nan

    distances = [500, 1000, 1500, 2000, 3000, 5000]
    for distance in distances:
        df.loc[
            df[f"big_church_count_{distance}"] > df[f"church_count_{distance}"],
            f"big_church_count_{distance}",
        ] = np.nan

    return df


def remove_outliers(df):
    outliers_removed_df = (
        df.pipe(remove_outliers_explicit, target_column, lower=3e6, upper=20e6)
        # .pipe(remove_outliers_explicit, "num_room", upper=5)
        .pipe(remove_outliers_explicit, "full_sq", lower=28, upper=100)
        # .pipe(remove_outliers_iqr, "life_sq")
        # .pipe(remove_outliers_iqr, "kitch_sq")
        # .pipe(remove_outliers_iqr, "build_year")
        # .pipe(remove_outliers_explicit, "kremlin_km", upper=30)
        # .pipe(remove_outliers_explicit, "area_m", upper=0.8e8)
        # .pipe(remove_outliers_iqr, "max_floor")
        # .pipe(remove_outliers_iqr, "floor")
        # .pipe(remove_outliers_iqr, "full_all")
        # .pipe(remove_outliers_percentile, features_to_remove_outliers_dynamically)
    )

    print(f"Original data size: {df.shape[0]} rows")
    print(f"Filtered data size: {outliers_removed_df.shape[0]} rows")

    return outliers_removed_df


def count_values_above(data, col_name, threshold):
    count_below_threshold = (data[col_name] > threshold).sum()

    print(f"Col_name: {col_name}")
    print(f"Number of rows > {threshold}: {count_below_threshold}")


def count_values_below(data, col_name, threshold):
    count_below_threshold = (data[col_name] < threshold).sum()

    print(f"Col_name: {col_name}")
    print(f"Number of rows < {threshold}: {count_below_threshold}")


def count_values_above_percentile(data, col_name, percentile=99):
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
