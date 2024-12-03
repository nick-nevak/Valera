import numpy as np
import pandas as pd

from utils.constants import date_column


import numpy as np
import pandas as pd


def apply_feature_engineering(df, target):
    return (
        df
        # .pipe(create_accessibility_and_adequacy_features)
        # .pipe(process_and_drop_cafe_price_columns)
        # .pipe(aggregate_affordability_accessibility)
        # .pipe(process_distance_features, feature_distance_weights)
        # .pipe(calculate_weighted_build_count, construction_weights)
        # .pipe(drop_redundant_columns, "build_count", construction_weights)
        # .pipe(calculate_weighted_material_score, material_weights)
        # .pipe(drop_redundant_columns, "build_count", material_weights)
        # .pipe(weighted_average_aggregation, "green_part", green_weights)
        # .pipe(weighted_average_aggregation, "prom_part", prom_weights)
        # .pipe(drop_redundant_columns, "green_part", green_weights)
        # .pipe(drop_redundant_columns, "prom_part", prom_weights)
        # .pipe(process_demographic_features)
        # .pipe(improve_top_features, target)
        # .pipe(fi_experimental)
        # .pipe(drop_initial_features)
        # .pipe(drop_testing)
        .pipe(process_timestamp)
        # .pipe(drop_cafe_columns)
        # .pipe(drop_weighted_columns)
    )


def merge_with_macro(transactions_df, macro_df):
    columns_to_clean = [
        "child_on_acc_pre_school",
        "modern_education_share",
        "old_education_build_share",
    ]

    for col in columns_to_clean:
        macro_df[col] = (
            macro_df[col]
            .replace(
                "#!", None
            )  # Replace '#!' with None (Python's native missing value)
            .str.replace(
                ",", ".", regex=True
            )  # Replace commas with periods for decimal handling
            .astype(float)  # Convert to numeric, None is treated as NaN
        )

    macro_most_important_features = ["timestamp"] + [
        "cpi",
        "micex_rgbi_tr",
        "rent_price_3room_bus",
        "rent_price_4+room_bus",
        "rts",
        "ppi",
        "rent_price_2room_bus",
        "mortgage_value",
        "deposits_rate",
        "mortgage_rate",
        "usdrub",
        "balance_trade",
        "oil_urals",
        "brent",
        "gdp_quart_growth",
        "average_provision_of_build_contract_moscow",
        "net_capital_export",
        "rent_price_1room_bus",
        "grp",
        "average_provision_of_build_contract",
        "deposits_growth",
        "income_per_cap",
        "micex",
        "balance_trade_growth",
        "eurrub",
        "rent_price_2room_eco",
        "rent_price_1room_eco",
        "gdp_quart",
        "mortgage_growth",
        "micex_cbi_tr",
        "real_dispos_income_per_cap_growth",
        "grp_growth",
    ]
    macro_df = macro_df[macro_most_important_features]

    time_span_col_name = "month"
    period = "M"

    # time_span_col_name = 'quarter'
    # period = 'Q'

    # time_span_col_name = 'year'
    # period = 'Y'

    # Step 1: Prepare transactions dataset
    transactions = transactions_df.copy()
    transactions[time_span_col_name] = transactions["timestamp"].dt.to_period(period)

    macro_df[time_span_col_name] = macro_df["timestamp"].dt.to_period(period)

    sum_columns = ["mortgage_value"]
    mean_columns = [
        col
        for col in macro_df.columns
        if col not in sum_columns + ["timestamp", time_span_col_name]
    ]
    macro_per_period = macro_df.groupby(macro_df[time_span_col_name]).agg(
        {
            **{col: "sum" for col in sum_columns},
            **{col: "median" for col in mean_columns},
        }
    )
    macro_per_period = macro_df.groupby(macro_df[time_span_col_name]).mean()
    macro_per_period = macro_per_period.drop(columns=["timestamp"])

    # Step 3: Merge transactions with aggregated macro data
    return pd.merge(
        transactions,
        macro_per_period,
        how="left",
        left_on=time_span_col_name,
        right_on=time_span_col_name,
    )


def drop_weighted_columns(df):
    columns_to_drop = [col for col in df.columns if col.endswith("_weighted")]
    return df.drop(columns=columns_to_drop)


def drop_cafe_columns(df):
    columns_to_drop = [col for col in df.columns if col.startswith("cafe_")]
    return df.drop(columns=columns_to_drop)


#  Showed +0.5%
def fi_experimental(df):
    # Living space ratio: life_sq / full_sq
    df["living_space_ratio"] = df["life_sq"] / df["full_sq"]

    # Kitchen to living ratio: kitch_sq / life_sq
    df["kitchen_to_living_ratio"] = df["kitch_sq"] / df["life_sq"]

    # Floor to max floor ratio: floor / max_floor
    df["floor_to_max_floor_ratio"] = df["floor"] / df["max_floor"]

    # Rooms per square meter: num_room / full_sq
    df["rooms_per_sq"] = df["num_room"] / df["full_sq"]

    # Elderly dependents ratio: ekder_all / work_all
    df["elderly_dependents"] = df["ekder_all"] / df["work_all"]

    # Youth dependents ratio: young_all / work_all
    df["youth_dependents"] = df["young_all"] / df["work_all"]

    # Green to industrial ratio: green_zone_part / indust_part
    df["green_to_industrial_ratio"] = df["green_zone_part"] / df["indust_part"]

    # Healthcare to population ratio: hospital_beds_raion / raion_popul
    df["healthcare_to_population_ratio"] = df["hospital_beds_raion"] / df["raion_popul"]

    # School to population ratio: children_school / raion_popul
    df["school_to_population_ratio"] = df["children_school"] / df["raion_popul"]

    # Distance to center per square meter: kremlin_km / full_sq
    # df["distance_to_center_per_sq"] = df["kremlin_km"] / df["full_sq"]

    # Age of building: timestamp.year - build_year
    df["age_of_building"] = pd.to_datetime(df["timestamp"]).dt.year - df["build_year"]

    # High floor indicator: floor / max_floor > 0.75
    df["is_high_floor"] = (df["floor"] / df["max_floor"] > 0.75).astype(int)

    # Handle divisions by zero and invalid operations
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def create_accessibility_and_adequacy_features(df):

    # Normalize population data per km² (area in km² already normalized)
    df["population_density"] = df["full_all"] / (df["area_m"] / 1_000_000)

    # Accessibility Features
    df["child_facility_accessibility"] = df["0_6_all"] / (
        df["kindergarten_km"] + df["preschool_km"] + 1e-6
    )
    df["school_accessibility"] = df["7_14_all"] / (df["school_km"] + 1e-6)
    df["university_accessibility"] = df["16_29_all"] / (df["university_km"] + 1e-6)
    df["workplace_accessibility"] = df["work_all"] / (df["office_km"] + 1e-6)
    df["elderly_healthcare_accessibility"] = df["ekder_all"] / (
        df["public_healthcare_km"] + 1e-6
    )

    # Adequacy Features (Reusing Accessibility)
    df["child_facility_adequacy"] = (
        df["child_facility_accessibility"] / df["population_density"]
    )
    df["school_adequacy"] = df["school_accessibility"] / df["population_density"]
    df["university_adequacy"] = (
        df["university_accessibility"] / df["population_density"]
    )
    df["workplace_adequacy"] = df["workplace_accessibility"] / df["population_density"]
    df["elderly_healthcare_adequacy"] = (
        df["elderly_healthcare_accessibility"] / df["population_density"]
    )

    # Log-transforming features to reduce skewness
    # for col in [
    #     'child_facility_accessibility', 'school_accessibility', 'university_accessibility',
    #     'workplace_accessibility', 'elderly_healthcare_accessibility',
    #     'child_facility_adequacy', 'school_adequacy', 'university_adequacy',
    #     'workplace_adequacy', 'elderly_healthcare_adequacy'
    # ]:
    #     df[f'log_{col}'] = np.log1p(df[col])

    return df


def clean_up_accessibility_columns(df):

    # List of columns to drop (used for calculations)
    columns_to_drop = [
        "population_density",
        "child_facility_accessibility",
        "school_accessibility",
        "university_accessibility",
        "workplace_accessibility",
        "elderly_healthcare_accessibility",
        "child_facility_adequacy",
        "school_adequacy",
        "university_adequacy",
        "workplace_adequacy",
        "elderly_healthcare_adequacy",
    ]

    # Drop the columns if they exist in the DataFrame
    df.drop(
        columns=[col for col in columns_to_drop if col in df.columns],
        inplace=True,
        errors="ignore",
    )

    return df


city_center_features = [
    "mkad_km",
    "ttk_km",
    "sadovoe_km",
    "bulvar_ring_km",
    "kremlin_km",
    "area_km",
]


def create_distance_based_features(df):

    # Convert area to km²
    df["area_km"] = df["area_m"] / 1_000_000

    # Normalize distance features by sub-area size, excluding city center features
    distance_features = [
        col for col in df.columns if "_km" in col and col not in city_center_features
    ]
    for col in distance_features:
        df[f"relative_{col}"] = df[col] / (df["area_km"] ** 0.5 + 1e-6)

    # Calculate weighted averages for accessibility groups, excluding city center features
    groups = {
        "public_transport_accessibility": [
            "metro_km_walk",
            "metro_km_avto",
            "public_transport_station_km",
        ],
        "education_accessibility": [
            "kindergarten_km",
            "school_km",
            "university_km",
            "preschool_km",
        ],
        "healthcare_accessibility": ["public_healthcare_km", "hospice_morgue_km"],
        "recreation_accessibility": [
            "park_km",
            "green_zone_km",
            "stadium_km",
            "fitness_km",
            "swim_pool_km",
        ],
        "commerce_accessibility": [
            "shopping_centers_km",
            "market_shop_km",
            "big_market_km",
        ],
    }

    for group_name, group_features in groups.items():
        group_distances = [df[col] for col in group_features if col in df]
        df[group_name] = sum(group_distances) / (len(group_features) + 1e-6)

    # Interaction with population density
    df["population_density"] = df["full_all"] / (df["area_km"] + 1e-6)
    for group_name in groups.keys():
        df[f"{group_name}_per_density"] = df[group_name] / (
            df["population_density"] + 1e-6
        )

    # Impact scores for distance features, excluding city center features
    for col in distance_features:
        if col in df.columns:
            df[f"impact_{col}"] = df[col] / (df["full_all"] + 1e-6)

    return df


def aggregate_affordability_accessibility(df):

    # Define weights for each distance range
    distance_weights = {
        "500": 6,
        "1000": 5,
        "1500": 4,
        "2000": 3,
        "3000": 2,
        "5000": 1,
    }

    accessibility_distance_weights = {
        "500": 1,
        "1000": 1,
        "1500": 1,
        "2000": 1,
        "3000": 1,
        "5000": 1,
    }

    # Identify affordability and accessibility columns
    affordability_columns = [
        col for col in df.columns if col.startswith("cafe_affordability_")
    ]
    accessibility_columns = [
        col for col in df.columns if col.startswith("cafe_accessibility_")
    ]

    # Aggregate affordability
    df["cafe_total_affordability"] = df[affordability_columns].dot(
        [distance_weights[col.split("_")[-1]] for col in affordability_columns]
    )

    # Aggregate accessibility
    df["cafe_total_accessibility"] = df[accessibility_columns].dot(
        [
            accessibility_distance_weights[col.split("_")[-1]]
            for col in accessibility_columns
        ]
    )

    # Drop individual distance-specific columns
    df.drop(columns=affordability_columns + accessibility_columns, inplace=True)

    return df


def process_and_drop_cafe_price_columns(df):
    # Create a copy of the DataFrame to avoid modifying the original

    # Define price level weights
    price_weights = {"500": 6, "1000": 5, "1500": 4, "2500": 3, "4000": 2, "high": 1}

    # Define distance weights
    distance_weights = {
        "500": 6,
        "1000": 5,
        "1500": 4,
        "2000": 3,
        "3000": 2,
        "5000": 1,
    }

    # Extract unique distance ranges from column names
    distance_ranges = sorted(
        set(col.split("_")[2] for col in df.columns if col.startswith("cafe_count_")),
        key=int,
    )

    # Process each distance range
    for distance in distance_ranges:
        # Select price-related columns corresponding to the current distance range
        relevant_price_columns = [
            col for col in df.columns if f"cafe_count_{distance}_price_" in col
        ]

        # Ensure columns are sorted by price level to align with weights
        relevant_price_columns.sort(
            key=lambda x: list(price_weights.keys()).index(x.split("_")[-1])
        )

        # Calculate weighted affordability score for this distance range
        df[f"cafe_affordability_{distance}"] = df[relevant_price_columns].dot(
            [price_weights[col.split("_")[-1]] for col in relevant_price_columns]
        )

        # Calculate accessibility score (weighted count) for this distance range
        count_column = f"cafe_count_{distance}"
        # Use provided weight or default to 1 if not specified
        weight = distance_weights.get(distance, 1)
        if count_column in df.columns:
            df[
                f"cafe_accessibility_{
                distance}"
            ] = (df[count_column] * weight)

    # Identify all columns to drop
    columns_to_drop = [
        col
        for col in df.columns
        if (
            # Drop price-related columns
            (col.startswith("cafe_count_") and "_price_" in col)
            # Drop na_price columns
            or (col.endswith("_na_price") and col.startswith("cafe_count_"))
            # Drop min/max/avg price columns
            or any(
                sub in col
                for sub in ["_min_price_avg", "_max_price_avg", "_avg_price_"]
            )
            # Drop total count columns per distance
            or any(col == f"cafe_count_{distance}" for distance in distance_ranges)
        )
    ]

    # Drop the identified columns
    df.drop(columns=columns_to_drop, inplace=True)

    # Sort new columns by distance range in ascending order
    sorted_columns = sorted(
        [
            col
            for col in df.columns
            if col.startswith("cafe_affordability_")
            or col.startswith("cafe_accessibility_")
        ],
        key=lambda x: int(x.split("_")[-1]),
    )

    # Reorder columns to place sorted new columns at the end of the DataFrame
    df = df[[col for col in df.columns if col not in sorted_columns] + sorted_columns]

    return df


def calculate_incremental_counts(df, base_column):

    distances = sorted(
        [
            col.split("_")[-1]
            for col in df.columns
            if col.startswith(base_column) and col.split("_")[-1].isdigit()
        ],
        key=int,
    )

    for i, distance in enumerate(distances):
        current_col = f"{base_column}{distance}"
        if i == 0:
            # First distance, no need to subtract
            df[f"incremental_{current_col}"] = df[current_col]
        else:
            prev_col = f"{base_column}{distances[i - 1]}"
            df[
                f"incremental_{
                current_col}"
            ] = (df[current_col] - df[prev_col])

    return df


def aggregate_weighted_counts(df, base_column, distance_weights):
    # Extract distances from columns and sort them numerically
    distances = sorted(
        [
            col.split("_")[-1]
            for col in df.columns
            if col.startswith(f"incremental_{base_column}")
        ],
        key=int,
    )

    # Initialize weighted score to 0
    weighted_score = 0

    # Accumulate weighted scores for each distance
    for distance in distances:
        col_name = f"incremental_{base_column}{distance}"
        if col_name in df.columns:  # Ensure the column exists in the dataframe
            weighted_score += df[col_name] * distance_weights.get(distance, 1)

    return weighted_score


# Drop columns used to calculate sport_count_weighted, but keep sport_count_weighted


def drop_sport_count_columns(df, base_column):
    # Identify columns to drop for incremental counts and base counts
    columns_to_drop = [
        col
        for col in df.columns
        if (col.startswith(f"incremental_{base_column}") or col.startswith(base_column))
        and col != f"{base_column}weighted"  # Keep sport_count_weighted
    ]

    # Drop the identified columns
    # inplace=False ensures a new DataFrame is returned
    df = df.drop(columns=columns_to_drop, inplace=False)

    return df


feature_distance_weights = {
    "sport_count_": {
        "500": 1,
        "1000": 1,
        "1500": 1,
        "2000": 1,
        "3000": 1,
        "5000": 1,
    },
    "office_count_": {"500": 6, "1000": 5, "1500": 4, "2000": 3, "3000": 2, "5000": 1},
    "trc_count_": {"500": 6, "1000": 5, "1500": 4, "2000": 3, "3000": 2, "5000": 1},
    "big_church_count_": {
        "500": 6,
        "1000": 5,
        "1500": 4,
        "2000": 3,
        "3000": 2,
        "5000": 1,
    },
    "church_count_": {"500": 6, "1000": 5, "1500": 4, "2000": 3, "3000": 2, "5000": 1},
    "mosque_count_": {"500": 6, "1000": 5, "1500": 4, "2000": 3, "3000": 2, "5000": 1},
    "leisure_count_": {"500": 6, "1000": 5, "1500": 4, "2000": 3, "3000": 2, "5000": 1},
    "market_count_": {"500": 6, "1000": 5, "1500": 4, "2000": 3, "3000": 2, "5000": 1},
    "trc_sqm_": {"500": 6, "1000": 5, "1500": 4, "2000": 3, "3000": 2, "5000": 1},
    "office_sqm_": {"500": 6, "1000": 5, "1500": 4, "2000": 3, "3000": 2, "5000": 1},
}

# Process and aggregate distance-weighted features


def process_distance_features(X, feature_distance_weights):
    for feature, distance_weights in feature_distance_weights.items():
        # Calculate incremental counts for the current feature
        X = calculate_incremental_counts(X, feature)

        # Aggregate the counts into a single weighted score
        X[f"{feature}weighted"] = aggregate_weighted_counts(
            X, feature, distance_weights
        )

        # Drop the incremental columns for the current feature
        # Ensure this function handles all *_distance* features
        X = drop_sport_count_columns(X, feature)

    return X


def drop_redundant_columns(df, feature_prefix, weight_dict):
    # Generate column names to drop based on the prefix and keys of the weight dictionary
    columns_to_drop = [
        f"{feature_prefix}_{
        suffix}"
        for suffix in weight_dict.keys()
    ]
    # Drop the columns
    return df.drop(columns=columns_to_drop, errors="ignore")


construction_weights = {
    "before_1920": 1,
    "1921-1945": 2,
    "1946-1970": 3,
    "1971-1995": 4,
    "after_1995": 5,
}


def calculate_weighted_build_count(df, construction_weights):
    df["weighted_build_count"] = sum(
        df[f"build_count_{period}"] * weight
        for period, weight in construction_weights.items()
    )
    return df


material_weights = {
    "block": 1.2,
    "wood": 0.5,
    "frame": 1.0,
    "brick": 1.5,
    "monolith": 2.0,
    "panel": 0.8,
    "foam": 0.7,
    "slag": 0.6,
    "mix": 1.1,
}


def calculate_weighted_material_score(df, material_weights):
    weighted_score = sum(
        df[f"build_count_{material}"] * weight
        for material, weight in material_weights.items()
    )
    df["weighted_material_score"] = weighted_score
    return df


green_weights = {"500": 5, "1000": 4, "1500": 3, "2000": 2, "3000": 1, "5000": 0.5}
prom_weights = {"500": 5, "1000": 4, "1500": 3, "2000": 2, "3000": 1, "5000": 0.5}


def weighted_average_aggregation(X, feature_base_name, distance_weights):
    # Calculate the weighted average
    weighted_column = f"{feature_base_name}_weighted"
    X[weighted_column] = sum(
        X[f"{feature_base_name}_{distance}"] * weight
        for distance, weight in distance_weights.items()
    ) / sum(distance_weights.values())

    return X


def process_demographic_features(df):
    # Ensure no division by zero
    df["full_all"] = df["full_all"].replace(0, pd.NA)

    # Normalize features by total population
    df["male_ratio"] = df["male_f"] / df["full_all"]
    df["female_ratio"] = df["female_f"] / df["full_all"]

    # Gender balance ratio
    df["male_female_ratio"] = df["male_f"] / df["female_f"]

    # Normalize age groups
    df["young_ratio"] = df["young_all"] / df["full_all"]
    df["working_ratio"] = df["work_all"] / df["full_all"]
    df["elderly_ratio"] = df["ekder_all"] / df["full_all"]

    # Aggregate age groups
    df["youth_ratio"] = df["16_29_all"] / df["full_all"]

    # Dependency ratios
    df["child_dependency_ratio"] = df["young_ratio"] / df["working_ratio"]
    df["elderly_dependency_ratio"] = df["elderly_ratio"] / df["working_ratio"]
    df["total_dependency_ratio"] = (df["young_ratio"] + df["elderly_ratio"]) / df[
        "working_ratio"
    ]

    # Gender ratios for age groups
    df["young_male_ratio"] = df["young_male"] / df["young_all"]
    df["work_male_ratio"] = df["work_male"] / df["work_all"]
    df["elderly_male_ratio"] = df["ekder_male"] / df["ekder_all"]

    return df


def improve_top_features(df, target):
    # Ensure target is aligned with df
    if len(df) != len(target):
        raise ValueError("The length of df and target must be the same.")

    # Add the target as a temporary column to calculate the group mean
    df["target_temp"] = target

    # Sub-area encoding: Compute mean of target and map to sub_area
    sub_area_mean_target = df.groupby("sub_area")["target_temp"].mean()
    df["sub_area_mean"] = (
        df["sub_area"].map(sub_area_mean_target).fillna(0)
    )  # Fill missing values

    # Drop the temporary target column
    df.drop(columns=["target_temp"], inplace=True)

    # Feature interactions
    df["log_full_sq"] = np.log1p(df["full_sq"].clip(lower=0))
    df["full_sq_per_room"] = df["full_sq"] / (df["num_room"] + 1e-6)
    df["avg_room_size"] = df["full_sq"] / (df["num_room"] + 1e-6)
    df["cafe_affordability_per_density"] = df["cafe_total_affordability"] / (
        df["population_density"] + 1e-6
    )
    df["num_room_category"] = pd.cut(
        df["num_room"].fillna(0),
        bins=[0, 1, 3, 5, np.inf],
        labels=["Small", "Medium", "Large", "Very Large"],
    )
    df["kremlin_proximity_bin"] = pd.cut(
        df["kremlin_km"].fillna(np.inf),
        bins=[0, 2, 5, 10, np.inf],
        labels=["Very Close", "Close", "Moderate", "Far"],
    )

    # Building age and categories
    current_year = 2024
    df["building_age"] = current_year - df["build_year"].fillna(current_year)
    df["building_age_bin"] = pd.cut(
        df["building_age"],
        bins=[0, 10, 30, 100, np.inf],
        labels=["New", "Modern", "Old", "Historic"],
    )
    df["floor_ratio"] = df["floor"] / (df["max_floor"] + 1e-6)

    # City center proximity
    df["city_center_proximity"] = df[
        ["kremlin_km", "sadovoe_km", "ttk_km", "mkad_km"]
    ].mean(axis=1, skipna=True)

    # Building age and state interaction
    df["state_age_interaction"] = df["state"] * df["building_age"]

    # Youth ratio normalization
    df["normalized_youth_ratio"] = df["youth_ratio"] / (df["area_m"] / 1_000_000 + 1e-6)

    return df


def drop_initial_population_columns(df):
    columns_to_drop = [
        "male_f",
        "female_f",
        "young_all",
        "young_male",
        "young_female",
        "work_all",
        "work_male",
        "work_female",
        "ekder_all",
        "ekder_male",
        "ekder_female",
        "0_6_all",
        "0_6_male",
        "0_6_female",
        "7_14_all",
        "7_14_male",
        "7_14_female",
        "0_17_all",
        "0_17_male",
        "0_17_female",
        "16_29_all",
        "16_29_male",
        "16_29_female",
        "0_13_all",
        "0_13_male",
        "0_13_female",
        "children_preschool",
        "children_school",
        "full_all",
    ]
    return df.drop(columns=columns_to_drop, inplace=False)


def drop_institutional_related_columns(df):
    columns_to_drop = [
        "preschool_quota",
        "preschool_education_centers_raion",
        "school_quota",
        "school_education_centers_raion",
        "school_education_centers_top_20_raion",
        "hospital_beds_raion",
        "healthcare_centers_raion",
        "university_top_20_raion",
        "sport_objects_raion",
        "additional_education_raion",
        "culture_objects_top_25_raion",
        "shopping_centers_raion",
    ]

    return df.drop(columns=columns_to_drop, errors="ignore")


def drop_initial_km_features(df):
    # Identify columns to drop
    columns_to_drop = [
        col for col in df.columns if "_km" in col and col not in city_center_features
    ]

    # Drop the identified columns
    return df.drop(columns=columns_to_drop, errors="ignore")


def drop_initial_features(df):
    df = drop_initial_population_columns(df)
    df = drop_initial_km_features(df)
    df = drop_institutional_related_columns(df)
    return df


def drop_testing(df):
    columns_to_drop = ["mkad_km", "ttk_km", "sadovoe_km", "bulvar_ring_km", "area_m"]

    return df.drop(columns=columns_to_drop, errors="ignore")


def process_timestamp(df):
    year_column = "year_of_transaction"

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df[year_column] = df[date_column].dt.year
    return df.drop(columns=[date_column])
