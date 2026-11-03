
# ==========================================
# POWERLIFT AI
# MODULE: FEATURE ENGINEERING
# OBJECTIVE: PREDICT NEXT COMPETITION TOTAL
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "data" /
    "openpowerlifting_with_dates.csv"
)

OUTPUT_PATH = (
    BASE_DIR / "data" /
    "athlete_features.csv"
)


def create_features():

    print("Loading historical athlete data...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Convert relevant columns to numeric
    numeric_columns = [
        "Age",
        "BodyweightKg",
        "BestSquatKg",
        "BestBenchKg",
        "BestDeadliftKg",
        "TotalKg"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Convert competition date
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    print("Original records:", len(df))

    # Remove records without essential identity/date information
    df = df.dropna(
        subset=["Name", "Sex", "Equipment", "Date"]
    ).copy()

    # Keep positive performance values only
    # Invalid or negative lifting results become missing
    lift_columns = [
        "BestSquatKg",
        "BestBenchKg",
        "BestDeadliftKg",
        "TotalKg"
    ]

    for col in lift_columns:
        df.loc[df[col] <= 0, col] = np.nan

    # Candidate identity:
    # Name + Sex + Equipment
    # This is NOT a verified unique athlete ID.

    identity_columns = [
        "Name",
        "Sex",
        "Equipment"
    ]

    # One record per candidate athlete per competition date.
    # Multiple entries on the same date are consolidated.

    aggregations = {
        "MeetID": "first",
        "Age": "median",
        "BodyweightKg": "median",
        "BestSquatKg": "max",
        "BestBenchKg": "max",
        "BestDeadliftKg": "max",
        "TotalKg": "max"
    }

    history = (
        df.groupby(
            identity_columns + ["Date"],
            as_index=False
        )
        .agg(aggregations)
    )

    # Sort records chronologically
    history = history.sort_values(
        identity_columns + ["Date"]
    ).reset_index(drop=True)

    # Create previous competition features
    grouped = history.groupby(
        identity_columns,
        sort=False
    )

    history["PreviousTotalKg"] = grouped["TotalKg"].shift(0)

    history["PreviousSquatKg"] = grouped["BestSquatKg"].shift(0)

    history["PreviousBenchKg"] = grouped["BestBenchKg"].shift(0)

    history["PreviousDeadliftKg"] = grouped["BestDeadliftKg"].shift(0)

    # Target: next competition's total
    history["NextTotalKg"] = grouped["TotalKg"].shift(-1)

    # Time since the previous recorded competition
    history["DaysSincePrevious"] = (
        grouped["Date"].diff().dt.days
    )

    # Remove rows without a valid previous and next total
    model_data = history.dropna(
        subset=[
            "PreviousTotalKg",
            "NextTotalKg"
        ]
    ).copy()

    # Remove impossible target values
    model_data = model_data[
        model_data["NextTotalKg"] > 0
    ]

    # Features used by our future model
    feature_columns = [
        "PreviousTotalKg",
        "PreviousSquatKg",
        "PreviousBenchKg",
        "PreviousDeadliftKg",
        "BodyweightKg",
        "Age",
        "DaysSincePrevious"
    ]

    # Keep rows with a valid previous total.
    # Other missing predictors will be handled later
    # using a training-only imputation pipeline.

    model_data = model_data.dropna(
        subset=["PreviousTotalKg"]
    )

    # Save feature dataset
    model_data.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n========== FEATURE ENGINEERING RESULTS ==========")

    print("Historical records:", len(history))

    print("Model-ready records:", len(model_data))

    print("\nFeature columns:")
    print(feature_columns)

    print("\nPrediction target:")
    print("NextTotalKg")

    print("\nSample model data:")

    print(
        model_data[
            identity_columns +
            ["Date"] +
            feature_columns +
            ["NextTotalKg"]
        ].head(10)
    )

    print("\nMissing values in features:")

    print(
        model_data[feature_columns].isnull().sum()
    )

    print("\nOutput saved to:")
    print(OUTPUT_PATH)

    print("\nFeature engineering completed!")


if __name__ == "__main__":

    create_features()
