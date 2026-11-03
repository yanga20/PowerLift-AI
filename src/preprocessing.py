
# ==========================================
# POWERLIFT AI
# MODULE: DATA PREPROCESSING
# ==========================================

import pandas as pd
import numpy as np
from pathlib import Path


# Locate project directory
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "openpowerlifting.csv"


def load_data():

    print("Loading PowerLift AI dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    print("Dataset loaded successfully!")

    return df


def inspect_data(df):

    print("\n========== DATA INSPECTION ==========")

    print("\n1. Dataset Shape:")
    print(df.shape)

    print("\n2. Dataset Columns:")
    print(df.columns.tolist())

    print("\n3. Missing Values:")
    print(df.isnull().sum())

    print("\n4. Duplicate Rows:")
    print(df.duplicated().sum())

    print("\n5. Unique Competitions:")
    print(df["MeetID"].nunique())

    print("\n6. Unique Athlete Names:")
    print(df["Name"].nunique())

    print("\n7. Available Date Columns:")

    date_columns = [
        col for col in df.columns
        if "date" in col.lower()
    ]

    print(date_columns if date_columns else "No date column found")


def inspect_athletes(df):

    print("\n========== ATHLETE HISTORY ==========")

    # Names alone may not uniquely identify athletes.
    # Name + Sex is only a candidate grouping,
    # not a verified athlete identity.

    athlete_history = (
        df.groupby(["Name", "Sex"])["MeetID"]
        .nunique()
        .reset_index(name="CompetitionCount")
    )

    repeated = athlete_history[
        athlete_history["CompetitionCount"] > 1
    ]

    print("\nPotential repeated athlete identities:")
    print(len(repeated))

    print("\nTop 10 candidates with multiple competitions:")

    print(
        repeated.sort_values(
            "CompetitionCount",
            ascending=False
        ).head(10).to_string(index=False)
    )

    print("\nCompetition count distribution:")

    print(
        athlete_history["CompetitionCount"]
        .value_counts()
        .sort_index()
        .head(10)
    )


def inspect_lifting_records(df):

    print("\n========== LIFTING RECORD ANALYSIS ==========")

    lifting_columns = [
        "BestSquatKg",
        "BestBenchKg",
        "BestDeadliftKg",
        "TotalKg"
    ]

    for col in lifting_columns:

        values = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        print(f"\n{col}")

        print("Valid records:", values.notna().sum())
        print("Missing records:", values.isna().sum())
        print("Minimum:", values.min())
        print("Maximum:", values.max())


def inspect_meet_information(df):

    print("\n========== COMPETITION INFORMATION ==========")

    print("MeetID data type:")
    print(df["MeetID"].dtype)

    print("\nSample MeetIDs:")
    print(df["MeetID"].head(15).tolist())

    print("\nImportant:")
    print(
        "MeetID is a competition identifier. "
        "Do not assume it represents chronological order."
    )


if __name__ == "__main__":

    df = load_data()

    inspect_data(df)

    inspect_athletes(df)

    inspect_lifting_records(df)

    inspect_meet_information(df)

    print("\nPowerLift AI inspection completed!")
