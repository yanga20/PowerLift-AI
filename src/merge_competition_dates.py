
# ==========================================
# POWERLIFT AI
# MODULE: COMPETITION DATE INTEGRATION
# ==========================================

import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Original dataset
ATHLETE_PATH = BASE_DIR / "data" / "openpowerlifting.csv"

# Competition dataset
MEET_PATH = BASE_DIR / "data" / "meets.csv"

# New output file (original remains untouched)
OUTPUT_PATH = BASE_DIR / "data" / "openpowerlifting_with_dates.csv"


def merge_competition_dates():

    print("Loading datasets...")

    athletes = pd.read_csv(
        ATHLETE_PATH,
        low_memory=False
    )

    meets = pd.read_csv(MEET_PATH)

    print("Athlete dataset:", athletes.shape)
    print("Meet dataset:", meets.shape)

    # Check required columns
    if "MeetID" not in athletes.columns:
        raise ValueError("MeetID missing in athlete dataset")

    if "MeetID" not in meets.columns:
        raise ValueError("MeetID missing in meets dataset")

    if "Date" not in meets.columns:
        raise ValueError("Date column missing in meets dataset")

    # Check for duplicate competition IDs
    if meets["MeetID"].duplicated().any():
        raise ValueError("Duplicate MeetIDs found in meets.csv")

    # Convert date column
    meets["Date"] = pd.to_datetime(
        meets["Date"],
        errors="coerce"
    )

    print("\nMissing competition dates:", meets["Date"].isna().sum())

    # Merge competition dates with athlete records
    merged = athletes.merge(
        meets[["MeetID", "Date"]],
        on="MeetID",
        how="left",
        validate="many_to_one",
        indicator=True
    )

    print("\n========== MERGE RESULTS ==========")

    print("Original records:", len(athletes))
    print("Merged records:", len(merged))

    print("\nMerge status:")
    print(merged["_merge"].value_counts())

    print("\nMissing dates:", merged["Date"].isna().sum())

    # Remove merge indicator
    merged.drop(columns=["_merge"], inplace=True)

    # Save as a separate file
    merged.to_csv(OUTPUT_PATH, index=False)

    print("\nDate integration completed!")

    print("New file saved at:")
    print(OUTPUT_PATH)

    print("\nFirst five records:")
    print(merged[["MeetID", "Name", "Date"]].head())


if __name__ == "__main__":

    merge_competition_dates()
