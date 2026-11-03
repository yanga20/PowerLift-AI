
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "openpowerlifting_with_dates.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

print("\n========== POWERLIFT AI DATE VERIFICATION ==========")

print("\nDataset shape:")
print(df.shape)

print("\nMissing dates:")
print(df["Date"].isna().sum())

print("\nEarliest competition:")
print(df["Date"].min())

print("\nLatest competition:")
print(df["Date"].max())

print("\nUnique competitions:")
print(df["MeetID"].nunique())

print("\nUnique athlete names:")
print(df["Name"].nunique())

print("\nSample chronological records:")

print(
    df.sort_values("Date")[
        ["Name", "MeetID", "Date", "BestSquatKg",
         "BestBenchKg", "BestDeadliftKg"]
    ].head(10)
)

print("\nDate verification completed!")
