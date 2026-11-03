
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "openpowerlifting.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

print("========== COMPETITION CHECK ==========")

print("\nTotal records:", len(df))

print("\nUnique MeetIDs:")
print(df["MeetID"].nunique())

print("\nMost common MeetIDs:")
print(df["MeetID"].value_counts().head(15))

print("\nMeetID minimum:", df["MeetID"].min())
print("MeetID maximum:", df["MeetID"].max())

print("\nPossible date columns:")
print([
    col for col in df.columns
    if "date" in col.lower()
])

print("\nAthletes with repeated names:")

counts = df.groupby(["Name", "Sex"])["MeetID"].nunique()

print("Repeated name-sex combinations:")
print((counts > 1).sum())

print("\n========== CHECK COMPLETED ==========")
