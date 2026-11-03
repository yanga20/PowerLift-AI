import pandas as pd
from pathlib import Path


# Locate the main project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Define dataset location
DATA_PATH = BASE_DIR / "data" / "openpowerlifting.csv"


# Function to load the dataset
def load_dataset():

    print("Loading PowerLift AI dataset...")

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False
    )

    print("Dataset loaded successfully!")

    return df


# Function to inspect the dataset
def inspect_dataset(df):

    print("\n========== POWERLIFT AI ==========")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nFirst Five Records:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nMissing Values:")
    print(df.isnull().sum())


# Main execution
if __name__ == "__main__":

    df = load_dataset()

    inspect_dataset(df)
