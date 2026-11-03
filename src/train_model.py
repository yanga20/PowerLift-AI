
import os
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


print("Starting PowerLift AI model training...")

# 1. Load the engineered dataset
df = pd.read_csv("data/athlete_features.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# 2. Select input features and prediction target
features = [
    "PreviousTotalKg",
    "PreviousSquatKg",
    "PreviousBenchKg",
    "PreviousDeadliftKg",
    "BodyweightKg",
    "Age",
    "DaysSincePrevious"
]

target = "NextTotalKg"

# 3. Prepare the data
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date", target])
df = df.sort_values("Date").reset_index(drop=True)

X = df[features]
y = df[target]

# 4. Split chronologically: earliest 70%, next 15%, final 15%
n = len(df)
train_end = int(n * 0.70)
validation_end = int(n * 0.85)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_val = X.iloc[train_end:validation_end]
y_val = y.iloc[train_end:validation_end]

X_test = X.iloc[validation_end:]
y_test = y.iloc[validation_end:]

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:", len(X_test))

# 5. Define candidate models
models = {
    "Linear Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            max_iter=150,
            max_leaf_nodes=20,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=42
        ))
    ])
}

# 6. Train and compare models using validation data
results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    rmse = mean_squared_error(y_val, predictions) ** 0.5
    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    trained_models[name] = model

    print(f"MAE: {mae:.2f} kg")
    print(f"RMSE: {rmse:.2f} kg")
    print(f"R2: {r2:.4f}")

# 7. Select the model with the lowest validation MAE
comparison = pd.DataFrame(results).sort_values("MAE")
best_name = comparison.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\nBest model:", best_name)

# 8. Evaluate the selected model on the untouched test period
test_predictions = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = mean_squared_error(y_test, test_predictions) ** 0.5
test_r2 = r2_score(y_test, test_predictions)

print("\nFinal test results:")
print(f"MAE: {test_mae:.2f} kg")
print(f"RMSE: {test_rmse:.2f} kg")
print(f"R2: {test_r2:.4f}")

# 9. Save model and output files
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

joblib.dump({
    "model": best_model,
    "features": features,
    "target": target,
    "model_name": best_name
}, "models/powerlift_model.pkl")

comparison.to_csv("outputs/model_comparison.csv", index=False)

sample_predictions = pd.DataFrame({
    "ActualNextTotalKg": y_test.reset_index(drop=True),
    "PredictedNextTotalKg": test_predictions
})

sample_predictions.to_csv(
    "outputs/sample_predictions.csv",
    index=False
)

print("\nTraining completed successfully!")
print("Saved model: models/powerlift_model.pkl")
print("Saved comparison: outputs/model_comparison.csv")
print("Saved predictions: outputs/sample_predictions.csv")
