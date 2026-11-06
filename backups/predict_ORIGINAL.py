
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# 1. LOAD SAVED MODEL AND PREPROCESSING
# ==========================================

artifacts = joblib.load("models/walmart_model.pkl")

model = artifacts["model"]
encoder = artifacts["encoder"]
imputer = artifacts["imputer"]
feature_columns = artifacts["feature_columns"]

print("Model loaded:", type(model).__name__)


# ==========================================
# 2. LOAD DATA
# ==========================================

features = pd.read_csv("data/features.csv", encoding="cp1252")
stores = pd.read_csv("data/stores.csv", encoding="cp1252")
test = pd.read_csv("data/test.csv", encoding="cp1252")

for df in (features, test):
    df["Date"] = pd.to_datetime(df["Date"])
    df["IsHoliday"] = df["IsHoliday"].astype(int)

stores = pd.get_dummies(
    stores,
    columns=["Type"],
    dtype=int
)


# ==========================================
# 3. MERGE DATASETS
# ==========================================

# Preserve original test row order
test["_row_order"] = np.arange(len(test))

test = test.merge(
    features,
    how="left",
    on=["Store", "Date"],
    validate="many_to_one"
)

test = test.merge(
    stores,
    how="left",
    on="Store",
    validate="many_to_one"
)

# Check holiday columns
if not (test["IsHoliday_x"] == test["IsHoliday_y"]).all():
    raise ValueError("IsHoliday columns do not match!")

test = test.drop(columns=["IsHoliday_y"])
test = test.rename(columns={"IsHoliday_x": "IsHoliday"})

# Exclude Markdown features
markdown_cols = [
    "MarkDown1", "MarkDown2", "MarkDown3",
    "MarkDown4", "MarkDown5"
]

test = test.drop(columns=markdown_cols)

# Exclude features not used during training
excluded_cols = ["Temperature", "Fuel_Price"]

test = test.drop(columns=excluded_cols)


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

test["year"] = test["Date"].dt.year
test["month"] = test["Date"].dt.month
test["week"] = test["Date"].dt.isocalendar().week

# Restore original row order
test = test.sort_values("_row_order").reset_index(drop=True)

prediction_info = test[["Store", "Dept", "Date"]].copy()


# ==========================================
# 5. PREPROCESSING
# ==========================================

X_test = test.drop(columns=["Date", "_row_order"])

# Apply the already fitted imputer
impute_cols = ["CPI", "Unemployment"]

X_test[impute_cols] = imputer.transform(
    X_test[impute_cols]
)

# Apply the already fitted encoder
categorical_cols = ["Store", "Dept"]

encoded_array = encoder.transform(
    X_test[categorical_cols]
)

encoded_df = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(categorical_cols),
    index=X_test.index
)

X_test_final = pd.concat(
    [
        X_test.drop(columns=categorical_cols),
        encoded_df
    ],
    axis=1
)

# Ensure exactly the same feature columns and order as training
missing_cols = set(feature_columns) - set(X_test_final.columns)
extra_cols = set(X_test_final.columns) - set(feature_columns)

if missing_cols or extra_cols:
    raise ValueError(
        f"Feature mismatch! Missing: {missing_cols}, Extra: {extra_cols}"
    )

X_test_final = X_test_final[feature_columns]


# ==========================================
# 6. PREDICTION
# ==========================================
print("Final test shape:", X_test_final.shape)
print("Model features:", len(feature_columns))


predictions = model.predict(X_test_final)

prediction_info["Predicted_Weekly_Sales"] = predictions


# ==========================================
# 7. SAVE PREDICTIONS
# ==========================================

output_dir = Path("predictions")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "walmart_test_predictions.csv"

prediction_info.to_csv(output_path, index=False)

print("\nPrediction completed successfully!")
print("Test rows:", len(X_test_final))
print("Predictions:", len(predictions))
print("\nSample predictions:")
print(prediction_info.head(10))

print("\nSaved to:", output_path)
