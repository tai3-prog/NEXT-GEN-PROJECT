"""
Generate predictions for test.csv with the saved Walmart model pipeline.

The saved artifact contains a complete scikit-learn Pipeline (imputation, scaling where
needed, one-hot encoding, and the regressor), so this script only has to rebuild the same
raw input columns that were used for training:

    Store, Dept, IsHoliday, CPI, Unemployment, Size, Type_A, Type_B, Type_C, year, month, week

Temperature, Fuel_Price and MarkDown1-5 are not model inputs and are dropped.
test.csv has no Weekly_Sales, so these predictions cannot be scored here.
"""
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ==========================================
# 1. PATHS AND SETTINGS (relative to this file)
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "walmart_model.pkl"
DATA_DIR = BASE_DIR / "data"
OUTPUT_PATH = BASE_DIR / "predictions" / "walmart_test_predictions.csv"
CSV_ENCODING = "cp1252"

# ==========================================
# 2. LOAD SAVED MODEL PIPELINE
# ==========================================
artifacts = joblib.load(MODEL_PATH)
if artifacts.get("artifact_version", 1) < 2:
    raise ValueError(
        "models/walmart_model.pkl is an old-format artifact (separate model/encoder/imputer). "
        "Run Walmart_Sales_analyst.py to create the pipeline artifact."
    )

pipeline = artifacts["model"]
feature_columns = artifacts["feature_columns"]  # raw input columns, in training order

print("Model loaded:", artifacts["model_name"], "|", type(pipeline.named_steps["model"]).__name__)

# ==========================================
# 3. LOAD DATA
# ==========================================
features = pd.read_csv(DATA_DIR / "features.csv", encoding=CSV_ENCODING)
stores = pd.read_csv(DATA_DIR / "stores.csv", encoding=CSV_ENCODING)
test = pd.read_csv(DATA_DIR / "test.csv", encoding=CSV_ENCODING)

for df in (features, test):
    df["Date"] = pd.to_datetime(df["Date"])
    df["IsHoliday"] = df["IsHoliday"].astype(int)

stores = pd.get_dummies(stores, columns=["Type"], dtype=int)

# ==========================================
# 4. MERGE DATASETS
# ==========================================
n_rows = len(test)
test["_row_order"] = np.arange(n_rows)  # preserve the original test row order

test = test.merge(features, how="left", on=["Store", "Date"], validate="many_to_one")
test = test.merge(stores, how="left", on="Store", validate="many_to_one")
if len(test) != n_rows:
    raise ValueError("Merging changed the number of test rows.")

if not (test["IsHoliday_x"] == test["IsHoliday_y"]).all():
    raise ValueError("IsHoliday columns do not match!")
test = test.drop(columns=["IsHoliday_y"]).rename(columns={"IsHoliday_x": "IsHoliday"})

# ==========================================
# 5. FEATURE ENGINEERING (same as training)
# ==========================================
test["year"] = test["Date"].dt.year
test["month"] = test["Date"].dt.month
test["week"] = test["Date"].dt.isocalendar().week.astype(int)

test = test.sort_values("_row_order").reset_index(drop=True)
prediction_info = test[["Store", "Dept", "Date"]].copy()

# ==========================================
# 6. PREDICTION (the pipeline imputes, encodes and scales internally)
# ==========================================
missing = [c for c in feature_columns if c not in test.columns]
if missing:
    raise ValueError(f"Missing model input columns: {missing}")
X_test = test[feature_columns]
print("Model input shape:", X_test.shape)

prediction_info["Predicted_Weekly_Sales"] = pipeline.predict(X_test)

# ==========================================
# 7. SAVE PREDICTIONS (an existing file is backed up first)
# ==========================================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
if OUTPUT_PATH.exists():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = OUTPUT_PATH.with_name(f"{OUTPUT_PATH.stem}_previous_{stamp}{OUTPUT_PATH.suffix}")
    shutil.copy2(OUTPUT_PATH, backup)
    print("Previous predictions preserved at", backup)
prediction_info.to_csv(OUTPUT_PATH, index=False)

print("\nPrediction completed successfully!")
print("Test rows:", len(X_test))
print("Predictions:", len(prediction_info))
print("\nSample predictions:")
print(prediction_info.head(10))
print("\nSaved to:", OUTPUT_PATH)
