"""
Walmart Store Sales Forecasting
===============================
Data preparation, EDA, model training, evaluation, and artifact saving.

Official pipeline: models are trained WITHOUT the MarkDown1-MarkDown5 features.
Pre-week forecasting: Temperature and Fuel_Price are also excluded from the model
inputs (feature-availability audit: they are not reasonably known before the sales
week begins). CPI and Unemployment are still used but are flagged as UNAUDITED.
Validation: chronological 80/20 split on unique dates (no random splitting).
Models:     Linear Regression (scaled features) and a tuned Decision Tree.
"""

# =============================================================================
# 1. IMPORTS AND CONFIGURATION
# =============================================================================
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# --- Paths ---
DATA_DIR = Path("data")
MODEL_PATH = Path("models") / "walmart_model.pkl"
CSV_ENCODING = "cp1252"

# --- Columns ---
TARGET = "Weekly_Sales"
MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
IMPUTE_COLS = ["CPI", "Unemployment"]

# --- Feature availability (pre-week forecasting) ---
# Excluded from model inputs: not reasonably available before the sales week starts.
EXCLUDED_FEATURES = ["Temperature", "Fuel_Price"]
# Still used, but they REQUIRE A SEPARATE feature-availability audit.
UNAUDITED_FEATURES = ["CPI", "Unemployment"]
CATEGORICAL_COLS = ["Store", "Dept"]

# --- Split and model settings ---
TRAIN_FRACTION = 0.8
EXPECTED_SPLIT_DATE = "2012-04-13"
RANDOM_STATE = 42
TREE_MIN_SAMPLES_LEAF = 10


def inspect_dataframe(name, df, head_rows=5):
    """Print shape, preview, dtypes, missing values, and duplicates."""
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print("Shape:", df.shape)
    print(df.head(head_rows))
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("Total missing values:", df.isnull().sum().sum())
    print("Duplicated rows:", df.duplicated().sum())


def evaluate(y_true, y_pred):
    """Return MAE, RMSE, and R2 for a set of predictions."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def print_metrics(title, metrics):
    print(f"\n{title}")
    for name, value in metrics.items():
        print(f"{name}: {value:,.4f}")


def backup_existing_artifact(path):
    """Copy an existing model artifact to a timestamped file; never overwrite a backup."""
    if not path.exists():
        return None
    try:
        old_columns = joblib.load(path)["feature_columns"]
        label = "with_temp_fuel" if "Temperature" in old_columns else "previous"
    except Exception:
        label = "previous"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}_{label}_{stamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


# =============================================================================
# 2. LOAD DATASETS
# =============================================================================
features = pd.read_csv(DATA_DIR / "features.csv", encoding=CSV_ENCODING)
stores = pd.read_csv(DATA_DIR / "stores.csv", encoding=CSV_ENCODING)
train = pd.read_csv(DATA_DIR / "train.csv", encoding=CSV_ENCODING)
# test.csv has no target, so it is NOT used for training or evaluation.
# It is processed only so that its structure and missing values can be inspected.
test = pd.read_csv(DATA_DIR / "test.csv", encoding=CSV_ENCODING)

# =============================================================================
# 3. INITIAL DATA INSPECTION
# =============================================================================
inspect_dataframe("TRAIN (raw)", train)
inspect_dataframe("TEST (raw)", test)
inspect_dataframe("FEATURES (raw)", features, head_rows=30)
inspect_dataframe("STORES (raw)", stores)

# =============================================================================
# 4. DATA TYPE CONVERSION
# =============================================================================
for df in (train, test, features):
    df["Date"] = pd.to_datetime(df["Date"])
    df["IsHoliday"] = df["IsHoliday"].astype(int)

# One-hot encode the store Type (A/B/C) into Type_A, Type_B, Type_C.
stores = pd.get_dummies(stores, columns=["Type"], dtype=int)
print("\nStores after Type encoding:")
print(stores.head())

# =============================================================================
# 5. MERGE DATASETS
# =============================================================================
def merge_context(df):
    """Attach weekly features and store info; drop MarkDown columns."""
    df = df.merge(features, how="left", on=["Store", "Date"])
    df = df.merge(stores, how="left", on="Store")
    # MarkDown features are excluded from the whole pipeline.
    return df.drop(columns=MARKDOWN_COLS)


def resolve_holiday_columns(df, name):
    """features.csv and train/test.csv both carry IsHoliday; keep a single copy."""
    flags_match = (df["IsHoliday_x"] == df["IsHoliday_y"]).all()
    print(f"IsHoliday columns identical in {name}: {flags_match}")
    if not flags_match:
        raise ValueError(f"IsHoliday differs between source files in {name}.")
    return df.drop(columns=["IsHoliday_y"]).rename(columns={"IsHoliday_x": "IsHoliday"})


print()
train = resolve_holiday_columns(merge_context(train), "train")
test = resolve_holiday_columns(merge_context(test), "test")

# =============================================================================
# 6. FEATURE ENGINEERING (YEAR, MONTH, WEEK)
# =============================================================================
# Date is kept on purpose: it is needed for the chronological split below.
for df in (train, test):
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week

# =============================================================================
# 7. DATA CLEANING AND MISSING-VALUE INSPECTION
# =============================================================================
print("\nTrain shape:", train.shape)
print("Train missing values:")
print(train.isnull().sum())
print("Train duplicated rows:", train.duplicated().sum())

print("\nTest shape:", test.shape)
print("Test missing values:")
print(test.isnull().sum())
print("Test duplicated rows:", test.duplicated().sum())

print("\nCPI / Unemployment missing values")
print("Original features file:")
print(features[IMPUTE_COLS].isnull().sum())
print("Merged train:")
print(train[IMPUTE_COLS].isnull().sum())
print("Merged test:")
print(test[IMPUTE_COLS].isnull().sum())

missing_features = features[features[IMPUTE_COLS].isna().any(axis=1)]
print("\nFeature rows with missing CPI/Unemployment:", len(missing_features))
print("Date range:", missing_features["Date"].min(), "to", missing_features["Date"].max())

# =============================================================================
# 8. EXPLORATORY DATA ANALYSIS
# =============================================================================
# --- Weekly_Sales distribution and outliers (IQR rule) ---
print(f"\n{TARGET} descriptive statistics:")
print(train[TARGET].describe())
print("Skewness:", train[TARGET].skew())

q1 = train[TARGET].quantile(0.25)
q3 = train[TARGET].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
n_outliers = ((train[TARGET] < lower_bound) | (train[TARGET] > upper_bound)).sum()

print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("Outliers count:", n_outliers)

# --- Sales by holiday, store type, department, year, and month ---
print("\nSales by holiday flag:")
print(train.groupby("IsHoliday")[TARGET].agg(["mean", "median", "max", "count"]))

print("\nSales by store type (Type_A, Type_B, Type_C):")
print(train.groupby(["Type_A", "Type_B", "Type_C"])[TARGET].agg(["mean", "median", "count"]))

print("\nTop 10 departments by mean sales:")
dept_sales = (
    train.groupby("Dept")[TARGET]
    .agg(["mean", "median", "count"])
    .sort_values("mean", ascending=False)
)
print(dept_sales.head(10))

print("\nSales by year:")
print(train.groupby("year")[TARGET].agg(["mean", "median", "sum", "count"]))

print("\nSales by month:")
print(train.groupby("month")[TARGET].agg(["mean", "median", "sum", "count"]))

# --- Correlations ---
# EDA only: Temperature and Fuel_Price may still appear here, but they are NOT
# model inputs (see EXCLUDED_FEATURES).
corr = train.corr(numeric_only=True)

print(f"\nCorrelation of each feature with {TARGET}:")
print(corr[TARGET].drop(TARGET).sort_values())

print("\nHighly correlated feature pairs (|r| > 0.8):")
for col1 in corr.columns:
    for col2 in corr.columns:
        if col1 < col2:
            value = corr.loc[col1, col2]
            if pd.notna(value) and abs(value) > 0.8:
                print(f"{col1} - {col2}: {value:.4f}")

# =============================================================================
# 9. CHRONOLOGICAL TRAIN / VALIDATION SPLIT (80/20 ON UNIQUE DATES)
# =============================================================================
train = train.sort_values("Date").reset_index(drop=True)

unique_dates = sorted(train["Date"].unique())
split_index = int(len(unique_dates) * TRAIN_FRACTION)
split_date = pd.Timestamp(unique_dates[split_index])

if split_date != pd.Timestamp(EXPECTED_SPLIT_DATE):
    warnings.warn(
        f"Split date is {split_date.date()}, expected {EXPECTED_SPLIT_DATE}. "
        "The input data may differ from the original dataset."
    )

train_data = train[train["Date"] < split_date].copy()
val_data = train[train["Date"] >= split_date].copy()

print("\nSplit date:", split_date.date())
print("Training shape:", train_data.shape)
print("Validation shape:", val_data.shape)
print("Training date range:", train_data["Date"].min(), "to", train_data["Date"].max())
print("Validation date range:", val_data["Date"].min(), "to", val_data["Date"].max())

# =============================================================================
# 10. SEPARATE FEATURES AND TARGET
# =============================================================================
# Date is dropped only now, after the split has been made.
# EXCLUDED_FEATURES (Temperature, Fuel_Price) are removed here so they never reach
# the imputer, encoder, scaler, or either model.
NON_MODEL_COLS = [TARGET, "Date"] + EXCLUDED_FEATURES

X_train = train_data.drop(columns=NON_MODEL_COLS)
y_train = train_data[TARGET]

X_val = val_data.drop(columns=NON_MODEL_COLS)
y_val = val_data[TARGET]

leaked = [c for c in EXCLUDED_FEATURES if c in X_train.columns or c in X_val.columns]
if leaked:
    raise ValueError(f"Excluded features still present in model inputs: {leaked}")

print("\nExcluded from model inputs (not available pre-week):", EXCLUDED_FEATURES)
print("WARNING - still used, availability NOT yet audited:", UNAUDITED_FEATURES)

print("\nX_train:", X_train.shape, "| y_train:", y_train.shape)
print("X_val:  ", X_val.shape, "| y_val:  ", y_val.shape)
print("\nFeature dtypes:")
print(X_train.dtypes)

# =============================================================================
# 11. HANDLE MISSING VALUES (SimpleImputer, fitted on training data only)
# =============================================================================
imputer = SimpleImputer(strategy="median")
X_train[IMPUTE_COLS] = imputer.fit_transform(X_train[IMPUTE_COLS])
X_val[IMPUTE_COLS] = imputer.transform(X_val[IMPUTE_COLS])

print("\nRemaining missing values after imputation:")
print("Train:", X_train.isnull().sum().sum())
print("Validation:", X_val.isnull().sum().sum())

# =============================================================================
# 12. ENCODE STORE AND DEPT (OneHotEncoder, fitted on training data only)
# =============================================================================
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

X_train_encoded = encoder.fit_transform(X_train[CATEGORICAL_COLS])
X_val_encoded = encoder.transform(X_val[CATEGORICAL_COLS])

encoded_names = encoder.get_feature_names_out(CATEGORICAL_COLS)

X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_names, index=X_train.index)
X_val_encoded = pd.DataFrame(X_val_encoded, columns=encoded_names, index=X_val.index)

# Numeric columns first, then the one-hot columns (same layout as the original pipeline).
X_train_final = pd.concat([X_train.drop(columns=CATEGORICAL_COLS), X_train_encoded], axis=1)
X_val_final = pd.concat([X_val.drop(columns=CATEGORICAL_COLS), X_val_encoded], axis=1)

# The column order seen at training time is the order used everywhere else.
feature_columns = X_train_final.columns.tolist()
if X_val_final.columns.tolist() != feature_columns:
    raise ValueError("Training and validation feature columns are not aligned.")

print("\nX_train_final:", X_train_final.shape)
print("X_val_final:  ", X_val_final.shape)
print("Total feature columns:", len(feature_columns))
print("Encoded columns:", len(encoded_names))
print("Non-categorical columns:", len(feature_columns) - len(encoded_names))
print("Missing values:", X_train_final.isnull().sum().sum(), "(train),",
      X_val_final.isnull().sum().sum(), "(validation)")

# =============================================================================
# 13. TRAIN AND EVALUATE LINEAR REGRESSION (requires scaled features)
# =============================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val_final)

lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

lr_metrics = evaluate(y_val, lr_model.predict(X_val_scaled))
print_metrics("Linear Regression (validation)", lr_metrics)

# =============================================================================
# 14. TRAIN AND EVALUATE TUNED DECISION TREE (no scaling required)
# =============================================================================
dt_model = DecisionTreeRegressor(
    min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
)
dt_model.fit(X_train_final, y_train)

y_pred_dt = dt_model.predict(X_val_final)
dt_metrics = evaluate(y_val, y_pred_dt)
print_metrics("Tuned Decision Tree (validation)", dt_metrics)

dt_train_r2 = r2_score(y_train, dt_model.predict(X_train_final))
dt_val_r2 = dt_metrics["R2"]
print(f"\nTraining R2:   {dt_train_r2:.4f}")
print(f"Validation R2: {dt_val_r2:.4f}")
print(f"Difference:    {dt_train_r2 - dt_val_r2:.4f}")

# =============================================================================
# 15. COMPARE BOTH MODELS
# =============================================================================
comparison = pd.DataFrame([
    {"Model": "Linear Regression", **lr_metrics},
    {"Model": "Tuned Decision Tree", **dt_metrics},
])
print("\nModel comparison (validation set):")
print(comparison.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

# =============================================================================
# 16. SAVE THE FINAL MODEL AND PREPROCESSING OBJECTS
# =============================================================================
# The tuned Decision Tree is the primary model. The imputer expects only the
# columns in IMPUTE_COLS and the encoder expects CATEGORICAL_COLS; inference
# data must be reindexed to `feature_columns` before calling model.predict().
artifacts = {
    "model": dt_model,
    "encoder": encoder,
    "imputer": imputer,
    "feature_columns": feature_columns,
}

if any(c in feature_columns for c in EXCLUDED_FEATURES):
    raise ValueError("Excluded features found in feature_columns; refusing to save.")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
backup_path = backup_existing_artifact(MODEL_PATH)
if backup_path is not None:
    print(f"\nPrevious artifact preserved at {backup_path}")
joblib.dump(artifacts, MODEL_PATH)

print(f"\nSaved artifacts to {MODEL_PATH}")
print("Saved keys:", list(artifacts.keys()))
print("Model type:", type(dt_model).__name__)
print("Number of feature columns:", len(feature_columns))
