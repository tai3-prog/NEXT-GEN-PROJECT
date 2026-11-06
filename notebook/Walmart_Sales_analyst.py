"""
Walmart Store Sales Forecasting
===============================
Data preparation, EDA, model training, time-series cross-validation, tuning,
holdout evaluation, error analysis, and artifact saving.

Official pipeline: models are trained WITHOUT the MarkDown1-MarkDown5 features.
Pre-week forecasting: Temperature and Fuel_Price are also excluded from the model
inputs (feature-availability audit: they are not reasonably known before the sales
week begins). CPI and Unemployment are still used but are flagged as UNAUDITED
(see audit_cpi_unemployment() and the README).

Evaluation design (no random splitting anywhere):
  1. DEVELOPMENT period = earliest 80% of unique dates in train.csv.
     Cross-validation and hyperparameter tuning use ONLY this period.
  2. HOLDOUT period = latest 20% of unique dates. It is not touched until the
     model has been selected from the cross-validation results.
  3. FINAL model = the selected pipeline re-fitted on ALL labeled data
     (development + holdout) after the evaluation is finished.
test.csv has no Weekly_Sales, so it is never used for evaluation.

Models: Mean baseline, Linear Regression, Decision Tree, Random Forest.
Set the environment variable WALMART_QUICK_TEST=1 for a fast smoke test with
tiny search grids (results from that mode must not be reported).
"""

# =============================================================================
# 1. IMPORTS AND CONFIGURATION
# =============================================================================
import os
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# --- Paths (relative to this file, so the script works from any directory) ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "models" / "walmart_model.pkl"
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR = REPORT_DIR / "figures"
CSV_ENCODING = "cp1252"

# --- Columns ---
TARGET = "Weekly_Sales"
MARKDOWN_COLS = ["MarkDown1", "MarkDown2", "MarkDown3", "MarkDown4", "MarkDown5"]
IMPUTE_COLS = ["CPI", "Unemployment"]
CATEGORICAL_COLS = ["Store", "Dept"]
NUMERIC_PASSTHROUGH_COLS = ["IsHoliday", "Size", "Type_A", "Type_B", "Type_C",
                            "year", "month", "week"]
# The raw columns every pipeline receives (and the app / predict.py must supply).
INPUT_COLUMNS = CATEGORICAL_COLS + IMPUTE_COLS + NUMERIC_PASSTHROUGH_COLS

# --- Feature availability (pre-week forecasting) ---
# Excluded from model inputs: not reasonably available before the sales week starts.
EXCLUDED_FEATURES = ["Temperature", "Fuel_Price"]
# Still used, but they REQUIRE A SEPARATE feature-availability audit.
UNAUDITED_FEATURES = ["CPI", "Unemployment"]

# --- Split and model settings ---
TRAIN_FRACTION = 0.8  # share of unique dates used for development (CV + tuning)
EXPECTED_SPLIT_DATE = "2012-04-13"
RANDOM_STATE = 42
TREE_MIN_SAMPLES_LEAF = 10  # the ORIGINAL fixed value; kept as a reference model
CV_SPLITS = 4

# Search spaces. The Random Forest is deliberately modest (50 trees, each grown on
# a 50% bootstrap sample, 4 sampled candidates) so tuning stays affordable.
QUICK_TEST = os.environ.get("WALMART_QUICK_TEST") == "1"
if QUICK_TEST:
    DT_PARAM_GRID = {"model__min_samples_leaf": [10], "model__max_depth": [None]}
    RF_PARAM_SPACE = {"model__min_samples_leaf": [10], "model__max_features": [0.33],
                      "model__max_depth": [None]}
    RF_N_ITER, RF_N_ESTIMATORS = 1, 5
else:
    DT_PARAM_GRID = {"model__min_samples_leaf": [2, 5, 10, 20, 40],
                     "model__max_depth": [None, 25]}
    RF_PARAM_SPACE = {"model__min_samples_leaf": [5, 10, 20],
                      "model__max_features": [0.33, 0.5],
                      "model__max_depth": [None, 25]}
    RF_N_ITER, RF_N_ESTIMATORS = 4, 50
RF_MAX_SAMPLES = 0.5

SCORING = {"mae": "neg_mean_absolute_error",
           "rmse": "neg_root_mean_squared_error",
           "r2": "r2"}

# --- Plot colours (Walmart-inspired) ---
BLUE, YELLOW, NAVY, GREY = "#0071CE", "#FFC220", "#041E42", "#8A94A6"

REPORT = {}  # collected results, written to reports/results_summary.md at the end
SCRIPT_START = time.time()


# =============================================================================
# 1b. HELPER FUNCTIONS
# =============================================================================
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


def save_figure(fig, filename):
    """Save a Matplotlib figure under reports/figures/ and close it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, dpi=150)
    plt.close(fig)
    print("  saved figure:", (FIG_DIR / filename).relative_to(BASE_DIR))


def dataframe_to_markdown(df, float_format="{:,.2f}"):
    """Minimal DataFrame -> Markdown table (avoids needing the 'tabulate' package)."""
    def fmt(value):
        return float_format.format(value) if isinstance(value, (float, np.floating)) else str(value)
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    divider = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider] + rows)


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
# This is a fixed, store-level mapping (not learned from sales), so it is done once here.
stores = pd.get_dummies(stores, columns=["Type"], dtype=int)
print("\nStores after Type encoding:")
print(stores.head())

# =============================================================================
# 4b. DATA INTEGRITY CHECKS BEFORE MERGING (keys, duplicates, weekly dates)
# =============================================================================
print("\n" + "=" * 60 + "\nDATA INTEGRITY CHECKS (before merge)\n" + "=" * 60)
integrity_notes = []


def check_unique_keys(df, keys, name):
    """A merge key that is not unique would silently duplicate rows."""
    n_dup = int(df.duplicated(subset=keys).sum())
    line = f"Duplicate {keys} keys in {name}: {n_dup}"
    print(line)
    integrity_notes.append(line)
    if n_dup:
        raise ValueError(line)


check_unique_keys(features, ["Store", "Date"], "features.csv")
check_unique_keys(stores, ["Store"], "stores.csv")
check_unique_keys(train, ["Store", "Dept", "Date"], "train.csv")
check_unique_keys(test, ["Store", "Dept", "Date"], "test.csv")

for name, df in (("train.csv", train), ("test.csv", test)):
    pairs = df[["Store", "Date"]].drop_duplicates()
    unmatched = pairs.merge(features[["Store", "Date"]], how="left", indicator=True)
    n_unmatched = int((unmatched["_merge"] == "left_only").sum())
    line = f"(Store, Date) pairs of {name} missing from features.csv: {n_unmatched}"
    print(line)
    integrity_notes.append(line)
    if n_unmatched:
        raise ValueError(line)
    n_missing_store = int((~df["Store"].isin(stores["Store"])).sum())
    integrity_notes.append(f"Rows of {name} whose Store is missing from stores.csv: {n_missing_store}")
    print(integrity_notes[-1])

# Weekly date convention: every date should be a Friday, spaced exactly 7 days apart.
all_dates = pd.Series(sorted(set(train["Date"]) | set(test["Date"])))
line = (f"Weekday of all train/test dates: {sorted(all_dates.dt.day_name().unique())}; "
        f"gaps between consecutive weeks (days): {[int(d) for d in sorted(all_dates.diff().dropna().dt.days.unique())]}")
print(line)
integrity_notes.append(line)

train_rows_raw, test_rows_raw = len(train), len(test)

# =============================================================================
# 5. MERGE DATASETS
# =============================================================================
def merge_context(df, name):
    """Attach weekly features and store info; drop MarkDown columns."""
    rows_before = len(df)
    # validate="many_to_one" raises if the right-hand keys are not unique.
    df = df.merge(features, how="left", on=["Store", "Date"], validate="many_to_one")
    df = df.merge(stores, how="left", on="Store", validate="many_to_one")
    line = f"{name}: rows before merge {rows_before:,} -> after merge {len(df):,}"
    print(line)
    integrity_notes.append(line)
    if len(df) != rows_before:
        raise ValueError(f"Merge changed the number of {name} rows.")
    if df[["Size", "Type_A", "Type_B", "Type_C"]].isnull().any().any():
        raise ValueError(f"Store information is missing after merging {name}.")
    # MarkDown features are excluded from the whole pipeline.
    return df.drop(columns=MARKDOWN_COLS)


def resolve_holiday_columns(df, name):
    """features.csv and train/test.csv both carry IsHoliday; keep a single copy."""
    flags_match = (df["IsHoliday_x"] == df["IsHoliday_y"]).all()
    print(f"IsHoliday columns identical in {name}: {flags_match}")
    integrity_notes.append(f"IsHoliday identical in train/test and features.csv ({name}): {flags_match}")
    if not flags_match:
        raise ValueError(f"IsHoliday differs between source files in {name}.")
    return df.drop(columns=["IsHoliday_y"]).rename(columns={"IsHoliday_x": "IsHoliday"})


print()
train = resolve_holiday_columns(merge_context(train, "train"), "train")
test = resolve_holiday_columns(merge_context(test, "test"), "test")
assert len(train) == train_rows_raw and len(test) == test_rows_raw
REPORT["integrity_notes"] = integrity_notes

# =============================================================================
# 6. FEATURE ENGINEERING (YEAR, MONTH, WEEK)
# =============================================================================
# Date is kept on purpose: it is needed for the chronological split below.
for df in (train, test):
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["week"] = df["Date"].dt.isocalendar().week.astype(int)  # ISO week of the Friday date

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

# Why MarkDown1-5 are excluded: they are mostly empty and only exist late in the data.
print("\nMarkDown availability in features.csv (these columns are excluded from modeling):")
for col in MARKDOWN_COLS:
    first_date = features.loc[features[col].notna(), "Date"].min()
    print(f"{col}: {features[col].notna().mean():.1%} non-null, first value on {first_date.date()}")
REPORT["markdown_share"] = {c: float(features[c].notna().mean()) for c in MARKDOWN_COLS}
REPORT["markdown_first_date"] = str(features.loc[features[MARKDOWN_COLS[0]].notna(), "Date"].min().date())
REPORT["negative_sales_rows"] = int((train[TARGET] < 0).sum())

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
is_outlier = (train[TARGET] < lower_bound) | (train[TARGET] > upper_bound)
n_outliers = is_outlier.sum()

print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("Outliers count:", n_outliers)

# --- Outlier investigation: are they errors or legitimate high sales? ---
outliers = train[is_outlier]
recurring = outliers.groupby(["Store", "Dept", "week"])["year"].transform("nunique") >= 2
outlier_notes = [
    f"IQR bounds: {lower_bound:,.0f} to {upper_bound:,.0f}; {n_outliers:,} of {len(train):,} rows "
    f"({n_outliers / len(train):.1%}) are flagged, all above the upper bound "
    f"({int((train[TARGET] < lower_bound).sum()):,} below the lower bound).",
    f"Holiday-week share: {outliers['IsHoliday'].mean():.1%} of outliers vs {train['IsHoliday'].mean():.1%} of all rows.",
    f"Nov-Dec share: {outliers['month'].isin([11, 12]).mean():.1%} of outliers vs "
    f"{train['month'].isin([11, 12]).mean():.1%} of all rows.",
    f"{recurring.mean():.1%} of outlier rows belong to a Store/Dept/ISO-week combination that is an "
    f"outlier in at least 2 different years (recurring seasonal peaks, not one-off glitches).",
    "Top departments among outliers (dept: count): "
    + ", ".join(f"{d}: {c:,}" for d, c in outliers["Dept"].value_counts().head(5).items())
    + f" (these 5 departments hold {outliers['Dept'].value_counts().head(5).sum() / len(outliers):.1%} of all outliers).",
    f"Negative Weekly_Sales rows (likely returns/adjustments): {REPORT['negative_sales_rows']:,}. "
    "They are kept because they are real recorded values and cannot be verified as errors.",
    "DECISION: outliers are NOT removed. Evidence: they are concentrated in a few high-volume departments, "
    "they recur for the same Store/Dept/week in different years, and they are only mildly over-represented in "
    "holiday and Nov-Dec weeks. That pattern points to structurally large departments and repeating seasonal "
    "peaks rather than one-off data errors. It does not prove that every single value is correct. "
    "Removing them would teach the model to under-predict the largest sales, and tree-based models split on "
    "thresholds instead of fitting one global line, so extreme values distort them less than a linear fit.",
]
print("\nOutlier investigation:")
for note in outlier_notes:
    print("-", note)
REPORT["outlier_notes"] = outlier_notes

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


# --- EDA charts (Matplotlib) ---
def create_eda_charts(df, correlation):
    """Save the EDA figures to reports/figures/."""
    print("\nCreating EDA charts:")
    sales = df[TARGET]
    store_type = df[["Type_A", "Type_B", "Type_C"]].idxmax(axis=1).str.replace("Type_", "", regex=False)

    # 1. Distribution (full range and trimmed range, because of the heavy right tail)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(sales, bins=100, color=BLUE)
    axes[0].set(title="Weekly_Sales distribution (all records)", xlabel="Weekly sales (USD)", ylabel="Records")
    trimmed = sales[sales.between(sales.quantile(0.001), sales.quantile(0.99))]
    axes[1].hist(trimmed, bins=100, color=BLUE)
    axes[1].set(title="Trimmed to 0.1st-99th percentile", xlabel="Weekly sales (USD)", ylabel="Records")
    save_figure(fig, "01_weekly_sales_distribution.png")

    # 2. Total weekly sales over time, with holiday weeks highlighted
    weekly_total = df.groupby("Date")[TARGET].sum() / 1e6
    holiday_dates = df.loc[df["IsHoliday"] == 1, "Date"].unique()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(weekly_total.index, weekly_total.values, color=BLUE, linewidth=1.8, label="All stores and departments")
    ax.scatter(holiday_dates, weekly_total.loc[holiday_dates], color=YELLOW, edgecolor=NAVY, zorder=3,
               label="Holiday week")
    ax.set(title="Total weekly sales over time", xlabel="Week (Friday date)", ylabel="Total weekly sales (million USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, "02_weekly_sales_over_time.png")

    # 3. Average sales by month
    monthly = df.groupby("month")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(monthly.index.astype(str), monthly.values, color=BLUE)
    ax.set(title="Average weekly sales per record by month", xlabel="Month", ylabel="Mean Weekly_Sales (USD)")
    save_figure(fig, "03_avg_sales_by_month.png")

    # 4. Average sales by holiday status
    holiday = df.groupby("IsHoliday")[TARGET].mean()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(["Regular week", "Holiday week"], holiday.values, color=[BLUE, YELLOW])
    ax.set(title="Average weekly sales by holiday status", ylabel="Mean Weekly_Sales (USD)")
    save_figure(fig, "04_avg_sales_by_holiday.png")

    # 5. Average sales by store type
    by_type = df.groupby(store_type)[TARGET].mean()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(by_type.index, by_type.values, color=BLUE)
    ax.set(title="Average weekly sales by store type", xlabel="Store type", ylabel="Mean Weekly_Sales (USD)")
    save_figure(fig, "05_avg_sales_by_store_type.png")

    # 6. Top 10 departments
    top = df.groupby("Dept")[TARGET].mean().sort_values(ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(top.index.astype(str), top.values, color=BLUE)
    ax.set(title="Top 10 departments by average weekly sales", xlabel="Mean Weekly_Sales (USD)", ylabel="Department")
    save_figure(fig, "06_top_departments.png")

    # 7. Correlation heatmap (EDA only)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(correlation.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(correlation.columns)), correlation.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(correlation.columns)), correlation.columns)
    for i in range(len(correlation)):
        for j in range(len(correlation)):
            ax.text(j, i, f"{correlation.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Correlation heatmap (EDA only; Temperature/Fuel_Price are not model inputs)")
    save_figure(fig, "07_correlation_heatmap.png")


create_eda_charts(train, corr)


def build_eda_insights(df):
    """Data-driven insights; every number below is computed from the dataset."""
    sales = df[TARGET]
    insights = []
    insights.append(
        f"Weekly_Sales is strongly right-skewed (skewness {sales.skew():.2f}; mean ${sales.mean():,.0f} vs "
        f"median ${sales.median():,.0f}). A small number of very large values dominate squared-error "
        f"metrics such as RMSE.")
    hol = df.groupby("IsHoliday")[TARGET].mean()
    insights.append(
        f"Holiday weeks average ${hol[1]:,.0f} per record vs ${hol[0]:,.0f} for regular weeks "
        f"({(hol[1] / hol[0] - 1) * 100:+.1f}%), but the labeled data contains only "
        f"{df.loc[df['IsHoliday'] == 1, 'Date'].nunique()} distinct holiday weeks, so this effect rests on few events.")
    month = df.groupby("month")[TARGET].mean()
    insights.append(
        f"Strong seasonality: month {month.idxmax()} has the highest average weekly sales "
        f"(${month.max():,.0f}) and month {month.idxmin()} the lowest (${month.min():,.0f}); this pattern is "
        f"consistent with seasonal demand, although the data cannot confirm the cause.")
    store_type = df[["Type_A", "Type_B", "Type_C"]].idxmax(axis=1).str.replace("Type_", "", regex=False)
    by_type = df.groupby(store_type)[TARGET].mean()
    insights.append(
        "Store type matters: average weekly sales per record are "
        + ", ".join(f"Type {t} ${v:,.0f}" for t, v in by_type.items())
        + f"; store Size correlates with sales (r = {df['Size'].corr(sales):.2f}).")
    dept_totals = df.groupby("Dept")[TARGET].sum().sort_values(ascending=False)
    insights.append(
        f"Sales are concentrated: the top 5 departments ({', '.join(map(str, dept_totals.head(5).index))}) "
        f"account for {dept_totals.head(5).sum() / dept_totals.sum():.1%} of all recorded sales, "
        f"so department identity is a major driver.")
    return insights


eda_insights = build_eda_insights(train)
print("\nEDA insights:")
for item in eda_insights:
    print("-", item)
REPORT["eda_insights"] = eda_insights

# =============================================================================
# 9. CHRONOLOGICAL DEVELOPMENT / HOLDOUT SPLIT (80/20 ON UNIQUE DATES)
# =============================================================================
# Deterministic order: by date, then Store and Dept (ties on Date are otherwise arbitrary).
train = train.sort_values(["Date", "Store", "Dept"]).reset_index(drop=True)

unique_dates = sorted(train["Date"].unique())
split_index = int(len(unique_dates) * TRAIN_FRACTION)
split_date = pd.Timestamp(unique_dates[split_index])

if split_date != pd.Timestamp(EXPECTED_SPLIT_DATE):
    warnings.warn(
        f"Split date is {split_date.date()}, expected {EXPECTED_SPLIT_DATE}. "
        "The input data may differ from the original dataset."
    )

# All rows of one date stay on the same side of the split.
dev_data = train[train["Date"] < split_date].reset_index(drop=True)
holdout_data = train[train["Date"] >= split_date].reset_index(drop=True)
assert dev_data["Date"].max() < holdout_data["Date"].min()
assert not set(dev_data["Date"].unique()) & set(holdout_data["Date"].unique())

print("\nSplit date:", split_date.date())
print("Development shape:", dev_data.shape)
print("Holdout shape:    ", holdout_data.shape)
print("Development date range:", dev_data["Date"].min().date(), "to", dev_data["Date"].max().date(),
      f"({dev_data['Date'].nunique()} weeks)")
print("Holdout date range:    ", holdout_data["Date"].min().date(), "to", holdout_data["Date"].max().date(),
      f"({holdout_data['Date'].nunique()} weeks)")
REPORT["split"] = {
    "split_date": str(split_date.date()),
    "dev_rows": len(dev_data), "dev_weeks": int(dev_data["Date"].nunique()),
    "dev_range": f"{dev_data['Date'].min().date()} to {dev_data['Date'].max().date()}",
    "holdout_rows": len(holdout_data), "holdout_weeks": int(holdout_data["Date"].nunique()),
    "holdout_range": f"{holdout_data['Date'].min().date()} to {holdout_data['Date'].max().date()}",
}

# =============================================================================
# 10. SEPARATE FEATURES AND TARGET
# =============================================================================
# Date is dropped only now, after the split has been made.
# EXCLUDED_FEATURES (Temperature, Fuel_Price) never reach any pipeline.
NON_MODEL_COLS = [TARGET, "Date"] + EXCLUDED_FEATURES

unexpected = set(dev_data.drop(columns=NON_MODEL_COLS).columns) ^ set(INPUT_COLUMNS)
if unexpected:
    raise ValueError(f"Model input columns differ from INPUT_COLUMNS: {sorted(unexpected)}")

X_dev, y_dev = dev_data[INPUT_COLUMNS], dev_data[TARGET]
X_holdout, y_holdout = holdout_data[INPUT_COLUMNS], holdout_data[TARGET]

print("\nExcluded from model inputs (not available pre-week):", EXCLUDED_FEATURES)
print("WARNING - still used, availability NOT yet audited:", UNAUDITED_FEATURES)
print("\nX_dev:    ", X_dev.shape, "| y_dev:    ", y_dev.shape)
print("X_holdout:", X_holdout.shape, "| y_holdout:", y_holdout.shape)
print("\nModel input dtypes:")
print(X_dev.dtypes)


# =============================================================================
# 11. PREPROCESSING + MODEL PIPELINES (fitted only on the data passed to fit())
# =============================================================================
def build_preprocessor(scale, impute_cols=IMPUTE_COLS):
    """ColumnTransformer: median imputation, optional scaling, one-hot Store/Dept.

    - CPI/Unemployment: SimpleImputer(median) [+ StandardScaler for linear models]
    - other numeric columns: scaled for linear models, passed through for trees
    - Store/Dept: OneHotEncoder(handle_unknown="ignore") so unseen categories are safe
    Inside a Pipeline, every statistic (median, mean/std, categories) is learned from
    the training rows of each fit() call only, so cross-validation cannot leak.
    """
    impute_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        impute_steps.append(("scaler", StandardScaler()))
    transformers = []
    if impute_cols:
        transformers.append(("impute", Pipeline(impute_steps), list(impute_cols)))
    transformers.append(("numeric", StandardScaler() if scale else "passthrough", NUMERIC_PASSTHROUGH_COLS))
    transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                         CATEGORICAL_COLS))
    return ColumnTransformer(transformers, sparse_threshold=0, verbose_feature_names_out=False)


def build_pipeline(model, scale=False, impute_cols=IMPUTE_COLS):
    return Pipeline([("preprocess", build_preprocessor(scale, impute_cols)), ("model", model)])


def make_date_folds(dates, n_splits):
    """Chronological expanding-window folds built from SORTED UNIQUE DATES.

    TimeSeriesSplit is applied to the unique dates, then mapped back to row
    positions, so every row of a given date lands in the same fold (many
    Store/Dept rows share one date).
    """
    dates = pd.Series(dates).reset_index(drop=True)
    unique = np.sort(dates.unique())
    folds = []
    for train_dates_idx, test_dates_idx in TimeSeriesSplit(n_splits=n_splits).split(unique):
        train_rows = np.flatnonzero(dates.isin(unique[train_dates_idx]).to_numpy())
        test_rows = np.flatnonzero(dates.isin(unique[test_dates_idx]).to_numpy())
        folds.append((train_rows, test_rows))
    return folds


def verify_date_folds(dates, folds):
    """Raise if any fold lets a date appear on both sides or train after test."""
    dates = pd.Series(dates).reset_index(drop=True)
    seen_test_dates = set()
    for number, (train_rows, test_rows) in enumerate(folds, start=1):
        train_dates, test_dates = dates.iloc[train_rows], dates.iloc[test_rows]
        assert train_dates.max() < test_dates.min(), f"fold {number}: training is not earlier than test"
        assert not set(train_dates.unique()) & set(test_dates.unique()), f"fold {number}: date overlap"
        assert dates.isin(test_dates.unique()).sum() == len(test_rows), f"fold {number}: test date split"
        assert dates.isin(train_dates.unique()).sum() == len(train_rows), f"fold {number}: train date split"
        assert not set(test_dates.unique()) & seen_test_dates, f"fold {number}: test dates reused"
        seen_test_dates |= set(test_dates.unique())
        print(f"Fold {number}: train {train_dates.min().date()} -> {train_dates.max().date()} "
              f"({train_dates.nunique()} weeks, {len(train_rows):,} rows) | "
              f"test {test_dates.min().date()} -> {test_dates.max().date()} "
              f"({test_dates.nunique()} weeks, {len(test_rows):,} rows)")
    print("Fold verification passed: chronological, no date shared between train and test, "
          "no test date reused.")


print("\n" + "=" * 60 + "\nTIME-SERIES CROSS-VALIDATION FOLDS (development data only)\n" + "=" * 60)
cv_folds = make_date_folds(dev_data["Date"], CV_SPLITS)
verify_date_folds(dev_data["Date"], cv_folds)
REPORT["cv_folds"] = [
    {"fold": i, "train_weeks": int(dev_data["Date"].iloc[tr].nunique()),
     "train_end": str(dev_data["Date"].iloc[tr].max().date()),
     "test_start": str(dev_data["Date"].iloc[te].min().date()),
     "test_end": str(dev_data["Date"].iloc[te].max().date()),
     "test_weeks": int(dev_data["Date"].iloc[te].nunique())}
    for i, (tr, te) in enumerate(cv_folds, start=1)
]


# =============================================================================
# 12. CROSS-VALIDATION AND HYPERPARAMETER TUNING (development data only)
# =============================================================================
def summarize_cv(mae, rmse, r2, fold_mae):
    return {"CV MAE": float(mae), "CV MAE std": float(np.std(fold_mae)), "CV RMSE": float(rmse),
            "CV R2": float(r2), "fold_mae": [float(v) for v in fold_mae]}


def cross_validate_fixed(pipeline):
    """CV for models without hyperparameters to tune (baseline, linear regression)."""
    res = cross_validate(pipeline, X_dev, y_dev, cv=cv_folds, scoring=SCORING, n_jobs=1)
    return summarize_cv(-res["test_mae"].mean(), -res["test_rmse"].mean(), res["test_r2"].mean(),
                        -res["test_mae"])


def run_search(name, search, csv_name):
    """Fit a GridSearchCV/RandomizedSearchCV on the development data and report it."""
    start = time.time()
    search.fit(X_dev, y_dev)
    results = pd.DataFrame(search.cv_results_)
    for col in ("mean_test_mae", "mean_test_rmse"):
        results[col] = -results[col]
    keep = [c for c in results.columns if c.startswith("param_")] + [
        "mean_test_mae", "std_test_mae", "mean_test_rmse", "mean_test_r2", "rank_test_mae"]
    results = results[keep].sort_values("rank_test_mae")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(REPORT_DIR / csv_name, index=False)
    idx = search.best_index_
    fold_mae = [-search.cv_results_[f"split{k}_test_mae"][idx] for k in range(len(cv_folds))]
    summary = summarize_cv(-search.cv_results_["mean_test_mae"][idx], -search.cv_results_["mean_test_rmse"][idx],
                           search.cv_results_["mean_test_r2"][idx], fold_mae)
    summary["best_params"] = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    summary["search_table"] = results
    print(f"\n{name}: {len(results)} candidates x {len(cv_folds)} folds in {time.time() - start:,.0f}s")
    print(results.head(8).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("Best parameters:", summary["best_params"], f"| CV MAE {summary['CV MAE']:,.2f}")
    return summary, search.best_estimator_


print("\n" + "=" * 60 + "\nMODEL SELECTION BY CROSS-VALIDATION (MAE is the primary metric)\n" + "=" * 60)
cv_results = {}   # model name -> CV summary
dev_models = {}   # model name -> pipeline fitted on the development data

# 12a. Mean baseline
baseline = build_pipeline(DummyRegressor(strategy="mean"))
cv_results["Mean Baseline"] = cross_validate_fixed(baseline)
dev_models["Mean Baseline"] = clone(baseline).fit(X_dev, y_dev)

# 12b. Linear Regression (scaled features, nothing to tune)
linear = build_pipeline(LinearRegression(), scale=True)
cv_results["Linear Regression"] = cross_validate_fixed(linear)
dev_models["Linear Regression"] = clone(linear).fit(X_dev, y_dev)

# 12c. Decision Tree: GridSearchCV (scoring on MAE, refit on the whole development set)
tree_pipeline = build_pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE))
dt_search = GridSearchCV(tree_pipeline, DT_PARAM_GRID, scoring=SCORING, refit="mae", cv=cv_folds,
                         n_jobs=1, verbose=0)
cv_results["Decision Tree (tuned)"], dev_models["Decision Tree (tuned)"] = run_search(
    "Decision Tree grid search", dt_search, "cv_results_decision_tree.csv")

# 12d. The ORIGINAL fixed Decision Tree (min_samples_leaf=10, unlimited depth) as a reference.
fixed_params = {"min_samples_leaf": TREE_MIN_SAMPLES_LEAF, "max_depth": None}
dt_table = cv_results["Decision Tree (tuned)"]["search_table"]
fixed_row = dt_table[(dt_table["param_model__min_samples_leaf"] == TREE_MIN_SAMPLES_LEAF)
                     & dt_table["param_model__max_depth"].isna()]
fixed_fold_mae = None
if not fixed_row.empty:  # that combination is part of the grid: reuse its CV scores
    fixed_cv = fixed_row.iloc[0]
    cv_results["Decision Tree (original, leaf=10)"] = summarize_cv(
        fixed_cv["mean_test_mae"], fixed_cv["mean_test_rmse"], fixed_cv["mean_test_r2"],
        [fixed_cv["mean_test_mae"]] * len(cv_folds))
    cv_results["Decision Tree (original, leaf=10)"]["CV MAE std"] = float(fixed_cv["std_test_mae"])
    cv_results["Decision Tree (original, leaf=10)"]["fold_mae"] = []
else:
    original_tree = build_pipeline(DecisionTreeRegressor(min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
                                                         random_state=RANDOM_STATE))
    cv_results["Decision Tree (original, leaf=10)"] = cross_validate_fixed(original_tree)
dev_models["Decision Tree (original, leaf=10)"] = build_pipeline(
    DecisionTreeRegressor(min_samples_leaf=TREE_MIN_SAMPLES_LEAF, random_state=RANDOM_STATE)
).fit(X_dev, y_dev)

# 12e. Random Forest: RandomizedSearchCV over a modest space
forest_pipeline = build_pipeline(RandomForestRegressor(
    n_estimators=RF_N_ESTIMATORS, max_samples=RF_MAX_SAMPLES, random_state=RANDOM_STATE, n_jobs=-1))
rf_search = RandomizedSearchCV(forest_pipeline, RF_PARAM_SPACE, n_iter=RF_N_ITER, scoring=SCORING,
                               refit="mae", cv=cv_folds, n_jobs=1, random_state=RANDOM_STATE, verbose=0)
cv_results["Random Forest (tuned)"], dev_models["Random Forest (tuned)"] = run_search(
    "Random Forest randomized search", rf_search, "cv_results_random_forest.csv")

# 12f. Model selection happens HERE, from CV results only (before any holdout score is computed).
CANDIDATES = ["Linear Regression", "Decision Tree (tuned)", "Random Forest (tuned)"]
best_name = min(CANDIDATES, key=lambda name: cv_results[name]["CV MAE"])
print(f"\nSELECTED MODEL (lowest mean CV MAE on development folds): {best_name} "
      f"[CV MAE {cv_results[best_name]['CV MAE']:,.2f}]")
REPORT["selected_model"] = best_name
REPORT["selected_cv"] = {k: v for k, v in cv_results[best_name].items() if k != "search_table"}

# =============================================================================
# 13. HOLDOUT EVALUATION (first time the holdout period is used)
# =============================================================================
print("\n" + "=" * 60 + "\nFINAL HOLDOUT EVALUATION (models fitted on development data only)\n" + "=" * 60)
MODEL_ORDER = ["Mean Baseline", "Linear Regression", "Decision Tree (original, leaf=10)",
               "Decision Tree (tuned)", "Random Forest (tuned)"]
rows, holdout_predictions = [], {}
for name in MODEL_ORDER:
    pipeline = dev_models[name]
    y_pred = pipeline.predict(X_holdout)
    holdout_predictions[name] = y_pred
    metrics = evaluate(y_holdout, y_pred)
    train_r2 = r2_score(y_dev, pipeline.predict(X_dev))  # kept from the original script (overfit check)
    print_metrics(f"{name} (holdout)", metrics)
    print(f"Training R2: {train_r2:.4f} | Holdout R2: {metrics['R2']:.4f} | gap: {train_r2 - metrics['R2']:.4f}")
    rows.append({
        "Model": name,
        "CV MAE": cv_results[name]["CV MAE"], "CV RMSE": cv_results[name]["CV RMSE"],
        "Holdout MAE": metrics["MAE"], "Holdout RMSE": metrics["RMSE"], "Holdout R2": metrics["R2"],
        "Train R2 (dev)": train_r2,
    })

comparison = pd.DataFrame(rows)
print("\nModel comparison (CV on development folds | metrics on the untouched holdout):")
print(comparison.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
REPORT["comparison"] = comparison
REPORT["best_params"] = cv_results[best_name].get("best_params", {})

best_metrics = evaluate(y_holdout, holdout_predictions[best_name])
REPORT["selected_holdout"] = best_metrics
baseline_mae = comparison.loc[comparison["Model"] == "Mean Baseline", "Holdout MAE"].iloc[0]
print(f"\nSelected model: {best_name} | holdout MAE {best_metrics['MAE']:,.2f} "
      f"({1 - best_metrics['MAE'] / baseline_mae:.1%} lower than the mean baseline)")

# Reference: does the original fixed tree reproduce the previously reported validation numbers?
REPORT["previous_reported"] = {"MAE": 1959.2630, "RMSE": 4132.2556, "R2": 0.9646}


# =============================================================================
# 14. ERROR ANALYSIS (selected model, holdout period)
# =============================================================================
def error_table(df, group_col):
    """MAE and relative error for each group."""
    grouped = df.groupby(group_col)
    table = pd.DataFrame({
        "Records": grouped.size(),
        "Mean actual": grouped["actual"].mean(),
        "MAE": grouped["abs_error"].mean(),
        "Bias (actual - predicted)": grouped["residual"].mean(),
    })
    table["MAE % of mean actual"] = 100 * table["MAE"] / table["Mean actual"].abs()
    return table.reset_index()


def run_error_analysis(model_name, y_pred):
    print("\n" + "=" * 60 + f"\nERROR ANALYSIS: {model_name} on the holdout period\n" + "=" * 60)
    err = holdout_data[["Store", "Dept", "Date", "IsHoliday", "month"]].copy()
    err["actual"], err["predicted"] = y_holdout.to_numpy(), y_pred
    err["residual"] = err["actual"] - err["predicted"]
    err["abs_error"] = err["residual"].abs()
    err["Week type"] = np.where(err["IsHoliday"] == 1, "Holiday week", "Regular week")

    by_store = error_table(err, "Store")
    by_dept = error_table(err, "Dept")
    by_month = error_table(err, "month")
    by_holiday = error_table(err, "Week type")
    by_store.to_csv(REPORT_DIR / "error_by_store.csv", index=False)
    by_dept.to_csv(REPORT_DIR / "error_by_dept.csv", index=False)
    by_month.to_csv(REPORT_DIR / "error_by_month.csv", index=False)
    by_holiday.to_csv(REPORT_DIR / "error_by_week_type.csv", index=False)

    # Aggregated view: errors partly cancel when summed to store-week totals (useful for planning).
    store_week = err.groupby(["Store", "Date"])[["actual", "predicted"]].sum()
    store_week_mae = (store_week["actual"] - store_week["predicted"]).abs().mean()
    store_week_pct = 100 * store_week_mae / store_week["actual"].mean()
    print(f"Record level (Store x Dept x Week): MAE ${err['abs_error'].mean():,.0f}")
    print(f"Store-week totals (sum over departments): MAE ${store_week_mae:,.0f} "
          f"({store_week_pct:.1f}% of mean store-week sales)")
    print("\nError by week type:\n", by_holiday.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nError by month:\n", by_month.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nWorst 5 stores by MAE:\n",
          by_store.sort_values("MAE", ascending=False).head(5).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    print("\nWorst 5 departments by MAE:\n",
          by_dept.sort_values("MAE", ascending=False).head(5).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    # 1. Actual vs predicted
    sample = err.sample(min(20000, len(err)), random_state=RANDOM_STATE)
    limit = float(max(err["actual"].max(), err["predicted"].max()))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(sample["actual"], sample["predicted"], s=4, alpha=0.25, color=BLUE)
    ax.plot([0, limit], [0, limit], color=YELLOW, linewidth=2, label="Perfect prediction")
    ax.set(title=f"Actual vs predicted weekly sales (holdout, {model_name})",
           xlabel="Actual Weekly_Sales (USD)", ylabel="Predicted Weekly_Sales (USD)")
    ax.legend()
    save_figure(fig, "08_actual_vs_predicted.png")

    # 2. Residual distribution
    fig, ax = plt.subplots(figsize=(7, 4.5))
    clipped = err["residual"].clip(err["residual"].quantile(0.005), err["residual"].quantile(0.995))
    ax.hist(clipped, bins=100, color=BLUE)
    ax.axvline(0, color=YELLOW, linewidth=2)
    ax.set(title="Residual distribution (actual - predicted, 0.5-99.5 percentile)",
           xlabel="Residual (USD)", ylabel="Records")
    save_figure(fig, "09_residual_distribution.png")

    # 3. Error by store
    ordered = by_store.sort_values("MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(ordered["Store"].astype(str), ordered["MAE"], color=BLUE)
    ax.set(title="Holdout MAE by store", xlabel="Store", ylabel="MAE (USD)")
    ax.tick_params(axis="x", labelsize=7)
    save_figure(fig, "10_error_by_store.png")

    # 4. Error by department (15 largest MAE)
    worst = by_dept.sort_values("MAE", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(worst["Dept"].astype(str), worst["MAE"], color=BLUE)
    ax.set(title="Holdout MAE by department (15 largest)", xlabel="MAE (USD)", ylabel="Department")
    save_figure(fig, "11_error_by_department.png")

    # 5. Error by month and week type
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].bar(by_month["month"].astype(str), by_month["MAE"], color=BLUE)
    axes[0].set(title="Holdout MAE by month", xlabel="Month", ylabel="MAE (USD)")
    axes[1].bar(by_holiday["Week type"], by_holiday["MAE"], color=[YELLOW, BLUE])
    axes[1].set(title="Holdout MAE by week type", ylabel="MAE (USD)")
    save_figure(fig, "12_error_by_period.png")

    REPORT["error_by_week_type"] = by_holiday
    REPORT["error_by_month"] = by_month
    REPORT["worst_stores"] = by_store.sort_values("MAE", ascending=False).head(5)
    REPORT["best_stores"] = by_store.sort_values("MAE").head(3)
    REPORT["worst_depts"] = by_dept.sort_values("MAE", ascending=False).head(5)
    REPORT["store_week"] = {"mae": float(store_week_mae), "pct": float(store_week_pct)}
    REPORT["holdout_mean_actual"] = float(err["actual"].mean())
    REPORT["holdout_median_abs_error"] = float(err["abs_error"].median())


run_error_analysis(best_name, holdout_predictions[best_name])


# =============================================================================
# 15. FEATURE-AVAILABILITY AUDIT: CPI AND UNEMPLOYMENT (still unresolved)
# =============================================================================
def audit_cpi_unemployment():
    """Collect what the data CAN say about CPI/Unemployment availability, plus a CV ablation."""
    print("\n" + "=" * 60 + "\nCPI / UNEMPLOYMENT AVAILABILITY AUDIT\n" + "=" * 60)
    notes = []
    last_cpi_date = features.loc[features["CPI"].notna(), "Date"].max()
    test_missing = float(test["CPI"].isna().mean())
    notes.append(f"features.csv has CPI/Unemployment values up to {last_cpi_date.date()}; "
                 f"{test_missing:.1%} of test.csv rows (later weeks) have none.")
    per_date_unique = features.dropna(subset=["CPI"]).groupby("Date")["CPI"].nunique()
    notes.append(f"CPI differs across stores at the same date (median {per_date_unique.median():.0f} distinct values "
                 f"across stores per week), so it is not a single national series.")
    monthly_unique = (features.dropna(subset=["CPI"]).assign(ym=lambda d: d["Date"].dt.to_period("M"))
                      .groupby(["Store", "ym"])["CPI"].nunique())
    notes.append(f"Within one store and calendar month CPI takes a median of {monthly_unique.median():.0f} distinct "
                 f"value(s) across {monthly_unique.groupby(level=0).size().median():.0f} months per store, "
                 f"so it changes gradually week to week.")
    notes.append("The dataset has NO publication/release timestamps. Whether the CPI and Unemployment value attached "
                 "to a week was actually known BEFORE that week begins CANNOT be verified from these files. "
                 "Official macro statistics are typically published after the period they describe, so the value "
                 "shown for a week may be later information than a true pre-week forecast would have. "
                 "STATUS: UNRESOLVED. The features are kept (project decision) but flagged.")

    # Ablation on the development folds only (no holdout use): how much do they matter?
    best_pipeline = clone(dev_models[best_name])
    no_macro = clone(best_pipeline).set_params(preprocess=build_preprocessor(
        scale=best_name == "Linear Regression", impute_cols=[]))
    X_dev_no_macro = X_dev.drop(columns=IMPUTE_COLS)
    res = cross_validate(no_macro, X_dev_no_macro, y_dev, cv=cv_folds, scoring=SCORING, n_jobs=1)
    mae_without = float(-res["test_mae"].mean())
    mae_with = cv_results[best_name]["CV MAE"]
    notes.append(f"Ablation ({best_name}, same CV folds): CV MAE {mae_with:,.2f} with CPI+Unemployment vs "
                 f"{mae_without:,.2f} without them ({(mae_without / mae_with - 1) * 100:+.1f}%).")
    for note in notes:
        print("-", note)
    REPORT["audit_notes"] = notes
    REPORT["ablation"] = {"with": mae_with, "without": mae_without}


audit_cpi_unemployment()


# =============================================================================
# 16. TRAIN THE FINAL DEPLOYMENT MODEL AND SAVE THE ARTIFACT
# =============================================================================
# Evaluation is finished. The selected pipeline (with its tuned hyperparameters) is now
# re-fitted on ALL labeled data (development + holdout); preprocessing is re-fitted too.
print("\n" + "=" * 60 + "\nFINAL DEPLOYMENT MODEL\n" + "=" * 60)
X_all, y_all = train[INPUT_COLUMNS], train[TARGET]
final_model = clone(dev_models[best_name]).fit(X_all, y_all)
print(f"Fitted '{best_name}' on {len(X_all):,} labeled rows "
      f"({train['Date'].min().date()} to {train['Date'].max().date()}).")

preprocess = final_model.named_steps["preprocess"]
store_categories = preprocess.named_transformers_["categorical"].categories_

artifacts = {
    "artifact_version": 2,
    "model": final_model,  # complete Pipeline: ColumnTransformer + estimator
    "feature_columns": INPUT_COLUMNS,  # RAW columns the pipeline expects (order matters)
    "model_name": best_name,
    "best_params": REPORT["best_params"],
    "cv_metrics": {k: v for k, v in cv_results[best_name].items()
                   if k in ("CV MAE", "CV MAE std", "CV RMSE", "CV R2")},
    "holdout_metrics": best_metrics,  # measured on the model fitted on development data only
    "baseline_holdout_mae": float(baseline_mae),
    "split_date": str(split_date.date()),
    "trained_date_range": (str(train["Date"].min().date()), str(train["Date"].max().date())),
    "n_train_rows": int(len(X_all)),
    "known_stores": [int(s) for s in store_categories[0]],
    "known_depts": [int(d) for d in store_categories[1]],
    "excluded_features": EXCLUDED_FEATURES,
    "unaudited_features": UNAUDITED_FEATURES,
    "random_state": RANDOM_STATE,
    "sklearn_version": sklearn.__version__,
    "created_at": datetime.now().isoformat(timespec="seconds"),
}
if any(c in INPUT_COLUMNS for c in EXCLUDED_FEATURES):
    raise ValueError("Excluded features found in feature_columns; refusing to save.")

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
backup_path = backup_existing_artifact(MODEL_PATH)
if backup_path is not None:
    print(f"Previous artifact preserved at {backup_path}")
joblib.dump(artifacts, MODEL_PATH, compress=3)  # compressed: forests can be large

# Verify that the saved artifact loads and reproduces the in-memory predictions.
reloaded = joblib.load(MODEL_PATH)
sample_rows = X_all.sample(200, random_state=RANDOM_STATE)
assert np.allclose(reloaded["model"].predict(sample_rows[reloaded["feature_columns"]]),
                   final_model.predict(sample_rows)), "Reloaded model predictions differ"
print(f"\nSaved artifacts to {MODEL_PATH}")
print("Saved keys:", list(artifacts.keys()))
print("Model:", best_name, "| pipeline steps:", [name for name, _ in final_model.steps])
print("Reload check passed (identical predictions on 200 sample rows).")

# Feature importance of the final model, grouped (tree-based models only).
estimator = final_model.named_steps["model"]
if hasattr(estimator, "feature_importances_"):
    names = preprocess.get_feature_names_out()
    importance = pd.Series(estimator.feature_importances_, index=names)
    grouped = importance.groupby(lambda n: "Store (one-hot)" if n.startswith("Store_")
                                 else "Dept (one-hot)" if n.startswith("Dept_") else n).sum()
    grouped = grouped.sort_values(ascending=False)
    print("\nFeature importance (grouped):")
    print(grouped.round(4).to_string())
    REPORT["importance"] = grouped.head(8)

REPORT["final"] = {"rows": len(X_all), "range": artifacts["trained_date_range"], "n_features_in": len(INPUT_COLUMNS),
                   "n_encoded_features": int(len(preprocess.get_feature_names_out()))}


# =============================================================================
# 17. WRITE reports/results_summary.md (all numbers come from this run)
# =============================================================================
def write_results_summary():
    sel, sc = REPORT["selected_model"], REPORT["selected_cv"]
    sp, hm = REPORT["split"], REPORT["selected_holdout"]
    lines = ["# Results summary (generated by Walmart_Sales_analyst.py)", "",
             f"Generated: {datetime.now():%Y-%m-%d %H:%M} | scikit-learn {sklearn.__version__} | "
             f"quick-test mode: {QUICK_TEST}", "",
             "## Data integrity checks", *[f"- {n}" for n in REPORT["integrity_notes"]], "",
             "## EDA insights", *[f"{i}. {t}" for i, t in enumerate(REPORT["eda_insights"], 1)], "",
             "## Outlier investigation", *[f"- {n}" for n in REPORT["outlier_notes"]], "",
             "## Splits", f"- Development: {sp['dev_range']} ({sp['dev_weeks']} weeks, {sp['dev_rows']:,} rows)",
             f"- Holdout (untouched until model selection): {sp['holdout_range']} ({sp['holdout_weeks']} weeks, "
             f"{sp['holdout_rows']:,} rows)",
             f"- Final model trained on all labeled data: {REPORT['final']['range'][0]} to {REPORT['final']['range'][1]} "
             f"({REPORT['final']['rows']:,} rows)", "",
             f"## Time-series CV folds ({CV_SPLITS}, built on unique dates)",
             dataframe_to_markdown(pd.DataFrame(REPORT["cv_folds"]), "{:,.2f}"), "",
             "## Model comparison", dataframe_to_markdown(REPORT["comparison"], "{:,.4f}"), "",
             f"Selected model (lowest mean CV MAE, chosen before looking at the holdout): **{sel}**",
             f"- Hyperparameters: `{REPORT['best_params']}`",
             f"- CV MAE {sc['CV MAE']:,.2f} (std across folds {sc['CV MAE std']:,.2f}); "
             f"holdout MAE {hm['MAE']:,.2f}, RMSE {hm['RMSE']:,.2f}, R2 {hm['R2']:.4f}",
             f"- Holdout mean weekly sales per record: ${REPORT['holdout_mean_actual']:,.0f}; "
             f"MAE = {100 * hm['MAE'] / REPORT['holdout_mean_actual']:.1f}% of it; median absolute error "
             f"${REPORT['holdout_median_abs_error']:,.0f}",
             f"- Store-week totals: MAE ${REPORT['store_week']['mae']:,.0f} "
             f"({REPORT['store_week']['pct']:.1f}% of mean store-week sales)", "",
             "## Error analysis (holdout)", "### By week type",
             dataframe_to_markdown(REPORT["error_by_week_type"], "{:,.2f}"), "### By month",
             dataframe_to_markdown(REPORT["error_by_month"], "{:,.2f}"), "### Worst 5 stores by MAE",
             dataframe_to_markdown(REPORT["worst_stores"], "{:,.2f}"), "### Worst 5 departments by MAE",
             dataframe_to_markdown(REPORT["worst_depts"], "{:,.2f}"), "",
             "## CPI / Unemployment availability audit (UNRESOLVED)", *[f"- {n}" for n in REPORT["audit_notes"]], ""]
    if "importance" in REPORT:
        lines += ["## Feature importance (final model, grouped)",
                  dataframe_to_markdown(REPORT["importance"].rename("importance").reset_index()
                                        .rename(columns={"index": "feature"}), "{:,.4f}"), ""]
    lines += [f"Total runtime: {(time.time() - SCRIPT_START) / 60:.1f} minutes."]
    (REPORT_DIR / "results_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nWrote", (REPORT_DIR / "results_summary.md").relative_to(BASE_DIR))


write_results_summary()
print(f"\nDone in {(time.time() - SCRIPT_START) / 60:.1f} minutes.")
