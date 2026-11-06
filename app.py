"""
Walmart Weekly Sales Forecasting — Streamlit dashboard
======================================================
Interactive front-end for the trained model saved in ``models/walmart_model.pkl``.

The artifact (version 2) holds a complete scikit-learn Pipeline, so inference is:

    1. store info (Size, Type -> Type_A/B/C via pd.get_dummies, as in training)
    2. year / month / week from the weekly reference date (ISO week, as in training)
    3. one raw input row with the saved ``feature_columns`` (Store, Dept, IsHoliday,
       CPI, Unemployment, Size, Type_A/B/C, year, month, week)
    4. pipeline.predict  (imputation, scaling if needed and one-hot encoding happen
       inside the saved pipeline using statistics learned at training time)

Temperature, Fuel_Price and MarkDown1-5 are NOT model inputs. Nothing is re-fitted here.

Run with:  python -m streamlit run app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from string import Template

import altair as alt
import joblib
import pandas as pd
import sklearn
import streamlit as st

# =============================================================================
# 1. CONFIGURATION
# =============================================================================
APP_DIR = Path(__file__).resolve().parent  # independent of the working directory
MODEL_PATH = APP_DIR / "models" / "walmart_model.pkl"
DATA_DIR = APP_DIR / "data"
CSV_ENCODING = "cp1252"  # same encoding used by the training script

ARTIFACT_KEYS = ("model", "feature_columns", "model_name", "holdout_metrics", "cv_metrics",
                 "baseline_holdout_mae", "split_date", "trained_date_range", "known_stores", "known_depts")
IMPUTE_COLS = ["CPI", "Unemployment"]

# Walmart sales weeks end on Friday. Verified on the data: every Date in
# train/test/features is a Friday, and each flagged holiday (Super Bowl, Labor
# Day, Thanksgiving, Christmas) falls in the 6 days *before* the flagged Friday,
# i.e. the week runs Saturday -> Friday and the Friday is its closing date.
WEEK_ENDING_WEEKDAY = 4  # Monday = 0 ... Friday = 4

# Selectable calendar = the weeks covered by the project's data files.
FIRST_SELECTABLE = date(2010, 1, 30)  # Saturday starting the first week (2010-02-05)
LAST_SELECTABLE = date(2013, 7, 26)  # last week in features.csv / test.csv
DEFAULT_DATE = date(2012, 11, 2)  # first week of the test (forecast) horizon

# Walmart-inspired palette.
BLUE, BLUE_DARK, NAVY, YELLOW = "#0071CE", "#004C91", "#041E42", "#FFC220"
TEXT, MUTED, BORDER, BG = "#1E293B", "#64748B", "#E2E8F0", "#F4F6F9"


# =============================================================================
# 2. DATE LOGIC (weekly reference date and model date features)
# =============================================================================
def week_ending_friday(selected: date) -> date:
    """Return the Friday that closes the Saturday-Friday sales week of ``selected``."""
    return selected + timedelta(days=(WEEK_ENDING_WEEKDAY - selected.weekday()) % 7)


def week_start(reference: date) -> date:
    """First day (Saturday) of the sales week closing on ``reference``."""
    return reference - timedelta(days=6)


def date_features(reference: date) -> dict[str, int]:
    """year / month / week exactly as built in training (dt.year, dt.month, ISO week)."""
    return {
        "year": reference.year,
        "month": reference.month,
        "week": int(reference.isocalendar()[1]),
    }


def trained_end_week(artifacts: dict) -> date:
    return date.fromisoformat(artifacts["trained_date_range"][1])


def period_label(reference: date, artifacts: dict) -> str:
    """Describe where a week sits relative to the deployed model's training data."""
    end = trained_end_week(artifacts)
    if reference <= end:
        return (f"Training data: the deployed model was refitted on all labeled weeks (through {end:%Y-%m-%d}), "
                "so this is not an out-of-sample forecast. Its accuracy is measured on the held-out weeks shown below.")
    return f"Forecast horizon: after the last week of recorded sales ({end:%Y-%m-%d}); no actual sales exist to compare with."


def fmt_day(d: date) -> str:
    return f"{d:%a} {d.day} {d:%b %Y}"


def money(value: float) -> str:
    return f"{'-' if value < 0 else ''}${abs(value):,.0f}"


# =============================================================================
# 3. DATA AND MODEL LOADING (cached)
# =============================================================================
def _read_csv(name: str, **kwargs) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    return pd.read_csv(path, encoding=CSV_ENCODING, **kwargs)


@st.cache_resource(show_spinner="Loading trained model…")
def load_artifacts() -> dict:
    """Load the saved pipeline artifact and check that it has the expected structure."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    artifacts = joblib.load(MODEL_PATH)
    if artifacts.get("artifact_version", 1) < 2:
        raise ValueError("models/walmart_model.pkl is an old-format artifact. "
                         "Run Walmart_Sales_analyst.py to create the pipeline artifact.")
    missing = [key for key in ARTIFACT_KEYS if key not in artifacts]
    if missing:
        raise KeyError(f"Model artifact is missing expected keys: {missing}")
    return artifacts


@st.cache_data(show_spinner="Loading datasets…")
def load_stores() -> pd.DataFrame:
    return _read_csv("stores.csv")


@st.cache_data(show_spinner=False)
def load_features() -> pd.DataFrame:
    df = _read_csv("features.csv", usecols=["Store", "Date", "IsHoliday", "CPI", "Unemployment"])
    df["Date"] = pd.to_datetime(df["Date"])
    df["IsHoliday"] = df["IsHoliday"].astype(int)
    return df


@st.cache_data(show_spinner=False)
def load_train() -> pd.DataFrame:
    df = _read_csv("train.csv", usecols=["Store", "Dept", "Date", "Weekly_Sales"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data(show_spinner=False)
def store_departments() -> dict[int, list[int]]:
    """Departments that have recorded sales, per store."""
    train = load_train()
    return {
        int(store): sorted(int(d) for d in depts)
        for store, depts in train.groupby("Store")["Dept"].unique().items()
    }


@st.cache_data(show_spinner=False)
def load_history(store: int, dept: int) -> pd.DataFrame:
    """Actual weekly sales of one Store x Department, sorted chronologically."""
    train = load_train()
    mask = (train["Store"] == store) & (train["Dept"] == dept)
    return train.loc[mask, ["Date", "Weekly_Sales"]].sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def holiday_weeks() -> set[date]:
    """Weekly reference dates flagged as holiday weeks in features.csv."""
    features = load_features()
    return set(features.loc[features["IsHoliday"] == 1, "Date"].dt.date)


@dataclass(frozen=True)
class Economics:
    """CPI / Unemployment for one store-week, plus whether the saved imputer fills them."""
    cpi_raw: float  # as recorded; NaN when missing (the saved imputer then fills it)
    unemployment_raw: float
    cpi: float  # value the model actually uses (after imputation)
    unemployment: float
    imputed: bool


def fitted_imputer_medians(artifacts: dict):
    """Medians learned by the SimpleImputer inside the saved pipeline (CPI, Unemployment)."""
    preprocess = artifacts["model"].named_steps["preprocess"]
    return preprocess.named_transformers_["impute"].named_steps["imputer"].statistics_


def economic_indicators(artifacts: dict, store: int, reference: date) -> Economics:
    """Read CPI / Unemployment from features.csv for this store and week.

    Missing values are left as NaN so the imputer inside the saved pipeline fills them,
    exactly as predict.py does for the test set (some 2013 weeks have no CPI/Unemployment).
    """
    features = load_features()
    row = features[(features["Store"] == store) & (features["Date"] == pd.Timestamp(reference))]
    raw = [float(row.iloc[0][col]) if not row.empty else float("nan") for col in IMPUTE_COLS]
    medians = fitted_imputer_medians(artifacts)
    used = [float(m) if pd.isna(r) else r for r, m in zip(raw, medians)]
    return Economics(raw[0], raw[1], used[0], used[1], imputed=any(pd.isna(r) for r in raw))


# =============================================================================
# 4. PREDICTION PIPELINE (mirrors predict.py)
# =============================================================================
def build_model_input(
    artifacts: dict,
    stores: pd.DataFrame,
    *,
    store: int,
    dept: int,
    reference: date,
    is_holiday: bool,
    cpi: float,
    unemployment: float,
) -> pd.DataFrame:
    """Build one raw input row with exactly the columns (and order) the pipeline was trained on."""
    # The saved one-hot encoder silently ignores unknown categories; refuse them instead.
    if store not in artifacts["known_stores"] or dept not in artifacts["known_depts"]:
        raise ValueError(f"Store {store} / Dept {dept} was not seen during training.")

    # Store Size and Type_A/B/C, built the same way as in training.
    stores_encoded = pd.get_dummies(stores, columns=["Type"], dtype=int)
    store_row = stores_encoded[stores_encoded["Store"] == store]
    if store_row.empty:
        raise ValueError(f"Store {store} not found in stores.csv.")

    row = {
        "Store": store,
        "Dept": dept,
        "IsHoliday": int(is_holiday),
        "CPI": float(cpi),  # NaN is allowed: the pipeline's imputer fills it
        "Unemployment": float(unemployment),
        **date_features(reference),
        **store_row.drop(columns="Store").iloc[0].to_dict(),  # Size, Type_A, Type_B, Type_C
    }
    feature_columns = list(artifacts["feature_columns"])
    missing, extra = set(feature_columns) - set(row), set(row) - set(feature_columns)
    if missing or extra:
        raise ValueError(f"Feature mismatch! Missing: {sorted(missing)}, Extra: {sorted(extra)}")
    return pd.DataFrame([row])[feature_columns]


def predict_weekly_sales(artifacts: dict, stores: pd.DataFrame, **inputs) -> float:
    X = build_model_input(artifacts, stores, **inputs)
    return float(artifacts["model"].predict(X)[0])


# =============================================================================
# 5. RESULT MODEL
# =============================================================================
@dataclass(frozen=True)
class Forecast:
    store: int
    dept: int
    reference: date
    is_holiday: bool
    cpi: float
    unemployment: float
    prediction: float
    store_type: str
    store_size: int
    benchmark: float | None
    history_weeks: int
    indicators_imputed: bool
    period_note: str


def run_forecast(artifacts, stores, *, store, dept, reference, is_holiday, econ: Economics) -> Forecast:
    prediction = predict_weekly_sales(
        artifacts, stores, store=store, dept=dept, reference=reference, is_holiday=is_holiday,
        cpi=econ.cpi_raw, unemployment=econ.unemployment_raw,  # NaN -> saved imputer
    )
    history = load_history(store, dept)
    info = stores[stores["Store"] == store].iloc[0]
    return Forecast(
        store=store, dept=dept, reference=reference, is_holiday=is_holiday,
        cpi=econ.cpi, unemployment=econ.unemployment, prediction=prediction,
        store_type=str(info["Type"]), store_size=int(info["Size"]),
        benchmark=float(history["Weekly_Sales"].mean()) if not history.empty else None,
        history_weeks=len(history), indicators_imputed=econ.imputed,
        period_note=period_label(reference, artifacts),
    )


# =============================================================================
# 6. STYLING
# =============================================================================
CSS = Template("""
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.stApp { background: $bg; color: $text; font-family: "Inter","Segoe UI",-apple-system,Roboto,Helvetica,Arial,sans-serif; }
.block-container { max-width: 1240px; padding: 1.6rem 2rem 3rem; }
.stApp h1, .stApp h2, .stApp h3 { color: $navy; letter-spacing: -0.01em; }

/* ---- Light theme enforced for every widget, whatever the browser/OS theme ----
   Streamlit paints widgets from the visitor's theme (dark text-on-light would
   otherwise become white-on-light), so colors are set explicitly below. */
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li, [data-testid="stMarkdownContainer"] strong { color: $text; }
[data-testid="stMarkdownContainer"] code { background: #EEF2F6; color: $navy; }
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stCheckbox"] label, [data-testid="stCheckbox"] p { color: $text !important; font-weight: 500; font-size: .86rem; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: $muted !important; opacity: 1 !important; }

/* Input surfaces */
[data-testid="stSelectbox"] [role="group"], [data-testid="stNumberInputContainer"], [data-testid="stDateInputField"] {
  background: #fff !important; border: 1px solid #CBD5E1 !important; border-radius: 8px !important; }
[data-testid="stSelectbox"] [role="group"] *, [data-testid="stNumberInputContainer"] *, [data-testid="stDateInputField"] * {
  color: $text !important; -webkit-text-fill-color: $text; }
[data-testid="stDateInputField"] input, [data-testid="stNumberInputField"] { background: transparent !important; }
[data-testid="stSelectbox"] [role="group"] svg { fill: $text !important; }
[data-testid="stCheckbox"] label:has(input:not(:checked)) > div:first-of-type { background: #fff !important; border: 1.5px solid #94A3B8 !important; }
[data-testid="stTooltipIcon"] svg, [data-testid="stTooltipIcon"] svg * { stroke: $muted !important; fill: none !important; opacity: 1 !important; }

/* Dropdown list and calendar popup */
[data-testid="stSelectboxVirtualDropdown"], [data-testid="stSelectboxVirtualDropdown"] ul { background: #fff !important; }
[data-testid="stSelectboxVirtualDropdown"] * { color: $text !important; }
[data-testid="stSelectboxVirtualDropdown"] [role="option"]:hover, [data-testid="stSelectboxVirtualDropdown"] [aria-selected="true"] { background: #EAF3FC !important; }

/* Cards (keyed containers) */
.st-key-chart_card { margin-top: .9rem; }
.st-key-config_card, .st-key-chart_card { background: #fff; border: 1px solid $border; border-radius: 12px; padding: 1.1rem 1.2rem; }
[data-testid="stExpander"] { background: #fff; border: 1px solid $border; border-radius: 12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary:hover { background: #F1F5F9 !important; border-radius: 12px; }
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * { color: $navy !important; font-weight: 600; }

/* Read-only indicators */
.sublabel { font-size: .86rem; font-weight: 500; color: $text; margin: .5rem 0 .4rem; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: .6rem; }

/* Header */
.hero { background: $navy; border-radius: 14px; padding: 1.5rem 1.75rem; margin-bottom: 1.4rem; border-bottom: 4px solid $yellow; }
.hero h1 { color: #fff !important; font-size: 1.85rem; margin: 0 0 .35rem; }
.hero p { color: #C9D6E8; margin: 0 0 .9rem; max-width: 760px; font-size: .95rem; line-height: 1.5; }
.pill { display: inline-flex; align-items: center; gap: .5rem; background: rgba(255,255,255,.1); color: #fff;
  border: 1px solid rgba(255,255,255,.25); border-radius: 999px; padding: .25rem .8rem; font-size: .78rem; }
.pill i { width: 8px; height: 8px; border-radius: 50%; background: $yellow; display: inline-block; }

/* Section titles */
.section { font-size: 1.05rem; font-weight: 700; color: $navy; margin: 0 0 .7rem; }
.section small { font-weight: 400; color: $muted; font-size: .82rem; margin-left: .4rem; }

/* Week preview */
.weekbox { background: #EAF3FC; border-left: 4px solid $blue; border-radius: 8px; padding: .6rem .8rem; margin: .1rem 0 .9rem; font-size: .83rem; color: $text; line-height: 1.5; }
.weekbox b { color: $navy; }

/* KPI cards */
.kpi-main { background: $blue; color: #fff; border-radius: 12px; padding: 1.3rem 1.5rem; margin-bottom: .9rem; border-bottom: 4px solid $yellow; }
.kpi-main .label { font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .85; }
.kpi-main .value { font-size: 2.7rem; font-weight: 750; line-height: 1.15; margin: .2rem 0; }
.kpi-main .sub { font-size: .87rem; opacity: .92; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: .75rem; margin-bottom: .6rem; }
.card { background: #fff; border: 1px solid $border; border-radius: 10px; padding: .85rem 1rem; }
.card .label { font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: $muted; }
.card .value { font-size: 1.3rem; font-weight: 700; color: $navy; margin-top: .15rem; }
.card .note { font-size: .76rem; color: $muted; margin-top: .2rem; line-height: 1.4; }
.tag { display: inline-block; font-size: .78rem; font-weight: 600; padding: .1rem .55rem; border-radius: 999px; }
.tag.up { background: #E3F4EA; color: #1B7F43; } .tag.down { background: #FDE9E7; color: #B42318; }
.tag.holiday { background: #FFF4D6; color: #8A6100; } .tag.regular { background: #EEF2F6; color: $muted; }
</style>
""").substitute(
    bg=BG, text=TEXT, navy=NAVY, muted=MUTED, border=BORDER,
    blue=BLUE, blue_dark=BLUE_DARK, yellow=YELLOW,
)


def card(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="note">{note}</div>' if note else ""
    return f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div>{note_html}</div>'


# =============================================================================
# 7. UI SECTIONS
# =============================================================================
def render_header(artifacts: dict) -> None:
    model_name = artifacts["model_name"]
    n_features = len(artifacts["feature_columns"])
    st.markdown(
        f"""
        <div class="hero">
          <h1>Walmart Sales Forecasting</h1>
          <p>Predict weekly sales for any Store × Department using a machine learning model
          trained on historical Walmart sales, then compare the result with recorded history.</p>
          <span class="pill"><i></i>Trained model loaded · {model_name} · {n_features} input features</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_config(artifacts: dict) -> dict:
    """Draw the configuration panel and return the current selections."""
    st.markdown('<div class="section">Forecast configuration</div>', unsafe_allow_html=True)
    departments = store_departments()

    with st.container(key="config_card"):
        store = st.selectbox("Store", sorted(departments), format_func=lambda s: f"Store {s}")
        dept = st.selectbox(
            "Department", departments[store], format_func=lambda d: f"Dept {d}", key=f"dept_{store}",
            help="Only departments with recorded sales in the selected store are listed.",
        )

        selected = st.date_input(
            "Forecast date", value=DEFAULT_DATE, min_value=FIRST_SELECTABLE, max_value=LAST_SELECTABLE,
            help="Any day of the sales week works; it is mapped to that week's closing Friday.",
        )
        reference = week_ending_friday(selected)
        features = date_features(reference)
        st.markdown(
            f'<div class="weekbox"><b>Sales week:</b> {fmt_day(week_start(reference))} → {fmt_day(reference)}<br>'
            f'<b>Weekly reference date:</b> {reference:%Y-%m-%d} (Friday) · '
            f'year {features["year"]}, month {features["month"]}, ISO week {features["week"]}</div>',
            unsafe_allow_html=True,
        )

        flagged = reference in holiday_weeks()
        is_holiday = st.checkbox("Holiday week", value=flagged, key=f"holiday_{reference}")
        if is_holiday != flagged:
            st.caption(f"Differs from the dataset's holiday calendar, which marks this week as "
                       f"{'a holiday' if flagged else 'a regular'} week.")

        # CPI / Unemployment are model inputs but not user inputs: read-only, taken from the data.
        econ = economic_indicators(artifacts, store, reference)
        st.markdown(
            '<div class="sublabel">Economic indicators (automatic)</div>'
            f'<div class="grid2">{card("CPI", f"{econ.cpi:,.3f}")}'
            f'{card("Unemployment", f"{econ.unemployment:.3f}%")}</div>',
            unsafe_allow_html=True,
        )
        st.caption("Read from the dataset for this store and week."
                   + (" Not recorded for this week, so the model's saved median values are used."
                      if econ.imputed else ""))

    return dict(store=store, dept=dept, reference=reference, is_holiday=is_holiday, econ=econ)


def render_results(result: Forecast) -> None:
    st.markdown('<div class="section">Forecast result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="kpi-main">
          <div class="label">Predicted weekly sales</div>
          <div class="value">{money(result.prediction)}</div>
          <div class="sub">Store {result.store} · Dept {result.dept} · week ending {fmt_day(result.reference)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    imputed_note = "Saved median (not recorded)" if result.indicators_imputed else "From the dataset"
    status = '<span class="tag holiday">Holiday week</span>' if result.is_holiday else '<span class="tag regular">Regular week</span>'
    cards = [
        card("Store", f"Store {result.store}", f"Type {result.store_type} · {result.store_size:,} sq ft"),
        card("Department", f"Dept {result.dept}"),
        card("Forecast week", f"{result.reference:%d %b %Y}",
             f"{fmt_day(week_start(result.reference))} → {fmt_day(result.reference)}"),
        card("Week status", status),
        card("CPI", f"{result.cpi:,.2f}", imputed_note),
        card("Unemployment", f"{result.unemployment:.2f}%", imputed_note),
    ]
    if result.benchmark is not None:
        delta_note = ""
        if abs(result.benchmark) >= 1:
            pct = (result.prediction / result.benchmark - 1) * 100
            tag = "up" if pct >= 0 else "down"
            delta_note = f'<span class="tag {tag}">{pct:+.1f}% vs benchmark</span><br>'
        cards.append(card(
            "Historical average (benchmark)", money(result.benchmark),
            f"{delta_note}Mean of {result.history_weeks} recorded weeks for this Store × Dept. "
            "Not last week's sales and not a guaranteed expected value.",
        ))
    st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.caption(result.period_note)


def render_history(store: int, dept: int) -> None:
    history = load_history(store, dept)
    st.markdown(
        f'<div class="section">Historical sales<small>Actual records from train.csv · '
        f"Store {store} · Dept {dept}</small></div>",
        unsafe_allow_html=True,
    )
    if history.empty:
        st.info("No historical sales are recorded for this Store and Department.")
        return

    sales = history["Weekly_Sales"]
    cols = st.columns(4)
    stats = [
        ("Historical average", money(sales.mean())),
        ("Maximum weekly sales", money(sales.max())),
        ("Minimum weekly sales", money(sales.min())),
        ("Recorded weeks", f"{len(sales):,}"),
    ]
    for col, (label, value) in zip(cols, stats):
        col.markdown(card(label, value), unsafe_allow_html=True)

    line = alt.Chart(history).mark_line(
        color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(color=BLUE, size=22)
    ).encode(
        x=alt.X("Date:T", title="Week ending (Friday)", axis=alt.Axis(format="%b %Y", labelAngle=0, grid=False)),
        y=alt.Y("Weekly_Sales:Q", title="Weekly sales (USD)", axis=alt.Axis(format="$,.0f")),
        tooltip=[alt.Tooltip("Date:T", title="Week ending", format="%a %d %b %Y"),
                 alt.Tooltip("Weekly_Sales:Q", title="Weekly sales", format="$,.2f")],
    )
    average = alt.Chart(pd.DataFrame({"avg": [float(sales.mean())]})).mark_rule(
        color="#B8860B", strokeDash=[6, 4], strokeWidth=1.5
    ).encode(y="avg:Q")
    chart = (line + average).properties(
        title=alt.TitleParams(f"Weekly sales — Store {store}, Department {dept}  (dashed line: historical average)",
                              anchor="start", color=NAVY, fontSize=14, fontWeight=600),
        width="container", height=340,
    ).configure(
        background="#FFFFFF",
        view={"stroke": None},
        axis={"labelColor": MUTED, "titleColor": TEXT, "gridColor": "#EEF2F6",
              "domainColor": BORDER, "tickColor": BORDER},
    )
    with st.container(key="chart_card"):
        st.altair_chart(chart, theme=None)


@st.cache_data(show_spinner=False)
def holdout_mean_sales(split_date: str) -> float:
    """Mean weekly sales per record in the held-out evaluation weeks (for relative error)."""
    train = load_train()
    return float(train.loc[train["Date"] >= pd.Timestamp(split_date), "Weekly_Sales"].mean())


def render_model_quality(artifacts: dict) -> None:
    """Show the evaluation metrics measured on the held-out weeks."""
    hm, cv = artifacts["holdout_metrics"], artifacts["cv_metrics"]
    mean_sales = holdout_mean_sales(artifacts["split_date"])
    mae_gain = 1 - hm["MAE"] / artifacts["baseline_holdout_mae"]
    st.markdown('<div class="section">Model quality<small>measured on held-out weeks</small></div>',
                unsafe_allow_html=True)
    cards = [
        card("MAE", money(hm["MAE"]), f"≈ {100 * hm['MAE'] / mean_sales:.1f}% of mean weekly sales ({money(mean_sales)})"),
        card("RMSE", money(hm["RMSE"]), "Penalises large misses more than MAE"),
        card("R²", f"{hm['R2']:.4f}", "Share of variance explained"),
        card("vs mean baseline", f"{mae_gain:.0%} lower MAE", f"Baseline MAE {money(artifacts['baseline_holdout_mae'])}"),
    ]
    st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    st.caption(
        f"Evaluated on weeks from {artifacts['split_date']} with the model fitted on earlier weeks only; "
        f"mean cross-validation MAE on the earlier weeks was {money(cv['CV MAE'])}. These are validation results, "
        "not a guarantee of future accuracy.")


def render_model_details(artifacts: dict) -> None:
    with st.expander("Model details and limitations"):
        params = ", ".join(f"{k}={v}" for k, v in artifacts.get("best_params", {}).items()) or "defaults"
        start, end = artifacts["trained_date_range"]
        st.markdown(
            f"""
- **Model:** {artifacts['model_name']} (`{params}`), trained on {artifacts['n_train_rows']:,} labeled rows ({start} to {end}).
  Inputs: Store, Department, holiday flag, CPI, unemployment, store size/type, year, month and ISO week.
- **Excluded inputs:** Temperature, Fuel Price and all MarkDown features.
- **Evaluation design:** hyperparameters were tuned with time-series cross-validation on the earlier 80% of weeks;
  the metrics above come from the later, untouched 20%. The deployed model was then refitted on all labeled weeks, so no
  untouched data remains to measure it directly.
- **Weekly convention:** Walmart weeks close on Friday (Saturday → Friday). Any selected day is mapped to that closing
  Friday before the year, month and ISO week features are generated.
- **Unresolved feature availability:** CPI and unemployment values come from the dataset. Whether they would truly be known
  before the forecast week is **not verified**, so treat pre-week use with caution.
- **No sales history as input:** the model does not see last week's sales; it learns store/department level, seasonality
  (ISO week) and a few economic and holiday signals. The year feature cannot extrapolate beyond the years it was trained on.
- **Holiday effect:** estimated from only a few holiday weeks in the data, so it is weakly supported.
- **Not production-validated:** no deployment, monitoring or prospective test has been performed.
            """
        )


# =============================================================================
# 8. MAIN
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="Walmart Sales Forecasting", page_icon="📈", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    try:
        artifacts = load_artifacts()
        stores = load_stores()
        load_features()
        store_departments()
    except (FileNotFoundError, KeyError, ValueError) as exc:
        st.error(f"Could not start the application: {exc}")
        st.stop()
    except Exception as exc:  # corrupted pickle, incompatible library version, ...
        st.error(f"Unexpected error while loading the model or data: {exc}")
        st.stop()

    render_header(artifacts)
    config_col, result_col = st.columns([1, 1.65], gap="large")

    with config_col:
        inputs = render_config(artifacts)

    # The forecast follows the selections live (one fast model call per change).
    with result_col:
        try:
            render_results(run_forecast(artifacts, stores, **inputs))
            render_model_quality(artifacts)
        except Exception as exc:
            st.error(f"The prediction could not be generated: {exc}")

    st.write("")
    try:
        render_history(inputs["store"], inputs["dept"])
    except Exception as exc:
        st.error(f"Historical data could not be displayed: {exc}")
    render_model_details(artifacts)


if __name__ == "__main__":
    main()
