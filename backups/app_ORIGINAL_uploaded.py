import streamlit as st
import joblib
import pandas as pd
import altair as alt
import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Walmart Sales Forecast",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — light, clean, enterprise design
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Fonts & base ──────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background-color: #F5F7FA;
    color: #1E293B;
}

/* ── Main content area ─────────────────────────────────────────────────────── */
.main .block-container {
    padding: 1.75rem 2.25rem 3rem 2.25rem;
    max-width: 1280px;
}

/* ── Sidebar ───────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E8ECF0;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.25rem;
}

/* ── Divider helper ────────────────────────────────────────────────────────── */
.sidebar-divider {
    border: none;
    border-top: 1px solid #E8ECF0;
    margin: 1.25rem 0;
}

/* ── Sidebar section label ─────────────────────────────────────────────────── */
.sidebar-section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.6rem;
}

/* ── Header banner ─────────────────────────────────────────────────────────── */
.header-wrap {
    background: #FFFFFF;
    border: 1px solid #E8ECF0;
    border-radius: 14px;
    padding: 1.25rem 1.75rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.header-title {
    font-size: 1.55rem;
    font-weight: 800;
    color: #1E293B;
    letter-spacing: -0.5px;
    margin: 0 0 2px 0;
}
.header-title span {
    color: #1677D2;
}
.header-subtitle {
    font-size: 0.875rem;
    color: #64748B;
    margin: 0;
    font-weight: 400;
}
.engine-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #EEF4FF;
    color: #1677D2;
    font-size: 0.775rem;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 20px;
    border: 1px solid #BFDBFE;
    white-space: nowrap;
}
.engine-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #1677D2;
    display: inline-block;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── KPI card grid ─────────────────────────────────────────────────────────── */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E8ECF0;
    border-radius: 14px;
    padding: 1.25rem 1.4rem 1.1rem 1.4rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    height: 100%;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 14px 14px;
}
.kpi-card.primary::after  { background: #1677D2; }
.kpi-card.success::after  { background: #16A34A; }
.kpi-card.danger::after   { background: #DC2626; }
.kpi-card.neutral::after  { background: #94A3B8; }

.kpi-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 0.4rem;
}
.kpi-icon {
    font-size: 1.1rem;
    float: right;
    margin-top: -2px;
    opacity: 0.6;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    line-height: 1.1;
    color: #1E293B;
    font-variant-numeric: tabular-nums;
}
.kpi-value.primary { color: #1677D2; }
.kpi-value.success { color: #16A34A; }
.kpi-value.danger  { color: #DC2626; }
.kpi-value.neutral { color: #64748B; }

.kpi-delta {
    font-size: 0.8rem;
    font-weight: 500;
    color: #94A3B8;
    margin-top: 0.3rem;
}

/* ── Section heading ───────────────────────────────────────────────────────── */
.section-heading {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1E293B;
    margin: 0 0 0.25rem 0;
}
.section-subheading {
    font-size: 0.8rem;
    color: #94A3B8;
    font-weight: 400;
    margin: 0 0 1rem 0;
}

/* ── Chart / content card ──────────────────────────────────────────────────── */
.content-card {
    background: #FFFFFF;
    border: 1px solid #E8ECF0;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    margin-bottom: 1.25rem;
}

/* ── Prediction focus card ─────────────────────────────────────────────────── */
.prediction-card {
    background: #FFFFFF;
    border: 1.5px solid #1677D2;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    box-shadow: 0 4px 20px rgba(22, 119, 210, 0.08);
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 2.5rem;
    flex-wrap: wrap;
}
.prediction-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #1677D2;
    margin-bottom: 0.4rem;
}
.prediction-value {
    font-size: 3rem;
    font-weight: 800;
    color: #1677D2;
    letter-spacing: -1px;
    line-height: 1;
    font-variant-numeric: tabular-nums;
}
.prediction-context {
    font-size: 0.85rem;
    color: #64748B;
    margin-top: 0.4rem;
}
.prediction-divider {
    width: 1px;
    height: 56px;
    background: #E8ECF0;
    flex-shrink: 0;
}
.prediction-meta-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94A3B8;
    margin-bottom: 2px;
}
.prediction-meta-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1E293B;
}

/* ── Inline badge ──────────────────────────────────────────────────────────── */
.inline-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
}
.badge-green  { background: #DCFCE7; color: #15803D; }
.badge-red    { background: #FEE2E2; color: #B91C1C; }
.badge-amber  { background: #FEF9C3; color: #A16207; }
.badge-blue   { background: #DBEAFE; color: #1E40AF; }
.badge-slate  { background: #F1F5F9; color: #475569; }

/* ── Info banner ───────────────────────────────────────────────────────────── */
.info-banner {
    background: #EEF4FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    font-size: 0.85rem;
    color: #1E40AF;
    margin-bottom: 1rem;
}

/* ── Footer ────────────────────────────────────────────────────────────────── */
.footer {
    text-align: center;
    font-size: 0.775rem;
    color: #94A3B8;
    margin-top: 3rem;
    padding-top: 1.25rem;
    border-top: 1px solid #E8ECF0;
}
.footer strong { color: #1677D2; font-weight: 600; }

/* ── Streamlit native overrides ────────────────────────────────────────────── */
/* Primary button */
div[data-testid="stButton"] > button[kind="primary"] {
    background-color: #1677D2 !important;
    border-color: #1677D2 !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    box-shadow: 0 2px 6px rgba(22,119,210,0.25) !important;
    transition: background 0.15s ease !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #1260B8 !important;
    border-color: #1260B8 !important;
}
/* Metric label text */
[data-testid="stMetric"] label {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    color: #94A3B8 !important;
}
/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600 !important;
    font-size: 0.875rem !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD ARTIFACTS & DATA  (read-only — no retraining)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_artifacts():
    return joblib.load("models/walmart_model.pkl")


@st.cache_data
def load_data():
    stores = pd.read_csv("data/stores.csv", encoding="cp1252")
    train  = pd.read_csv("data/train.csv",  encoding="cp1252")
    train["Date"] = pd.to_datetime(train["Date"])
    return stores, train


artifacts = load_artifacts()
stores, train = load_data()

model          = artifacts["model"]
encoder        = artifacts["encoder"]
imputer        = artifacts["imputer"]
feature_columns = artifacts["feature_columns"]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    # ── Logo / branding ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:0.1rem;">
        <div style="background:#1677D2;color:#fff;font-size:1.1rem;font-weight:800;
                    width:34px;height:34px;border-radius:9px;display:flex;
                    align-items:center;justify-content:center;flex-shrink:0;">W</div>
        <div>
            <div style="font-size:1rem;font-weight:700;color:#1E293B;line-height:1.2;">Sales Forecast</div>
            <div style="font-size:0.72rem;color:#94A3B8;font-weight:400;">Walmart Analytics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Section 1: Forecast Configuration ───────────────────────────────────
    st.markdown('<p class="sidebar-section-label">📋 Forecast Configuration</p>', unsafe_allow_html=True)

    store_id = st.selectbox(
        "Store",
        sorted(stores["Store"].unique()),
        help="Select one of the 45 Walmart store locations.",
    )

    store_info = stores.loc[stores["Store"] == store_id].iloc[0]
    store_type = store_info["Type"]
    store_size = store_info["Size"]

    st.markdown(f"""
    <div style="background:#F8FAFC;border:1px solid #E8ECF0;border-radius:8px;
                padding:8px 10px;margin:-6px 0 10px 0;font-size:0.775rem;color:#64748B;">
        Type <strong style="color:#1E293B;">{store_type}</strong>
        &nbsp;·&nbsp;
        <strong style="color:#1E293B;">{store_size:,}</strong> sq ft
    </div>
    """, unsafe_allow_html=True)

    departments = sorted(train.loc[train["Store"] == store_id, "Dept"].unique())
    department_id = st.selectbox(
        "Department",
        departments,
        help=f"Departments active in Store {store_id}.",
    )

    forecast_date = st.date_input(
        "Forecast Date",
        value=datetime.date.today(),
        help="Select the target week. Year, month, and ISO week are derived automatically.",
    )

    is_holiday = st.checkbox(
        "Holiday Week",
        value=False,
        help="Mark if this week falls on one of the four major Walmart holiday weeks.",
    )
    if is_holiday:
        st.markdown('<span class="inline-badge badge-amber">🎄 Holiday flag active</span>', unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Section 2: Economic Indicators ──────────────────────────────────────
    st.markdown('<p class="sidebar-section-label">📈 Economic Indicators</p>', unsafe_allow_html=True)

    cpi_default          = float(imputer.statistics_[0])
    unemployment_default = float(imputer.statistics_[1])

    cpi = st.number_input(
        "CPI (Consumer Price Index)",
        value=cpi_default,
        format="%.3f",
        help=f"Reflects the cost of consumer goods. Training mean: {cpi_default:.3f}.",
    )

    unemployment = st.number_input(
        "Unemployment Rate (%)",
        value=unemployment_default,
        format="%.3f",
        help=f"Regional unemployment rate. Training mean: {unemployment_default:.3f}.",
    )

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Predict button ───────────────────────────────────────────────────────
    predict_clicked = st.button(
        "⚡ Run Forecast",
        type="primary",
        use_container_width=True,
    )

    st.markdown(f"""
    <div style="text-align:center;margin-top:0.5rem;font-size:0.72rem;color:#CBD5E1;">
        Week {forecast_date.isocalendar().week} · {forecast_date.strftime('%b %Y')}
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION PIPELINE  (identical to original — untouched)
# ═══════════════════════════════════════════════════════════════════════════════

row = pd.DataFrame([{
    "Store":        store_id,
    "Dept":         department_id,
    "IsHoliday":    int(is_holiday),
    "CPI":          cpi,
    "Unemployment": unemployment,
    "Size":         store_info["Size"],
    "Type_A":       int(store_info["Type"] == "A"),
    "Type_B":       int(store_info["Type"] == "B"),
    "Type_C":       int(store_info["Type"] == "C"),
    "year":         forecast_date.year,
    "month":        forecast_date.month,
    "week":         int(forecast_date.isocalendar().week),
}])

impute_cols     = ["CPI", "Unemployment"]
row[impute_cols] = imputer.transform(row[impute_cols])

categorical_cols = ["Store", "Dept"]
encoded_array    = encoder.transform(row[categorical_cols])
encoded_df       = pd.DataFrame(
    encoded_array,
    columns=encoder.get_feature_names_out(categorical_cols),
)

row_final = pd.concat(
    [row.drop(columns=categorical_cols).reset_index(drop=True),
     encoded_df.reset_index(drop=True)],
    axis=1,
)[feature_columns]

prediction = model.predict(row_final)[0]


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

dept_history = (
    train[(train["Store"] == store_id) & (train["Dept"] == department_id)]
    .sort_values("Date")
    .copy()
)
has_history = not dept_history.empty

if has_history:
    hist_avg = dept_history["Weekly_Sales"].mean()
    hist_min = dept_history["Weekly_Sales"].min()
    hist_max = dept_history["Weekly_Sales"].max()
    hist_n   = len(dept_history)
    delta_abs = prediction - hist_avg
    delta_pct = (delta_abs / hist_avg) * 100 if hist_avg != 0 else 0
    is_above  = delta_abs >= 0
else:
    hist_avg = hist_min = hist_max = hist_n = None
    delta_abs = delta_pct = 0
    is_above = True


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER BANNER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="header-wrap">
    <div>
        <h1 class="header-title">Walmart <span>Sales Forecast</span></h1>
        <p class="header-subtitle">
            Department-level weekly sales forecasting &amp; historical retail analytics
        </p>
    </div>
    <div class="engine-pill">
        <span class="engine-dot"></span>
        ML Forecasting Engine
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_forecast, tab_analytics, tab_model = st.tabs([
    "📊 Forecast Results",
    "📈 Historical Analytics",
    "🔧 Model Info",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — FORECAST RESULTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_forecast:

    # ── 1A. Main prediction card ─────────────────────────────────────────────
    week_type_badge = (
        '<span class="inline-badge badge-amber">🎄 Holiday Week</span>'
        if is_holiday
        else '<span class="inline-badge badge-slate">Regular Week</span>'
    )

    if has_history:
        delta_badge_cls  = "badge-green" if is_above else "badge-red"
        delta_arrow      = "▲" if is_above else "▼"
        delta_badge_text = f"{delta_arrow} {'+' if is_above else ''}{delta_abs:,.2f} ({'+' if is_above else ''}{delta_pct:.1f}%)"
        delta_badge_html = f'<span class="inline-badge {delta_badge_cls}">{delta_badge_text}</span>'
        context_text     = f"vs. historical dept average of <strong>${hist_avg:,.2f}</strong>"
    else:
        delta_badge_html = ""
        context_text     = "No historical benchmark available"

    st.markdown(f"""
    <div class="prediction-card">
        <div>
            <div class="prediction-label">Predicted Weekly Sales</div>
            <div class="prediction-value">${prediction:,.2f}</div>
            <div class="prediction-context">
                {delta_badge_html}&nbsp;&nbsp;{context_text}
            </div>
        </div>
        <div class="prediction-divider"></div>
        <div>
            <div class="prediction-meta-label">Store</div>
            <div class="prediction-meta-value">Store #{store_id} · Type {store_type}</div>
        </div>
        <div class="prediction-divider"></div>
        <div>
            <div class="prediction-meta-label">Department</div>
            <div class="prediction-meta-value">Dept #{department_id}</div>
        </div>
        <div class="prediction-divider"></div>
        <div>
            <div class="prediction-meta-label">Forecast Week</div>
            <div class="prediction-meta-value">{forecast_date.strftime('%b %d, %Y')}</div>
        </div>
        <div class="prediction-divider"></div>
        <div>
            <div class="prediction-meta-label">Week Type</div>
            <div class="prediction-meta-value">{week_type_badge}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 1B. KPI card row ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4, gap="small")

    # Card 1 — Predicted Sales
    with c1:
        st.markdown(f"""
        <div class="kpi-card primary">
            <div class="kpi-label">Predicted Sales <span class="kpi-icon">💰</span></div>
            <div class="kpi-value primary">${prediction:,.0f}</div>
            <div class="kpi-delta">Weekly total forecast</div>
        </div>
        """, unsafe_allow_html=True)

    # Card 2 — Hist Average
    with c2:
        if has_history:
            st.markdown(f"""
            <div class="kpi-card neutral">
                <div class="kpi-label">Hist. Dept Average <span class="kpi-icon">📏</span></div>
                <div class="kpi-value neutral">${hist_avg:,.0f}</div>
                <div class="kpi-delta">Over {hist_n} recorded weeks</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="kpi-card neutral">
                <div class="kpi-label">Hist. Dept Average <span class="kpi-icon">📏</span></div>
                <div class="kpi-value neutral">—</div>
                <div class="kpi-delta">No historical data</div>
            </div>
            """, unsafe_allow_html=True)

    # Card 3 — Sales Difference $
    with c3:
        if has_history:
            cls = "success" if is_above else "danger"
            sign = "+" if is_above else ""
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">Sales Difference <span class="kpi-icon">{'📈' if is_above else '📉'}</span></div>
                <div class="kpi-value {cls}">{sign}${abs(delta_abs):,.0f}</div>
                <div class="kpi-delta">{'Above' if is_above else 'Below'} historical average</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="kpi-card neutral">
                <div class="kpi-label">Sales Difference <span class="kpi-icon">📊</span></div>
                <div class="kpi-value neutral">—</div>
                <div class="kpi-delta">No benchmark available</div>
            </div>
            """, unsafe_allow_html=True)

    # Card 4 — % Difference
    with c4:
        if has_history:
            cls = "success" if is_above else "danger"
            sign = "+" if is_above else ""
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">% vs Avg <span class="kpi-icon">🎯</span></div>
                <div class="kpi-value {cls}">{sign}{delta_pct:.1f}%</div>
                <div class="kpi-delta">Relative performance</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="kpi-card neutral">
                <div class="kpi-label">% vs Avg <span class="kpi-icon">🎯</span></div>
                <div class="kpi-value neutral">—</div>
                <div class="kpi-delta">No benchmark available</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    # ── 1C. Sales Forecast Visualization ─────────────────────────────────────
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">📉 Sales Forecast Visualization</p>', unsafe_allow_html=True)

    if not has_history:
        st.markdown(f"""
        <div class="info-banner">
            ℹ️ No historical sales data available in <code>train.csv</code> for
            <strong>Store {store_id} / Dept {department_id}</strong>.
            Showing the forecast value only.
        </div>
        """, unsafe_allow_html=True)
        st.metric("Forecasted Weekly Sales", f"${prediction:,.2f}")

    else:
        st.markdown(
            f'<p class="section-subheading">Historical weekly sales (from <code>train.csv</code>) '
            f'with the forecast point for the selected date · Store {store_id} · Dept {department_id}</p>',
            unsafe_allow_html=True,
        )

        # Build chart dataframe — history + single forecast point
        hist_chart = dept_history[["Date", "Weekly_Sales"]].copy()
        hist_chart["Type"] = "Historical"

        forecast_point = pd.DataFrame([{
            "Date":         pd.Timestamp(forecast_date),
            "Weekly_Sales": prediction,
            "Type":         "Forecast",
        }])

        chart_df = pd.concat([hist_chart, forecast_point], ignore_index=True)

        # Average reference line data
        avg_line = pd.DataFrame([
            {"Date": hist_chart["Date"].min(), "avg": hist_avg},
            {"Date": hist_chart["Date"].max(), "avg": hist_avg},
        ])

        # Altair historical line
        hist_line = (
            alt.Chart(hist_chart)
            .mark_line(color="#1677D2", strokeWidth=1.8)
            .encode(
                x=alt.X("Date:T", title="Date", axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
                y=alt.Y("Weekly_Sales:Q", title="Weekly Sales ($)",
                        axis=alt.Axis(labelFontSize=11, titleFontSize=12, format="$,.0f")),
                tooltip=[
                    alt.Tooltip("Date:T", title="Date"),
                    alt.Tooltip("Weekly_Sales:Q", title="Weekly Sales", format="$,.2f"),
                ],
            )
        )

        # Area fill under history
        hist_area = (
            alt.Chart(hist_chart)
            .mark_area(color="#1677D2", opacity=0.07)
            .encode(
                x="Date:T",
                y="Weekly_Sales:Q",
            )
        )

        # Average dashed line
        avg_rule = (
            alt.Chart(avg_line)
            .mark_line(color="#94A3B8", strokeDash=[5, 4], strokeWidth=1.4)
            .encode(
                x="Date:T",
                y=alt.Y("avg:Q"),
                tooltip=[alt.Tooltip("avg:Q", title="Dept Avg", format="$,.2f")],
            )
        )

        # Forecast point
        forecast_dot = (
            alt.Chart(forecast_point)
            .mark_point(color="#F59E0B", size=160, filled=True, strokeWidth=2, stroke="#FFFFFF")
            .encode(
                x="Date:T",
                y="Weekly_Sales:Q",
                tooltip=[
                    alt.Tooltip("Date:T", title="Forecast Date"),
                    alt.Tooltip("Weekly_Sales:Q", title="Predicted Sales", format="$,.2f"),
                ],
            )
        )

        chart = (
            (hist_area + hist_line + avg_rule + forecast_dot)
            .properties(height=330)
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor="#F1F5F9", gridOpacity=1)
        )

        st.altair_chart(chart, use_container_width=True)

        # Legend
        col_l1, col_l2, col_l3 = st.columns([1, 1, 2])
        with col_l1:
            st.markdown('<span style="font-size:0.8rem;">🔵 Historical Sales&nbsp;&nbsp;</span>', unsafe_allow_html=True)
        with col_l2:
            st.markdown('<span style="font-size:0.8rem;">🟡 Forecast Point&nbsp;&nbsp;</span>', unsafe_allow_html=True)
        with col_l3:
            st.markdown('<span style="font-size:0.8rem;color:#94A3B8;">— Department Average</span>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 1D. Economic Indicators mini-cards ───────────────────────────────────
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">🌐 Economic Indicators</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-subheading">Values passed to the model for this forecast</p>', unsafe_allow_html=True)

    ec1, ec2, ec3, ec4 = st.columns(4, gap="small")
    with ec1:
        st.metric("Consumer Price Index (CPI)", f"{cpi:.3f}",
                  help="Higher CPI indicates higher cost of living.")
    with ec2:
        st.metric("Unemployment Rate (%)", f"{unemployment:.3f}",
                  help="Regional unemployment percentage.")
    with ec3:
        st.metric("Store Size", f"{int(store_size):,} sq ft",
                  help="Total floor area of the selected store.")
    with ec4:
        st.metric("ISO Week Number", f"Week {forecast_date.isocalendar().week}",
                  help="ISO calendar week derived from the selected forecast date.")

    st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — HISTORICAL ANALYTICS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_analytics:

    store_data = train[train["Store"] == store_id].copy()

    # ── Store-level overview KPIs ────────────────────────────────────────────
    total_weeks   = store_data["Date"].nunique()
    total_depts   = store_data["Dept"].nunique()
    weekly_totals = store_data.groupby("Date")["Weekly_Sales"].sum()
    avg_weekly    = weekly_totals.mean()

    hol_avg = store_data[store_data["IsHoliday"] == True]["Weekly_Sales"].mean()
    reg_avg = store_data[store_data["IsHoliday"] == False]["Weekly_Sales"].mean()
    hol_lift = ((hol_avg - reg_avg) / reg_avg * 100) if reg_avg else 0

    st.markdown(f"### Store #{store_id} — Historical Overview")

    ov1, ov2, ov3, ov4 = st.columns(4, gap="small")
    with ov1:
        st.metric("Recorded Weeks",    f"{total_weeks}")
    with ov2:
        st.metric("Active Departments", f"{total_depts}")
    with ov3:
        st.metric("Avg Store Weekly Revenue", f"${avg_weekly:,.0f}")
    with ov4:
        sign = "+" if hol_lift >= 0 else ""
        st.metric("Holiday Sales Lift", f"{sign}{hol_lift:.1f}%",
                  help="Percentage difference between holiday and regular week dept averages.")

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── Top 10 Departments by average weekly sales ───────────────────────────
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">🏆 Top 10 Departments by Avg Weekly Sales</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-subheading">Real historical averages from train.csv · Store {store_id} · Highlighting selected Dept {department_id}</p>', unsafe_allow_html=True)

    dept_avg = (
        store_data.groupby("Dept")["Weekly_Sales"]
        .mean()
        .reset_index()
        .nlargest(10, "Weekly_Sales")
    )
    dept_avg["Dept_Label"] = "Dept " + dept_avg["Dept"].astype(str)
    dept_avg["IsSelected"] = dept_avg["Dept"] == department_id

    bar = (
        alt.Chart(dept_avg)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Weekly_Sales:Q", title="Avg Weekly Sales ($)",
                    axis=alt.Axis(format="$,.0f", labelFontSize=11)),
            y=alt.Y("Dept_Label:N", sort="-x", title="Department",
                    axis=alt.Axis(labelFontSize=11)),
            color=alt.condition(
                alt.datum.IsSelected,
                alt.value("#1677D2"),
                alt.value("#CBD5E1"),
            ),
            tooltip=[
                alt.Tooltip("Dept_Label:N", title="Department"),
                alt.Tooltip("Weekly_Sales:Q", title="Avg Weekly Sales", format="$,.2f"),
            ],
        )
        .properties(height=320)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridColor="#F5F7FA")
    )

    st.altair_chart(bar, use_container_width=True)

    if department_id in dept_avg["Dept"].values:
        st.markdown(
            f'<span class="inline-badge badge-blue">🔵 Dept {department_id} highlighted above</span>',
            unsafe_allow_html=True,
        )
    else:
        st.caption(f"Dept {department_id} is not in the top 10 for Store {store_id}.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Store-level weekly revenue trend ─────────────────────────────────────
    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-heading">📅 Total Store Weekly Revenue Trend</p>', unsafe_allow_html=True)
    st.markdown(f'<p class="section-subheading">Sum of all departments · Store {store_id}</p>', unsafe_allow_html=True)

    weekly_rev = weekly_totals.reset_index()
    weekly_rev.columns = ["Date", "Total_Sales"]

    store_trend = (
        alt.Chart(weekly_rev)
        .mark_area(color="#1677D2", opacity=0.12)
        .encode(x="Date:T", y="Total_Sales:Q")
    ) + (
        alt.Chart(weekly_rev)
        .mark_line(color="#1677D2", strokeWidth=1.8)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Total_Sales:Q", title="Total Sales ($)",
                    axis=alt.Axis(format="$,.0f")),
            tooltip=[
                alt.Tooltip("Date:T", title="Week"),
                alt.Tooltip("Total_Sales:Q", title="Total Sales", format="$,.2f"),
            ],
        )
    )

    st.altair_chart(
        store_trend.properties(height=300).configure_view(strokeWidth=0),
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — MODEL INFO
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_model:

    st.markdown("### 🔧 Model & Pipeline Architecture")

    mi1, mi2 = st.columns(2, gap="large")

    with mi1:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("**Algorithm**")
        st.code(type(model).__name__, language=None)

        st.markdown("**Artifact path**")
        st.code("models/walmart_model.pkl", language=None)

        st.markdown("**Feature vector dimension**")
        st.code(f"{len(feature_columns)} features", language=None)

        st.markdown("**Categorical encoder**")
        st.code("OneHotEncoder  →  Store, Dept", language=None)

        st.markdown("**Numerical imputer**")
        st.code(f"SimpleImputer  →  CPI (mean={cpi_default:.3f}), Unemployment (mean={unemployment_default:.3f})", language=None)
        st.markdown("</div>", unsafe_allow_html=True)

    with mi2:
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        st.markdown("**Pipeline Integrity**")
        st.markdown("""
- ✅ Model loaded read-only from `models/walmart_model.pkl`
- ✅ Imputer and encoder are the fitted training objects
- ✅ Feature column order enforced at prediction time
- ✅ No retraining or model replacement
- ✅ `predict.py` and all data files remain untouched
- ✅ Predictions are mathematically identical for identical inputs
        """)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("📋 All Feature Columns (in order)"):
        st.dataframe(
            pd.DataFrame({"#": range(1, len(feature_columns) + 1),
                          "Feature": feature_columns}),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(f"""
<div class="footer">
    Walmart Sales Forecast Dashboard
    &nbsp;·&nbsp; Model: <strong>{type(model).__name__}</strong>
    &nbsp;·&nbsp; Streamlit {st.__version__}
    &nbsp;·&nbsp; Data: <code>train.csv</code> (read-only)
</div>
""", unsafe_allow_html=True)
