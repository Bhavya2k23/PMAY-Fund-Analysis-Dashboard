import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# ==========================================================
# PMAY Fund Analysis Dashboard — Enhanced Version 1.0
# ==========================================================

# --- Robust path resolution ---------------------------------------------
# This script lives at: PMAY Fund Analysis/dashboard/app.py
# We resolve paths from THIS FILE's location rather than the terminal's
# current working directory. This fixes the "File does not exist" /
# "master_dataset.csv not found" class of errors that happen whenever
# Streamlit is launched from a different folder than expected.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # one level up from dashboard/

MASTER_FILE = os.path.join(PROJECT_ROOT, "data", "cleaned", "master_dataset.csv")
INDIA_GEOJSON_URL = "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"

st.set_page_config(
    page_title="PMAY Fund Analysis Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1e2530 0%, #262d3a 100%);
        border: 1px solid #333c4d;
        border-radius: 12px;
        padding: 16px 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.25);
    }
    div[data-testid="stMetricLabel"] { font-size: 14px; opacity: 0.8; }
    div[data-testid="stMetricValue"] { font-size: 26px; font-weight: 700; }
    .performer-card {
        background: linear-gradient(135deg, #1e2530 0%, #262d3a 100%);
        border-radius: 12px;
        padding: 18px;
        border-left: 4px solid #2A9D8F;
        margin-bottom: 10px;
    }
    .performer-card.low { border-left: 4px solid #E76F51; }
    .section-title { font-size: 20px; font-weight: 700; margin-top: 8px; margin-bottom: 4px; }
    .insight-box {
        background: linear-gradient(135deg, #1e2530 0%, #262d3a 100%);
        border: 1px solid #333c4d;
        border-left: 4px solid #4C9AFF;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 15px;
    }
    .warning-box {
        background: #3a2416;
        border: 1px solid #a8611b;
        border-left: 4px solid #f0a03c;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 12px;
        font-size: 14px;
    }
    .empty-state {
        background: linear-gradient(135deg, #1e2530 0%, #262d3a 100%);
        border: 1px dashed #4a5568;
        border-radius: 12px;
        padding: 40px 20px;
        text-align: center;
        font-size: 16px;
        opacity: 0.85;
        margin: 20px 0;
    }
    .footer-box {
        text-align:center;
        opacity:0.85;
        padding: 14px;
        font-size: 13px;
        line-height: 1.6;
    }
    /* Mobile responsiveness tweaks */
    @media (max-width: 768px) {
        div[data-testid="stMetricValue"] { font-size: 20px; }
        div[data-testid="stMetricLabel"] { font-size: 12px; }
        .section-title { font-size: 17px; }
        .insight-box { font-size: 13px; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


df = load_data(MASTER_FILE)

if df is None:
    st.error(f"Master dataset not found at:\n`{MASTER_FILE}`")
    st.info(
        "This usually means Phase 4 (merge_data.py) hasn't been run yet, or the "
        "cleaned data folder is in a different location than expected. "
        "Run `python src/merge_data.py` from the project root first."
    )
    st.stop()

REQUIRED_COLS = {"state_name", "total", "funds_released_crore",
                 "funds_utilized_crore", "fund_utilization_percent"}
missing_required = REQUIRED_COLS - set(df.columns)
if missing_required:
    st.warning(f"Some expected columns are missing: {missing_required}. Certain charts will be skipped.")

# ----------------------------------------------------------
# Data quality flags (Problem 1 / Problem 3)
# ----------------------------------------------------------
has_util = "fund_utilization_percent" in df.columns
has_released = "funds_released_crore" in df.columns
has_utilized = "funds_utilized_crore" in df.columns

if has_util:
    # Flag rows where released/utilized are both zero -> should read "No Data", not 0%
    if has_released and has_utilized:
        df["_no_fund_data"] = (df["funds_released_crore"].fillna(0) == 0) & (df["funds_utilized_crore"].fillna(0) == 0)
    else:
        df["_no_fund_data"] = False
    # Flag rows where utilization exceeds 100% (possible data inconsistency)
    df["_util_over_100"] = df["fund_utilization_percent"] > 100

any_over_100 = has_util and df["_util_over_100"].any()

# ----------------------------------------------------------
# Precompute default filter bounds (used for widget defaults AND reset)
# ----------------------------------------------------------
all_states = sorted(df["state_name"].dropna().unique().tolist())

if has_util:
    UTIL_MIN_DEFAULT, UTIL_MAX_DEFAULT = 0, int(np.ceil(df["fund_utilization_percent"].max())) + 1
else:
    UTIL_MIN_DEFAULT, UTIL_MAX_DEFAULT = 0, 100

if "total" in df.columns:
    BEN_MIN_DEFAULT, BEN_MAX_DEFAULT = int(df["total"].min()), int(df["total"].max()) + 1
else:
    BEN_MIN_DEFAULT, BEN_MAX_DEFAULT = 0, 1

if has_released:
    REL_MIN_DEFAULT, REL_MAX_DEFAULT = float(df["funds_released_crore"].min()), float(df["funds_released_crore"].max()) + 1
else:
    REL_MIN_DEFAULT, REL_MAX_DEFAULT = 0.0, 1.0

if has_utilized:
    UTZ_MIN_DEFAULT, UTZ_MAX_DEFAULT = float(df["funds_utilized_crore"].min()), float(df["funds_utilized_crore"].max()) + 1
else:
    UTZ_MIN_DEFAULT, UTZ_MAX_DEFAULT = 0.0, 1.0

# Keys for every filter widget — used by the Reset Filters button below.
FILTER_KEYS = [
    "flt_states", "flt_util_range", "flt_beneficiary_range",
    "flt_released_range", "flt_utilized_range", "flt_exclude_outliers",
]

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
st.sidebar.markdown("## 🏠 PMAY Fund Analysis")
st.sidebar.markdown("---")

# Reset Filters button (Improvement #2) — clears widget state so every
# filter snaps back to its full-range default on the next rerun.
reset_col1, reset_col2 = st.sidebar.columns([3, 2])
with reset_col1:
    st.markdown("### 📍 Filters")
with reset_col2:
    if st.button("↺ Reset", use_container_width=True, help="Clear all filters back to defaults"):
        for k in FILTER_KEYS:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

selected_states = st.sidebar.multiselect(
    "State(s)", all_states, default=all_states, key="flt_states"
)

if has_util:
    util_range = st.sidebar.slider(
        "Utilization % range", UTIL_MIN_DEFAULT, UTIL_MAX_DEFAULT,
        (UTIL_MIN_DEFAULT, UTIL_MAX_DEFAULT), key="flt_util_range",
        help="Utilization % = (Funds Utilized ÷ Funds Released) × 100",
    )
else:
    util_range = None

# Problem 13 — additional range filters (beneficiaries / released / utilized)
if "total" in df.columns:
    beneficiary_range = st.sidebar.slider(
        "Beneficiary range", BEN_MIN_DEFAULT, BEN_MAX_DEFAULT,
        (BEN_MIN_DEFAULT, BEN_MAX_DEFAULT), key="flt_beneficiary_range",
    )
else:
    beneficiary_range = None

if has_released:
    released_range = st.sidebar.slider(
        "Funds Released (Cr) range", REL_MIN_DEFAULT, REL_MAX_DEFAULT,
        (REL_MIN_DEFAULT, REL_MAX_DEFAULT), key="flt_released_range",
    )
else:
    released_range = None

if has_utilized:
    utilized_range = st.sidebar.slider(
        "Funds Utilized (Cr) range", UTZ_MIN_DEFAULT, UTZ_MAX_DEFAULT,
        (UTZ_MIN_DEFAULT, UTZ_MAX_DEFAULT), key="flt_utilized_range",
    )
else:
    utilized_range = None

exclude_outliers = False
if any_over_100:
    exclude_outliers = st.sidebar.checkbox(
        "Exclude utilization > 100% (data inconsistency)", value=False,
        key="flt_exclude_outliers",
    )

# --- Apply filters (with a lightweight spinner — Improvement #5) --------
with st.spinner("Loading dashboard..."):
    filtered_df = df[df["state_name"].isin(selected_states)].copy()

    if util_range is not None:
        filtered_df = filtered_df[filtered_df["fund_utilization_percent"].between(util_range[0], util_range[1])]
    if beneficiary_range is not None:
        filtered_df = filtered_df[filtered_df["total"].between(beneficiary_range[0], beneficiary_range[1])]
    if released_range is not None:
        filtered_df = filtered_df[filtered_df["funds_released_crore"].between(released_range[0], released_range[1])]
    if utilized_range is not None:
        filtered_df = filtered_df[filtered_df["funds_utilized_crore"].between(utilized_range[0], utilized_range[1])]
    if exclude_outliers and has_util:
        filtered_df = filtered_df[~filtered_df["_util_over_100"]]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")
st.sidebar.caption(f"**{len(filtered_df)}** states/UTs shown")

if "total" in df.columns:
    st.sidebar.caption(f"**{df['total'].sum()/1e6:.2f} Million** total beneficiaries (all data)")
if has_released:
    st.sidebar.caption(f"**₹{df['funds_released_crore'].sum()/1000:.1f}K Cr** released (all data)")

# Problem 12 — richer quick stats (computed on filtered data, excluding no-data rows for utilization)
if has_util and not filtered_df.empty:
    valid_util = filtered_df.loc[~filtered_df.get("_no_fund_data", False), "fund_utilization_percent"]
    if not valid_util.empty:
        st.sidebar.markdown("**Utilization stats (selected states)**")
        st.sidebar.caption(f"Average: **{valid_util.mean():.1f}%**")
        st.sidebar.caption(f"Median: **{valid_util.median():.1f}%**")
        highest_row = filtered_df.loc[valid_util.idxmax()]
        lowest_row = filtered_df.loc[valid_util.idxmin()]
        st.sidebar.caption(f"Highest: **{highest_row['state_name']}** ({highest_row['fund_utilization_percent']:.1f}%)")
        st.sidebar.caption(f"Lowest: **{lowest_row['state_name']}** ({lowest_row['fund_utilization_percent']:.1f}%)")
if "total" in filtered_df.columns and not filtered_df.empty:
    top_beneficiary_row = filtered_df.loc[filtered_df["total"].idxmax()]
    st.sidebar.caption(f"Most beneficiaries: **{top_beneficiary_row['state_name']}** ({top_beneficiary_row['total']/1e6:.2f}M)")

# About section (Improvement #10)
with st.sidebar.expander("ℹ️ About this Dashboard"):
    st.markdown(
        """
        **Objective**
        Analyze state-wise fund release, utilization, and beneficiary
        reach under the Pradhan Mantri Awas Yojana (PMAY) scheme.

        **Data Source**
        PMAY-G Open Government Data (data.gov.in) — cleaned and
        aggregated for analysis.

        **Technologies Used**
        - Python
        - Pandas / NumPy
        - Plotly
        - Streamlit
        - Exploratory Data Analysis (EDA)

        **Author**
        Bhavya — Data Analyst Portfolio Project
        """
    )

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.title("🏠 PMAY Fund Analysis Dashboard")
st.caption("Pradhan Mantri Awas Yojana — Fund Utilization & Beneficiary Analytics")
st.markdown("---")

# Problem 1 — data-inconsistency warning banner
if any_over_100:
    n_over = int(df["_util_over_100"].sum())
    st.markdown(
        f"""<div class="warning-box">⚠️ <b>Data note:</b> {n_over} state(s)/UT(s) report utilization
        above 100% (funds utilized exceed funds released). This is likely due to carry-forward or
        revised expenditure figures in the source data rather than an error in this dashboard.
        Use the sidebar checkbox to exclude these rows if you want a stricter 0–100% view.</div>""",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------
# Better Empty State (Improvement #6)
# ----------------------------------------------------------
if filtered_df.empty:
    st.markdown(
        """<div class="empty-state">
        🔍 <b>No states found.</b><br>
        Try changing or resetting your filters in the sidebar.
        </div>""",
        unsafe_allow_html=True,
    )
    st.stop()

# ----------------------------------------------------------
# KPI Row (with trend icons — Improvement #4)
# ----------------------------------------------------------
def safe_sum(frame, col):
    return frame[col].sum() if col in frame.columns else None


total_beneficiaries = safe_sum(filtered_df, "total")
released = safe_sum(filtered_df, "funds_released_crore")
utilized = safe_sum(filtered_df, "funds_utilized_crore")
utilization_pct = round((utilized / released) * 100, 1) if released and released > 0 else None

# Compare current filtered utilization against the all-data baseline to
# decide which way the trend arrow should point.
baseline_util_pct = None
if has_released and has_utilized:
    base_released = df["funds_released_crore"].sum()
    base_utilized = df["funds_utilized_crore"].sum()
    if base_released > 0:
        baseline_util_pct = round((base_utilized / base_released) * 100, 1)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("States/UTs", len(filtered_df), help="Number of states/UTs matching the current filters.")
k2.metric(
    "Beneficiaries",
    f"{total_beneficiaries/1e6:.2f}M" if total_beneficiaries is not None else "N/A",
    help="Total beneficiaries across the current filter selection.",
)
k3.metric(
    "Funds Released",
    f"₹{released/1000:.1f}K Cr" if released is not None else "N/A",
    help="Sum of funds released across the current filter selection.",
)
k4.metric(
    "Funds Utilized",
    f"₹{utilized/1000:.1f}K Cr" if utilized is not None else "N/A",
    help="Sum of funds utilized across the current filter selection.",
)
if utilization_pct is not None:
    delta_val = None
    if baseline_util_pct is not None:
        delta_val = round(utilization_pct - baseline_util_pct, 1)
    k5.metric(
        "Utilization",
        f"{utilization_pct}%",
        delta=f"{delta_val:+.1f}% vs all-data avg" if delta_val is not None else None,
        help="Utilization % = (Funds Utilized ÷ Funds Released) × 100, aggregated across the current filter selection.",
    )
else:
    k5.metric("Utilization", "N/A")

st.markdown("<br>", unsafe_allow_html=True)


# ----------------------------------------------------------
# Top / Bottom performer — Problem 2 & 3: show Top 5 / Bottom 5 instead
# of a single outlier-prone "Top Performer" card, and label true
# no-data rows as "No Data" rather than 0%.
# ----------------------------------------------------------
if has_util and not filtered_df.empty:
    display_df = filtered_df.copy()
    display_df["Utilization Display"] = display_df.apply(
        lambda r: "No Data" if r.get("_no_fund_data", False) else f"{r['fund_utilization_percent']:.1f}%",
        axis=1,
    )

    ranked = display_df[~display_df.get("_no_fund_data", False)].sort_values(
        "fund_utilization_percent", ascending=False
    )

    st.markdown('<p class="section-title">🏆 Top & Bottom Performers (by Utilization %)</p>', unsafe_allow_html=True)
    tcol, bcol = st.columns(2)
    with tcol:
        st.markdown("**Top 5 States**")
        top5 = ranked.head(5)[["state_name", "Utilization Display"]].rename(
            columns={"state_name": "State", "Utilization Display": "Utilization"}
        )
        st.dataframe(top5, use_container_width=True, hide_index=True)
    with bcol:
        st.markdown("**Bottom 5 States**")
        bottom5 = ranked.tail(5)[["state_name", "Utilization Display"]].rename(
            columns={"state_name": "State", "Utilization Display": "Utilization"}
        ).sort_values("Utilization")
        st.dataframe(bottom5, use_container_width=True, hide_index=True)

    no_data_states = display_df[display_df.get("_no_fund_data", False)]["state_name"].tolist()
    if no_data_states:
        st.caption(f"ℹ️ No fund data reported for: {', '.join(no_data_states)}")

st.markdown("---")

# ----------------------------------------------------------
# Problem 15 — Auto-generated insights (expanded to 8-10 — Improvement #9)
# ----------------------------------------------------------
st.markdown('<p class="section-title">📌 Auto-Generated Insights</p>', unsafe_allow_html=True)
insights = []

valid_util_df = filtered_df[~filtered_df.get("_no_fund_data", False)] if has_util else filtered_df

if "total" in filtered_df.columns and total_beneficiaries and total_beneficiaries > 0 and not filtered_df.empty:
    top_b = filtered_df.loc[filtered_df["total"].idxmax()]
    share = top_b["total"] / total_beneficiaries * 100
    insights.append(f"📈 **{top_b['state_name']}** contributes **{share:.1f}%** of total beneficiaries in the current selection.")

    low_b = filtered_df.loc[filtered_df["total"].idxmin()]
    insights.append(f"📉 **{low_b['state_name']}** has the lowest number of beneficiaries in the current selection.")

    insights.append(f"📊 Average beneficiaries per state/UT is **{filtered_df['total'].mean()/1e3:.1f}K**.")

if has_util and not valid_util_df.empty:
    low = valid_util_df.loc[valid_util_df["fund_utilization_percent"].idxmin()]
    high = valid_util_df.loc[valid_util_df["fund_utilization_percent"].idxmax()]
    insights.append(f"📉 **{low['state_name']}** has the lowest utilization at **{low['fund_utilization_percent']:.1f}%** — the least efficient in the current selection.")
    insights.append(f"🏅 **{high['state_name']}** is the most efficient state, with utilization at **{high['fund_utilization_percent']:.1f}%**.")
    insights.append(f"📊 Median utilization across selected states is **{valid_util_df['fund_utilization_percent'].median():.1f}%** (median is more robust to outliers than the average).")

if has_released and not filtered_df.empty:
    top_rel = filtered_df.loc[filtered_df["funds_released_crore"].idxmax()]
    insights.append(f"💰 **{top_rel['state_name']}** received the highest funds released, at **₹{top_rel['funds_released_crore']:.1f} Cr**.")
    insights.append(f"💰 Average funds released per state/UT is **₹{filtered_df['funds_released_crore'].mean():.1f} Cr**.")

if has_utilized and not filtered_df.empty:
    top_utz = filtered_df.loc[filtered_df["funds_utilized_crore"].idxmax()]
    insights.append(f"💵 **{top_utz['state_name']}** recorded the highest funds utilized, at **₹{top_utz['funds_utilized_crore']:.1f} Cr**.")

if has_released and has_utilized and not filtered_df.empty:
    corr_cols = filtered_df[["funds_released_crore", "funds_utilized_crore"]].dropna()
    if len(corr_cols) > 2:
        corr_val = corr_cols.corr().iloc[0, 1]
        insights.append(f"🔗 Funds Released and Funds Utilized are correlated at **{corr_val:.2f}** ({'strong' if abs(corr_val) > 0.7 else 'moderate' if abs(corr_val) > 0.4 else 'weak'} relationship).")

if not insights:
    st.caption("No insights available for the current filter selection.")
else:
    for ins in insights[:10]:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

st.markdown("---")


# ----------------------------------------------------------
# Tabs
# ----------------------------------------------------------
tab_overview, tab_analytics, tab_map, tab_data = st.tabs(
    ["📈 Overview", "🔬 Deep Analytics", "🗺 India Map", "📋 Dataset"]
)

with tab_overview:
    col_left, col_right = st.columns(2)

    with col_left:
        if {"state_name", "total"}.issubset(filtered_df.columns):
            top10 = filtered_df.sort_values("total", ascending=False).head(10)
            fig = px.bar(
                top10, x="total", y="state_name", orientation="h",
                color="total", color_continuous_scale="Blues",
                title="Top 10 States by Beneficiaries",
                text_auto=".2s",  # Problem 5 — show values on bars
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if {"state_name", "fund_utilization_percent"}.issubset(filtered_df.columns):
            fig2 = px.bar(
                filtered_df.sort_values("fund_utilization_percent", ascending=False),
                x="state_name", y="fund_utilization_percent",
                color="fund_utilization_percent", color_continuous_scale="RdYlGn",
                title="Fund Utilization % by State",
                text_auto=".1f",  # Problem 5
                hover_data={"fund_utilization_percent": ":.1f"},
            )
            fig2.update_layout(xaxis_tickangle=-45, showlegend=False)
            fig2.add_hline(y=100, line_dash="dash", line_color="white", opacity=0.4,
                            annotation_text="100% line", annotation_position="top left")
            fig2.update_traces(
                hovertemplate="<b>%{x}</b><br>Utilization = Funds Utilized ÷ Funds Released × 100<br>Value: %{y:.1f}%<extra></extra>"
            )
            st.plotly_chart(fig2, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        cat_cols = [c for c in ["sc", "st", "others", "minority"] if c in filtered_df.columns]
        if cat_cols:
            totals = filtered_df[cat_cols].sum().reset_index()
            totals.columns = ["category", "count"]
            fig3 = px.pie(
                totals, names="category", values="count", hole=0.5,
                title="Beneficiary Distribution by Category",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col_right2:
        if {"funds_released_crore", "funds_utilized_crore", "total"}.issubset(filtered_df.columns):
            hover_cols = [c for c in ["fund_utilization_percent", "total", "funds_released_crore", "funds_utilized_crore"] if c in filtered_df.columns]
            fig4 = px.scatter(
                filtered_df, x="funds_released_crore", y="funds_utilized_crore",
                size="total", color="fund_utilization_percent",
                hover_name="state_name",
                hover_data=hover_cols,  # Problem 6 — richer hover info
                color_continuous_scale="RdYlGn",
                title="Funds Released vs Utilized (bubble size = beneficiaries)",
                size_max=45,
            )
            st.plotly_chart(fig4, use_container_width=True)

with tab_analytics:
    st.markdown('<p class="section-title">Correlation Heatmap</p>', unsafe_allow_html=True)
    numeric_df = filtered_df.select_dtypes(include="number").drop(
        columns=[c for c in ["_util_over_100", "_no_fund_data"] if c in filtered_df.columns], errors="ignore"
    )
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True).round(2)
        fig5 = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r",
            aspect="auto",
        )
        fig5.update_layout(height=600)
        st.plotly_chart(fig5, use_container_width=True)

        # Problem 7 — automatic interpretation of strongest relationship
        corr_pairs = corr.where(~np.eye(len(corr), dtype=bool)).unstack().dropna()
        if not corr_pairs.empty:
            strongest = corr_pairs.abs().idxmax()
            strongest_val = corr.loc[strongest[0], strongest[1]]
            direction = "Positive" if strongest_val > 0 else "Negative"
            strength = "Strong" if abs(strongest_val) > 0.7 else "Moderate" if abs(strongest_val) > 0.4 else "Weak"
            st.markdown(
                f'<div class="insight-box">🔍 <b>{strongest[0]}</b> and <b>{strongest[1]}</b> — '
                f'Correlation = <b>{strongest_val:.2f}</b> ({strength} {direction} relationship)</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Not enough numeric columns for a correlation heatmap.")

    st.markdown('<p class="section-title">Distribution of Fund Utilization %</p>', unsafe_allow_html=True)
    if has_util:
        util_for_plot = filtered_df[~filtered_df.get("_no_fund_data", False)]["fund_utilization_percent"].dropna()
        d1, d2 = st.columns(2)
        with d1:
            fig6 = px.box(filtered_df, y="fund_utilization_percent", points="all",
                          title="Boxplot — Utilization %")
            st.plotly_chart(fig6, use_container_width=True)
        with d2:
            fig7 = px.histogram(filtered_df, x="fund_utilization_percent", nbins=15,
                                title="Histogram — Utilization %")
            if not util_for_plot.empty:
                mean_val = util_for_plot.mean()
                median_val = util_for_plot.median()
                fig7.add_vline(x=mean_val, line_dash="solid", line_color="#4C9AFF",
                                annotation_text=f"Mean {mean_val:.1f}%", annotation_position="top")
                fig7.add_vline(x=median_val, line_dash="dash", line_color="#E76F51",
                                annotation_text=f"Median {median_val:.1f}%", annotation_position="bottom")
            st.plotly_chart(fig7, use_container_width=True)

        # Statistical summary — supports Problem 7/12 style analytical depth
        if not util_for_plot.empty:
            st.markdown('<p class="section-title">Statistical Summary — Utilization %</p>', unsafe_allow_html=True)
            stats = {
                "Mean": util_for_plot.mean(),
                "Median": util_for_plot.median(),
                "Std Dev": util_for_plot.std(),
                "Min": util_for_plot.min(),
                "Max": util_for_plot.max(),
                "Skewness": util_for_plot.skew(),
            }
            stat_cols = st.columns(len(stats))
            for col, (label, val) in zip(stat_cols, stats.items()):
                col.metric(label, f"{val:.2f}")

with tab_map:
    st.markdown('<p class="section-title">India Map — Fund Utilization by State</p>', unsafe_allow_html=True)
    st.caption(
        "Note: rendering this map requires internet access to fetch India state boundaries "
        "(GeoJSON) at runtime. If it doesn't render, the bar chart in the Overview tab shows "
        "the same data by state."
    )
    if {"state_name", "fund_utilization_percent"}.issubset(filtered_df.columns):
        try:
            map_hover_cols = [c for c in ["total", "funds_released_crore", "funds_utilized_crore"] if c in filtered_df.columns]
            fig_map = px.choropleth(
                filtered_df,
                geojson=INDIA_GEOJSON_URL,
                featureidkey="properties.NAME_1",
                locations="state_name",
                color="fund_utilization_percent",
                color_continuous_scale="RdYlGn",
                hover_name="state_name",
                hover_data=map_hover_cols,  # Problem 4 — hover tooltip with detail
                title="Fund Utilization % by State",
            )
            fig_map.update_geos(fitbounds="locations", visible=False)
            fig_map.update_layout(height=650)
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.info(f"Map could not be rendered ({e}). Showing ranked bar chart instead.")
            fig_fallback = px.bar(
                filtered_df.sort_values("fund_utilization_percent", ascending=False),
                x="state_name", y="fund_utilization_percent",
                title="Fund Utilization % by State (map fallback)",
            )
            st.plotly_chart(fig_fallback, use_container_width=True)

with tab_data:
    st.markdown('<p class="section-title">Full Filtered Dataset</p>', unsafe_allow_html=True)

    # Problem 10 — global search across all text/number columns, not just state
    search = st.text_input("🔍 Search across all columns (state, funds, beneficiaries, utilization, etc.)")

    table = filtered_df.drop(columns=[c for c in ["_util_over_100", "_no_fund_data"] if c in filtered_df.columns], errors="ignore").copy()

    if search:
        mask = table.apply(lambda row: row.astype(str).str.contains(search, case=False, na=False).any(), axis=1)
        table = table[mask]

    if table.empty:
        st.markdown(
            """<div class="empty-state">
            🔍 <b>No states found.</b><br>
            Try changing filters or clearing your search.
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        numeric_cols_for_sort = table.select_dtypes(include="number").columns.tolist()
        if numeric_cols_for_sort:
            sort_col = st.selectbox("Sort by", options=numeric_cols_for_sort)
            sort_dir = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
            table = table.sort_values(sort_col, ascending=(sort_dir == "Ascending"))

        # Problem 9 — rows-per-page pagination control
        total_rows = len(table)
        page_size_options = [10, 20, 50, 100, "All"]
        page_size = st.selectbox("Rows per page", options=page_size_options, index=1)

        if page_size == "All" or total_rows == 0:
            page_table = table
        else:
            n_pages = max(1, int(np.ceil(total_rows / page_size)))
            page_num = st.number_input("Page", min_value=1, max_value=n_pages, value=1, step=1)
            start = (page_num - 1) * page_size
            end = start + page_size
            page_table = table.iloc[start:end]
            st.caption(f"Showing rows {start + 1}–{min(end, total_rows)} of {total_rows}")

        # Improvement #8 — conditional formatting / heat colors on the table,
        # closer to a Power BI-style look, instead of a plain grid.
        style_subset = [c for c in ["fund_utilization_percent", "funds_released_crore", "funds_utilized_crore", "total"] if c in page_table.columns]
        if style_subset:
            styled_table = page_table.style.background_gradient(
                subset=style_subset, cmap="RdYlGn", axis=0
            )
            if "fund_utilization_percent" in page_table.columns:
                styled_table = styled_table.format({"fund_utilization_percent": "{:.1f}"})
            st.dataframe(styled_table, use_container_width=True, height=400)
        else:
            st.dataframe(page_table, use_container_width=True, height=400)

        # Problem 11 — downloads reflect the current filtered (and searched/sorted) selection
        st.caption(f"Downloads below include all **{total_rows}** filtered rows (not just the current page).")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download filtered data as CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="pmay_filtered_data.csv",
                mime="text/csv",
            )
        with dl2:
            try:
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    table.to_excel(writer, index=False, sheet_name="PMAY Data")
                st.download_button(
                    "⬇️ Download filtered data as Excel",
                    data=buffer.getvalue(),
                    file_name="pmay_filtered_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            except ImportError:
                st.caption("Install `openpyxl` to enable Excel export.")


# ----------------------------------------------------------
# Footer (Improvement #1)
# ----------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer-box">
<b>PMAY Fund Analysis Dashboard</b> — Version 1.0<br>
Created by <b>Bhavya</b> &nbsp;|&nbsp; Data Analyst Portfolio Project<br>
Python | Pandas | Plotly | Streamlit<br>
Data source: PMAY-G Open Government Data (data.gov.in) — cleaned and aggregated for analysis
</div>
""", unsafe_allow_html=True)