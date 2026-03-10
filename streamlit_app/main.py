"""
Federal Reserve Interest Rate Prediction
Professional ML Analysis — Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
import json, os
from PIL import Image

st.set_page_config(
    page_title="FED Rate ML Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS = os.path.join(BASE, "ml_analysis", "outputs", "charts")
RES    = os.path.join(BASE, "ml_analysis", "outputs", "results")
DATA   = os.path.join(BASE, "App", "Tabs", "Datasets", "finaldataset.csv")

@st.cache_data
def load_results():
    with open(f"{RES}/regression_results.json")      as f: reg   = json.load(f)
    with open(f"{RES}/classification_results.json")  as f: cls   = json.load(f)
    with open(f"{RES}/clustering_results.json")       as f: clust = json.load(f)
    with open(f"{RES}/arm_results.json")              as f: arm   = json.load(f)
    with open(f"{RES}/pca_results.json")              as f: pca   = json.load(f)
    with open(f"{RES}/statistical_analysis.json")     as f: stat  = json.load(f)
    return reg, cls, clust, arm, pca, stat

@st.cache_data
def load_data():
    df = pd.read_csv(DATA)
    df['date'] = pd.to_datetime(df['date'])
    return df

def show_chart(cat, fname, caption=""):
    path = os.path.join(CHARTS, cat, fname)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Chart missing: {cat}/{fname}")

def insight_box(text, kind="info"):
    icons = {"info": "💡", "success": "✅", "warning": "⚠️", "finding": "🔍", "conclusion": "📌"}
    icon  = icons.get(kind, "💡")
    colors = {
        "info":       ("#e8f4fd", "#1565c0", "#bbdefb"),
        "success":    ("#e8f5e9", "#2e7d32", "#c8e6c9"),
        "warning":    ("#fff8e1", "#f57f17", "#ffe082"),
        "finding":    ("#f3e5f5", "#6a1b9a", "#e1bee7"),
        "conclusion": ("#fce4ec", "#880e4f", "#f8bbd0"),
    }
    bg, text_c, border = colors.get(kind, colors["info"])
    st.markdown(f"""
    <div style="background:{bg};border-left:4px solid {border};border-radius:0 8px 8px 0;
                padding:0.9rem 1.2rem;margin:0.7rem 0;color:{text_c};">
      <strong>{icon} {text}</strong>
    </div>""", unsafe_allow_html=True)

def section_divider(title):
    st.markdown(f"""
    <div style="background:linear-gradient(90deg,#1e3a5f,#2196F3);color:white;
                padding:0.5rem 1.2rem;border-radius:8px;margin:1.5rem 0 1rem;">
      <strong style="font-size:1rem;">▸ {title}</strong>
    </div>""", unsafe_allow_html=True)

def metric_row(metrics: dict):
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)

reg_res, cls_res, clust_res, arm_res, pca_res, stat_res = load_results()
df = load_data()

# ── Professional CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1e3a5f 60%, #2c5282 100%) !important;
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio > label { font-size: 0.78rem !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

/* Main header */
h1 { color: #1e3a5f !important; border-bottom: 3px solid #2196F3;
     padding-bottom: 0.4rem; margin-bottom: 1.2rem; }
h2 { color: #1e3a5f !important; }
h3 { color: #2c5282 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: white; border-radius: 10px;
    padding: 0.8rem 1rem; border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
[data-testid="stMetricValue"] { color: #1e3a5f !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #718096 !important; font-size: 0.78rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-weight: 600; color: #4a5568;
    border-radius: 8px 8px 0 0;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1e3a5f; border-bottom: 3px solid #2196F3;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Expander */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
}

/* General text */
p { color: #2d3748; line-height: 1.7; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:0.5rem 0 1rem;">
      <div style="font-size:2rem;">📈</div>
      <div style="font-weight:800;font-size:1rem;color:white;">FED Rate Prediction</div>
      <div style="font-size:0.72rem;color:rgba(255,255,255,0.65);margin-top:3px;">
        Professional ML Analysis
      </div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate to:", [
        "🏠  Overview",
        "🧹  Data Cleaning",
        "🔍  EDA",
        "⚙️  Feature Engineering",
        "📊  Statistical Analysis",
        "🔵  PCA",
        "📉  Regression Models",
        "🎯  Classification Models",
        "🔮  Clustering",
        "🔗  Association Rules",
        "🏆  Model Comparison",
        "📋  Conclusion",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("""
    <div style="font-size:0.75rem;color:rgba(255,255,255,0.6);line-height:1.9;">
    <b style="color:rgba(255,255,255,0.9);">Dataset</b><br>
    FRED Economic Data<br>
    1954–2024 (70 years)<br>
    834 samples · 67 features<br>
    <br>
    <b style="color:rgba(255,255,255,0.9);">Models Trained</b><br>
    8 Regression · 7 Classification<br>
    2 Clustering · 1 ARM<br>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.title("Federal Reserve Rate Prediction — ML Analysis")
    st.markdown("""
    This project applies a **complete industry-standard ML pipeline** to 70 years of US macroeconomic
    data (1954–2024) from the Federal Reserve Economic Database (FRED). The goal is to predict
    Federal Reserve interest rate decisions — one of the most consequential economic signals in global finance.
    """)

    best_reg = max(reg_res, key=lambda x: reg_res[x]['R2'])
    best_cls = max(cls_res, key=lambda x: cls_res[x]['Accuracy'])
    metric_row({
        "Dataset Size": "834 samples",
        "Engineered Features": "67",
        "Algorithms": "15+",
        f"Best R² ({best_reg})": f"{reg_res[best_reg]['R2']:.3f}",
        f"Best Accuracy ({best_cls})": f"{cls_res[best_cls]['Accuracy']:.1%}",
        "Historical Coverage": "70 years",
    })

    st.markdown("---")
    section_divider("End-to-End ML Pipeline")
    steps = [
        ("🗃️", "Raw Data",        "FRED API\n1954–2024"),
        ("🧹", "Cleaning",        "Imputation\nWinsorization"),
        ("⚙️", "Feature Eng.",    "67 features\nLag, Rolling"),
        ("📊", "Stat Analysis",   "T-test, ANOVA\nADF, Shapiro"),
        ("🔵", "PCA",             "4 components\n95.7% variance"),
        ("🤖", "Model Train",     "8 Reg. · 7 Cls.\nClustering · ARM"),
        ("🔁", "Cross-Val",       "TimeSeriesSplit\n5 folds"),
        ("📈", "Evaluation",      "RMSE · R² · ACC\nROC · Confusion"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, title, desc) in zip(cols, steps):
        col.markdown(f"""
        <div style="background:white;border:1px solid #e2e8f0;border-top:3px solid #2196F3;
                    border-radius:10px;padding:0.8rem 0.5rem;text-align:center;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);">
          <div style="font-size:1.4rem;">{icon}</div>
          <div style="font-weight:700;font-size:0.82rem;color:#1e3a5f;margin:4px 0;">{title}</div>
          <div style="font-size:0.7rem;color:#718096;white-space:pre-line;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        section_divider("Problem Statements")
        st.success("**📉 Regression:** Predict the exact Federal Funds Rate as a continuous value. This is useful for bond pricing, risk models, and macroeconomic forecasting.")
        st.info("**🎯 Classification:** Predict the *direction* of the next rate change — Increase, Decrease, or No Change. Relevant for trading strategy and portfolio allocation.")
        st.warning("**🔮 Unsupervised:** Identify hidden economic regimes via DBSCAN and KMeans clustering. Discover economic co-movement rules using Apriori ARM.")

        section_divider("Algorithms Implemented")
        algos = [
            ("Regression", "#2196F3",
             ["Linear Regression","Ridge (L2)","Lasso (L1)","SVR",
              "Decision Tree","Random Forest","Gradient Boosting","XGBoost"]),
            ("Classification", "#E91E63",
             ["Logistic Regression","Naive Bayes","SVM","Decision Tree",
              "Random Forest","Gradient Boosting","XGBoost"]),
            ("Unsupervised", "#FF9800",
             ["KMeans","DBSCAN","Apriori ARM"]),
        ]
        for group, color, alg_list in algos:
            tags = "".join(f'<span style="display:inline-block;background:{color}18;color:{color};border:1px solid {color}44;border-radius:20px;padding:2px 10px;font-size:0.75rem;font-weight:600;margin:2px;">{a}</span>' for a in alg_list)
            st.markdown(f'<p style="margin:6px 0 2px;font-weight:700;font-size:0.85rem;color:#4a5568;">{group}</p>{tags}', unsafe_allow_html=True)

    with col2:
        show_chart("eda", "01_fedfunds_timeseries.png")
        insight_box(
            "The FED Funds Rate peaked at 19.10% in 1981 during the Volcker shock to curb runaway inflation. "
            "Post-2008, rates were held near 0% for 7 years (ZIRP). The 2022–2023 hiking cycle was the "
            "steepest in 40 years. These structural shifts make forward prediction genuinely challenging.",
            "finding"
        )

    st.markdown("---")
    section_divider("Dataset — FRED Economic Indicators (1954–2024)")
    feat_df = pd.DataFrame({
        "Feature": ["FEDRates","ConsumerPriceIndex","GDP","InflationConsumerPrice",
                    "MedianCPI","RealGDP","RealGDPPerCapita","RealPotentialGDP","UnemploymentRate"],
        "Description": [
            "Effective Federal Funds Rate — monthly average",
            "CPI percent change (month-over-month)",
            "Nominal Gross Domestic Product (Billions USD)",
            "Annual consumer price inflation rate (%)",
            "Median Consumer Price Index (Dallas Fed)",
            "Real GDP — inflation-adjusted (Billions 2017 USD)",
            "Real GDP per person (2017 USD)",
            "CBO estimate of economy's productive capacity",
            "U-3 Unemployment Rate (%)"],
        "Role": ["🎯 Target (Regression)","Feature","Feature","Feature","Feature",
                 "Feature","Feature","Feature","Feature"],
        "Source": ["FRED: FEDFUNDS","FRED: CPIAUCSL","FRED: GDP","World Bank","Dallas Fed",
                   "FRED: GDPCA","FRED: A939RX0Q048SBEA","FRED: GDPPOT","FRED: UNRATE"],
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧹  Data Cleaning":
    st.title("Data Cleaning & Quality Assurance")
    st.markdown("""
    Raw economic data from FRED arrives with several quality issues that must be resolved before
    any analysis. This section documents every transformation applied and the reasoning behind each decision.
    """)

    section_divider("Raw Data Overview")
    with st.expander("🗃️ Inspect Raw Dataset (first 20 rows)", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Raw Rows", df.shape[0])
        c2.metric("Columns", df.shape[1]-1, "features + target")
        c3.metric("Date Start", str(df['date'].min().year))
        c4.metric("Date End", str(df['date'].max().year))

    st.markdown("---")
    section_divider("Missing Value Analysis")
    col1, col2 = st.columns([1, 1])
    with col1:
        show_chart("cleaning", "01_missing_values.png")
    with col2:
        null_data = df.isnull().sum().reset_index()
        null_data.columns = ["Feature","Missing Count"]
        null_data["Missing %"] = (null_data["Missing Count"] / len(df) * 100).round(2)
        st.dataframe(null_data, use_container_width=True, hide_index=True)
        insight_box(
            "No structural NaN values were found. However, 111 entries in ConsumerPriceIndexAllItems "
            "contain literal 0.0 values — a sentinel used for early FRED data where percent change "
            "couldn't be computed (data unavailability, not true zero change). These are treated as "
            "missing and interpolated.", "warning"
        )
        insight_box(
            "Strategy: Forward-fill followed by backward-fill preserves the time-series continuity "
            "of economic data. Simply dropping rows would destroy 13% of the dataset and create "
            "temporal gaps.", "info"
        )

    st.markdown("---")
    section_divider("Outlier Detection & Winsorization")
    col1, col2 = st.columns(2)
    with col1:
        show_chart("cleaning", "02_boxplots_before_after.png")
        insight_box(
            "Box plots before (top row) and after (bottom row) Winsorization at the 1st/99th "
            "percentile. The key change is in InflationConsumerPrice — extreme values from the "
            "1970s oil shocks are capped rather than removed, preserving the economic signal "
            "while preventing these points from dominating model training.", "finding"
        )
    with col2:
        show_chart("cleaning", "03_outlier_counts.png")
        insight_box(
            "IQR × 3 extreme outliers found: Inflation (12 entries — 1970s stagflation, 2022 surge), "
            "CPI (4 entries), Unemployment (1 entry — COVID-19 spike to 14.8% in April 2020). "
            "These represent genuine rare economic events, not data errors.", "warning"
        )
        insight_box(
            "Why Winsorization over deletion? Deleting outliers would erase historically significant "
            "economic events (Volcker shock, 2008 crisis, COVID). Winsorization preserves the "
            "observation while limiting its leverage on the model.", "success"
        )

    st.markdown("---")
    section_divider("Cleaning Steps Summary")
    steps_df = pd.DataFrame({
        "Step": ["1. Drop missing target","2. Forward/Backward fill","3. Fix zero CPI",
                 "4. Winsorization","5. Result"],
        "Action": [
            "Remove rows where FEDRates is NaN",
            "Fill remaining NaNs with adjacent values (time-series safe)",
            "Replace 111 zero-CPI entries with linear interpolation",
            "Clip each feature at 1st and 99th percentile",
            "Clean, complete dataset ready for feature engineering"],
        "Rows Affected": ["0","~26","111","17","846 clean rows"],
        "Rationale": [
            "Target must be present for supervised learning",
            "Preserves temporal continuity; no data loss",
            "Economic data was unavailable, not actually zero",
            "Prevents extreme leverage without discarding rare events",
            "100% complete — no missing values remaining"],
    })
    st.dataframe(steps_df, use_container_width=True, hide_index=True)
    insight_box(
        "Conclusion: All 846 monthly observations were retained. No rows were dropped, ensuring "
        "the full 70-year economic cycle is available for training. The cleaned dataset captures "
        "multiple full business cycles across different monetary policy regimes.", "conclusion"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  EDA":
    st.title("Exploratory Data Analysis")
    st.markdown("""
    EDA reveals the distributional properties, temporal structure, and inter-relationships
    of all economic indicators before modelling. Key questions: What does the data look like?
    Are there trends? Which features correlate with FED rates?
    """)

    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "📊 Distributions", "🔥 Correlations"])

    with tab1:
        section_divider("Federal Funds Rate — 70-Year History")
        show_chart("eda", "01_fedfunds_timeseries.png")
        st.markdown("""
        **Reading the chart:** The blue line is the monthly Federal Funds Rate. Red shaded bands
        mark NBER-defined recession periods.

        **Key observations:**
        - **1954–1965:** Rates gradually rose from 0.8% to ~4%, reflecting post-WWII economic expansion
        - **1965–1981:** Progressive tightening to combat Vietnam-era and oil-shock inflation, peaking at **19.1%** (Volcker Shock)
        - **1982–2000:** Secular decline as inflation was tamed; the "Great Moderation" period
        - **2001–2004:** Rapid cuts to 1% following the dot-com crash and 9/11 shock
        - **2008–2015:** Zero Interest Rate Policy (ZIRP) — rates held near 0% for 7 years post-GFC
        - **2022–2023:** Fastest hiking cycle in 40 years (+525bps in 16 months) to combat post-COVID inflation
        """)
        insight_box(
            "The data exhibits clear non-stationarity — the rate level has completely different "
            "statistical properties across decades. This creates the core modelling challenge: "
            "a model trained on 1970s high-inflation data cannot simply extrapolate to the 2010s "
            "ZIRP environment. This is precisely why TimeSeriesSplit CV gives lower scores than "
            "standard K-Fold.", "warning"
        )
        st.markdown("---")
        section_divider("All Economic Indicators — Time Series")
        show_chart("eda", "02_all_features_timeseries.png")
        insight_box(
            "GDP and RealGDP show consistent upward trends driven by 70 years of economic growth — "
            "these are non-stationary and will require differencing or growth-rate transformation "
            "for linear models. InflationConsumerPrice is mean-reverting and stationary. "
            "Unemployment shows clear cyclical spikes at recession periods.", "finding"
        )

    with tab2:
        section_divider("Feature Distributions")
        show_chart("eda", "03_distributions.png")
        st.markdown("""
        **What to look for:** The red dashed line = mean, green = median. Large mean-median divergence
        indicates skewness; flat, wide distributions indicate high variance.

        **Analysis:**
        - **FEDRates:** Bimodal — high-rate 1970s-80s cluster and low-rate modern cluster. Mean ≠ Median confirms right-skew
        - **GDP/RealGDP/RealPotentialGDP:** Strong right-skew from decades of compounding economic growth
        - **InflationConsumerPrice:** Near-normal with a positive tail from the 1970s inflation surge
        - **UnemploymentRate:** Moderately right-skewed; most months cluster around 4–6%
        """)
        insight_box(
            "All features fail the Shapiro-Wilk normality test (p << 0.05), which is confirmed by "
            "the non-Gaussian shapes shown here. This justifies using non-parametric statistical "
            "tests (Spearman vs Pearson) and motivates tree-based ML models over linear classifiers "
            "that assume normality.", "finding"
        )
        st.markdown("---")
        section_divider("Scatter Matrix — Key Indicators")
        show_chart("eda", "06_scatter_matrix.png")
        insight_box(
            "GDP and RealGDP show near-perfect collinearity (diagonal alignment in scatter). "
            "This multicollinearity is why PCA reduces 8 features to 4 components while retaining 95.7% "
            "of variance — the GDP-family features are largely redundant. Lasso regression exploits "
            "this by zeroing out 55% of features.", "finding"
        )

    with tab3:
        section_divider("Pearson Correlation Matrix")
        show_chart("eda", "04_correlation_heatmap.png")
        st.markdown("""
        **Interpreting correlations with FEDRates:**
        - **InflationConsumerPrice (r=0.72):** Strongest positive correlation — high inflation historically drives high rates
        - **MedianCPI (r=0.45):** Second strongest — confirms inflation signal
        - **GDP/RealGDP/RealPotentialGDP (r≈-0.40 to -0.45):** Negative — as the overall economy has grown over decades, rates have also come down (secular trend)
        - **UnemploymentRate (r≈-0.02):** Nearly no linear correlation — but highly non-linear (high rates during high unemployment in stagflation vs low rates during high unemployment in recessions)

        **Warning:** The negative GDP correlation reflects a secular trend (rates trending down as economy grows), not a causal relationship. Do not interpret as "economic growth lowers rates."
        """)
        insight_box(
            "The high inter-feature correlation among GDP, RealGDP, RealGDPPerCapita, and "
            "RealPotentialGDP (r > 0.99) confirms severe multicollinearity. Ridge and Lasso "
            "regularization are specifically designed to handle this — Ridge shrinks correlated "
            "coefficients together while Lasso eliminates the redundant ones.", "conclusion"
        )
        st.markdown("---")
        section_divider("Scatter Plots vs FED Rate (Time-Colored)")
        show_chart("eda", "05_scatter_vs_fedrate.png")
        insight_box(
            "Color encodes time (dark = 1954, bright = 2024). The curved, non-linear patterns "
            "confirm that simple linear relationships cannot capture the full story — "
            "the Inflation-Rate relationship has completely different regimes pre- and post-2000. "
            "This motivates ensemble methods (Random Forest, XGBoost) over linear models.", "finding"
        )
        st.markdown("---")
        section_divider("Descriptive Statistics")
        st.dataframe(df.select_dtypes(include=np.number).describe().T.round(4),
                     use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Feature Engineering":
    st.title("Feature Engineering")
    st.markdown("""
    Raw economic indicators tell us *what the economy looks like today*. But the Federal Reserve
    makes policy based on *trends, momentum, and forward expectations*. Feature engineering
    transforms static snapshots into a rich temporal feature set that captures these dynamics.
    """)

    metric_row({
        "Raw Features": "8",
        "Lag Features": "32  (×4 lags)",
        "Rolling Stats": "24  (mean + std)",
        "Interaction Terms": "5",
        "Calendar Features": "3",
        "Total Features": "67",
    })

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Classification Target",
        "📅 Lag Features",
        "📊 Rolling Statistics",
        "🔗 Interactions & Growth",
    ])

    with tab1:
        section_divider("Rate Direction — Classification Target")
        col1, col2 = st.columns(2)
        with col1:
            show_chart("features", "01_class_distribution.png")
        with col2:
            st.markdown("""
            **How the target is defined:**

            The FED Funds Rate changes are computed month-over-month (Δ = Rate_t - Rate_{t-1}):
            - **Increase:** Δ > +0.05% (a meaningful rate hike)
            - **Decrease:** Δ < -0.05% (a meaningful rate cut)
            - **No Change:** -0.05% ≤ Δ ≤ +0.05% (unchanged or trivial movement)

            The 0.05% threshold filters out rounding noise while capturing genuine policy decisions.
            """)
            insight_box(
                "Class distribution: No_Change (40.8%), Increase (33.6%), Decrease (25.6%). "
                "The mild imbalance is handled via stratified train/test splitting — each split "
                "maintains the original class proportions. Weighted F1-score is used as the "
                "primary metric to account for this imbalance.", "finding"
            )
            insight_box(
                "The Federal Reserve makes discrete, deliberate rate changes at FOMC meetings "
                "(typically 8 per year). Between meetings, rates don't change — explaining the "
                "large 'No Change' class. This creates a natural challenge: predicting whether "
                "a meeting will produce a change.", "info"
            )

    with tab2:
        section_divider("Lag Features — Capturing Monetary Policy Momentum")
        col1, col2 = st.columns(2)
        with col1:
            show_chart("features", "03_lag_correlations.png")
        with col2:
            st.markdown("""
            **What lag features represent:**
            For each of the 8 raw features, we create 4 lags:
            - **Lag-1:** Previous month's value (most recent signal)
            - **Lag-3:** Quarter-ago value (short-term trend)
            - **Lag-6:** Half-year ago value (medium-term trend)
            - **Lag-12:** Year-ago value (annual momentum)

            **Why this matters:** The Fed doesn't react instantly to economic data.
            Policy decisions reflect a forward-looking view of trends that built up over
            months. A 12-month lag of inflation captures whether inflation has been
            *persistently* high — a key trigger for rate hikes.
            """)
            insight_box(
                "Inflation lag features (1–12 months) consistently show the highest correlation "
                "with FED rates. The 12-month lag of InflationConsumerPrice achieves r ≈ 0.70 — "
                "comparable to the raw feature. This confirms that the Fed responds to persistent "
                "inflation trends rather than single-month spikes.", "conclusion"
            )

    with tab3:
        section_divider("Rolling Statistics — Smoothing Economic Noise")
        st.markdown("""
        Monthly economic data is noisy. A single month's CPI can jump for one-off reasons
        (energy price spikes, seasonal effects) that don't reflect underlying trends.
        Rolling windows smooth this noise:
        - **3-month rolling mean:** Short-term smoothed trend
        - **6-month rolling mean:** Medium-term smoothed trend
        - **3-month rolling std:** Volatility measure — high std = uncertain/volatile period
        """)
        insight_box(
            "Rolling standard deviation features capture economic volatility regimes. "
            "High inflation volatility (1970s-80s) is a distinct signal from stable low "
            "inflation (2000s). This temporal volatility clustering is well-captured by "
            "ensemble models but not by linear regression.", "info"
        )
        section_divider("Top 20 Correlated Features")
        show_chart("features", "02_feature_correlations.png")
        insight_box(
            "The dominance of lag and rolling features over raw features in the correlation "
            "ranking confirms that temporal context (what was happening 1–12 months ago) is more "
            "predictive than the current snapshot alone. Lasso regression later independently "
            "confirms this by retaining mostly lag features.", "conclusion"
        )

    with tab4:
        section_divider("Interaction & Growth Features")
        st.markdown("""
        **Derived features created:**

        | Feature | Formula | Economic Meaning |
        |---------|---------|-----------------|
        | Inflation_x_Unemployment | Inflation × Unemployment | Phillips Curve proxy — stagflation indicator |
        | GDP_growth | Δ GDP / GDP × 100 | Nominal economic growth rate |
        | RealGDP_growth | Δ RealGDP / RealGDP × 100 | Real economic growth (inflation-adjusted) |
        | Month, Quarter | Calendar | Seasonality — Fed meets on a fixed schedule |
        | Year | Calendar | Long-term regime indicator |
        """)
        insight_box(
            "The Phillips Curve interaction term (Inflation × Unemployment) captures stagflation "
            "periods — the 1970s combination of high inflation AND high unemployment that was "
            "historically unprecedented. This interaction is nonzero only in these regime-breaking "
            "periods and helps the model distinguish them from normal inflationary periods.", "finding"
        )
        with st.expander("📄 Complete Feature Engineering Code", expanded=False):
            st.code("""
# ── Lag features ──
for col in features:
    for lag in [1, 3, 6, 12]:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# ── Rolling statistics ──
for col in features:
    df[f'{col}_roll3_mean'] = df[col].rolling(3).mean()
    df[f'{col}_roll6_mean'] = df[col].rolling(6).mean()
    df[f'{col}_roll3_std']  = df[col].rolling(3).std()

# ── Interaction & growth features ──
df['Inflation_x_Unemployment'] = df['InflationConsumerPrice'] * df['UnemployemenrRate']
df['GDP_growth']     = df['GDP'].pct_change() * 100
df['RealGDP_growth'] = df['RealGDP'].pct_change() * 100

# ── Classification target ──
df['RateChange']    = df['FEDRates'].diff()
df['RateDirection'] = 'No_Change'
df.loc[df['RateChange'] >  0.05, 'RateDirection'] = 'Increase'
df.loc[df['RateChange'] < -0.05, 'RateDirection'] = 'Decrease'

# ── Calendar ──
df['Month'] = df.index.month
df['Year']  = df.index.year
df['Quarter'] = df.index.quarter

# Drop NaN rows from lag/rolling creation
df = df.dropna()   # 846 → 834 rows
""", language="python")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Statistical Analysis":
    st.title("Statistical Analysis & Hypothesis Testing")
    st.markdown("""
    Before building ML models, we rigorously validate statistical assumptions and test key
    economic hypotheses. This step ensures our modelling choices are grounded in the data's
    statistical properties and reveals actionable economic insights.
    """)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📐 Normality Tests",
        "📉 Stationarity (ADF)",
        "🔬 T-Test & ANOVA",
        "📈 Rank Correlations",
    ])

    with tab1:
        section_divider("Shapiro-Wilk Normality Test")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            show_chart("stats", "01_qq_plots.png")
        with col2:
            norm_df = pd.DataFrame(stat_res['normality']).T
            norm_df['Normal?'] = norm_df['is_normal'].map({True:'✅ Yes', False:'❌ No'})
            norm_df = norm_df[['statistic','p_value','Normal?']].rename(
                columns={'statistic':'W-stat','p_value':'p-value'})
            st.dataframe(norm_df.round(6), use_container_width=True)
            st.markdown("""
            **Q-Q Plot interpretation:** Points lying on the red diagonal line = normally distributed.
            Deviations at the tails = heavy tails. S-curves = skewness.

            **All features show significant deviations** — confirmed by Shapiro-Wilk p << 0.05.
            """)
        insight_box(
            "Non-normality confirmed across all features. This has three important implications: "
            "(1) Pearson correlation may understate true relationships — Spearman rank correlation "
            "is more appropriate; (2) Parametric models assuming normality (e.g., basic linear "
            "regression inference) need careful interpretation; (3) Non-parametric ensemble methods "
            "(Random Forest, XGBoost) are naturally better suited to this data.", "conclusion"
        )

    with tab2:
        section_divider("Augmented Dickey-Fuller Stationarity Test")
        col1, col2 = st.columns([1.2, 1])
        with col1:
            show_chart("stats", "02_hypothesis_tests.png")
        with col2:
            stat_df = pd.DataFrame(stat_res['stationarity']).T
            stat_df['Stationary?'] = stat_df['is_stationary'].map(
                {True:'✅ Stationary', False:'❌ Non-Stationary'})
            stat_df = stat_df[['adf_stat','p_value','Stationary?']].rename(
                columns={'adf_stat':'ADF Stat','p_value':'p-value'})
            st.dataframe(stat_df.round(6), use_container_width=True)
            st.markdown("""
            **ADF Null Hypothesis:** Series has a unit root (non-stationary).
            **p < 0.05** → Reject null → Series is stationary.
            """)
        insight_box(
            "GDP, RealGDP, RealGDPPerCapita, and RealPotentialGDP are all non-stationary — "
            "they have clear upward trends driven by decades of economic growth. FEDRates itself "
            "is non-stationary (it has stayed in different regimes for decades). "
            "InflationConsumerPrice and UnemploymentRate are stationary, mean-reverting series. "
            "This motivates using growth-rate features (GDP_growth) and lag features rather than "
            "raw levels for the non-stationary variables.", "finding"
        )

    with tab3:
        section_divider("T-Test: FED Rates in High vs Low Inflation Periods")
        show_chart("stats", "03_ttest_spearman.png")
        ttest = stat_res['ttest_inflation']
        metric_row({
            "High Inflation Mean FED Rate": f"{ttest['high_mean']:.2f}%",
            "Low Inflation Mean FED Rate":  f"{ttest['low_mean']:.2f}%",
            "Difference":                   f"+{ttest['high_mean']-ttest['low_mean']:.2f}%",
            "t-statistic":                  f"{ttest['t_stat']:.3f}",
            "p-value":                      f"{ttest['p_value']:.2e}",
            "Conclusion":                   "✅ Highly Significant",
        })
        insight_box(
            "The t-test confirms that FED rates are significantly higher during high-inflation "
            f"periods ({ttest['high_mean']:.2f}% vs {ttest['low_mean']:.2f}%, Δ={ttest['high_mean']-ttest['low_mean']:.2f}pp). "
            "With t=20.17 and p≈5.7×10⁻⁷⁴, this is one of the strongest economic relationships "
            "in the dataset and directly validates the Federal Reserve's inflation-targeting mandate "
            "(the Taylor Rule).", "success"
        )
        st.markdown("---")
        section_divider("ANOVA: FED Rates Across Economic Regimes")
        anova = stat_res['anova_regimes']
        insight_box(
            f"One-way ANOVA across four economic regime quartiles (Very Low / Low / High / Very High rates): "
            f"F-statistic = {anova['f_stat']:.2f}, p ≈ {anova['p_value']:.1e}. "
            "The near-zero p-value proves that the mean FED rate is statistically different across "
            "all four regimes — there is no single 'typical' rate; the economy operates in "
            "fundamentally different monetary policy states.", "conclusion"
        )
        section_divider("Descriptive Statistics Heatmap")
        show_chart("stats", "04_descriptive_stats_heatmap.png")

    with tab4:
        section_divider("Spearman Rank Correlations with FED Rate")
        col1, col2 = st.columns([1.3, 1])
        with col1:
            show_chart("stats", "03_ttest_spearman.png")
        with col2:
            spear_df = pd.DataFrame(stat_res['spearman']).T.round(4)
            spear_df.columns = ['Spearman ρ', 'p-value']
            spear_df = spear_df.sort_values('Spearman ρ', key=abs, ascending=False)
            spear_df['Significant?'] = spear_df['p-value'].apply(
                lambda p: '✅' if p < 0.05 else '❌')
            st.dataframe(spear_df, use_container_width=True)
        insight_box(
            "InflationConsumerPrice dominates with ρ = 0.676 — the strongest predictor. "
            "All GDP-related features are negatively correlated (secular trend: rates fell as "
            "the economy grew over 70 years). UnemploymentRate shows near-zero rank correlation "
            "(ρ = -0.006, p = 0.87) — not because unemployment is irrelevant, but because the "
            "relationship is highly non-linear (high rates occur with both high AND low unemployment "
            "in different regimes).", "conclusion"
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PCA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵  PCA":
    st.title("Principal Component Analysis (PCA)")
    st.markdown("""
    PCA identifies the axes of maximum variance in our 8-dimensional feature space.
    This reveals the underlying structure of economic data and shows which combinations
    of indicators are most informative — without using the target variable.
    """)

    metric_row({
        "PC1 Variance": f"{pca_res['explained_variance_ratio'][0]*100:.1f}%",
        "PC2 Variance": f"{pca_res['explained_variance_ratio'][1]*100:.1f}%",
        "PC3 Variance": f"{pca_res['explained_variance_ratio'][2]*100:.1f}%",
        "PC4 Variance": f"{pca_res['explained_variance_ratio'][3]*100:.1f}%",
        "PCs for 95%": str(pca_res['n_components_95pct']),
        "PCs for 99%": "5",
    })

    tab1, tab2, tab3 = st.tabs(["📊 Scree & Loadings", "🗺️ Biplot", "📈 Cumulative Variance"])

    with tab1:
        show_chart("pca", "01_scree_loadings.png")
        st.markdown("""
        **Scree plot (left):** The blue bars show each component's individual contribution.
        The red line is cumulative variance. The green dashed line marks 95%.
        The steep drop after PC1 (57.3%) confirms that one axis dominates — economic scale.

        **Loadings heatmap (right):** Shows how each original feature contributes to each PC.
        - **PC1:** All GDP-family features load positively (economic scale factor)
        - **PC2:** Inflation and CPI features load positively (inflation factor)
        - **PC3:** Unemployment loads distinctly (labor market factor)
        """)
        col1, col2, col3 = st.columns(3)
        col1.info("**PC1 (57.3%) — Economic Scale**\nGDP, RealGDP, RealGDPPerCapita, and RealPotentialGDP all load > 0.5. This component essentially measures the *size* of the economy — larger in recent decades.")
        col2.info("**PC2 (18.7%) — Inflation Dynamics**\nInflationConsumerPrice and MedianCPI load strongly. This component captures the inflationary state — high in the 1970s-80s, low in the 2000s-2010s.")
        col3.info("**PC3 (13.7%) — Labor Market**\nUnemploymentRate and CPI change load on this component. Captures cyclical labor market dynamics independent of long-term economic scale.")
        insight_box(
            "Only 4 components are needed to explain 95.7% of variance in 8 original features. "
            "This extreme compression is caused by the near-perfect multicollinearity among "
            "GDP-related features (PC1 alone captures most of their shared variance). "
            "This validates Lasso's strategy of zeroing out redundant GDP features.", "conclusion"
        )

    with tab2:
        show_chart("pca", "02_biplot.png")
        st.markdown("""
        **Biplot interpretation:**
        - **Points** represent monthly observations, colored by rate direction (Green=Increase, Yellow=No Change, Red=Decrease)
        - **Arrows** show how original features map into PC space (arrow direction = feature's influence)
        - Points clustered together are months with similar economic conditions

        **Left panel (PC1 vs PC2):** The GDP arrows all point in the same direction (right), confirming multicollinearity.
        Inflation arrows point upward-right, separating the high-inflation 1970s-80s months from the modern low-inflation era.

        **Right panel (PC1 vs PC3):** Unemployment separates out more distinctly here, capturing cyclical downturns.
        """)
        insight_box(
            "Rate increases (green) cluster in the PC2-positive region (high inflation area), "
            "confirming the inflation-rate relationship. Rate decreases (red) tend to cluster "
            "in recession periods (low PC1, high PC3 = low economy, high unemployment). "
            "This visual separation validates that PCA captures economically meaningful structure.", "finding"
        )

    with tab3:
        show_chart("pca", "03_cumulative_variance.png")
        loadings = pd.DataFrame(pca_res['loadings'])
        st.subheader("PCA Loadings — Full Table")
        st.dataframe(loadings.round(4), use_container_width=True)
        insight_box(
            "PC5 (4.02%) and beyond capture residual variance. Retaining 4 components for ML "
            "would reduce the feature space from 8 to 4 while losing only 4.33% of information. "
            "However, since tree-based models can handle multicollinear features natively (they "
            "select the most informative at each split), we use the full feature set with lag/rolling "
            "features for the final models rather than PCA-reduced features.", "info"
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: REGRESSION MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉  Regression Models":
    st.title("Regression Models — Predicting the Exact FED Rate")
    st.markdown("""
    Eight regression algorithms are trained to predict the Federal Funds Rate as a continuous
    numeric value. Models are evaluated on a held-out 20% test set (most recent observations)
    using RMSE, MAE, and R².
    """)

    st.info("""
    **TimeSeriesSplit Cross-Validation:** Standard K-Fold shuffles time and creates data leakage
    with our lag features (future months leak into training via their lag values). TimeSeriesSplit
    always trains on the past and tests on the future — the only correct protocol for time-series.
    CV R² scores are negative for some models because economic regimes from decades ago don't
    generalize to structurally different future periods. Test set R² values (20% holdout from
    the recent end) remain strong and are the primary evaluation metric.
    """)

    # Summary table
    section_divider("Performance Summary — All Regression Models")
    reg_data = [{
        'Model': m,
        'RMSE': r['RMSE'], 'MAE': r['MAE'], 'R²': r['R2'],
        'CV R² (TimeSeriesSplit)': f"{r['cv']['mean']:.4f} ± {r['cv']['std']:.4f}",
        'Fit Status': r.get('learning_curve',{}).get('status','—'),
    } for m, r in sorted(reg_res.items(), key=lambda x: x[1]['R2'], reverse=True)]
    reg_df = pd.DataFrame(reg_data)
    st.dataframe(
        reg_df.style.highlight_min(subset=['RMSE','MAE'], color='#c8e6c9')
                    .highlight_max(subset=['R²'], color='#c8e6c9'),
        use_container_width=True, hide_index=True
    )
    insight_box(
        "XGBoost achieves R²=0.9816 on the test set — meaning 98.2% of variance in FED rates "
        "is explained by engineered economic features. This strong test performance reflects "
        "that recent economic patterns (2000s–2020s, the test period) are reasonably predictable "
        "given the previous 60+ years of training data.", "success"
    )

    st.markdown("---")
    section_divider("Visual Comparison")
    show_chart("comparison", "reg_model_comparison.png")
    st.markdown("""
    **Reading the charts:**
    - **RMSE (left):** Average prediction error in percentage points. XGBoost's RMSE of 0.44% means
      predictions are typically within ±0.44% of the actual rate.
    - **R² (center):** Proportion of variance explained. Values above 0.95 (green line) indicate
      excellent predictive power on the test set.
    - **MAE (right):** Mean absolute error — less sensitive to large outliers than RMSE.
      Stars indicate the best-performing model in each metric.
    """)
    insight_box(
        "Clear performance tiers emerge: (1) Ensemble methods — XGBoost, GBM, Random Forest "
        "(R² > 0.97); (2) SVR (R²=0.905); (3) Decision Tree (R²=0.874); (4) Linear models — "
        "Linear, Lasso, Ridge (R²=0.71–0.74). The ensemble advantage comes from capturing "
        "non-linear interactions between economic indicators that linear models miss.", "conclusion"
    )

    st.markdown("---")
    model_choice = st.selectbox("🔎 Deep-dive into a specific model:", list(reg_res.keys()))
    m = model_choice
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("RMSE", reg_res[m]['RMSE'], "lower is better")
    c2.metric("MAE",  reg_res[m]['MAE'],  "lower is better")
    c3.metric("R²",   reg_res[m]['R2'],   "higher is better")
    c4.metric("Fit Status", reg_res[m].get('learning_curve',{}).get('status','—'))

    chart_map = {
        "Linear Regression": ("02_lr_actual_vs_pred.png","01_lr_learning_curve.png","03_lr_coefficients.png"),
        "Ridge Regression":  ("05_ridge_actual_vs_pred.png","04_ridge_learning_curve.png","06_ridge_alpha_tuning.png"),
        "Lasso Regression":  ("08_lasso_actual_vs_pred.png","07_lasso_learning_curve.png","09_lasso_feature_selection.png"),
        "SVR":               ("11_svr_actual_vs_pred.png","10_svr_learning_curve.png", None),
        "Decision Tree":     ("13_dt_actual_vs_pred.png","14_dt_depth_overfitting.png","12_dt_learning_curve.png"),
        "Random Forest":     ("16_rf_actual_vs_pred.png","15_rf_learning_curve.png","17_rf_feature_importance.png"),
        "Gradient Boosting": ("19_gb_actual_vs_pred.png","18_gb_learning_curve.png", None),
        "XGBoost":           ("21_xgb_actual_vs_pred.png","20_xgb_learning_curve.png", None),
    }
    charts_m = chart_map.get(m, (None, None, None))
    valid = [c for c in charts_m if c]
    cols = st.columns(len(valid))
    for col_w, fname in zip(cols, valid):
        p = os.path.join(CHARTS, "regression", fname)
        if os.path.exists(p):
            col_w.image(p, use_container_width=True)

    # Per-model description
    descriptions = {
        "Linear Regression": ("**Baseline model.** Assumes a linear relationship between features and FED rate. "
            "R²=0.74 is reasonable but misses the non-linear rate dynamics across economic regimes. "
            "The coefficient chart shows that CPI lag features have the largest positive coefficients. "
            "The residual plot shows systematic over-prediction at high rates and under-prediction at low rates "
            "— confirming that non-linearity is not captured."),
        "Ridge Regression": ("**L2 regularization** penalizes large coefficients, shrinking them toward zero "
            "proportionally. Best alpha=10 means moderate regularization is optimal. Ridge handles "
            "multicollinearity well by distributing weights across correlated GDP features rather than "
            "arbitrarily selecting one. The alpha tuning chart shows CV R² plateau — further increasing "
            "alpha over-constrains the model."),
        "Lasso Regression": ("**L1 regularization** performs automatic feature selection by driving "
            "exactly zero coefficients for redundant features. Best alpha=0.01 (light regularization). "
            "37 of 67 features were zeroed out — mostly redundant GDP-family features. "
            "The retained features (Inflation lags, CPI lags) confirm our EDA findings about the most "
            "predictive variables. Lasso's feature selection is a form of embedded domain knowledge."),
        "SVR": ("**Support Vector Regression** with RBF kernel finds a hyperplane in a high-dimensional "
            "feature space that maximizes the margin around predictions. RBF kernel (γ='scale') "
            "handles non-linear relationships. C=10 allows more flexibility. R²=0.905 — strong performance "
            "but slower to train than ensemble methods. The kernel trick enables SVR to model "
            "complex economic regime transitions."),
        "Decision Tree": ("**Single tree with optimal depth=4.** The depth analysis chart is the key "
            "insight here — it clearly demonstrates the bias-variance tradeoff. Shallow trees "
            "(depth 1-2) underfit both train and test data. Deep trees (depth 10+) memorize training "
            "data (train R²→1.0) but test R² drops — textbook overfitting. Depth=4 balances this tradeoff."),
        "Random Forest": ("**200 trees averaged.** The ensemble effect dramatically reduces overfitting "
            "vs a single Decision Tree. Each tree sees a random feature subset, preventing co-adaptation. "
            "R²=0.975 on test set. Feature importance confirms inflation lag features are the most "
            "informative splits across all 200 trees."),
        "Gradient Boosting": ("**Sequential boosting** corrects residual errors from each previous tree. "
            "Learning rate=0.05 (slow, careful learning) with 200 stages achieves R²=0.979. "
            "The gradual residual correction process captures subtle non-linear patterns that "
            "single trees miss. GBM is particularly good at the extreme rate values "
            "(low post-2008, high 1980s) that other models struggle with."),
        "XGBoost": ("**Best model (R²=0.9816, RMSE=0.44%).** XGBoost adds regularization terms "
            "to standard GBM, preventing overfitting. subsample=0.8 and colsample_bytree=0.8 "
            "add stochasticity. The actual-vs-predicted chart shows near-perfect alignment "
            "across the full rate range. Residuals are homoscedastic — no systematic bias at "
            "any rate level."),
    }
    if m in descriptions:
        insight_box(descriptions[m], "finding")

    st.markdown("---")
    section_divider("Regularization: Ridge vs Lasso Comparison")
    col1, col2, col3 = st.columns(3)
    with col1: show_chart("regression", "06_ridge_alpha_tuning.png")
    with col2: show_chart("regression", "09_lasso_feature_selection.png")
    with col3: show_chart("regression", "14_dt_depth_overfitting.png")
    insight_box(
        "Ridge retains all features (shrinks coefficients to small values). Lasso performs hard "
        "feature selection (zeros out 55% of features). The Decision Tree depth plot demonstrates "
        "the universal bias-variance tradeoff: too simple → underfitting; too complex → overfitting; "
        "the sweet spot is found by cross-validation.", "conclusion"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLASSIFICATION MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Classification Models":
    st.title("Classification Models — Predicting Rate Direction")
    st.markdown("""
    Seven classifiers predict whether the FED will **Increase**, **Decrease**, or **Hold** rates.
    This is inherently harder than regression — it requires predicting the *decision* of a
    committee of economists, not just the level of an economic indicator.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.warning("""
        **Why ~63% accuracy?**

        A random 3-class baseline = **33.3%**. Our best models reach **63.5%** — a +90% improvement.

        The remaining ~37% error reflects genuinely unpredictable factors not in our data:
        FOMC meeting transcripts, Fed chair signals, market expectations, geopolitical shocks.
        No purely economic model can achieve 100% accuracy on this task.
        """)
    with col2:
        st.success("""
        **What 63% accuracy means in practice:**

        If used as a signal in a portfolio strategy, +63% directional accuracy consistently beats
        the 50% breakeven threshold needed for profitability. Academic literature suggests
        4-5% annualized alpha from rate-direction signals — significant in bond markets.
        """)

    section_divider("Performance Summary — All Classification Models")
    cls_data = [{
        'Model': m,
        'Accuracy': r['Accuracy'], 'F1 (Weighted)': r['F1'],
        'Precision': r['Precision'], 'Recall': r['Recall'],
        'CV Accuracy': f"{r['cv']['mean']:.4f} ± {r['cv']['std']:.4f}",
        'Fit': r.get('learning_curve',{}).get('status','—'),
    } for m, r in sorted(cls_res.items(), key=lambda x: x[1]['Accuracy'], reverse=True)]
    cls_df = pd.DataFrame(cls_data)
    st.dataframe(
        cls_df.style.highlight_max(subset=['Accuracy','F1 (Weighted)'], color='#c8e6c9'),
        use_container_width=True, hide_index=True
    )

    show_chart("comparison", "cls_model_comparison.png")
    insight_box(
        "Gradient Boosting (63.5%) and XGBoost (62.9%) top the classification rankings. "
        "The red dashed baseline (33.3%) shows all models learn meaningful patterns. "
        "Naive Bayes (44.3%) struggles most — its feature independence assumption is "
        "severely violated since all economic indicators are highly correlated.", "finding"
    )

    st.markdown("---")
    model_choice_c = st.selectbox("🔎 Explore model in detail:", list(cls_res.keys()))
    mc = model_choice_c
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy",  cls_res[mc]['Accuracy'])
    c2.metric("F1 Score",  cls_res[mc]['F1'])
    c3.metric("Precision", cls_res[mc]['Precision'])
    c4.metric("Recall",    cls_res[mc]['Recall'])

    cm_map = {
        "Logistic Regression": ("02_logreg_cm.png","03_logreg_roc.png","01_logreg_lc.png"),
        "Naive Bayes":         ("05_nb_cm.png","06_nb_roc.png","04_nb_lc.png"),
        "SVM":                 ("08_svm_cm.png","09_svm_roc.png","07_svm_lc.png"),
        "Decision Tree":       ("11_dt_cm.png","12_dt_roc.png","13_dt_depth_overfitting.png"),
        "Random Forest":       ("15_rf_cm.png","16_rf_roc.png","14_rf_lc.png"),
        "Gradient Boosting":   ("19_gb_cm.png","20_gb_roc.png","18_gb_lc.png"),
        "XGBoost":             ("22_xgb_cm.png","23_xgb_roc.png","21_xgb_lc.png"),
    }
    c_charts = cm_map.get(mc, (None, None, None))
    c_cols = st.columns(3)
    for col_w, fname in zip(c_cols, c_charts):
        if fname:
            p = os.path.join(CHARTS, "classification", fname)
            if os.path.exists(p):
                col_w.image(p, use_container_width=True)

    cls_descriptions = {
        "Logistic Regression": ("**Probabilistic linear classifier.** Models the log-odds of each class as a linear "
            "combination of features. Accuracy=55.1%. The ROC curves show AUC > 0.70 for all classes — "
            "the model assigns meaningful probabilities even when the hard prediction is wrong. "
            "The No_Change class has the highest recall (66%) — the model correctly identifies "
            "most stable periods. Rate hikes and cuts are harder because they depend on "
            "cumulative evidence the linear model struggles to synthesize."),
        "Naive Bayes": ("**Probabilistic classifier assuming feature independence.** This assumption "
            "is badly violated here — all economic indicators are correlated. Despite this, "
            "Gaussian NB still achieves 44.3% (well above 33% baseline). The confusion matrix "
            "shows the model heavily predicts No_Change (majority class bias). "
            "Best used as a fast baseline, not a deployment model for this dataset."),
        "SVM": ("**Support Vector Machine with RBF kernel.** Maps features into a high-dimensional "
            "space where a maximum-margin hyperplane separates classes. C=10, γ='scale'. "
            "Accuracy=59.9% — strong performance for a kernel method. "
            "The RBF kernel captures non-linear decision boundaries between rate regimes. "
            "Slower to train than ensemble methods on this dataset size."),
        "Decision Tree": ("**Single tree with depth=1 (found via TimeSeriesCV).** The shallow optimal "
            "depth reflects that any deeper tree overfits — as shown in the depth-vs-accuracy chart. "
            "A depth-1 tree makes predictions based on a single feature split. "
            "Accuracy=46.1%. The lesson: for this complex multi-class problem, single trees "
            "need ensemble support to be effective."),
        "Random Forest": ("**200 trees, depth=10.** The ensemble dramatically improves over a single "
            "tree. Accuracy=59.9% — matches SVM. Feature importance shows rate momentum "
            "(FEDRates_lag1) and inflation lags are the most informative splits. "
            "The confusion matrix shows balanced performance across all three classes, "
            "unlike simpler models that bias toward the majority class."),
        "Gradient Boosting": ("**Best classifier (63.5% accuracy).** Sequential boosting corrects "
            "previous trees' errors, progressively refining the decision boundary. "
            "200 stages, learning rate=0.05. The confusion matrix shows the best balance "
            "across all classes — Decrease (60%), Increase (54%), No_Change (74%). "
            "The learning curve shows convergence without overfitting."),
        "XGBoost": ("**Second best (62.9%).** Regularized gradient boosting with subsample=0.8, "
            "colsample=0.8. The ROC curves show the highest AUC for the Decrease class (~0.77) "
            "— rate cuts tend to follow clear deterioration signals (recessions, financial crises) "
            "that XGBoost captures well. Feature importance confirms inflation lag features "
            "and rate momentum as top predictors."),
    }
    if mc in cls_descriptions:
        insight_box(cls_descriptions[mc], "finding")

    st.markdown("---")
    section_divider("Feature Importance — RF vs XGBoost")
    col1, col2 = st.columns(2)
    with col1: show_chart("classification", "17_rf_feature_importance.png")
    with col2: show_chart("classification", "24_xgb_feature_importance.png")
    insight_box(
        "Both RF and XGBoost independently confirm: lag features of InflationConsumerPrice and "
        "ConsumerPriceIndexAllItems dominate. The 1-month lag of FEDRates (rate momentum) is also "
        "critical — the current rate level provides strong prior information about the next decision. "
        "This is consistent with the Fed's stated practice of 'gradualism' in rate adjustments.", "conclusion"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮  Clustering":
    st.title("Clustering — Discovering Economic Regimes")
    st.markdown("""
    Unsupervised clustering discovers *hidden structure* in economic data without using
    any labels. The goal: do distinct monetary policy regimes exist in the data?
    Can we identify them automatically?
    """)

    metric_row({
        "KMeans Best k": str(clust_res['kmeans']['best_k']),
        "KMeans Silhouette": f"{clust_res['kmeans']['best_silhouette']:.4f}",
        "DBSCAN Best eps": str(clust_res['dbscan']['best_eps']),
        "DBSCAN Clusters": str(clust_res['dbscan']['n_clusters']),
        "DBSCAN Noise %": f"{clust_res['dbscan']['noise_pct']:.1f}%",
    })

    tab1, tab2, tab3 = st.tabs(["📊 KMeans", "🔮 DBSCAN", "📋 Parameter Search"])

    with tab1:
        show_chart("clustering", "01_clustering_overview.png")
        st.markdown("""
        **Top-left (Elbow Curve):** Inertia drops steeply until k=2–3, then flattens.
        The elbow suggests 2 clusters is the natural break point.

        **Top-center (Silhouette Score):** Confirms k=2 as optimal — the highest silhouette
        score (0.366) indicates meaningful cluster separation.

        **Top-right (KMeans Visualization in PCA 2D):** The two clusters clearly separate in
        the principal component space — Cluster 0 is the low-rate modern era (post-2000),
        Cluster 1 is the high-rate inflationary era (1965–1990).
        """)
        show_chart("clustering", "02_cluster_profiles.png")
        insight_box(
            f"KMeans identifies two fundamentally different monetary policy regimes with "
            f"silhouette score = {clust_res['kmeans']['best_silhouette']:.4f}: "
            "(1) **High-rate inflationary regime** (Cluster 1): 1965–1990, mean FED rate ~7–10%, "
            "high inflation, characterized by the Great Inflation and Volcker disinflation. "
            "(2) **Low-rate modern regime** (Cluster 0): 1990–2024, mean FED rate ~2–3%, "
            "low inflation, Great Moderation + ZIRP era. These two regimes have fundamentally "
            "different causal structures — a model trained only on one regime will fail in the other.", "conclusion"
        )

    with tab2:
        st.markdown("""
        **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** finds clusters
        based on density rather than distance to centroids. It can find arbitrary-shaped clusters
        and classifies low-density points as 'noise' (outliers).

        **Parameters:** eps (neighborhood radius), min_samples (minimum points to form a cluster core).
        """)
        insight_box(
            f"DBSCAN with eps={clust_res['dbscan']['best_eps']} finds "
            f"{clust_res['dbscan']['n_clusters']} clusters with "
            f"{clust_res['dbscan']['noise_pct']:.1f}% noise points. "
            "The extremely high noise rate reveals a crucial insight: **economic transitions are "
            "gradual, not discrete**. Monthly economic data lies on a continuous manifold — "
            "there are no sharp boundaries between regimes, only slow drifts. "
            "This makes DBSCAN less suitable for this dataset than KMeans, which forces "
            "clear regime boundaries.", "finding"
        )
        insight_box(
            "The high DBSCAN noise rate (97.8%) is actually informative: most months are "
            "'transition months' between the two major regimes rather than settled within a "
            "single cluster. This gradual nature of economic change is precisely why the Federal "
            "Reserve uses forward guidance — to prepare markets for regime transitions.", "info"
        )

    with tab3:
        section_divider("DBSCAN Parameter Grid Search")
        dbscan_df = pd.DataFrame(clust_res['dbscan_grid'])
        dbscan_df.columns = ['eps', 'Clusters Found', 'Noise %', 'Silhouette Score']
        st.dataframe(
            dbscan_df.style.highlight_max(subset=['Silhouette Score'], color='#c8e6c9'),
            use_container_width=True, hide_index=True
        )
        st.markdown("""
        **Interpreting the grid search:**
        - Small eps (0.5): Too restrictive — almost everything is noise (99%)
        - Large eps (3.0): Too permissive — 12 clusters with 45% noise — over-segmenting
        - Optimal eps=1.0: Best silhouette score (0.46) with 2 clusters
        """)
        insight_box(
            "The k-distance plot (DBSCAN panel, bottom-left in overview) shows the 'knee' in "
            "5th-nearest-neighbor distances — this knee empirically identifies the optimal eps. "
            "This is the standard methodology for DBSCAN parameter selection. The knee at ~1.0 "
            "confirms our grid search result.", "info"
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ASSOCIATION RULES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔗  Association Rules":
    st.title("Association Rule Mining — Apriori Algorithm")
    st.markdown("""
    Association Rule Mining discovers **co-occurrence patterns** in economic data.
    Instead of predicting a value, we ask: *When the economy is in state X, what other
    states tend to occur together?* Treating each month as a "transaction" of economic states,
    Apriori finds statistically robust patterns.
    """)

    metric_row({
        "Frequent Itemsets": str(arm_res['n_frequent_itemsets']),
        "Rules Generated":   str(arm_res['n_rules']),
        "Max Lift":          f"{arm_res['top_lift']:.4f}",
        "Max Confidence":    f"{arm_res['top_confidence']:.4f}",
        "Min Support Used":  "0.30",
        "Min Confidence":    "0.60",
    })

    tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "📋 Top Rules", "🔧 Methodology"])

    with tab1:
        show_chart("arm", "01_arm_overview.png")
        st.markdown("""
        **Top-left (Support vs Confidence):** Points represent individual rules. Color encodes lift.
        Rules in the upper-right are both frequent AND accurate. Rules with high lift (yellow-green)
        are non-random associations.

        **Top-right (Support vs Lift):** Rules with lift > 1.0 (above baseline) are positively associated.
        The GDP↔RealGDP rule (lift ≈ 2.0) is the strongest association.

        **Bottom (Top 15 by Lift/Confidence):** GDP_Mid → RealGDP_Mid dominates — economically expected
        since nominal and real GDP are tightly coupled in moderate-growth periods.
        """)
        if os.path.exists(os.path.join(CHARTS, "arm", "02_arm_heatmap.png")):
            show_chart("arm", "02_arm_heatmap.png")

    with tab2:
        rules_path = os.path.join(RES, "top_arm_rules.json")
        if os.path.exists(rules_path):
            with open(rules_path) as f:
                rules_data = json.load(f)
            if rules_data:
                rules_df = pd.DataFrame(rules_data)
                for col in ['antecedents','consequents']:
                    rules_df[col] = rules_df[col].apply(
                        lambda x: ', '.join(x) if isinstance(x,list) else str(x))
                disp_cols = [c for c in ['antecedents','consequents','support',
                                          'confidence','lift'] if c in rules_df.columns]
                st.dataframe(rules_df[disp_cols].round(4).sort_values('lift',ascending=False),
                             use_container_width=True, hide_index=True)

        insight_box(
            "Rule 1 (GDP_Mid → RealGDP_Mid, Lift=1.96): When nominal GDP is in its mid range, "
            "real GDP is almost always also mid-range. This makes economic sense — in moderate "
            "growth periods, inflation doesn't drive a wedge between nominal and real GDP.", "success"
        )
        insight_box(
            "Rule 2 & 3 (FEDRates_Mid ↔ Inflation_Mid, Lift=1.22): Bidirectional rule confirms "
            "the Taylor Rule — moderate inflation co-occurs with moderate interest rates. "
            "Lift > 1.0 means this is not random; moderate inflation genuinely tends to "
            "occur alongside moderate rates more than chance would predict.", "success"
        )
        insight_box(
            "The limited number of strong rules (4 rules with confidence ≥ 0.60) reflects the "
            "continuous nature of economic data. When discretized into Low/Mid/High buckets, "
            "many months fall in boundary zones — weakening rule support. More granular "
            "discretization would yield more rules but lower support.", "warning"
        )

    with tab3:
        st.markdown("""
        **Discretization Process:**
        Each continuous feature is binned into three categories using tertile boundaries:
        - **Low:** Below 25th percentile
        - **Mid:** Between 25th and 75th percentile
        - **High:** Above 75th percentile

        **Apriori Parameters:**
        - `min_support = 0.30`: A rule must appear in ≥30% of all monthly observations
        - `min_confidence = 0.60`: The antecedent must predict the consequent ≥60% of the time
        - `Lift > 1.0`: Only rules that outperform random co-occurrence are meaningful

        **Metrics:**
        | Metric | Formula | Interpretation |
        |--------|---------|----------------|
        | Support | P(A ∪ B) | How often the rule appears in data |
        | Confidence | P(B\|A) | How reliably A predicts B |
        | Lift | Conf / P(B) | Is the relationship above random? (>1 = yes) |
        """)
        with st.expander("📄 ARM Code", expanded=False):
            st.code("""
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Discretize features into Low/Mid/High
def discretize(series, col_name):
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    def label(v):
        return f'{col_name}_Low' if v <= q25 else (f'{col_name}_High' if v > q75 else f'{col_name}_Mid')
    return series.apply(label)

# Build transaction list
transactions = [[discretize(arm_df[col], col[:8]).loc[idx] for col in ARM_FEATURES]
                for idx in arm_df.index]

# Encode and mine
te = TransactionEncoder()
te_df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)
freq_items = apriori(te_df, min_support=0.30, use_colnames=True)
rules = association_rules(freq_items, metric='confidence', min_threshold=0.60)
rules = rules.sort_values('lift', ascending=False)
""", language="python")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆  Model Comparison":
    st.title("Complete Model Comparison & Evaluation")
    st.markdown("""
    This section synthesizes results from all 15 algorithms into a unified comparison,
    analyzes bias-variance tradeoffs across models, and examines why temporal cross-validation
    is critical for honest evaluation of time-series ML models.
    """)

    tab1, tab2, tab3 = st.tabs([
        "📊 Performance Overview",
        "⚖️ Overfitting Analysis",
        "🔁 Cross-Validation",
    ])

    with tab1:
        show_chart("comparison", "master_comparison.png")
        st.markdown("""
        **Reading the four panels:**
        - **Regression R² (top-left):** XGBoost and GBM achieve R² > 0.979 on the test set.
          Even the weakest model (Ridge, R²=0.71) is far above random (R²=0).
        - **Regression RMSE (top-right):** XGBoost's RMSE of 0.44% means predictions are
          within ±0.44 percentage points of the actual rate.
        - **Classification Accuracy (bottom-left):** All models beat the 33% random baseline.
          GBM at 63.5% is the best achievable with economic-only features.
        - **Classification F1 (bottom-right):** Consistent with accuracy — GBM leads, followed
          by XGBoost. No model sacrifices precision for recall disproportionately.
        """)
        insight_box(
            "The performance hierarchy is consistent: XGBoost/GBM > Random Forest > SVR/SVM > "
            "Decision Tree > Linear/Logistic > Naive Bayes. This ordering reflects the models' "
            "capacity to capture non-linear interactions — ensemble boosting methods handle "
            "complex economic regime dynamics that simpler methods cannot.", "conclusion"
        )

        st.markdown("---")
        section_divider("Final Rankings")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🏆 Regression — by R²**")
            for i, (m, r) in enumerate(sorted(reg_res.items(),
                                              key=lambda x: x[1]['R2'], reverse=True)):
                medal = ["🥇","🥈","🥉"][i] if i < 3 else f"  {i+1}."
                st.markdown(f"{medal} **{m}** — R²=`{r['R2']}` RMSE=`{r['RMSE']}`")
        with col2:
            st.markdown("**🏆 Classification — by Accuracy**")
            for i, (m, r) in enumerate(sorted(cls_res.items(),
                                              key=lambda x: x[1]['Accuracy'], reverse=True)):
                medal = ["🥇","🥈","🥉"][i] if i < 3 else f"  {i+1}."
                st.markdown(f"{medal} **{m}** — Acc=`{r['Accuracy']:.4f}` F1=`{r['F1']}`")

    with tab2:
        show_chart("comparison", "overfitting_analysis.png")
        st.markdown("""
        **Top panel (Train vs Validation):** Blue solid bars = training score, faded bars = validation score.
        Numbers above bars show the gap (Δ). Green Δ < 0.05 = good fit. Orange 0.05–0.15 = mild overfit. Red > 0.15 = strong overfit.

        **Bottom panel (Gap chart):** Directly shows overfitting magnitude. Dashed lines mark thresholds.
        """)
        insight_box(
            "Most ensemble models show mild overfitting (orange) — their training scores are "
            "higher than validation scores, but the gap is manageable. "
            "Simple models (Linear Regression, Naive Bayes) show good fit or underfitting "
            "(low absolute scores). The Decision Tree classifier shows the clearest overfitting "
            "pattern — deep trees memorize training data perfectly (train→1.0) while validation "
            "performance degrades.", "finding"
        )
        insight_box(
            "**How to handle overfitting in this project:**\n"
            "1. **Regularization:** Ridge and Lasso penalize complex coefficients\n"
            "2. **Max depth limits:** Decision Trees and GBM use max_depth=4–10\n"
            "3. **Subsampling:** XGBoost uses subsample=0.8, colsample=0.8 to add stochasticity\n"
            "4. **Early stopping:** Can be applied to XGBoost/GBM with a validation set\n"
            "5. **Ensemble averaging:** Random Forest's tree averaging reduces variance",
            "info"
        )

    with tab3:
        show_chart("comparison", "cv_all_models.png")
        st.markdown("""
        **Left panel (Regression — Test R²):** Shows actual test-set R² scores with an explanatory
        note about why TimeSeriesSplit CV scores are negative (temporal generalization challenge).

        **Right panel (Classification — TimeSeriesSplit CV Accuracy):** Error bars show ±1 std
        across 5 temporal folds. Gradient Boosting achieves the most consistent CV accuracy.
        """)
        st.warning("""
        **Why TimeSeriesSplit matters for this dataset:**

        With lag features (1–12 month lags), standard K-Fold CV creates **temporal data leakage**:
        - Training fold may contain month T+6
        - Validation fold contains month T
        - But month T's lag-6 feature IS the value at T+6!

        This means the model "sees" future data during training, artificially inflating CV scores.
        TimeSeriesSplit prevents this by always training on the past (months 1…N) and testing on
        the future (months N+1…M), mirroring real-world deployment conditions.

        **Impact:** Standard K-Fold can inflate regression R² by 40–80% on this dataset.
        Our TimeSeriesSplit scores are conservative and honest.
        """)
        insight_box(
            "The Gradient Boosting classifier shows the most stable TimeSeriesSplit CV "
            "accuracy (0.417 ± 0.119). The XGBoost classifier shows lower variance (0.456 ± 0.071) "
            "— more consistent fold-to-fold. Both are significantly above the 0.333 baseline. "
            "These CV scores represent the model's expected performance on genuinely unseen "
            "future economic data.", "conclusion"
        )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Conclusion":
    st.title("Conclusions & Key Findings")
    st.markdown("""
    This analysis applies a complete, industry-standard ML pipeline to 70 years of Federal Reserve
    economic data. Here we synthesize the most important technical and economic findings.
    """)

    best_reg = max(reg_res, key=lambda x: reg_res[x]['R2'])
    best_cls = max(cls_res, key=lambda x: cls_res[x]['Accuracy'])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best Regressor",    best_reg, f"R²={reg_res[best_reg]['R2']}")
    col2.metric("Best Classifier",   best_cls, f"Acc={cls_res[best_cls]['Accuracy']:.1%}")
    col3.metric("Above Cls Baseline",f"+{(cls_res[best_cls]['Accuracy']-0.333)*100:.1f}pp", "vs 33.3% random")
    col4.metric("Lasso Feature Sel.","55%", "features zeroed out")

    st.markdown("---")
    section_divider("Key Technical Findings")
    findings = [
        ("🏆", "success",   "Ensemble Methods Dominate",
         f"XGBoost achieves R²={reg_res[best_reg]['R2']} and Gradient Boosting achieves "
         f"{cls_res[best_cls]['Accuracy']:.1%} classification accuracy. Sequential boosting's "
         "error-correction mechanism captures complex non-linear economic relationships that "
         "linear models fundamentally cannot represent."),
        ("🔑", "finding",   "Feature Engineering Was Critical",
         "Lag features (1–12 months) and rolling statistics were the most predictive features "
         "in both regression and classification, confirmed independently by Lasso, Random Forest, "
         "and XGBoost feature importances. Economic momentum and persistence are the strongest "
         "predictors of future rate decisions."),
        ("📐", "info",      "PCA Reveals Deep Multicollinearity",
         f"4 principal components capture 95.7% of variance in 8 raw features. PC1 (57.3%) "
         "alone captures the entire GDP-family variance. This explains why Lasso zeroed out "
         "55% of features — most were informationally redundant."),
        ("⏱️", "warning",   "Temporal CV Is Non-Negotiable for Time Series",
         "Standard K-Fold inflates regression R² by 40–80% due to lag-feature leakage. "
         "TimeSeriesSplit provides honest, forward-prediction scores. All CV results reported "
         "use TimeSeriesSplit — any competing analysis using K-Fold should be questioned."),
        ("📉", "info",      "Regularization Provides Automatic Feature Selection",
         "Lasso (α=0.01) zeroed out 37/67 features. The retained features align with economic "
         "theory: inflation lags, CPI lags, and rate momentum. Ridge distributed weights evenly "
         "across correlated features. Both outperform unregularized linear regression on this dataset."),
        ("🔮", "finding",   "Economic Transitions Are Gradual (DBSCAN Evidence)",
         "DBSCAN's 97.8% noise rate confirms that monthly economic data lies on a continuous "
         "manifold without sharp regime boundaries. KMeans (k=2) provides a useful stylized "
         "picture of two regimes, but real transitions happen over years, not months."),
        ("🎯", "conclusion", "Classification Ceiling ~63% With Economic Data Only",
         "Gradient Boosting at 63.5% represents the practical ceiling for economic-data-only "
         "models. Improving further requires FOMC transcripts (NLP), Federal Funds Futures "
         "market prices, and economic surprise indices. This is consistent with academic "
         "literature on monetary policy predictability."),
    ]
    for icon, kind, title, body in findings:
        with st.expander(f"{icon} {title}", expanded=True):
            insight_box(body, kind)

    st.markdown("---")
    section_divider("Economic Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **The Taylor Rule Confirmed:**
        - Spearman ρ = 0.676 for Inflation ↔ FED Rate
        - T-test: High inflation periods → 6.67% vs 2.68% rates (p ≈ 10⁻⁷⁴)
        - ARM Rule: FEDRates_Mid ↔ Inflation_Mid (Lift=1.22)

        Three independent statistical methods confirm the same economic relationship:
        the Federal Reserve raises rates when inflation is high. This is the core of
        the Taylor Rule, empirically validated by 70 years of data.
        """)
    with col2:
        st.markdown("""
        **Two Distinct Monetary Policy Regimes:**
        - KMeans k=2 identifies: High-rate era (1965–1990) vs Low-rate modern era (1990–2024)
        - The Volcker Shock (1979–1983) is the boundary event
        - Models trained only on pre-1990 data cannot generalize to post-1990 structure
        - This structural break is the primary reason TimeSeriesSplit CV scores are lower

        The 2022–2023 hiking cycle (rate: 0.08% → 5.33%) represents a potential
        transition back toward the high-rate regime — an out-of-sample test for all models.
        """)

    st.markdown("---")
    section_divider("Limitations & Future Work")
    col1, col2 = st.columns(2)
    with col1:
        st.error("""
        **Current Limitations:**
        - Monthly frequency only — misses intra-meeting signals
        - No FOMC meeting transcripts or Fed speech sentiment
        - No market expectations (Fed Funds Futures, yield curve)
        - No geopolitical features (oil shocks, wars, pandemics)
        - TimeSeriesSplit CV shows regime-crossing generalization limits
        """)
    with col2:
        st.success("""
        **Recommended Improvements:**
        - NLP on FOMC minutes using BERT/FinBERT for sentiment
        - Add Federal Funds Futures implied rates (market expectations)
        - Treasury yield curve shape (inversion = recession signal)
        - Economic surprise indices (actual vs forecast)
        - LSTM or Transformer for long-range temporal dependencies
        - Stacking ensemble of top 3 models
        """)
