# Federal Reserve Interest Rate Prediction
### An End-to-End Professional Machine Learning Project

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive ML pipeline applied to 70 years of US Federal Reserve economic data (1954–2024). Covers data cleaning, statistical analysis, feature engineering, PCA, and 15+ models across regression, classification, clustering, and association rule mining. Built with a static website, Streamlit app, and Jupyter notebooks.

---

## Key Results at a Glance

| Task | Best Model | Score |
|---|---|---|
| Regression (predict exact FED rate) | XGBoost | **R² = 0.9816, RMSE = 0.44%** |
| Classification (predict rate direction) | Gradient Boosting | **Accuracy = 63.5%** (+91% vs random) |
| Clustering | KMeans | **k=2 economic regimes** (Silhouette = 0.366) |
| Association Rules | Apriori | **4 rules, max lift = 1.96** |
| PCA | — | **4 components = 95.7% variance** |
| Features engineered | — | **67 from 8 raw variables** |

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [ML Workflow](#ml-workflow)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [Key Findings](#key-findings)
- [Contact](#contact)

---

## Project Overview

The Federal Reserve's interest rate decisions influence every corner of the economy — mortgage rates, business investment, consumer credit, and financial markets. This project treats rate prediction as a rigorous ML problem, applying the full industry-standard workflow to 847 monthly observations spanning 1954 to 2024.

**Three problem statements are addressed:**

1. **Regression** — Predict the exact Federal Funds Rate as a continuous value. Eight algorithms are compared: Linear Regression, Ridge, Lasso, SVR, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.

2. **Classification** — Predict the direction of the next rate change: Increase, Decrease, or No Change. Seven algorithms are compared: Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, Gradient Boosting, and XGBoost.

3. **Unsupervised Learning** — Discover hidden economic regimes using DBSCAN and KMeans clustering. Mine co-movement patterns between economic indicators using the Apriori association rule mining algorithm.

**What makes this project different from a basic ML tutorial:**
- TimeSeriesSplit cross-validation (not K-Fold) to prevent temporal data leakage from lag features
- Honest reporting of regime-shift challenges — negative CV R² scores are explained, not hidden
- Statistical validation of all claims (Shapiro-Wilk, ADF, T-test, ANOVA, Spearman)
- 67 domain-informed engineered features, not just raw columns
- Economic interpretation of every result, connecting ML findings to monetary policy theory

---

## Dataset

**Source:** [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/), Federal Reserve Bank of St. Louis.

**Coverage:** January 1954 — December 2024 | Monthly frequency | 847 observations

| Feature | Description | Role |
|---|---|---|
| **FEDRates** | Federal Funds Rate (% per annum) | Target variable |
| ConsumerPriceIndexAllItems | CPI month-over-month % change | Predictor |
| GDP | Nominal GDP (Billions USD) | Predictor |
| InflationConsumerPrice | Annual consumer inflation rate (%) | Predictor |
| MedianConsumerPriceIndex | Trimmed-mean inflation measure | Predictor |
| RealGDP | Inflation-adjusted GDP | Predictor |
| RealGDPPerCapita | Real GDP per person | Predictor |
| RealPotentialGDP | CBO estimate of potential output | Predictor |
| UnemploymentRate | U-3 unemployment rate (%) | Predictor |

The dataset is stored at `App/Tabs/Datasets/finaldataset.csv`. To refresh the data using the FRED API, see `App/API_Data_Collection.py` (requires a free FRED API key from [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)).

---

## ML Workflow

```
Raw CSV Data
    │
    ▼
Step 1: Data Cleaning
    Winsorization (1st/99th percentile) · Zero CPI interpolation · Forward/backward fill for quarterly GDP
    │
    ▼
Step 2: Exploratory Data Analysis (EDA)
    Time series · Distributions · Correlation heatmaps · Scatter plots vs target
    │
    ▼
Step 3: Feature Engineering  (8 → 67 features)
    Lag features (1, 3, 6, 12 months) · Rolling mean & std (3m, 6m) · Interaction terms · Calendar features
    │
    ▼
Step 4: Statistical Analysis & Hypothesis Testing
    Shapiro-Wilk normality · ADF stationarity · T-test · ANOVA · Spearman correlation
    │
    ▼
Step 5: Principal Component Analysis (PCA)
    4 components → 95.7% variance · Economic interpretation of each component
    │
    ▼
Step 6: Regression Models (8 algorithms)
    Evaluated on R², RMSE, MAE · TimeSeriesSplit CV · Bias-variance analysis
    │
    ▼
Step 7: Classification Models (7 algorithms)
    Evaluated on Accuracy, Weighted F1 · Confusion matrices · ROC-AUC per class
    │
    ▼
Step 8: Clustering (DBSCAN + KMeans)
    Economic regime discovery · Silhouette scoring · Cluster profile interpretation
    │
    ▼
Step 9: Association Rule Mining (Apriori)
    Tertile discretization · Support / Confidence / Lift · Taylor Rule validation
    │
    ▼
Step 10: Model Comparison & Evaluation
    Master comparison charts · Overfitting analysis · Learning curves · Final rankings
```

---

## Results

### Regression

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **XGBoost** ⭐ | **0.9816** | **0.4377** | **0.2955** |
| Gradient Boosting | 0.9791 | 0.4666 | 0.3112 |
| Random Forest | 0.9749 | 0.5109 | 0.3182 |
| SVR | 0.9050 | 0.9950 | 0.6701 |
| Decision Tree | 0.8742 | 1.1448 | 0.7987 |
| Linear Regression | 0.7403 | 1.6449 | 1.2329 |
| Lasso Regression | 0.7205 | 1.7063 | 1.2994 |
| Ridge Regression | 0.7101 | 1.7378 | 1.3275 |

XGBoost predictions are within **0.44 percentage points** of the actual FED rate on average. Lasso automatically zeroed out 55% of features (37 of 67), confirming that recent inflation lags are the dominant predictors.

> **Note on CV scores:** All regression TimeSeriesSplit CV R² scores are negative — not a bug. Models trained on earlier economic regimes (high-rate 1970s–80s) cannot accurately predict structurally different future regimes (near-zero 2010s), an honest reflection of economic regime shifts. Test-set R² values are strong because the held-out 20% is chronologically adjacent to the training period.

### Classification

| Model | Accuracy | Weighted F1 | CV Accuracy |
|---|---|---|---|
| **Gradient Boosting** ⭐ | **63.5%** | **0.6327** | 41.7% ± 11.9% |
| XGBoost | 62.9% | 0.6273 | 45.6% ± 7.1% |
| SVM | 59.9% | 0.5989 | 41.7% ± 3.0% |
| Random Forest | 59.9% | 0.5986 | 42.3% ± 17.3% |
| Logistic Regression | 55.1% | 0.5475 | 47.8% ± 13.0% |
| Decision Tree | 46.1% | 0.3986 | 42.6% ± 15.2% |
| Naive Bayes | 44.3% | 0.4322 | 54.1% ± 13.3% |

Random baseline = 33.3%. Best model achieves **+30.2 percentage points above random** (+91% improvement). The ceiling with macroeconomic-only data is approximately 65–70% — Fed decisions also incorporate FOMC speeches, market expectations, and geopolitical context not captured in economic statistics.

### Statistical Analysis

| Test | Result |
|---|---|
| Shapiro-Wilk (normality) | All 9 features non-normal (p < 0.05) |
| ADF (stationarity) | 7 of 9 features non-stationary (unit root) |
| T-test (high vs low inflation) | Mean rates: 6.67% vs 2.68%, p = 5.68×10⁻⁷⁴ |
| ANOVA (economic regimes) | F = 1440.28, p ≈ 0 |
| Spearman (inflation vs rates) | ρ = 0.676 (strongest predictor) |

### PCA

| Component | Variance Explained | Interpretation |
|---|---|---|
| PC1 | 57.3% | Economic Scale (GDP growth) |
| PC2 | 18.7% | Inflation Cycle |
| PC3 | 13.7% | Labor Market (Unemployment) |
| PC4 | 6.0% | Residual monetary dynamics |
| **Total (4 components)** | **95.7%** | — |

### Clustering

| Algorithm | Result |
|---|---|
| KMeans (k=2) | Cluster 0: Low-rate modern era (1990–2024) · Cluster 1: High-rate inflationary era (1965–1990) |
| DBSCAN (eps=1.0) | 97.8% noise — economic transitions are gradual, not discrete |

### Association Rules (Apriori)

| Rule | Confidence | Lift |
|---|---|---|
| GDP_Mid → RealGDP_Mid | 97.8% | 1.957 |
| FEDRates_Mid → Inflation_Mid | 61.3% | 1.217 |
| Inflation_Mid → FEDRates_Mid | 60.7% | 1.217 |

The Apriori algorithm independently rediscovered the **Taylor Rule** — the central principle of modern monetary policy — purely from co-occurrence patterns, without any domain knowledge encoded in the algorithm.

---

## Project Structure

```
ML-Project/
│
├── App/
│   ├── API_Data_Collection.py       # FRED API data fetching script
│   └── Tabs/
│       └── Datasets/
│           └── finaldataset.csv     # Main dataset (847 rows × 9 columns)
│
├── ml_analysis/
│   ├── part1_eda_features.py        # Step 1–5: Cleaning, EDA, Features, Stats, PCA
│   ├── part2_models.py              # Step 6–10: All ML models, clustering, ARM, comparisons
│   ├── fix_charts.py                # Chart corrections and styling improvements
│   ├── generate_notebooks.py        # Programmatic Jupyter notebook generation
│   └── outputs/
│       ├── charts/                  # All generated charts (72 PNG files)
│       │   ├── eda/                 # EDA charts
│       │   ├── cleaning/            # Data cleaning charts
│       │   ├── features/            # Feature engineering charts
│       │   ├── stats/               # Statistical analysis charts
│       │   ├── pca/                 # PCA charts
│       │   ├── regression/          # Per-model regression charts (21 files)
│       │   ├── classification/      # Per-model classification charts (24 files)
│       │   ├── clustering/          # Clustering charts
│       │   ├── arm/                 # Association rule mining charts
│       │   └── comparison/          # Master comparison charts
│       └── results/                 # JSON result files + preprocessed pickle
│           ├── regression_results.json
│           ├── classification_results.json
│           ├── clustering_results.json
│           ├── arm_results.json
│           ├── pca_results.json
│           ├── statistical_analysis.json
│           └── preprocessed_data.pkl
│
├── streamlit_app/
│   ├── main.py                      # Multi-page Streamlit application (1500+ lines)
│   └── requirements.txt             # Streamlit-specific dependencies
│
├── notebooks/                       # Jupyter notebooks (one per ML step)
│   ├── 01_Data_Cleaning_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Statistical_Analysis.ipynb
│   ├── 04_PCA_Analysis.ipynb
│   ├── 05_Regression_Models.ipynb
│   ├── 06_Classification_Models.ipynb
│   ├── 07_Clustering.ipynb
│   ├── 08_Association_Rules.ipynb
│   └── 09_Model_Comparison.ipynb
│
├── docs/
│   ├── index.html                   # Static website (open directly in browser, or via GitHub Pages)
│   └── images/                      # All chart images referenced by the website
│
├── build_simple_site.py             # Script to regenerate the static website
├── PROJECT_REPORT.md                # Full written report (~9,800 words)
├── requirements.txt                 # Full Python dependencies
└── README.md                        # This file
```

---

## Installation

### Prerequisites
- Python 3.11 or higher
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ML-Project.git
cd ML-Project
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Create
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option 1 — Static Website (no install required)

Simply open the file in any browser:

```
docs/index.html
```

Or on Windows:
```bash
start docs/index.html
```

**Live on GitHub Pages:** `https://YOUR_USERNAME.github.io/ML-Project/`

This is the fastest way to view all results, charts, and analysis.

---

### Option 2 — Streamlit Interactive App

```bash
streamlit run streamlit_app/main.py
```

Opens at `http://localhost:8501`. Navigate using the sidebar. Contains 12 pages covering every analysis step with interactive charts, metric cards, and full written analysis.

---

### Option 3 — Jupyter Notebooks (step-by-step walkthrough)

```bash
pip install jupyter
jupyter lab
```

Open notebooks in order from `notebooks/01_Data_Cleaning_EDA.ipynb`. Each notebook is self-contained with markdown explanations and runnable code. Notebooks 02–09 load the preprocessed data from `ml_analysis/outputs/results/preprocessed_data.pkl`.

---

### Option 4 — Run the full pipeline from scratch

This re-runs all ML analysis from the raw CSV and regenerates all charts and result files.

```bash
# Step 1: Data cleaning, EDA, feature engineering, PCA
python ml_analysis/part1_eda_features.py

# Step 2: All ML models (regression, classification, clustering, ARM)
python ml_analysis/part2_models.py

# Step 3 (optional): Regenerate fixed comparison charts
python ml_analysis/fix_charts.py

# Step 4 (optional): Regenerate the static website
python build_simple_site.py

# Step 5 (optional): Regenerate Jupyter notebooks
python ml_analysis/generate_notebooks.py
```

**Expected runtime:** `part1_eda_features.py` ~2 min | `part2_models.py` ~10–15 min (SVR and SVM are the slowest)

---

### Option 5 — Refresh data from FRED API

```bash
python App/API_Data_Collection.py
```

> **Requires a free FRED API key.** Register at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html) and replace the `api_key` value in `App/API_Data_Collection.py` with your own key.

FRED Series IDs used in this project:

| Series ID | Description |
|---|---|
| FEDFUNDS | Federal Funds Rate |
| CPIAUCSL | Consumer Price Index (All Items) |
| GDP | Nominal GDP |
| FPCPITOTLZGUSA | Inflation, Consumer Prices |
| MEDCPIM158SFRBCLE | Median Consumer Price Index |
| GDPC1 | Real GDP |
| A939RX0Q048SBEA | Real GDP Per Capita |
| GDPPOT | Real Potential GDP |
| UNRATE | Unemployment Rate |

---

## Technologies Used

| Category | Library | Purpose |
|---|---|---|
| Core ML | scikit-learn ≥ 1.3 | All models, CV, preprocessing, PCA |
| Gradient Boosting | XGBoost ≥ 2.0 | Best regression and classification model |
| Association Rules | mlxtend ≥ 0.23 | Apriori algorithm |
| Data | pandas ≥ 2.0 | DataFrames, time series |
| Numerical | numpy ≥ 1.24 | Array operations |
| Visualization | matplotlib ≥ 3.7 + seaborn ≥ 0.12 | All 72 charts |
| Statistics | scipy ≥ 1.10 + statsmodels ≥ 0.14 | Hypothesis tests, ADF test |
| Web App | Streamlit ≥ 1.32 | Interactive dashboard |
| Notebooks | nbformat ≥ 5.0 | Programmatic notebook generation |
| Images | Pillow ≥ 10.0 | Image loading in Streamlit |
| Data Source | FRED API | Economic data |

---

## Key Findings

### 1. The Taylor Rule Is Empirically Confirmed

The Taylor Rule states central banks set rates based on inflation and output gaps. Every analytical technique in this project independently validated this:
- Inflation has the highest Spearman correlation with rates (ρ = 0.676)
- T-test confirms rates average **6.67% during high inflation vs 2.68% during low inflation** (p = 5.68×10⁻⁷⁴)
- Apriori independently discovered inflation ↔ FED rate co-occurrence rules from data alone
- All tree-based models rank inflation lag features as most important

### 2. Feature Engineering Was the Most Impactful Step

Without lag and rolling features, best R² ≈ 0.50. With the full 67-feature engineered set, XGBoost achieves R² = 0.9816. The 1-month lag of inflation is the single most predictive feature — the Fed responds to sustained trends, not individual data points.

### 3. Ensemble Methods Dominate Both Tasks

XGBoost and Gradient Boosting outperform all other algorithms on both regression and classification. The reasons are specific to economic data: non-linear regime-dependent dynamics, three-way feature interactions, and natural robustness of tree splits to outliers.

### 4. Economic Transitions Are Gradual, Not Discrete

DBSCAN classified 97.8% of months as noise — not a failure, but a finding. Economic regime changes happen over years and decades, not overnight. KMeans' hard boundaries are a useful simplification, but DBSCAN reveals the continuous manifold on which economic data actually lies.

### 5. Classification Ceiling Is ~65% With Economic Data Alone

Federal Reserve decisions incorporate FOMC member sentiment, market expectations, and geopolitical context not available in macroeconomic statistics. Reaching 63.5% accuracy with only 8 economic indicators is a strong result, and represents a +91% improvement over random guessing.

### 6. Temporal Cross-Validation Matters Enormously

Standard K-Fold CV on time-series data with lag features causes data leakage. TimeSeriesSplit reveals the true forward-prediction difficulty. Reporting negative CV R² scores honestly is more valuable than reporting inflated K-Fold scores that don't reflect real-world deployment performance.

---

## Contact

**Sangram More**

- GitHub: [@Sangram-More](https://github.com/Sangram-More)
- LinkedIn: [linkedin.com/in/sangrammore](https://www.linkedin.com/in/sangrammore)
- Email: sangrammoreus@gmail.com

---

## License

This project is licensed under the MIT License.

---

*Data: FRED (Federal Reserve Economic Data), Federal Reserve Bank of St. Louis · Coverage: 1954–2024*
