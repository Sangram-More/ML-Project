# Federal Reserve Interest Rate Prediction
## A Comprehensive End-to-End Machine Learning Project Report

---

## Table of Contents

1. Project Overview
2. Dataset Description
3. Problem Statements
4. Step 1 — Data Cleaning & Quality Assurance
5. Step 2 — Exploratory Data Analysis (EDA)
6. Step 3 — Feature Engineering
7. Step 4 — Statistical Analysis & Hypothesis Testing
8. Step 5 — Principal Component Analysis (PCA)
9. Step 6 — Regression Models
10. Step 7 — Classification Models
11. Step 8 — Clustering Analysis
12. Step 9 — Association Rule Mining (Apriori)
13. Step 10 — Model Comparison, Cross-Validation & Overfitting Analysis
14. Key Findings & Conclusions
15. Technical Stack

---

## 1. Project Overview

This project is a professional, industry-standard machine learning pipeline applied to one of the most consequential economic decisions in the United States — the Federal Reserve's interest rate policy. The Federal Funds Rate is the benchmark short-term interest rate set by the Federal Open Market Committee (FOMC). It directly influences mortgage rates, business borrowing costs, consumer credit, and the broader economy. When the Fed raises rates, it slows inflation but risks reducing growth; when it cuts rates, it stimulates growth but risks overheating.

Using 70 years of macroeconomic data from the Federal Reserve Economic Data (FRED) database spanning January 1954 to December 2024, this project builds and evaluates more than 15 machine learning models across three categories of learning: supervised regression, supervised classification, and unsupervised learning (clustering and association rule mining).

The project demonstrates the full ML workflow as practiced in industry:
- Raw data ingestion and quality auditing
- Data cleaning and outlier treatment
- Exploratory analysis and visual storytelling
- Domain-driven feature engineering
- Statistical validation and hypothesis testing
- Dimensionality reduction via PCA
- Model training, evaluation, and comparison
- Cross-validation with temporal data considerations
- Bias-variance tradeoff analysis and overfitting diagnosis

The result is not just a predictive model but a complete analytical narrative connecting economic theory (the Taylor Rule, Phillips Curve, monetary policy transmission) to data-driven evidence.

---

## 2. Dataset Description

### Source
Federal Reserve Economic Data (FRED), maintained by the Federal Reserve Bank of St. Louis.

### Coverage
- Period: January 1954 — December 2024
- Frequency: Monthly
- Observations: 847 rows (after cleaning)
- Raw Features: 9 columns (1 target + 8 predictors)

### Features

| Feature | Full Name | Unit | Description |
|---|---|---|---|
| FEDRates | Federal Funds Rate | % per annum | Target variable. The overnight lending rate set by the Federal Reserve. |
| ConsumerPriceIndexAllItems | CPI (All Items) | % month-over-month | Percentage change in the Consumer Price Index for all goods and services. Measures monthly inflation. |
| GDP | Nominal GDP | Billions USD | Total economic output in current dollars. Not adjusted for inflation. |
| InflationConsumerPrice | Annual Inflation Rate | % per annum | Year-over-year percentage change in consumer prices. |
| MedianConsumerPriceIndex | Median CPI | % per annum | A trimmed measure of inflation that excludes the most extreme price changes, giving a more stable inflation signal. |
| RealGDP | Real GDP | Billions chained 2017 USD | GDP adjusted for inflation, measuring true economic growth. |
| RealGDPPerCapita | Real GDP Per Capita | Chained 2017 USD | Real GDP divided by population — a measure of living standards. |
| RealPotentialGDP | Real Potential GDP | Billions chained 2017 USD | The Congressional Budget Office's estimate of the economy's maximum sustainable output. When actual GDP exceeds potential, inflation tends to rise. |
| UnemploymentRate | Unemployment Rate | % | The U-3 measure — percentage of the labor force that is jobless and actively seeking work. |

### Target Variable

The FED Funds Rate ranged from a historic low of approximately 0.04% (post-2008 quantitative easing and COVID-19 response) to a historic high of 19.10% in June 1981, during the Volcker-era tightening campaign to break the inflationary spiral of the 1970s. The unconditional mean over the full dataset is approximately 4.6%, though this masks substantial structural variation across monetary policy eras.

### Key Historical Phases

- 1954–1965: Post-war reconstruction era; moderate rates (1–4%)
- 1965–1981: The Great Inflation; rates rose from 4% to nearly 20% as the Fed fought successive inflation shocks (Vietnam War spending, oil embargoes, supply shocks)
- 1981–2001: The Great Moderation; rates fell steadily from 19% to ~6% as inflation was tamed and macroeconomic volatility declined
- 2001–2009: Dot-com bust and Global Financial Crisis; aggressive rate cuts to near zero
- 2009–2015: Near-zero interest rate policy (ZIRP); unprecedented accommodation in the wake of the financial crisis
- 2015–2019: Gradual normalization; rates rose from 0% to 2.4%
- 2020: COVID-19 emergency; rates cut back to near zero in March 2020
- 2022–2023: The fastest tightening cycle since the 1980s; 525 basis points of hikes to combat post-pandemic inflation

---

## 3. Problem Statements

Three complementary prediction problems are defined, each requiring different ML techniques.

### Problem 1: Regression — Predict the Exact FED Rate
Predict the Federal Funds Rate as a continuous numerical value given the current month's economic indicators and their recent history. This tests whether economic momentum signals can explain the level of monetary policy.

Success metric: R² Score (coefficient of determination), RMSE (root mean squared error), MAE (mean absolute error).

### Problem 2: Classification — Predict Rate Direction
Classify the Fed's next action as one of three outcomes: Increase (rate change > +0.05%), Decrease (rate change < -0.05%), or No Change (|rate change| ≤ 0.05%). This tests whether the direction of monetary policy is predictable from economic data.

Success metric: Accuracy, Weighted F1 Score.

### Problem 3: Unsupervised — Discover Economic Regimes and Patterns
Without using any labels, discover whether the 70 years of economic data naturally cluster into distinct monetary policy regimes. Additionally, use association rule mining to find which economic indicator states tend to co-occur, and whether these rules correspond to known economic principles (such as the Taylor Rule).

Success metric: Silhouette Score (clustering), Lift (association rules).

---

## 4. Step 1 — Data Cleaning & Quality Assurance

Before any analysis or modeling, the raw data must be systematically audited and corrected. Poor data quality propagates forward through the entire pipeline, corrupting every downstream result.

### 4.1 Initial Assessment

Upon loading the raw CSV, an inspection revealed:
- Shape: 847 rows × 9 columns
- Temporal span: January 1954 to December 2024
- Apparent data types: all numeric, but with problematic entries

### 4.2 Issue 1: Zero-Value CPI Entries

The ConsumerPriceIndexAllItems column contained 111 entries with a value of exactly 0.0 in the earlier part of the dataset (roughly 1954–1969). These zeros were not genuine zero-inflation readings; rather, they represented a data availability issue — the granular monthly CPI data was not collected or reported for this early period, and zero was used as a placeholder.

Treatment: All zero values in the CPI column were replaced with NaN (Not a Number) and then filled using linear interpolation — computing the expected value at each missing point by drawing a straight line between the nearest valid observations before and after the gap.

Justification: Linear interpolation was chosen over forward-fill because inflation changes gradually over time, and a linear bridge between known values is more defensible than assuming the last known value persisted unchanged for years.

### 4.3 Issue 2: Missing Values from Quarterly Reporting

Several GDP-related features (GDP, RealGDP, RealGDPPerCapita, RealPotentialGDP) are reported quarterly by government agencies but stored in a monthly dataset. This creates two months of missing data for every reported quarter.

Treatment: Forward-fill followed by backward-fill. Forward-fill propagates the last quarterly observation forward until the next release; backward-fill handles any remaining gaps at the beginning of the series.

Justification: GDP does not change meaningfully month-to-month within a quarter; using the most recently reported value is the standard approach in economic data processing.

### 4.4 Issue 3: Statistical Outliers

Even after gap-filling, some features contained extreme values reflecting genuine but rare economic shocks (the 1973 oil embargo, the 2008 financial crisis, the 2020 COVID lockdown). These extreme values can distort model training by creating undue leverage on parameter estimates.

Treatment: Winsorization at the 1st and 99th percentile. Values below the 1st percentile were capped at the 1st percentile value; values above the 99th percentile were capped at the 99th percentile value.

Justification: Winsorization was chosen over deletion because:
1. It preserves all 847 observations (no sample reduction)
2. It acknowledges that extreme events happened and should influence the model, but with bounded influence
3. It is more appropriate than log transformation for variables that can take negative values (e.g., inflation can be negative)

Features with the most extreme outliers identified (using IQR × 3 rule):
- InflationConsumerPrice: 12 extreme outliers (1970s stagflation era)
- ConsumerPriceIndexAllItems: 4 extreme outliers
- UnemploymentRate: 1 extreme outlier (COVID unemployment spike, April 2020)

### 4.5 Final Clean Dataset

After all cleaning steps:
- Rows: 834 (13 rows removed due to NaN target values from lag creation in a subsequent step)
- Missing values: 0
- Outliers: Capped (not removed)
- All features: numeric, bounded, and properly scaled for modeling

---

## 5. Step 2 — Exploratory Data Analysis (EDA)

EDA is the process of understanding the data visually and statistically before any formal modeling. Good EDA prevents wasted effort on models that cannot work and reveals opportunities for feature engineering.

### 5.1 FED Funds Rate Time Series

The most important visualization is the FED rate over 70 years, overlaid with NBER-defined recession periods. Key observations:

- The rate exhibits clear regime behavior — it is not a random walk but follows sustained directional trends lasting years or decades.
- Every recession (marked by shaded bands) is accompanied by rate cuts, without exception. This confirms the Fed's counter-cyclical mandate: ease policy during downturns to stimulate the economy.
- The peak rate of ~19% in 1981 was a deliberate policy choice by Federal Reserve Chairman Paul Volcker to break the inflationary expectations of the 1970s. This "Volcker shock" succeeded in reducing inflation from 14% to under 4% over three years, at the cost of a severe recession.
- The 2022–2023 hiking cycle, which raised rates from 0.08% to 5.33%, was the fastest and most aggressive tightening since the Volcker era, responding to post-pandemic inflation that peaked at 9.1% in June 2022.

### 5.2 Feature Time Series (All Variables, 1954–2024)

Examining all eight predictors over time reveals:

- GDP and RealGDP: Exhibit near-continuous upward trends driven by long-run economic growth. This non-stationarity (the series has no stable mean) is important for model selection — it means raw GDP levels are not directly informative about rate changes, but GDP growth rates and deviations from trend are.
- InflationConsumerPrice: Shows a clear arch shape — rising through the 1960s–1970s, peaking dramatically in 1980, then declining and stabilizing below 4% through the 1990s–2010s, before spiking again in 2021–2022.
- UnemploymentRate: Follows a sawtooth business cycle pattern — gradual declines during expansions punctuated by sharp spikes during recessions. The COVID spike to ~14.7% in April 2020 is the sharpest in the dataset.
- CPI (month-over-month): Relatively stationary compared to other features, oscillating around a low positive mean with occasional spikes.

### 5.3 Feature Distributions

Histograms with KDE overlays for all features confirm:

- None of the features are normally distributed (confirmed statistically in Step 4)
- GDP and RealGDP are strongly right-skewed (exponential growth over 70 years)
- FEDRates is right-skewed with a mode near 0–2% but a long right tail from the 1980s
- InflationConsumerPrice is approximately bell-shaped but with heavy tails
- UnemploymentRate is bimodal — most months cluster around 4–6%, with a secondary cluster from recessionary periods

The pervasive non-normality justifies:
1. Using non-parametric statistical tests (Spearman correlation instead of Pearson; Mann-Whitney instead of t-test)
2. Preferring tree-based models over linear regression that assumes normally distributed residuals
3. Applying standardization before distance-based models (SVR, SVM, KMeans, DBSCAN)

### 5.4 Correlation Analysis

The Pearson correlation matrix reveals:

- Strongest positive correlations with FEDRates: InflationConsumerPrice (r ≈ 0.72), MedianConsumerPriceIndex (r ≈ 0.65)
- Negative correlations with FEDRates: GDP (r ≈ -0.39), RealGDP (r ≈ -0.37), RealGDPPerCapita (r ≈ -0.35)

The negative GDP-rate correlation is counterintuitive at first glance — the Fed raises rates when the economy is strong, so one might expect a positive correlation. The negative correlation arises because both GDP and rates show strong time trends over 70 years, but in opposite directions. GDP has grown continuously; interest rates peaked in 1981 and trended downward for 40 years thereafter. This is a spurious correlation driven by long-run secular trends, not a causal economic relationship.

The GDP features are also highly correlated with each other (r > 0.95), indicating multicollinearity. This is addressed in Step 5 (PCA) and by regularized regression methods (Ridge, Lasso).

### 5.5 Scatter Plots vs FED Rate (Time-Colored)

Plotting each feature against the FED rate with points colored by time period reveals important structural breaks:

- The inflation-rate scatter shows a tight positive relationship in the 1970s–1980s (yellow points, upper right) but a much flatter relationship in the 2000s–2020s (blue/green points). This structural shift — the same inflation level produces different rate responses across eras — is the fundamental challenge for ML models.
- The unemployment-rate scatter shows no clear monotonic relationship across the full sample, but within any given decade the relationship is clearer. This suggests non-linear, era-dependent dynamics.

These scatter plots provided the key motivation for adding lag features and rolling statistics in Step 3: the Fed responds to trends, not just levels.

---

## 6. Step 3 — Feature Engineering

Raw economic features are insufficient for predicting interest rates. The Federal Reserve responds to sustained economic trends, not individual monthly data points. Feature engineering encodes economic momentum and dynamics into the feature set, transforming 8 raw variables into 67 informative predictors.

### 6.1 Lag Features (32 features)

For each of the 8 raw predictors, four temporal lags were created:
- 1-month lag (feature_lag1): Last month's value
- 3-month lag (feature_lag3): Value from three months ago
- 6-month lag (feature_lag6): Value from six months ago
- 12-month lag (feature_lag12): Value from twelve months ago

8 features × 4 lags = 32 lag features

Rationale: Monetary policy operates with a lag — economic conditions today influence rate decisions over the next several months, and current rates reflect decisions made months ago. The 12-month lag is particularly important because the Fed often states its policy intentions a year in advance, and annual inflation comparisons are the standard economic metric.

### 6.2 Rolling Statistics (24 features)

For each of the 8 raw predictors, two window sizes and two statistics were computed:
- 3-month rolling mean (feature_roll3_mean)
- 3-month rolling standard deviation (feature_roll3_std)
- 6-month rolling mean (feature_roll6_mean)
- 6-month rolling standard deviation (feature_roll6_std)

8 features × 2 windows × 2 statistics = 32, but 24 non-redundant rolling features used

Rationale:
- Rolling means smooth out monthly noise and capture sustained trends — a key input to Fed decisions
- Rolling standard deviations capture economic volatility and uncertainty. High volatility in inflation or GDP often leads the Fed to "wait and watch" before acting (No Change), while low volatility with a clear trend makes action more likely

### 6.3 Interaction Features (5 features)

Several economically motivated interaction terms were created:

- Inflation × Unemployment: A proxy for the Phillips Curve relationship. Economic theory predicts a tradeoff between inflation and unemployment; high values of both simultaneously (stagflation) create the most difficult policy environment.
- GDP growth rate: Month-over-month percentage change in nominal GDP. Captures acceleration/deceleration rather than level.
- Real GDP growth rate: Same as above for inflation-adjusted GDP.

### 6.4 Calendar Features (3 features)

- Month (1–12): Captures seasonal patterns in economic reporting and FOMC meeting schedules. The Fed meets 8 times per year, creating a discretization of policy decisions.
- Quarter (1–4): Captures quarterly reporting cycles for GDP and other statistics.
- Year: Captures long-run secular trends not fully captured by other features.

### 6.5 Classification Target Engineering

For Problem 2 (Classification), a discrete rate-change direction label was created:
- Increase: Month-over-month change in FEDRates > +0.05 percentage points
- Decrease: Month-over-month change in FEDRates < -0.05 percentage points
- No Change: |change| ≤ 0.05 percentage points

The ±0.05 threshold was chosen because:
1. The Federal Reserve typically moves rates in increments of 25 basis points (0.25%) minimum
2. Values within ±0.05 reflect rounding artifacts and data revisions, not actual policy changes
3. This threshold produces a reasonably balanced three-class distribution

### 6.6 Feature Count Summary

| Category | Count |
|---|---|
| Lag features (1, 3, 6, 12 months) | 32 |
| Rolling mean & std (3m, 6m windows) | 24 |
| Interaction & growth features | 5 |
| Calendar features | 3 |
| Original raw features | 8 |
| **Total features for modeling** | **67** |

Note: Original raw features are retained alongside engineered features, giving models the choice to use raw levels or engineered derivatives.

---

## 7. Step 4 — Statistical Analysis & Hypothesis Testing

Statistical testing provides the scientific rigor that differentiates a professional ML project from exploratory analysis. Before claiming that economic variables predict interest rates, we must formally test whether the observed relationships are statistically significant or could arise by chance.

### 7.1 Normality Testing — Shapiro-Wilk Test

Null hypothesis (H₀): The feature is drawn from a normal distribution.
Significance level: α = 0.05

Results for all 9 variables:

| Feature | W-statistic | p-value | Normal? |
|---|---|---|---|
| FEDRates | 0.9347 | ≈ 0.000 | No |
| ConsumerPriceIndexAllItems | 0.9800 | 2.0 × 10⁻⁶ | No |
| GDP | 0.8640 | ≈ 0.000 | No |
| InflationConsumerPrice | 0.8425 | ≈ 0.000 | No |
| MedianConsumerPriceIndex | 0.8639 | ≈ 0.000 | No |
| RealGDP | 0.9239 | ≈ 0.000 | No |
| RealGDPPerCapita | 0.9432 | ≈ 0.000 | No |
| RealPotentialGDP | 0.9230 | ≈ 0.000 | No |
| UnemploymentRate | 0.9445 | ≈ 0.000 | No |

Result: All features reject normality at the 0.05 significance level. This has direct implications:
- Parametric tests (t-test, ANOVA with equal-variance assumptions) are less reliable; non-parametric alternatives are preferred
- Linear regression's assumption of normally distributed residuals is likely violated; tree-based models are more appropriate
- Distance-based algorithms (KMeans, DBSCAN, SVR, SVM) require feature standardization before application

### 7.2 Stationarity Testing — Augmented Dickey-Fuller (ADF) Test

Null hypothesis (H₀): The series has a unit root (is non-stationary — it has a trending mean).
Significance level: α = 0.05

A time series is stationary if its mean and variance do not change over time. Modeling non-stationary series directly can lead to spurious correlations (two trending series appear highly correlated even if economically unrelated).

| Feature | ADF Statistic | p-value | Stationary? |
|---|---|---|---|
| FEDRates | -2.1276 | 0.2336 | No |
| ConsumerPriceIndexAllItems | -3.3857 | 0.0115 | **Yes** |
| GDP | +3.8007 | 1.0000 | No |
| InflationConsumerPrice | -2.6986 | 0.0743 | No |
| MedianConsumerPriceIndex | -2.4245 | 0.1349 | No |
| RealGDP | +2.7020 | 0.9991 | No |
| RealGDPPerCapita | +1.0220 | 0.9945 | No |
| RealPotentialGDP | +0.6176 | 0.9880 | No |
| UnemploymentRate | -3.2565 | 0.0169 | **Yes** |

Key findings:
- Only CPI (monthly change) and Unemployment Rate are stationary. Both are rate-of-change measures that naturally mean-revert.
- FEDRates itself is non-stationary — there is a clear downward trend from 1981 to 2021, meaning the mean rate in the 1980s (~8%) is completely different from the mean rate in the 2010s (~0.5%). This regime-shift behavior is the root cause of the negative TimeSeriesSplit CV scores for regression (discussed in Step 10).
- All GDP variants are strongly non-stationary (positive ADF statistics, p ≈ 1.0), confirming the continuous upward economic growth trend. This motivates the use of GDP growth rate features (first differences) rather than GDP levels.

### 7.3 T-Test: High vs Low Inflation Periods

Research question: Do high-inflation periods have significantly higher FED rates than low-inflation periods?

The dataset was split at the median annual inflation rate to create two groups:
- High inflation group: Months where annual inflation > median (≈ 3.2%)
- Low inflation group: Months where annual inflation ≤ median

Results:
- High inflation mean FED rate: **6.67%**
- Low inflation mean FED rate: **2.68%**
- Difference: **+3.99 percentage points**
- t-statistic: 20.17
- p-value: 5.68 × 10⁻⁷⁴ (effectively zero)

Interpretation: The probability of observing this large a difference by chance if there were no relationship between inflation and rates is astronomically small (essentially zero). This provides rigorous statistical confirmation of what the Taylor Rule predicts: the Federal Reserve systematically raises interest rates in high-inflation environments. The difference of nearly 4 percentage points between the two groups is both statistically significant and economically substantial.

### 7.4 One-Way ANOVA: Rate Differences Across Economic Regimes

Research question: Are FED rates significantly different across periods of different economic strength?

The dataset was divided into four economic regimes based on Real GDP growth rates: Very Low Growth, Low Growth, High Growth, Very High Growth (quartile-based).

Results:
- F-statistic: 1440.28 (extremely large)
- p-value: ≈ 0.000

Interpretation: The F-statistic of 1440 means there is more than 1,440 times more variance in rates between economic regimes than within them. This rejects the null hypothesis that all regimes have the same mean rate with overwhelming confidence. Economic regime (proxied by GDP growth) is a highly significant predictor of the interest rate level.

### 7.5 Spearman Rank Correlation

Spearman rank correlation is used instead of Pearson because:
1. All features are non-normally distributed (confirmed in 7.1)
2. The relationships between economic variables and rates are monotonic but not necessarily linear
3. Spearman is robust to outliers and non-linear monotonic relationships

Results (correlation with FEDRates, all p-values < 0.001 unless noted):

| Feature | Spearman ρ | Direction |
|---|---|---|
| InflationConsumerPrice | **+0.676** | Strong positive |
| MedianConsumerPriceIndex | **+0.450** | Moderate positive |
| ConsumerPriceIndexAllItems | +0.313 | Moderate positive |
| GDP | -0.390 | Moderate negative |
| RealPotentialGDP | -0.392 | Moderate negative |
| RealGDP | -0.386 | Moderate negative |
| RealGDPPerCapita | -0.376 | Moderate negative |
| UnemploymentRate | -0.006 | Near zero (p = 0.87) |

Key finding: Inflation (ρ = 0.676) is by far the strongest predictor. Unemployment rate has essentially zero correlation with the FED rate across the full 70-year period — a surprising finding that reflects the changing nature of the Fed's reaction function over time. In recent decades, unemployment became less important as inflation-targeting became the dominant framework.

---

## 8. Step 5 — Principal Component Analysis (PCA)

PCA is a dimensionality reduction technique that transforms the original correlated features into a new set of uncorrelated components (principal components, PCs). Each PC is a linear combination of the original features, and they are ordered by the amount of variance they explain.

### 8.1 Why PCA?

The 8 raw economic features are highly correlated (particularly the four GDP variants, which all measure similar economic growth). Feeding correlated features into distance-based models inflates the influence of the correlated dimensions. PCA decorrelates the features, providing a clean geometric representation of the data.

Additionally, PCA reveals the underlying structure of economic variation: which combinations of economic variables co-move, and what economic concept do they represent?

### 8.2 Explained Variance

| Component | Explained Variance | Cumulative Variance |
|---|---|---|
| PC1 | 57.31% | 57.31% |
| PC2 | 18.74% | 76.05% |
| PC3 | 13.66% | 89.71% |
| PC4 | 5.95% | 95.67% |
| PC5 | 4.02% | 99.69% |
| PC6 | 0.29% | 99.97% |
| PC7 | 0.02% | 99.99% |
| PC8 | <0.01% | 100.00% |

Key result: **4 principal components explain 95.67% of total variance**. This means that almost all the economically relevant information in the 8 raw features can be captured in just 4 dimensions. The remaining 4 components together explain only 4.33% — mostly measurement noise.

### 8.3 Interpretation of Principal Components

**PC1 — Economic Scale (57.31% of variance)**

PC1 loads most heavily on the four GDP variants (loadings: GDP = +0.444, RealGDP = +0.456, RealGDPPerCapita = +0.453, RealPotentialGDP = +0.457) with smaller contributions from inflation-related features. This component essentially captures the size of the US economy — it increases monotonically over 70 years as the economy grows. When a month has a high PC1 score, the economy is large and mature (recent decades); when it has a low PC1 score, the economy was smaller (early decades).

This component does not directly predict interest rates well in isolation, because large economies can have either high or low rates depending on the inflation environment.

**PC2 — Inflation Cycle (18.74% of variance)**

PC2 loads most heavily on InflationConsumerPrice (+0.638) and ConsumerPriceIndexAllItems (+0.613), with smaller contributions from MedianCPI (+0.288). This component captures the level of price pressure in the economy. High PC2 scores correspond to inflationary periods (1970s–1980s); low PC2 scores to deflationary or low-inflation periods (2009–2020).

This is the most economically meaningful component for predicting the FED rate, and it explains why inflation is the strongest individual predictor in the Spearman correlation analysis.

**PC3 — Labor Market (13.66% of variance)**

PC3 is dominated almost entirely by the UnemploymentRate loading (+0.926). This component isolates labor market conditions from economic size and inflation. It captures the business cycle: high during recessions, low during expansions.

Together, PC1 + PC2 + PC3 explain 89.71% of variance, capturing economic scale, price level, and employment conditions — the three core inputs to the Taylor Rule.

**PC4 — Residual Monetary Dynamics (5.95% of variance)**

PC4 captures remaining variance not explained by the first three components, including some of the MedianCPI vs InflationConsumerPrice differential (trimmed vs headline inflation). It pushes cumulative explanation over the 95.67% threshold.

### 8.4 PCA Biplot

Projecting all 847 months onto PC1 vs PC2 and coloring by rate direction reveals that:
- Rate increase months (green) tend to cluster at moderate-to-high PC2 values (inflation building up)
- Rate decrease months (red) cluster at various PC1 values but tend to follow periods of elevated PC2
- No Change months (yellow) are spread throughout the space, consistent with the Fed's preference for policy stability

The biplot arrows show that inflation-related features (InflationConsumerPrice, CPI) point in a direction that partially separates rate increases from rate decreases, confirming their classification utility.

---

## 9. Step 6 — Regression Models

Eight regression algorithms were trained to predict the Federal Funds Rate as a continuous numerical value. The dataset was split 80% training / 20% test, maintaining chronological order (no shuffling). Models are evaluated on RMSE, MAE, R², and TimeSeriesSplit cross-validation.

### 9.1 Train-Test Split Methodology

The most recent 20% of observations (approximately 2015–2024, the final ~170 months) are held out as the test set. All training data comes from the period before 2015. This design:
- Prevents data leakage from lag features (a lag-1 feature in a test sample would contain training data if the split were random)
- Simulates real-world deployment: we use historical data to predict the future
- Makes the evaluation conservative — the test period includes the 2022–2023 hiking cycle, which was historically unusual

### 9.2 Cross-Validation: TimeSeriesSplit

Standard K-Fold cross-validation randomly splits data into folds. For time-series data with lag features, this causes temporal leakage: a lag-12 feature computed for month M in a validation fold contains data from months M-1 through M-12, which may include months that are in the training fold of the same split. The model effectively "sees the future" during training.

TimeSeriesSplit solves this by always training on the past and validating on the future. With 5 splits:
- Split 1: Train on months 1–130, validate on months 131–165
- Split 2: Train on months 1–165, validate on months 166–200
- ...and so on, expanding the training window each split

The resulting CV scores are often negative for regression models. This is not a bug — it reflects the genuine challenge of predicting across economic regime changes. A model trained on data from 1954–1990 (high-rate regime) systematically overestimates rates when asked to predict 2010–2020 data (near-zero rate regime). This temporal instability is a real and important property of the data, and reporting it honestly is more valuable than inflating scores with inappropriate validation.

### 9.3 Regression Results

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| **XGBoost** ★ | **0.9816** | **0.4377** | **0.2955** |
| Gradient Boosting | 0.9791 | 0.4666 | 0.3112 |
| Random Forest | 0.9749 | 0.5109 | 0.3182 |
| SVR | 0.9050 | 0.9950 | 0.6701 |
| Decision Tree | 0.8742 | 1.1448 | 0.7987 |
| Lasso Regression | 0.7205 | 1.7063 | 1.2994 |
| Linear Regression | 0.7403 | 1.6449 | 1.2329 |
| Ridge Regression | 0.7101 | 1.7378 | 1.3275 |

### 9.4 Linear Regression

Linear regression assumes the target is a linear combination of the features. With 67 features, it can capture many relationships but assumes no interactions between features and no non-linearity.

Result: R² = 0.7403, RMSE = 1.6449 percentage points.

While 74% of rate variance is explained, the model systematically fails during rapid rate change periods. The residual plot shows heteroscedasticity — errors are larger during high-rate periods (1970s–1980s), suggesting non-linear dynamics that linear regression cannot capture.

The standardized coefficients reveal the most important features: rolling means and lags of InflationConsumerPrice and MedianCPI have the largest positive coefficients, consistent with the Taylor Rule.

### 9.5 Ridge Regression

Ridge regression (L2 regularization) adds a penalty equal to the sum of squared coefficients to the loss function. This shrinks all coefficients toward zero proportionally, addressing the multicollinearity problem identified in EDA.

Regularization parameter tuning: The optimal alpha (regularization strength) was selected by cross-validation across alpha values from 0.001 to 1000. Best alpha = 10.

Result: R² = 0.7101, RMSE = 1.7378 — slightly lower than plain linear regression.

This counterintuitive result occurs because the test period (2015–2024) contains the extreme 2022–2023 hiking cycle, and Ridge's coefficient shrinkage slightly reduces the model's ability to capture the full magnitude of this unprecedented move.

### 9.6 Lasso Regression

Lasso (L1 regularization) adds a penalty equal to the sum of absolute coefficient values. Unlike Ridge, L1 regularization drives weak coefficients exactly to zero, performing automatic feature selection.

Result: R² = 0.7205, RMSE = 1.7063 — comparable to Ridge.

More importantly, Lasso zeroed out **37 of 67 features (55.2%)**, retaining only 30 predictors. The surviving features are:
- Short-lag inflation and CPI features (lags 1, 3 months)
- 6-month rolling mean of inflation
- Recent FED rate lag (rate momentum)
- Calendar features

The 37 eliminated features were mostly long-lag GDP variants and interaction terms that are redundant given the inflation features. This Lasso selection provides an interpretable, parsimonious model that aligns with economic intuition: recent inflation history is sufficient for rate prediction.

### 9.7 Support Vector Regression (SVR)

SVR uses a kernel function (RBF — Radial Basis Function) to implicitly map features into a high-dimensional space where non-linear relationships become linear. The model finds a "tube" around the regression function within which errors are not penalized.

Result: R² = 0.9050, RMSE = 0.9950 — a significant improvement over linear models.

The non-linear kernel allows SVR to capture the curved, regime-dependent relationship between economic features and rates. However, it is slower to train and requires careful feature scaling (all features were standardized before SVR training).

### 9.8 Decision Tree Regression

Decision trees partition the feature space using binary splits, creating a piecewise constant prediction function.

Result: R² = 0.8742, RMSE = 1.1448. Moderate performance.

The most important analysis with Decision Trees is the bias-variance tradeoff demonstration through depth tuning:
- Depth = 1: Training R² = 0.52, Validation R² = 0.48 (high bias, underfitting)
- Depth = 4: Training R² = 0.88, Validation R² = 0.87 (balanced)
- Depth = 8: Training R² = 0.96, Validation R² = 0.82 (moderate overfitting)
- Depth = 20: Training R² = 1.00, Validation R² ≈ 0.68 (severe overfitting — memorizing training data)

Best depth = 4, achieving the optimal balance between capturing complexity and generalizing.

### 9.9 Random Forest Regression

Random Forest builds hundreds of decision trees, each trained on a bootstrap sample of the data and a random subset of features. The final prediction is the average across all trees.

Result: R² = 0.9749, RMSE = 0.5109 — strong performance.

Feature importance analysis (mean impurity decrease across all trees):
- Top features: InflationConsumerPrice_lag1 (1-month inflation lag), MedianCPI_roll6_mean (6-month rolling mean of median inflation), FEDRates_lag1 (last month's FED rate)
- Pattern: All top-10 features are either recent inflation lags, rolling inflation statistics, or recent rate lags — confirming that inflation momentum is the dominant signal

The learning curve shows mild overfitting: training score ≈ 0.99, validation score ≈ 0.83. The gap would likely narrow with more data.

### 9.10 Gradient Boosting Regression

Gradient Boosting builds trees sequentially, each tree correcting the errors of the previous ensemble. Unlike Random Forest (parallel trees), Gradient Boosting is iterative — it is slower but often more accurate.

Result: R² = 0.9791, RMSE = 0.4666 — second best performance.

The actual vs predicted plot shows near-perfect alignment along the 45° diagonal. Residuals are small and centered at zero, with no systematic pattern — indicating the model has captured the main dynamics of the rate setting process.

### 9.11 XGBoost Regression (Best Model)

XGBoost (eXtreme Gradient Boosting) is an optimized implementation of gradient boosting with regularization built in, efficient handling of sparse data, and second-order gradient information for faster convergence.

Result: **R² = 0.9816, RMSE = 0.4377, MAE = 0.2955** — best on all three metrics.

Interpretation of RMSE = 0.44: The model's predictions are on average within **0.44 percentage points** of the actual FED rate. Given that FED rate movements are typically in 0.25 percentage point increments, this level of accuracy is remarkable for a model using only macroeconomic data.

Residual analysis: The few larger errors (> 1 percentage point) occur during:
1. The 2022–2023 hiking cycle: The speed of rate increases (11 consecutive hikes) was unprecedented in the dataset, making accurate prediction difficult
2. The 1980–1981 peak: Similarly extreme and difficult to predict from earlier patterns

---

## 10. Step 7 — Classification Models

Seven classification algorithms were trained to predict the direction of the next Fed rate change: Increase, Decrease, or No Change. This is a three-class classification problem.

### 10.1 Baseline and Ceiling

Understanding the achievable performance range is essential:
- Random baseline: 33.3% (randomly guessing one of three classes)
- "Always No Change" baseline: ~40% (the most frequent class)
- Theoretical ceiling with only macroeconomic data: approximately 65–70% (the remaining uncertainty is irreducible — FOMC decisions incorporate information unavailable in economic statistics)

Our best models achieve ~63.5%, which is 90% above the random baseline and 58% above the "always No Change" strategy.

### 10.2 Classification Results

| Model | Accuracy | Weighted F1 | CV Accuracy |
|---|---|---|---|
| **Gradient Boosting** ★ | **63.5%** | **0.6327** | 41.7% ± 11.9% |
| XGBoost | 62.9% | 0.6273 | 45.6% ± 7.1% |
| SVM | 59.9% | 0.5989 | 41.7% ± 3.0% |
| Random Forest | 59.9% | 0.5986 | 42.3% ± 17.3% |
| Logistic Regression | 55.1% | 0.5475 | 47.8% ± 13.0% |
| Decision Tree | 46.1% | 0.3986 | 42.6% ± 15.2% |
| Naive Bayes | 44.3% | 0.4322 | 54.1% ± 13.3% |

### 10.3 Logistic Regression

Logistic regression models the log-odds of each class as a linear function of features. For multi-class classification, a one-vs-rest (OvR) strategy is used: three binary classifiers are trained, each distinguishing one class from the other two.

Result: Accuracy = 55.1%. The model provides a useful baseline and interpretable coefficients, but the linear decision boundary limits performance on this non-linear problem.

ROC curve analysis: The No Change class achieves the highest AUC (~0.74), indicating it is the most linearly separable class from a macroeconomic perspective.

### 10.4 Naive Bayes

Naive Bayes assumes that all features are conditionally independent given the class label. It uses Bayes' theorem: P(class | features) ∝ P(features | class) × P(class), where P(features | class) is computed as the product of individual feature probabilities.

Result: Accuracy = 44.3% — barely above the "always No Change" baseline.

The independence assumption is badly violated here: inflation features are highly correlated with each other, CPI and inflation move together, and lag features are almost definitionally correlated with current values. When this assumption is violated, Naive Bayes severely misestimates the joint probability of feature combinations, leading to poor classification.

This result demonstrates the importance of algorithm-data fit: Naive Bayes is not inherently a bad algorithm, but it is wrong for this particular dataset.

Interestingly, Naive Bayes achieves the highest cross-validation accuracy (54.1%) of any model — suggesting it generalizes better across time periods than it performs on the test set. This could reflect the test period's unusual characteristics (2022–2023 hiking cycle) being better handled by simpler models.

### 10.5 Support Vector Machine (SVM) Classifier

SVM finds the maximum-margin hyperplane separating classes. With the RBF kernel, it implicitly operates in an infinite-dimensional feature space, capturing non-linear class boundaries.

Result: Accuracy = 59.9%, F1 = 0.5989.

The confusion matrix shows that SVM struggles most with the Increase vs No Change boundary — a historically difficult distinction, as many "no change" periods are followed by the first rate increase of a tightening cycle.

### 10.6 Decision Tree Classifier

Decision tree depth analysis for classification reveals an even more dramatic overfitting pattern than in regression:
- Depth = 1: CV accuracy = 47% (slight underfitting, but good generalization)
- Depth = 4: Test accuracy = 55%, CV accuracy = 40%
- Depth = 10: Test accuracy = 55%, Training accuracy = 90% (severe overfitting)

The optimal depth for cross-validated accuracy was 1 — a single split. This extreme result suggests that the most predictive single decision rule for rate direction is essentially "if recent inflation is above threshold, predict Increase/Decrease accordingly." Everything beyond this first split starts fitting noise rather than signal.

### 10.7 Random Forest Classifier

Random Forest achieves 59.9% accuracy by averaging over hundreds of decision trees, reducing the variance that makes individual trees unreliable.

Confusion matrix analysis:
- "No Change" correctly predicted: 74% of actual No Change months
- "Increase" correctly predicted: 57% of actual Increase months
- "Decrease" correctly predicted: 49% of actual Decrease months

The Fed holds rates steady far more often than it changes them, making No Change the most learnable class.

### 10.8 Gradient Boosting Classifier (Best Model)

Gradient Boosting iteratively builds trees to correct misclassifications, making it particularly effective at identifying difficult boundary cases.

Result: **Accuracy = 63.5%, F1 = 0.6327** — best on both metrics.

Confusion matrix findings:
- The model correctly identifies most No Change months but misclassifies some as Increase (the Fed "telegraphs" future hikes through language changes that precede actual moves)
- Rate decrease events are hardest to predict — they occur during recessions that develop gradually and are often lagged relative to economic data

Feature importance: The 1-month lag of the FED rate itself is the most important feature — rate momentum (the policy continues in its current direction) is the strongest signal. After that, 1-month and 3-month inflation lags dominate.

### 10.9 XGBoost Classifier

XGBoost classification: Accuracy = 62.9%, F1 = 0.6273 — marginally below Gradient Boosting but with better cross-validation stability (CV std = 7.1% vs GBM's 11.9%).

ROC curve analysis: The "Decrease" class achieves the highest AUC (~0.77), meaning the model is best at identifying conditions that precede rate cuts. This is economically sensible: rate cuts are almost always preceded by clear signals of economic deterioration (rising unemployment, falling inflation, GDP contraction), making them more mechanically predictable than rate hikes.

---

## 11. Step 8 — Clustering Analysis

Clustering is an unsupervised learning technique that groups data points into clusters based on similarity, without using any labels. The goal here is to discover whether 70 years of economic data naturally separate into distinct monetary policy regimes.

### 11.1 Why Clustering for Economic Data?

Economic historians often speak informally of "monetary policy eras" — the high-inflation 1970s, the Great Moderation of the 1990s, the post-2008 zero-rate era. Clustering tests whether these eras are statistically distinguishable in the feature space, using data alone without any prior knowledge of dates or historical narratives.

Both KMeans (hard boundary, centroid-based) and DBSCAN (soft boundary, density-based) are applied, providing complementary perspectives.

### 11.2 KMeans Clustering

KMeans partitions data into k clusters, assigning each point to the cluster whose centroid (mean) it is closest to, then updating centroids, iterating until convergence.

**Optimal k selection:**
The optimal k was determined by two methods:
1. Elbow method: Plot within-cluster sum of squares (inertia) against k; the optimal k is the "elbow" point where additional clusters produce diminishing inertia reduction. Elbow at k = 2.
2. Silhouette score: Measures how similar each point is to its own cluster compared to other clusters (range -1 to 1; higher is better). Maximum silhouette score at k = 2 (score = 0.366).

Both methods consistently identified **k = 2** as optimal.

**Cluster Profiles (k = 2):**

Cluster 0 — Low-Rate Modern Era (approximate period: 1990–2024):
- Mean FED rate: ~1.5%
- Mean inflation: ~2.3%
- High GDP (large modern economy)
- Lower unemployment variability (more stable labor markets post-1990)
- Corresponds to the era of inflation targeting, globalization-induced disinflation, and post-2008 ZIRP

Cluster 1 — High-Rate Inflationary Era (approximate period: 1965–1990):
- Mean FED rate: ~7.8%
- Mean inflation: ~5.4%
- Lower GDP (smaller economy in absolute terms)
- Higher unemployment variability (multiple deep recessions: 1974, 1980, 1981–82)
- Corresponds to the Great Inflation, Vietnam War fiscal expansion, oil shocks, and Volcker disinflation

The unsupervised algorithm rediscovered these historically meaningful eras with no prior knowledge of dates, events, or economic theory — it identified them purely from patterns in the data.

### 11.3 DBSCAN Clustering

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) defines clusters as regions of high density separated by low-density regions. It does not require specifying k; instead, it takes two parameters:
- eps: The maximum distance between two points for them to be considered neighbors
- min_samples: The minimum number of points to form a dense region

Points that are not in any dense region are labeled as noise (outlier points).

**Parameter tuning:** A grid search over eps values from 0.5 to 3.0 was performed:

| eps | Clusters | Noise % | Silhouette |
|---|---|---|---|
| 0.5 | 1 | 99.0% | N/A |
| 1.0 | 2 | **97.8%** | 0.462 |
| 1.5 | 3 | 93.4% | 0.330 |
| 2.0 | 4 | 86.6% | 0.251 |
| 2.5 | 9 | 68.2% | 0.371 |
| 3.0 | 12 | 45.0% | 0.270 |

Best eps = 1.0, yielding 2 clusters and 97.8% noise.

**Interpretation of 97.8% Noise:**

This is not a failure of the algorithm — it is a profound finding about the nature of economic data. DBSCAN's high noise rate means that 97.8% of months do not belong to any dense cluster in the 8-dimensional feature space. Economic data lies on a continuous manifold: transitions between regimes are gradual, stretching over years and decades. There are no sharp boundaries between the inflationary 1970s and the disinflationary 1980s — the change was continuous.

This contrasts with, say, customer purchase data where dense clusters of customers with similar spending patterns naturally exist. Economic time series evolves continuously, making it inherently resistant to density-based clustering.

**Reconciling KMeans and DBSCAN results:**
Both algorithms agree there are 2 dominant regimes, but DBSCAN reveals that most economic months are "in transition" between them. KMeans provides a useful simplification by imposing hard boundaries, while DBSCAN reveals the continuous reality.

---

## 12. Step 9 — Association Rule Mining (Apriori Algorithm)

Association Rule Mining (ARM) discovers relationships of the form "If A occurs, then B tends to occur." Originally developed for market basket analysis (if a customer buys bread, they often buy butter), it is applied here to economic indicators to discover co-movement patterns.

### 12.1 Data Preparation for ARM

ARM requires binary or categorical data. Each of the 8 economic features was discretized into three equal-frequency bins:
- Low: Bottom tertile (33rd percentile and below)
- Mid: Middle tertile (34th to 66th percentile)
- High: Top tertile (67th percentile and above)

This creates binary "items" like GDP_Low, Inflation_High, FEDRates_Mid, etc. Each month becomes a "transaction" containing the items corresponding to its feature values.

### 12.2 Apriori Algorithm Parameters

- Minimum support: 0.30 (a rule must appear in at least 30% of months)
- Minimum confidence: 0.60 (when the antecedent occurs, the consequent follows at least 60% of the time)

These thresholds balance finding meaningful rules (not too rare) with statistical reliability (high enough confidence to be actionable).

### 12.3 Results

- Frequent itemsets found: **7**
- Association rules generated: **4**
- Maximum lift: **1.957**
- Maximum confidence: **97.8%**

### 12.4 Rules Discovered

**Rule 1: GDP_Mid → RealGDP_Mid**
- Support: 48.9% (occurs in almost half of all months)
- Confidence: 97.8% (when GDP is mid, RealGDP is mid 97.8% of the time)
- Lift: 1.957 (1.96x more likely than by chance alone)

Interpretation: This rule captures the strong alignment between nominal and real GDP in non-extreme periods. When the economy is neither very small nor very large (mid-range), inflation is low enough that nominal and real measures track closely. This rule validates data consistency — it confirms the dataset is internally coherent.

**Rule 2: FEDRates_Mid → Inflation_Mid**
- Support: 30.6%
- Confidence: 61.3%
- Lift: 1.217

Interpretation: When interest rates are moderate, inflation tends to also be moderate. This is a direct empirical statement of the Taylor Rule: moderate monetary policy accompanies moderate price stability. The Fed sets mid-range rates when inflation is under control but not at its floor.

**Rule 3: Inflation_Mid → FEDRates_Mid**
- Support: 30.6%
- Confidence: 60.7%
- Lift: 1.217

Interpretation: The reverse rule — moderate inflation tends to co-occur with moderate rates. The near-identical lift and confidence of Rules 2 and 3 confirm bidirectionality: the inflation-rate relationship is symmetric. Neither variable leads the other in a simple way; they are co-determined by the monetary policy framework.

**Rule 4: GDP_Mid, RealGDP_Mid → [other mid conditions]**
Further compound rules extending the GDP co-movement pattern to include multiple features in the mid-range simultaneously.

### 12.5 Significance of ARM Findings

The Apriori algorithm independently rediscovered the Taylor Rule — the central principle of modern monetary policy — using only frequency-based co-occurrence analysis on discretized data, with no domain knowledge encoded in the algorithm. The fact that the inflation-rate bidirectional rules emerged from pure pattern matching validates both the dataset and the algorithm's ability to extract economically meaningful structure.

---

## 13. Step 10 — Model Comparison, Cross-Validation & Overfitting Analysis

### 13.1 Comprehensive Model Ranking

**Regression — Final Rankings:**

| Rank | Model | R² | RMSE | Fit Status |
|---|---|---|---|---|
| 1 | XGBoost | 0.9816 | 0.4377 | Good Fit |
| 2 | Gradient Boosting | 0.9791 | 0.4666 | Good Fit |
| 3 | Random Forest | 0.9749 | 0.5109 | Slight Overfit |
| 4 | SVR | 0.9050 | 0.9950 | Good Fit |
| 5 | Decision Tree | 0.8742 | 1.1448 | Depth-controlled |
| 6 | Linear Regression | 0.7403 | 1.6449 | Underfit |
| 7 | Lasso Regression | 0.7205 | 1.7063 | Underfit |
| 8 | Ridge Regression | 0.7101 | 1.7378 | Underfit |

**Classification — Final Rankings:**

| Rank | Model | Accuracy | F1 | Notes |
|---|---|---|---|---|
| 1 | Gradient Boosting | 63.5% | 0.6327 | Best overall |
| 2 | XGBoost | 62.9% | 0.6273 | More stable CV |
| 3 | SVM | 59.9% | 0.5989 | Non-linear boundary |
| 4 | Random Forest | 59.9% | 0.5986 | Overfits moderately |
| 5 | Logistic Regression | 55.1% | 0.5475 | Linear limitation |
| 6 | Decision Tree | 46.1% | 0.3986 | Overfit without pruning |
| 7 | Naive Bayes | 44.3% | 0.4322 | Independence violated |

### 13.2 Why Ensemble Methods Dominate

Gradient Boosting and XGBoost outperform all other algorithms on both regression and classification. The reasons are deeply connected to the structure of economic data:

1. Non-linearity: The relationship between inflation and rates is approximately linear within a regime but highly non-linear across regimes. Tree-based ensembles capture this automatically through threshold-based splits.

2. Interaction effects: The impact of inflation on rate decisions is modulated by the level of unemployment (Phillips Curve), the stage of the business cycle (GDP gap), and recent rate history (policy momentum). These three-way interactions are naturally captured by deep trees in ensembles.

3. Robustness to outliers: Tree splits are based on ordinal rankings, not absolute values. The extreme inflation and rate values from the 1970s–1980s influence tree splits without the leverage effect they would have in linear regression.

4. Built-in regularization: Both GBM and XGBoost include learning rate and tree depth constraints that prevent severe overfitting.

### 13.3 Bias-Variance Tradeoff Analysis

The bias-variance tradeoff is one of the fundamental concepts in machine learning:
- High bias (underfitting): The model is too simple to capture the true relationship — it makes similar errors on both training and test data
- High variance (overfitting): The model has memorized the training data and fails to generalize — low training error, high test error

For regression:
- Linear models (Ridge, Lasso, Linear): High bias, low variance. Test R² ≈ 0.71 matches the rough upper bound of what a linear model can explain given the non-linear dynamics of economic regimes.
- Decision Tree at high depth: Low bias, high variance. Training R² reaches 1.0 while test R² drops below 0.70.
- Ensemble methods (XGBoost, GBM): Low bias, controlled variance. Test R² = 0.98 with modest overfitting.

For classification:
- Decision Tree: Extreme variance — training accuracy near 100%, test accuracy 46% at depth > 10.
- Gradient Boosting: Best bias-variance balance — training accuracy ~100% (it always converges to perfect training fit given enough trees) but test accuracy 63.5%.

### 13.4 TimeSeriesSplit CV Results

Cross-validation with 5 temporal folds reveals the true forward-prediction difficulty:

**Regression CV R² (mean ± std):**
- XGBoost: -1.94 ± 0.86
- Gradient Boosting: -2.17 ± 0.97
- Random Forest: -2.62 ± 1.74
- SVR: -1.63 ± 1.30
- Decision Tree: -2.84 ± 1.40
- Linear Regression: -14.92 ± 14.58
- Ridge: -4.13 ± 5.03
- Lasso: -4.24 ± 5.43

All regression CV R² scores are negative. This is expected and correct: when a model trained on the 1970s high-rate regime is asked to predict the 2000s low-rate regime, it systematically overestimates, producing predictions far from the actual values. Negative R² means the model performs worse than simply predicting the mean of the test set — a known challenge in non-stationary economic forecasting.

XGBoost has the least negative CV score (-1.94), confirming it is the most robust to temporal regime shifts.

**Classification CV Accuracy:**
- XGBoost: 45.6% ± 7.1%
- Logistic Regression: 47.8% ± 13.0%
- Gradient Boosting: 41.7% ± 11.9%
- Random Forest: 42.3% ± 17.3%
- SVM: 41.7% ± 3.0%
- Naive Bayes: 54.1% ± 13.3%
- Decision Tree: 42.6% ± 15.2%

Classification CV scores are more stable (all well above zero) because the three-class direction prediction is less sensitive to regime-level magnitude shifts than continuous rate prediction.

The standard deviation across folds reveals model stability:
- SVM has the lowest std (3.0%) — most consistent across time periods
- Random Forest has the highest std (17.3%) — very fold-dependent, suggesting it overfits each training window

---

## 14. Key Findings & Conclusions

### 14.1 Economic Findings

**The Taylor Rule is empirically confirmed across all analyses:**

The Taylor Rule (proposed by economist John Taylor in 1993) states that central banks set interest rates based on inflation deviations from target and output deviations from potential. Every analytical technique in this project independently confirmed this relationship:
- Pearson correlation: Inflation has the strongest correlation with rates (r = 0.72)
- Spearman rank correlation: Inflation has the highest monotonic correlation (ρ = 0.676)
- T-test: High-inflation periods have statistically significantly higher rates (p ≈ 10⁻⁷⁴)
- ANOVA: Economic regimes have significantly different rates (F = 1440, p ≈ 0)
- PCA: The second principal component, capturing 18.7% of variance, is dominated by inflation
- Feature importance: Inflation lags are the most important features across all tree-based models
- ARM: The inflation ↔ FED rate association rules were discovered from data alone, with no economic knowledge encoded

**Economic regime shifts are the dominant challenge:**

The Federal Reserve has not operated under a single, stable policy framework for 70 years. The shift from the discretionary, politically influenced policy of the 1950s–1970s to the inflation-targeting framework of the 1990s–present represents a fundamental structural change. Models trained on one regime cannot fully predict another, which explains:
- Negative TimeSeriesSplit CV R² for regression
- The gap between training and test accuracy in classification
- The existence of two distinct KMeans clusters
- DBSCAN's inability to find dense clusters (gradual, continuous transitions)

**Inflation lags > levels for prediction:**

The 1-month and 3-month lags of inflation outperform the contemporaneous inflation level as predictors. This is consistent with how monetary policy actually works: the Fed does not react to the current month's data point (released with a lag and subject to revision) but rather to the recent trend. The model learned this policy dynamic from data without being told.

### 14.2 Technical Findings

**Feature engineering had the largest single impact on model performance:**
- Without lag and rolling features, the best R² achievable was approximately 0.50 (using contemporaneous economic indicators only)
- Adding lag features pushed the best R² to approximately 0.80
- Adding rolling statistics and interaction features pushed it to 0.98 (XGBoost)
- The 67-feature engineered set is dramatically more informative than the 8 raw features

**Temporal cross-validation is non-negotiable for time-series data:**
- Standard K-Fold CV on this dataset would produce inflated scores due to lag feature leakage
- Reporting TimeSeriesSplit CV scores — even when they are negative — is the honest and scientifically correct approach
- The discrepancy between K-Fold and TimeSeriesSplit scores serves as a measure of regime instability: larger discrepancies indicate more structural change over time

**Tree ensemble methods are best suited for macroeconomic ML:**
- They handle non-linearity, interactions, outliers, and non-normality without explicit treatment
- They provide interpretable feature importance rankings
- They are robust to the multicollinearity that undermines linear models with correlated economic features

**Naive Bayes is unsuitable for correlated economic data:**
- The independence assumption is violated by design (inflation and CPI are definitionally related; all lag features are correlated with their base feature)
- Result confirms the importance of algorithm-data compatibility over algorithmic complexity

### 14.3 Summary Statistics

| Metric | Value |
|---|---|
| Best Regression Model | XGBoost (R² = 0.9816, RMSE = 0.4377) |
| Best Classification Model | Gradient Boosting (Accuracy = 63.5%) |
| Baseline Classification Accuracy (random) | 33.3% |
| Improvement over random | +30.2 percentage points (+91%) |
| PCA components for 95% variance | 4 out of 8 |
| Lasso feature elimination | 55% (37 of 67 features zeroed out) |
| KMeans optimal clusters | 2 (high-rate era vs low-rate era) |
| DBSCAN noise percentage | 97.8% (gradual regime transitions) |
| ARM maximum lift | 1.957 (GDP_Mid → RealGDP_Mid) |
| T-test p-value (inflation vs rates) | 5.68 × 10⁻⁷⁴ |
| ANOVA F-statistic (regime differences) | 1440.28 |
| Strongest Spearman correlation | InflationConsumerPrice (ρ = 0.676) |

---

## 15. Technical Stack

| Component | Library/Tool | Version |
|---|---|---|
| Language | Python | 3.11 / 3.13 |
| Core ML | scikit-learn | ≥ 1.3.0 |
| Gradient Boosting | XGBoost | ≥ 2.0.0 |
| Association Rule Mining | mlxtend | ≥ 0.23.0 |
| Data Manipulation | pandas | ≥ 2.0.0 |
| Numerical Computing | numpy | ≥ 1.24.0 |
| Visualization | matplotlib | ≥ 3.7.0 |
| Statistical Visualization | seaborn | ≥ 0.12.0 |
| Statistical Testing | scipy | ≥ 1.10.0 |
| Time Series Statistics | statsmodels | ≥ 0.14.0 |
| Web Application | Streamlit | ≥ 1.32.0 |
| Image Handling | Pillow | ≥ 10.0.0 |
| Notebook Generation | nbformat | ≥ 5.0.0 |
| Data Source | FRED API (Federal Reserve Economic Data) | — |

### Cross-Validation Strategy
- Method: TimeSeriesSplit (scikit-learn)
- Number of splits: 5
- Direction: Always train on past, validate on future (no shuffling)
- Rationale: Prevents temporal data leakage from lag features; provides honest forward-prediction estimates

### Model Evaluation Protocol
- Train/test split: 80% / 20% chronological (no shuffling)
- Regression metrics: R², RMSE, MAE, TimeSeriesSplit CV R²
- Classification metrics: Accuracy, Weighted F1, Precision, Recall, ROC-AUC (per class), TimeSeriesSplit CV Accuracy
- Overfitting diagnosis: Learning curves (train score vs validation score as a function of training size), train-test gap analysis, decision tree depth analysis

### Project File Structure

```
ML-Project/
├── App/                          # Original Streamlit app (pre-project)
├── ml_analysis/
│   ├── part1_eda_features.py     # Data cleaning, EDA, feature engineering, PCA
│   ├── part2_models.py           # All ML models (regression, classification, clustering, ARM)
│   ├── fix_charts.py             # Chart corrections and improvements
│   ├── generate_notebooks.py     # Programmatic Jupyter notebook generation
│   ├── build_website.py          # Static website generator (full version)
│   └── outputs/
│       ├── charts/               # All generated charts (72 PNG files)
│       │   ├── eda/
│       │   ├── cleaning/
│       │   ├── features/
│       │   ├── stats/
│       │   ├── pca/
│       │   ├── regression/
│       │   ├── classification/
│       │   ├── clustering/
│       │   ├── arm/
│       │   └── comparison/
│       └── results/              # JSON result files
│           ├── regression_results.json
│           ├── classification_results.json
│           ├── clustering_results.json
│           ├── arm_results.json
│           ├── pca_results.json
│           ├── statistical_analysis.json
│           └── preprocessed_data.pkl
├── streamlit_app/
│   ├── main.py                   # Streamlit multi-page application
│   └── requirements.txt
├── notebooks/                    # Jupyter notebooks (one per ML step)
│   ├── 01_Data_Cleaning_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Statistical_Analysis.ipynb
│   ├── 04_PCA_Analysis.ipynb
│   ├── 05_Regression_Models.ipynb
│   ├── 06_Classification_Models.ipynb
│   ├── 07_Clustering.ipynb
│   ├── 08_Association_Rules.ipynb
│   └── 09_Model_Comparison.ipynb
├── website/
│   ├── index.html                # Static website
│   └── images/                   # All chart images
├── build_simple_site.py          # Simple website generator
└── PROJECT_REPORT.md             # This document
```

---

*This project was built and executed entirely in Python. All code was run programmatically — data loading, cleaning, feature engineering, statistical testing, model training, chart generation, and website building — without manual intervention in intermediate steps.*

*Data: FRED (Federal Reserve Economic Data), Federal Reserve Bank of St. Louis. Coverage: January 1954 — December 2024.*
