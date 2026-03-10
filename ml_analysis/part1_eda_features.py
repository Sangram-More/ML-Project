"""
PART 1: Data Cleaning, EDA, Feature Engineering, Statistical Analysis, PCA
Federal Reserve Interest Rate Prediction — Professional ML Pipeline
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, kstest, ttest_ind, f_oneway, pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import json, os, pickle

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = "d:/Projects/ML website/ML-Project"
DATA   = f"{BASE}/App/Tabs/Datasets/finaldataset.csv"
OUT    = f"{BASE}/ml_analysis/outputs"
CHARTS = f"{OUT}/charts"
RES    = f"{OUT}/results"

plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = ['#2196F3','#F44336','#4CAF50','#FF9800','#9C27B0',
           '#00BCD4','#E91E63','#795548','#607D8B','#FF5722']
sns.set_palette(PALETTE)

def savefig(path, dpi=150):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"  Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & INITIAL EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 1 — DATA LOADING & INITIAL EXPLORATION")
print("="*70)

df_raw = pd.read_csv(DATA)
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.sort_values('date').reset_index(drop=True)

print(f"Shape       : {df_raw.shape}")
print(f"Date range  : {df_raw['date'].min().date()} to {df_raw['date'].max().date()}")
print(f"Columns     : {list(df_raw.columns)}")
print("\nFirst 5 rows:")
print(df_raw.head())
print("\nDtype info:")
print(df_raw.dtypes)
print("\nDescriptive statistics:")
print(df_raw.describe())

FEATURES = ['ConsumerPriceIndexAllItems','GDP','InflationConsumerPrice',
            'MedianConsumerPriceIndex','RealGDP','RealGDPPerCapita',
            'RealPotentialGDP','UnemployemenrRate']
TARGET_REG = 'FEDRates'

# ── Missing values before cleaning ──
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
missing_df = pd.DataFrame({'Missing':missing,'Missing_%':missing_pct})
print("\nMissing values (raw):")
print(missing_df[missing_df['Missing']>0])

# ── Fig 1: Missing Value Heatmap ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(df_raw.isnull(), cbar=False, yticklabels=False, ax=axes[0],
            cmap='RdYlGn_r')
axes[0].set_title('Missing Value Map (Raw Data)', fontsize=13, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

bars = axes[1].bar(missing_df.index, missing_df['Missing_%'],
                   color=[PALETTE[i % len(PALETTE)] for i in range(len(missing_df))])
axes[1].set_title('Missing Values % per Column', fontsize=13, fontweight='bold')
axes[1].set_xticklabels(missing_df.index, rotation=45, ha='right')
axes[1].set_ylabel('% Missing')
for bar, val in zip(bars, missing_df['Missing_%']):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
savefig(f"{CHARTS}/cleaning/01_missing_values.png")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 2 — DATA CLEANING")
print("="*70)

df = df_raw.copy()
df_raw_backup = df_raw.copy()

# 2a. Drop rows with missing FEDRates (the target)
before = len(df)
df = df.dropna(subset=['FEDRates'])
print(f"Rows dropped (missing FEDRates): {before - len(df)}")

# 2b. Forward-fill then backward-fill remaining numeric NaNs
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].ffill().bfill()
print(f"Rows after cleaning: {len(df)}")
print(f"Remaining NaNs: {df[numeric_cols].isnull().sum().sum()}")

# 2c. Fix ConsumerPriceIndexAllItems — replace 0.0 sentinel with NaN then interpolate
# Many early values are 0.0 because percent change wasn't available; interpolate linearly
zero_mask = df['ConsumerPriceIndexAllItems'] == 0.0
print(f"\nZero CPI entries (likely missing data): {zero_mask.sum()}")
df.loc[zero_mask, 'ConsumerPriceIndexAllItems'] = np.nan
df['ConsumerPriceIndexAllItems'] = df['ConsumerPriceIndexAllItems'].interpolate(method='linear')
df['ConsumerPriceIndexAllItems'] = df['ConsumerPriceIndexAllItems'].ffill().bfill()

# 2d. Outlier Detection using IQR
print("\nOutlier detection (IQR method):")
outlier_info = {}
for col in [TARGET_REG] + FEATURES:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 3.0 * IQR
    upper = Q3 + 3.0 * IQR
    out_mask = (df[col] < lower) | (df[col] > upper)
    outlier_info[col] = {'count': int(out_mask.sum()), 'lower': lower, 'upper': upper}
    if out_mask.sum() > 0:
        print(f"  {col}: {out_mask.sum()} extreme outliers (bounds [{lower:.2f}, {upper:.2f}])")

# Cap extreme outliers (Winsorization at 1st/99th percentile)
for col in [TARGET_REG] + FEATURES:
    p01 = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    df[col] = df[col].clip(lower=p01, upper=p99)

# ── Fig 2: Box plots before vs after cleaning ──
fig, axes = plt.subplots(2, len(FEATURES), figsize=(22, 8))
for i, col in enumerate(FEATURES):
    axes[0, i].boxplot(df_raw_backup[col].dropna(), patch_artist=True,
                       boxprops=dict(facecolor=PALETTE[i % len(PALETTE)], alpha=0.7))
    axes[0, i].set_title(col[:14], fontsize=7)
    axes[0, i].set_xticklabels(['Raw'])

    axes[1, i].boxplot(df[col].dropna(), patch_artist=True,
                       boxprops=dict(facecolor=PALETTE[i % len(PALETTE)], alpha=0.7))
    axes[1, i].set_xticklabels(['Clean'])

axes[0, 0].set_ylabel('Before Cleaning', fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel('After Cleaning', fontsize=10, fontweight='bold')
fig.suptitle('Outlier Detection — Box Plots Before vs After Cleaning', fontsize=14, fontweight='bold', y=1.02)
savefig(f"{CHARTS}/cleaning/02_boxplots_before_after.png")

# ── Fig 3: Outlier count bar chart ──
fig, ax = plt.subplots(figsize=(10, 4))
cols_out = list(outlier_info.keys())
counts = [outlier_info[c]['count'] for c in cols_out]
bars = ax.bar(cols_out, counts, color=PALETTE[:len(cols_out)])
ax.set_title('Extreme Outlier Counts per Feature (IQR × 3)', fontsize=13, fontweight='bold')
ax.set_xticklabels(cols_out, rotation=45, ha='right')
ax.set_ylabel('Outlier Count')
for bar, val in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            str(val), ha='center', va='bottom', fontsize=9)
savefig(f"{CHARTS}/cleaning/03_outlier_counts.png")

print(f"\nFinal clean dataset shape: {df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. EDA — EXPLORATORY DATA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 3 — EXPLORATORY DATA ANALYSIS")
print("="*70)

# ── Fig 4: Time series of FED Rates ──
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['date'], df['FEDRates'], color='#2196F3', linewidth=1.5, label='Federal Funds Rate')
ax.fill_between(df['date'], df['FEDRates'], alpha=0.1, color='#2196F3')
recessions = [('1973-11','1975-03'),('1980-01','1980-07'),('1981-07','1982-11'),
              ('1990-07','1991-03'),('2001-03','2001-11'),('2007-12','2009-06'),
              ('2020-02','2020-04')]
for r in recessions:
    ax.axvspan(pd.to_datetime(r[0]), pd.to_datetime(r[1]),
               alpha=0.15, color='red', label='_nolegend_')
ax.set_title('US Federal Funds Rate (1954–2024) with Recession Periods', fontsize=14, fontweight='bold')
ax.set_xlabel('Year'); ax.set_ylabel('Fed Funds Rate (%)')
ax.legend(['Fed Funds Rate', 'NBER Recessions'], loc='upper right')
savefig(f"{CHARTS}/eda/01_fedfunds_timeseries.png")

# ── Fig 5: All features time series ──
fig, axes = plt.subplots(4, 2, figsize=(16, 18))
axes = axes.flatten()
for i, col in enumerate(FEATURES):
    axes[i].plot(df['date'], df[col], color=PALETTE[i], linewidth=1.2)
    axes[i].fill_between(df['date'], df[col], alpha=0.08, color=PALETTE[i])
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Year'); axes[i].tick_params(axis='x', rotation=30)
fig.suptitle('All Economic Indicators — Time Series (1954–2024)', fontsize=14, fontweight='bold')
savefig(f"{CHARTS}/eda/02_all_features_timeseries.png")

# ── Fig 6: Distributions ──
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
all_cols = [TARGET_REG] + FEATURES
for i, col in enumerate(all_cols):
    axes[i].hist(df[col], bins=40, color=PALETTE[i % len(PALETTE)], alpha=0.75, edgecolor='white')
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.5, label='Mean')
    axes[i].axvline(df[col].median(), color='green', linestyle='--', linewidth=1.5, label='Median')
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].legend(fontsize=7)
fig.suptitle('Feature Distributions — Histograms with Mean/Median', fontsize=14, fontweight='bold')
savefig(f"{CHARTS}/eda/03_distributions.png")

# ── Fig 7: Correlation heatmap ──
fig, ax = plt.subplots(figsize=(11, 9))
corr = df[all_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, ax=ax, linewidths=0.5,
            annot_kws={'size': 9})
ax.set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')
savefig(f"{CHARTS}/eda/04_correlation_heatmap.png")

# ── Fig 8: Scatter plots vs FED Rates ──
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(FEATURES):
    sc = axes[i].scatter(df[col], df['FEDRates'], c=df['date'].astype(np.int64),
                         cmap='viridis', alpha=0.4, s=15)
    r, p = pearsonr(df[col], df['FEDRates'])
    axes[i].set_title(f'{col}\nr={r:.3f}, p={p:.2e}', fontsize=9, fontweight='bold')
    axes[i].set_xlabel(col, fontsize=8); axes[i].set_ylabel('FEDRates', fontsize=8)
    plt.colorbar(sc, ax=axes[i], label='Time →', pad=0.02)
fig.suptitle('Feature vs FED Rate Scatter Plots (Color = Time Progression)',
             fontsize=13, fontweight='bold')
savefig(f"{CHARTS}/eda/05_scatter_vs_fedrate.png")

# ── Fig 9: Pair plot (sample) ──
sample_cols = ['FEDRates','InflationConsumerPrice','UnemployemenrRate','GDP','RealGDP']
pair_df = df[sample_cols].copy()
fig = plt.figure(figsize=(12, 12))
from pandas.plotting import scatter_matrix
axes_sm = scatter_matrix(pair_df, alpha=0.3, figsize=(12, 12), diagonal='kde',
                         color='#2196F3')
plt.suptitle('Scatter Matrix — Key Economic Indicators', fontsize=13, fontweight='bold', y=1.01)
savefig(f"{CHARTS}/eda/06_scatter_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# 4. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 4 — FEATURE ENGINEERING")
print("="*70)

df = df.set_index('date')

# 4a. Rate change direction labels (classification target)
df['RateChange'] = df['FEDRates'].diff()
df['RateDirection'] = 'No_Change'
df.loc[df['RateChange'] >  0.05, 'RateDirection'] = 'Increase'
df.loc[df['RateChange'] < -0.05, 'RateDirection'] = 'Decrease'

print("Rate direction class distribution:")
print(df['RateDirection'].value_counts())

# 4b. Lag features (1, 3, 6, 12 months)
lag_months = [1, 3, 6, 12]
for col in FEATURES:
    for lag in lag_months:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

# 4c. Rolling statistics (3 and 6 month windows)
for col in FEATURES:
    df[f'{col}_roll3_mean'] = df[col].rolling(3).mean()
    df[f'{col}_roll6_mean'] = df[col].rolling(6).mean()
    df[f'{col}_roll3_std']  = df[col].rolling(3).std()

# 4d. Interaction features
df['Inflation_x_Unemployment'] = df['InflationConsumerPrice'] * df['UnemployemenrRate']
df['GDP_growth'] = df['GDP'].pct_change() * 100
df['RealGDP_growth'] = df['RealGDP'].pct_change() * 100

# 4e. Calendar features
df['Month']  = df.index.month
df['Year']   = df.index.year
df['Quarter'] = df.index.quarter

# 4f. Economic regime (based on FED rate quartiles)
df['RateRegime'] = pd.qcut(df['FEDRates'], q=4,
                            labels=['Very_Low','Low','High','Very_High'])

# Drop NaN rows introduced by lag/rolling features
n_before = len(df)
df = df.dropna()
print(f"Rows after dropping lag-NaN: {n_before} → {len(df)}")

print(f"\nEngineered dataset shape: {df.shape}")
print(f"Total features created   : {df.shape[1]}")

# ── Fig 10: Class distribution ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vc = df['RateDirection'].value_counts()
colors_cls = ['#F44336','#4CAF50','#FF9800']
axes[0].bar(vc.index, vc.values, color=colors_cls[:len(vc)])
for i, (k, v) in enumerate(vc.items()):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')
axes[0].set_title('Rate Direction Class Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Direction'); axes[0].set_ylabel('Count')

axes[1].pie(vc.values, labels=vc.index, autopct='%1.1f%%',
            colors=colors_cls[:len(vc)], startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=2))
axes[1].set_title('Rate Direction Class Proportion', fontsize=12, fontweight='bold')
savefig(f"{CHARTS}/features/01_class_distribution.png")

# ── Fig 11: Feature importances preview (correlation with target) ──
num_df = df.select_dtypes(include=[np.number])
corr_with_target = num_df.corr()['FEDRates'].drop('FEDRates').sort_values(key=abs, ascending=False)
top20 = corr_with_target.head(20)
fig, ax = plt.subplots(figsize=(10, 7))
colors_bar = ['#F44336' if v < 0 else '#2196F3' for v in top20.values]
ax.barh(range(len(top20)), top20.values, color=colors_bar)
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20.index, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Top 20 Features Correlated with FED Rate', fontsize=13, fontweight='bold')
ax.set_xlabel('Pearson Correlation Coefficient')
savefig(f"{CHARTS}/features/02_feature_correlations.png")

# ── Fig 12: Lag features correlation ──
lag_cols = [c for c in df.columns if '_lag' in c and 'FEDRates' not in c][:16]
if lag_cols:
    lag_corr = df[lag_cols + ['FEDRates']].corr()['FEDRates'].drop('FEDRates')
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(lag_corr)), lag_corr.values,
           color=[PALETTE[i % len(PALETTE)] for i in range(len(lag_corr))])
    ax.set_xticks(range(len(lag_corr)))
    ax.set_xticklabels(lag_corr.index, rotation=45, ha='right', fontsize=7)
    ax.set_title('Lag Feature Correlations with FED Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('Pearson Correlation')
    ax.axhline(0, color='black', linewidth=0.8)
    savefig(f"{CHARTS}/features/03_lag_correlations.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. STATISTICAL ANALYSIS & HYPOTHESIS TESTING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 5 — STATISTICAL ANALYSIS & HYPOTHESIS TESTING")
print("="*70)

stats_results = {}

# 5a. Normality tests
print("\n--- Normality Tests (Shapiro-Wilk) ---")
normality = {}
for col in [TARGET_REG] + FEATURES:
    sample = df[col].dropna().sample(min(500, len(df)), random_state=42)
    stat, p = shapiro(sample)
    normality[col] = {'statistic': round(float(stat), 4), 'p_value': round(float(p), 6),
                      'is_normal': bool(p > 0.05)}
    print(f"  {col:40s} W={stat:.4f}  p={p:.4e}  Normal={'YES' if p>0.05 else 'NO'}")
stats_results['normality'] = normality

# 5b. Stationarity test (ADF)
from statsmodels.tsa.stattools import adfuller
print("\n--- ADF Stationarity Tests ---")
stationarity = {}
for col in [TARGET_REG] + FEATURES:
    result = adfuller(df[col].dropna(), autolag='AIC')
    stationarity[col] = {'adf_stat': round(float(result[0]), 4),
                         'p_value':  round(float(result[1]), 6),
                         'is_stationary': bool(result[1] < 0.05)}
    status = 'STATIONARY' if result[1] < 0.05 else 'NON-STATIONARY'
    print(f"  {col:40s} ADF={result[0]:.4f}  p={result[1]:.4e}  {status}")
stats_results['stationarity'] = stationarity

# 5c. T-test: Compare FED rates in high vs low inflation periods
med_inflation = df['InflationConsumerPrice'].median()
high_inf_rates = df.loc[df['InflationConsumerPrice'] > med_inflation, 'FEDRates']
low_inf_rates  = df.loc[df['InflationConsumerPrice'] <= med_inflation, 'FEDRates']
t_stat, t_p = ttest_ind(high_inf_rates, low_inf_rates)
print(f"\n--- T-Test: FED Rate in High vs Low Inflation ---")
print(f"  High Inflation mean FED rate : {high_inf_rates.mean():.3f}%")
print(f"  Low Inflation mean FED rate  : {low_inf_rates.mean():.3f}%")
print(f"  t-stat={t_stat:.4f}  p={t_p:.4e}  Significant={'YES' if t_p<0.05 else 'NO'}")
stats_results['ttest_inflation'] = {'t_stat': float(t_stat), 'p_value': float(t_p),
                                     'high_mean': float(high_inf_rates.mean()),
                                     'low_mean': float(low_inf_rates.mean())}

# 5d. ANOVA: FED rates across rate regimes
groups = [df.loc[df['RateRegime'] == r, 'FEDRates'].values
          for r in df['RateRegime'].cat.categories]
f_stat, f_p = f_oneway(*groups)
print(f"\n--- ANOVA: FED Rate across Economic Regimes ---")
print(f"  F-stat={f_stat:.4f}  p={f_p:.4e}  Significant={'YES' if f_p<0.05 else 'NO'}")
stats_results['anova_regimes'] = {'f_stat': float(f_stat), 'p_value': float(f_p)}

# 5e. Spearman correlation
print("\n--- Spearman Rank Correlations with FED Rate ---")
spearman_results = {}
for col in FEATURES:
    rho, p = spearmanr(df[col], df['FEDRates'])
    spearman_results[col] = {'rho': round(float(rho), 4), 'p_value': round(float(p), 6)}
    print(f"  {col:40s} rho={rho:.4f}  p={p:.4e}")
stats_results['spearman'] = spearman_results

with open(f"{RES}/statistical_analysis.json", 'w') as f:
    json.dump(stats_results, f, indent=2)

# ── Fig 13: Normality check — QQ plots ──
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(all_cols):
    from scipy.stats import probplot
    probplot(df[col], dist='norm', plot=axes[i])
    axes[i].set_title(f'{col}\nShapiro p={normality.get(col,{}).get("p_value","N/A")}',
                      fontsize=9, fontweight='bold')
    axes[i].get_lines()[0].set(markersize=2, alpha=0.5, color=PALETTE[i % len(PALETTE)])
    axes[i].get_lines()[1].set(color='red', linewidth=1.5)
fig.suptitle('Q-Q Plots — Normality Assessment', fontsize=14, fontweight='bold')
savefig(f"{CHARTS}/stats/01_qq_plots.png")

# ── Fig 14: Hypothesis test summary ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
norm_df = pd.DataFrame(normality).T
norm_df['color'] = norm_df['is_normal'].map({True:'#4CAF50', False:'#F44336'})
axes[0].barh(norm_df.index, norm_df['p_value'],
             color=norm_df['color'])
axes[0].axvline(0.05, color='black', linestyle='--', linewidth=1.5, label='α = 0.05')
axes[0].set_title('Shapiro-Wilk Normality Test p-values\n(Green = Normal, Red = Non-Normal)',
                  fontsize=11, fontweight='bold')
axes[0].set_xlabel('p-value')
axes[0].legend()

stat_df = pd.DataFrame(stationarity).T
stat_df['color'] = stat_df['is_stationary'].map({True:'#4CAF50', False:'#F44336'})
axes[1].barh(stat_df.index, stat_df['p_value'],
             color=stat_df['color'])
axes[1].axvline(0.05, color='black', linestyle='--', linewidth=1.5, label='α = 0.05')
axes[1].set_title('ADF Stationarity Test p-values\n(Green = Stationary, Red = Non-Stationary)',
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel('p-value')
axes[1].legend()
savefig(f"{CHARTS}/stats/02_hypothesis_tests.png")

# ── Fig 15: T-test visualization ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].hist(high_inf_rates, bins=40, alpha=0.6, color='#F44336', label=f'High Inflation (μ={high_inf_rates.mean():.2f}%)')
axes[0].hist(low_inf_rates,  bins=40, alpha=0.6, color='#2196F3', label=f'Low Inflation (μ={low_inf_rates.mean():.2f}%)')
axes[0].axvline(high_inf_rates.mean(), color='#F44336', linestyle='--', linewidth=2)
axes[0].axvline(low_inf_rates.mean(),  color='#2196F3', linestyle='--', linewidth=2)
axes[0].set_title(f'T-Test: FED Rate by Inflation Level\nt={t_stat:.3f}, p={t_p:.2e}', fontsize=11, fontweight='bold')
axes[0].set_xlabel('FED Rate (%)'); axes[0].legend()

spear_vals = pd.DataFrame(spearman_results).T['rho'].sort_values()
colors_sp = ['#F44336' if v < 0 else '#2196F3' for v in spear_vals.values]
axes[1].barh(spear_vals.index, spear_vals.values, color=colors_sp)
axes[1].set_title('Spearman Rank Correlations with FED Rate', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Spearman ρ')
axes[1].axvline(0, color='black', linewidth=0.8)
savefig(f"{CHARTS}/stats/03_ttest_spearman.png")

# ── Fig 16: Descriptive statistics heatmap ──
desc = df[[TARGET_REG]+FEATURES].describe().T
fig, ax = plt.subplots(figsize=(14, 6))
desc_norm = (desc - desc.min()) / (desc.max() - desc.min() + 1e-9)
sns.heatmap(desc_norm, annot=desc.round(2), fmt='.2f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, annot_kws={'size': 7})
ax.set_title('Descriptive Statistics Heatmap (Normalized for Color Scale)',
             fontsize=13, fontweight='bold')
savefig(f"{CHARTS}/stats/04_descriptive_stats_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# 6. PCA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 6 — PRINCIPAL COMPONENT ANALYSIS")
print("="*70)

scaler_pca = StandardScaler()
X_pca_raw  = df[FEATURES].values
X_pca      = scaler_pca.fit_transform(X_pca_raw)

pca_full = PCA()
pca_full.fit(X_pca)
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("Explained variance per component:")
for i, (ev, cv) in enumerate(zip(explained, cumulative)):
    print(f"  PC{i+1}: {ev:.4f} ({ev*100:.2f}%)  Cumulative: {cv*100:.2f}%")

n_components_95 = np.argmax(cumulative >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components_95}")

# ── Fig 17: PCA scree plot ──
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
components = range(1, len(explained) + 1)
axes[0].bar(components, explained * 100, color='#2196F3', alpha=0.8, label='Individual')
axes[0].plot(components, cumulative * 100, 'ro-', linewidth=2, markersize=6, label='Cumulative')
axes[0].axhline(95, color='green', linestyle='--', linewidth=1.5, label='95% threshold')
axes[0].set_xlabel('Principal Component'); axes[0].set_ylabel('Explained Variance (%)')
axes[0].set_title('PCA Scree Plot', fontsize=13, fontweight='bold')
axes[0].legend(); axes[0].set_xticks(components)

# Loadings heatmap
pca_3 = PCA(n_components=3)
pca_3.fit(X_pca)
loadings = pd.DataFrame(pca_3.components_.T, index=FEATURES,
                        columns=[f'PC{i+1}' for i in range(3)])
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r',
            ax=axes[1], linewidths=0.5, annot_kws={'size': 9})
axes[1].set_title('PCA Loadings (Top 3 Components)', fontsize=13, fontweight='bold')
savefig(f"{CHARTS}/pca/01_scree_loadings.png")

# ── Fig 18: PCA biplot (PC1 vs PC2) ──
X_pca_3 = pca_3.transform(X_pca)
direction_map = {'Increase': 0, 'No_Change': 1, 'Decrease': 2}
color_vals = df['RateDirection'].map(direction_map).fillna(1)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sc = axes[0].scatter(X_pca_3[:, 0], X_pca_3[:, 1], c=color_vals,
                     cmap='RdYlGn', alpha=0.5, s=20)
for i, feat in enumerate(FEATURES):
    scale = 3
    axes[0].arrow(0, 0, pca_3.components_[0, i]*scale, pca_3.components_[1, i]*scale,
                  head_width=0.08, head_length=0.05, fc='#333333', ec='#333333', alpha=0.8)
    axes[0].text(pca_3.components_[0, i]*scale*1.15,
                 pca_3.components_[1, i]*scale*1.15, feat[:10], fontsize=7)
axes[0].set_xlabel(f'PC1 ({explained[0]*100:.1f}% var)')
axes[0].set_ylabel(f'PC2 ({explained[1]*100:.1f}% var)')
axes[0].set_title('PCA Biplot (PC1 vs PC2)', fontsize=12, fontweight='bold')
plt.colorbar(sc, ax=axes[0], label='Rate Direction (0=Inc,1=No,2=Dec)')

axes[1].scatter(X_pca_3[:, 0], X_pca_3[:, 2], c=color_vals, cmap='RdYlGn', alpha=0.5, s=20)
axes[1].set_xlabel(f'PC1 ({explained[0]*100:.1f}% var)')
axes[1].set_ylabel(f'PC3 ({explained[2]*100:.1f}% var)')
axes[1].set_title('PCA Biplot (PC1 vs PC3)', fontsize=12, fontweight='bold')
savefig(f"{CHARTS}/pca/02_biplot.png")

# ── Fig 19: PCA cumulative variance ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(components, cumulative*100, alpha=0.2, color='#2196F3')
ax.plot(components, cumulative*100, 'b-o', linewidth=2, markersize=8)
for i, cv in enumerate(cumulative):
    ax.annotate(f'{cv*100:.1f}%', (i+1, cv*100), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=8)
ax.axhline(95, color='red', linestyle='--', linewidth=1.5, label='95%')
ax.axhline(99, color='green', linestyle='--', linewidth=1.5, label='99%')
ax.set_xlabel('Number of Components'); ax.set_ylabel('Cumulative Explained Variance (%)')
ax.set_title('PCA Cumulative Explained Variance', fontsize=13, fontweight='bold')
ax.legend(); ax.set_xticks(components)
savefig(f"{CHARTS}/pca/03_cumulative_variance.png")

pca_results = {
    'explained_variance_ratio': [float(x) for x in explained],
    'cumulative_variance': [float(x) for x in cumulative],
    'n_components_95pct': int(n_components_95),
    'loadings': loadings.to_dict()
}
with open(f"{RES}/pca_results.json", 'w') as f:
    json.dump(pca_results, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# 7. SAVE PREPROCESSED DATA FOR MODELLING
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 7 — SAVING PREPROCESSED DATA")
print("="*70)

# Select only numeric columns + target labels
model_df = df.select_dtypes(include=[np.number]).copy()
model_df['RateDirection'] = df['RateDirection']
model_df = model_df.dropna()

# Regression target
y_reg = model_df['FEDRates'].values

# Classification target (encode)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_cls = le.fit_transform(model_df['RateDirection'].values)
label_names = le.classes_

# Feature columns (exclude targets & derived target cols)
exclude = ['FEDRates','RateChange','Month','Year','Quarter']
feature_cols = [c for c in model_df.columns
                if c not in exclude and model_df[c].dtype != object
                and c != 'RateDirection']

X = model_df[feature_cols].values
print(f"Feature matrix shape: {X.shape}")
print(f"Regression target shape: {y_reg.shape}")
print(f"Classification target shape: {y_cls.shape}")
print(f"Classes: {label_names}")
print(f"Class distribution: {dict(zip(label_names, np.bincount(y_cls)))}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save everything for Part 2
pkl_data = {
    'X': X, 'X_scaled': X_scaled, 'y_reg': y_reg, 'y_cls': y_cls,
    'feature_cols': feature_cols, 'label_names': list(label_names),
    'scaler': scaler, 'le': le, 'df': df, 'model_df': model_df,
    'pca_3': pca_3, 'X_pca_3': X_pca_3,
    'explained': explained, 'cumulative': cumulative
}
with open(f"{RES}/preprocessed_data.pkl", 'wb') as f:
    pickle.dump(pkl_data, f)

print("\nPart 1 complete. All charts and data saved.")
print(f"Charts in: {CHARTS}")
print(f"Results in: {RES}")
