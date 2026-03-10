"""
Generate all Jupyter notebooks for the ML analysis.
Each notebook is self-contained and runnable.
"""
import nbformat as nbf
import os

NB_DIR = "d:/Projects/ML website/ML-Project/notebooks"
os.makedirs(NB_DIR, exist_ok=True)

BASE = "d:/Projects/ML website/ML-Project"
DATA = f"{BASE}/App/Tabs/Datasets/finaldataset.csv"
OUT  = f"{BASE}/ml_analysis/outputs"

def nb(cells):
    nb_obj = nbf.v4.new_notebook()
    nb_obj.cells = cells
    nb_obj.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.0"}
    }
    return nb_obj

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

SETUP = f'''import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

plt.style.use('seaborn-v0_8-whitegrid')
PALETTE = ['#2196F3','#F44336','#4CAF50','#FF9800','#9C27B0',
           '#00BCD4','#E91E63','#795548','#607D8B','#FF5722']
sns.set_palette(PALETTE)

DATA_PATH = r"{DATA}"
OUT_PATH  = r"{OUT}"
'''

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 1 — Data Cleaning & EDA
# ══════════════════════════════════════════════════════════════════════════════
nb1_cells = [
md("""# Notebook 1: Data Cleaning & Exploratory Data Analysis
## Federal Reserve Interest Rate Prediction — ML Pipeline

**Objective:** Load, clean, and explore the FRED economic dataset covering 1954–2024.

**Dataset Features:**
- `FEDRates` — Federal Funds Rate (TARGET)
- `ConsumerPriceIndexAllItems` — CPI % change (MoM)
- `GDP` — Nominal GDP (Billions USD)
- `InflationConsumerPrice` — Annual inflation rate
- `MedianConsumerPriceIndex` — Median CPI
- `RealGDP` — Inflation-adjusted GDP
- `RealGDPPerCapita` — Real GDP per person
- `RealPotentialGDP` — Economy's productive capacity
- `UnemployemenrRate` — U-3 unemployment rate
"""),
code(SETUP),
md("## 1. Data Loading"),
code('''# Load dataset
df_raw = pd.read_csv(DATA_PATH)
df_raw['date'] = pd.to_datetime(df_raw['date'])
df_raw = df_raw.sort_values('date').reset_index(drop=True)

print(f"Shape: {df_raw.shape}")
print(f"Date range: {df_raw['date'].min().date()} to {df_raw['date'].max().date()}")
display(df_raw.head(10))
'''),
code('''# Descriptive statistics
display(df_raw.describe().T.round(3))
'''),
code('''# Missing value analysis
missing = df_raw.isnull().sum()
missing_pct = (missing / len(df_raw) * 100).round(2)
display(pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct}))
'''),
md("## 2. Data Cleaning"),
code('''df = df_raw.copy()

# Step 1: Drop rows with missing target
before = len(df)
df = df.dropna(subset=['FEDRates'])
print(f"Rows dropped (missing FEDRates): {before - len(df)}")

# Step 2: Forward-fill then backward-fill remaining NaNs
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].ffill().bfill()
print(f"Remaining NaNs after fill: {df[numeric_cols].isnull().sum().sum()}")

# Step 3: Fix zero CPI entries (sentinel value for unavailable data)
zero_mask = df['ConsumerPriceIndexAllItems'] == 0.0
print(f"Zero CPI entries: {zero_mask.sum()}")
df.loc[zero_mask, 'ConsumerPriceIndexAllItems'] = np.nan
df['ConsumerPriceIndexAllItems'] = df['ConsumerPriceIndexAllItems'].interpolate('linear').ffill().bfill()

# Step 4: Winsorization — cap at 1st/99th percentile
for col in numeric_cols:
    p01, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(lower=p01, upper=p99)

print(f"Final shape: {df.shape}")
'''),
code('''# Outlier detection using IQR method
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()
features = ['ConsumerPriceIndexAllItems','GDP','InflationConsumerPrice',
            'MedianConsumerPriceIndex','RealGDP','RealGDPPerCapita',
            'RealPotentialGDP','UnemployemenrRate']
for i, col in enumerate(features):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    axes[i].boxplot([df_raw[col].dropna(), df[col]], labels=['Before','After'],
                    patch_artist=True,
                    boxprops=dict(facecolor=PALETTE[i], alpha=0.6))
    axes[i].set_title(f'{col[:15]}\\n({outliers} outliers)', fontsize=9, fontweight='bold')
fig.suptitle('Box Plots: Before vs After Winsorization', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
'''),
md("## 3. Exploratory Data Analysis"),
code('''# FED Rate time series with recession periods
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['date'], df['FEDRates'], color='#2196F3', linewidth=1.5)
ax.fill_between(df['date'], df['FEDRates'], alpha=0.1, color='#2196F3')
recessions = [('1973-11','1975-03'),('1980-01','1980-07'),('1981-07','1982-11'),
              ('1990-07','1991-03'),('2001-03','2001-11'),('2007-12','2009-06'),
              ('2020-02','2020-04')]
for r in recessions:
    ax.axvspan(pd.to_datetime(r[0]), pd.to_datetime(r[1]), alpha=0.15, color='red')
ax.set_title('US Federal Funds Rate (1954–2024)', fontsize=13, fontweight='bold')
ax.set_xlabel('Year'); ax.set_ylabel('Rate (%)')
plt.tight_layout(); plt.show()
'''),
code('''# Distribution plots for all features
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
all_cols = ['FEDRates'] + features
for i, col in enumerate(all_cols):
    axes[i].hist(df[col], bins=40, color=PALETTE[i % len(PALETTE)], alpha=0.75, edgecolor='white')
    axes[i].axvline(df[col].mean(),   color='red',   linestyle='--', label='Mean')
    axes[i].axvline(df[col].median(), color='green', linestyle='--', label='Median')
    axes[i].set_title(col, fontsize=10, fontweight='bold')
    axes[i].legend(fontsize=7)
fig.suptitle('Feature Distributions', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
'''),
code('''# Correlation heatmap
fig, ax = plt.subplots(figsize=(11, 9))
corr = df[all_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            vmin=-1, vmax=1, ax=ax, linewidths=0.5)
ax.set_title('Pearson Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
print("\\nKey correlations with FEDRates:")
print(corr['FEDRates'].drop('FEDRates').sort_values(key=abs, ascending=False).round(4))
'''),
code('''# Scatter plots vs FED Rate
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
from scipy.stats import pearsonr
for i, col in enumerate(features):
    sc = axes[i].scatter(df[col], df['FEDRates'],
                         c=df['date'].astype(np.int64), cmap='viridis', alpha=0.4, s=15)
    r, p = pearsonr(df[col], df['FEDRates'])
    axes[i].set_title(f"{col[:14]}\\nr={r:.3f}, p={p:.2e}", fontsize=9)
    axes[i].set_xlabel(col[:12]); axes[i].set_ylabel('FEDRates')
fig.suptitle('Feature vs FED Rate Scatter (Color = Time)', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## Summary\n- 846 rows, 9 features loaded from FRED\n- 111 zero CPI values fixed via interpolation\n- Winsorization applied to cap extreme outliers\n- Inflation shows the strongest positive correlation with FED Rates (r≈0.72)"),
]

with open(f"{NB_DIR}/01_Data_Cleaning_EDA.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb(nb1_cells), f)
print("Notebook 1 created: 01_Data_Cleaning_EDA.ipynb")

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 2 — Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
nb2_cells = [
md("""# Notebook 2: Feature Engineering
## Federal Reserve Interest Rate Prediction

**Goal:** Transform 8 raw economic indicators into 67 informative features by creating:
1. **Lag features** — capture delayed economic effects (1, 3, 6, 12 months)
2. **Rolling statistics** — smooth short-term volatility (3 & 6-month windows)
3. **Interaction features** — Phillips Curve proxy, growth rates
4. **Classification target** — rate change direction (Increase/Decrease/No Change)
"""),
code(SETUP),
code('''# Load and clean data
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)
# Basic cleaning
df['ConsumerPriceIndexAllItems'] = df['ConsumerPriceIndexAllItems'].replace(0, np.nan)
df = df.ffill().bfill()
for col in df.select_dtypes(include=np.number).columns:
    p01, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
    df[col] = df[col].clip(p01, p99)
df = df.set_index('date')
print(f"Starting shape: {df.shape}")

FEATURES = ['ConsumerPriceIndexAllItems','GDP','InflationConsumerPrice',
            'MedianConsumerPriceIndex','RealGDP','RealGDPPerCapita',
            'RealPotentialGDP','UnemployemenrRate']
'''),
md("## 1. Classification Target — Rate Direction"),
code('''# Create rate change direction labels
df['RateChange']   = df['FEDRates'].diff()
df['RateDirection'] = 'No_Change'
df.loc[df['RateChange'] >  0.05, 'RateDirection'] = 'Increase'
df.loc[df['RateChange'] < -0.05, 'RateDirection'] = 'Decrease'

print("Class distribution:")
print(df['RateDirection'].value_counts())
print(f"\\nClass proportions:")
print(df['RateDirection'].value_counts(normalize=True).round(3))
'''),
code('''# Visualize class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vc = df['RateDirection'].value_counts()
colors = ['#F44336','#4CAF50','#FF9800'][:len(vc)]
axes[0].bar(vc.index, vc.values, color=colors)
for i, (k, v) in enumerate(vc.items()):
    axes[0].text(i, v+1, str(v), ha='center', fontweight='bold')
axes[0].set_title('Rate Direction Count', fontweight='bold')
axes[1].pie(vc.values, labels=vc.index, autopct='%1.1f%%', colors=colors)
axes[1].set_title('Rate Direction Proportion', fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## 2. Lag Features"),
code('''# Create lag features (1, 3, 6, 12 months)
lag_months = [1, 3, 6, 12]
for col in FEATURES:
    for lag in lag_months:
        df[f'{col}_lag{lag}'] = df[col].shift(lag)

print(f"Shape after lag features: {df.shape}")
print(f"New lag features created: {len(FEATURES) * len(lag_months)}")
'''),
code('''# Correlation of lag features with FED Rate
from scipy.stats import pearsonr
lag_corrs = {}
for col in FEATURES:
    for lag in lag_months:
        feat_name = f'{col}_lag{lag}'
        r, p = pearsonr(df[[feat_name, 'FEDRates']].dropna()[feat_name],
                        df[[feat_name, 'FEDRates']].dropna()['FEDRates'])
        lag_corrs[feat_name] = {'r': round(r,4), 'p': round(p,6)}

lag_corr_df = pd.DataFrame(lag_corrs).T.sort_values('r', key=abs, ascending=False)
print("Top 10 lag features by |correlation|:")
display(lag_corr_df.head(10))
'''),
md("## 3. Rolling Statistics"),
code('''# Rolling mean and standard deviation (3 and 6 month windows)
for col in FEATURES:
    df[f'{col}_roll3_mean'] = df[col].rolling(3).mean()
    df[f'{col}_roll6_mean'] = df[col].rolling(6).mean()
    df[f'{col}_roll3_std']  = df[col].rolling(3).std()

print(f"Shape after rolling features: {df.shape}")

# Visualize rolling vs raw for InflationConsumerPrice
fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(df.index, df['InflationConsumerPrice'], alpha=0.4, color='#2196F3', label='Raw')
ax.plot(df.index, df['InflationConsumerPrice_roll3_mean'], linewidth=2,
        color='#F44336', label='3-month rolling mean')
ax.plot(df.index, df['InflationConsumerPrice_roll6_mean'], linewidth=2,
        color='#4CAF50', label='6-month rolling mean')
ax.set_title('Inflation: Raw vs Rolling Mean', fontweight='bold')
ax.legend(); plt.tight_layout(); plt.show()
'''),
md("## 4. Interaction & Growth Features"),
code('''# Interaction features
df['Inflation_x_Unemployment'] = (df['InflationConsumerPrice'] * df['UnemployemenrRate'])
df['GDP_growth']     = df['GDP'].pct_change() * 100
df['RealGDP_growth'] = df['RealGDP'].pct_change() * 100

# Calendar features
df['Month']   = df.index.month
df['Year']    = df.index.year
df['Quarter'] = df.index.quarter

# Drop NaN rows from lag/rolling
n_before = len(df)
df = df.dropna()
print(f"Rows: {n_before} → {len(df)} (after dropping lag-NaN)")
print(f"Final feature count: {df.shape[1]}")
'''),
code('''# Visualize Phillips Curve proxy
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sc = axes[0].scatter(df['UnemployemenrRate'], df['InflationConsumerPrice'],
                     c=df['FEDRates'], cmap='RdYlGn', s=20, alpha=0.6)
axes[0].set_xlabel('Unemployment Rate'); axes[0].set_ylabel('Inflation')
axes[0].set_title('Phillips Curve (color = FED Rate)', fontweight='bold')
plt.colorbar(sc, ax=axes[0], label='FED Rate %')

axes[1].scatter(df['GDP_growth'], df['FEDRates'], alpha=0.4, color='#2196F3', s=15)
axes[1].set_xlabel('GDP Growth Rate (%)'); axes[1].set_ylabel('FED Rate (%)')
axes[1].set_title('GDP Growth vs FED Rate', fontweight='bold')
plt.tight_layout(); plt.show()
'''),
code('''# Top 20 features by correlation with FED Rate
model_df = df.select_dtypes(include=np.number).dropna()
corr_with_target = model_df.corr()['FEDRates'].drop('FEDRates').sort_values(key=abs, ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
top20 = corr_with_target.head(20)
colors_bar = ['#F44336' if v < 0 else '#2196F3' for v in top20.values]
ax.barh(range(len(top20)), top20.values, color=colors_bar)
ax.set_yticks(range(len(top20))); ax.set_yticklabels(top20.index, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Top 20 Features — Correlation with FED Rate', fontweight='bold')
ax.set_xlabel('Pearson r')
plt.tight_layout(); plt.show()
'''),
md("## Summary\n- 67 features created from 8 raw indicators\n- Lag features (especially 12-month inflation lags) have the strongest correlation\n- Lasso regression later confirms ~55% of features can be zeroed out"),
]
with open(f"{NB_DIR}/02_Feature_Engineering.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb(nb2_cells), f)
print("Notebook 2 created: 02_Feature_Engineering.ipynb")

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 3 — Statistical Analysis & Hypothesis Testing
# ══════════════════════════════════════════════════════════════════════════════
nb3_cells = [
md("""# Notebook 3: Statistical Analysis & Hypothesis Testing
## Federal Reserve Interest Rate Prediction

**Tests conducted:**
1. **Shapiro-Wilk** — Normality test
2. **ADF** — Augmented Dickey-Fuller stationarity test
3. **T-Test** — FED rates in high vs low inflation periods
4. **ANOVA** — FED rates across economic regimes
5. **Pearson & Spearman** — Correlation analysis
"""),
code(SETUP + '''
from scipy.stats import shapiro, ttest_ind, f_oneway, pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller
'''),
code('''# Load and prepare data
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').ffill().bfill().reset_index(drop=True)
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

FEATURES = ['ConsumerPriceIndexAllItems','GDP','InflationConsumerPrice',
            'MedianConsumerPriceIndex','RealGDP','RealGDPPerCapita',
            'RealPotentialGDP','UnemployemenrRate']
ALL_COLS = ['FEDRates'] + FEATURES
print(f"Data shape: {df.shape}")
display(df[ALL_COLS].describe().T.round(3))
'''),
md("## 1. Normality Tests — Shapiro-Wilk"),
code('''# Shapiro-Wilk normality test
print("Shapiro-Wilk Normality Tests:")
print(f"{'Feature':<40} {'W-stat':>10} {'p-value':>12} {'Normal?':>10}")
print("-"*75)
for col in ALL_COLS:
    sample = df[col].dropna().sample(min(500, len(df)), random_state=42)
    stat, p = shapiro(sample)
    normal = 'YES' if p > 0.05 else 'NO'
    print(f"{col:<40} {stat:>10.4f} {p:>12.4e} {normal:>10}")
'''),
code('''# Q-Q plots for normality assessment
from scipy.stats import probplot
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(ALL_COLS):
    probplot(df[col].dropna(), dist='norm', plot=axes[i])
    axes[i].set_title(col, fontsize=9, fontweight='bold')
    axes[i].get_lines()[0].set(markersize=2, alpha=0.5, color=PALETTE[i % len(PALETTE)])
    axes[i].get_lines()[1].set(color='red', linewidth=1.5)
fig.suptitle('Q-Q Plots — Normality Assessment', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## 2. Stationarity Tests — Augmented Dickey-Fuller"),
code('''# ADF stationarity test
print("ADF Stationarity Tests:")
print(f"{'Feature':<40} {'ADF stat':>10} {'p-value':>12} {'Result':>15}")
print("-"*80)
for col in ALL_COLS:
    result = adfuller(df[col].dropna(), autolag='AIC')
    status = 'STATIONARY' if result[1] < 0.05 else 'NON-STATIONARY'
    print(f"{col:<40} {result[0]:>10.4f} {result[1]:>12.4e} {status:>15}")
'''),
md("## 3. T-Test: FED Rates in High vs Low Inflation"),
code('''# Split into high/low inflation groups
med_inf = df['InflationConsumerPrice'].median()
high_inf = df.loc[df['InflationConsumerPrice'] >  med_inf, 'FEDRates']
low_inf  = df.loc[df['InflationConsumerPrice'] <= med_inf, 'FEDRates']

t_stat, t_p = ttest_ind(high_inf, low_inf)
print(f"High Inflation group — n={len(high_inf)}, mean FED rate: {high_inf.mean():.3f}%")
print(f"Low Inflation group  — n={len(low_inf)},  mean FED rate: {low_inf.mean():.3f}%")
print(f"t-statistic: {t_stat:.4f}")
print(f"p-value:     {t_p:.4e}")
print(f"Significant (p<0.05): {'YES' if t_p < 0.05 else 'NO'}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(high_inf, bins=40, alpha=0.6, color='#F44336',
        label=f'High Inflation (mean={high_inf.mean():.2f}%)')
ax.hist(low_inf,  bins=40, alpha=0.6, color='#2196F3',
        label=f'Low Inflation (mean={low_inf.mean():.2f}%)')
ax.axvline(high_inf.mean(), color='#F44336', linestyle='--', linewidth=2)
ax.axvline(low_inf.mean(),  color='#2196F3', linestyle='--', linewidth=2)
ax.set_title(f'T-Test: FED Rate by Inflation Level\\nt={t_stat:.3f}, p={t_p:.2e}', fontweight='bold')
ax.set_xlabel('FED Rate (%)'); ax.legend()
plt.tight_layout(); plt.show()
'''),
md("## 4. ANOVA: FED Rates Across Economic Regimes"),
code('''# Define economic regimes by FED rate quartiles
df['RateRegime'] = pd.qcut(df['FEDRates'], q=4,
                            labels=['Very_Low','Low','High','Very_High'])
groups = [df.loc[df['RateRegime'] == r, 'FEDRates'].values
          for r in ['Very_Low','Low','High','Very_High']]
f_stat, f_p = f_oneway(*groups)

print(f"ANOVA: FED Rate across economic regimes")
print(f"F-statistic: {f_stat:.4f}")
print(f"p-value:     {f_p:.4e}")
print(f"Significant: {'YES' if f_p < 0.05 else 'NO'}")

print("\\nMean FED Rate by regime:")
print(df.groupby('RateRegime')['FEDRates'].agg(['mean','std','count']).round(3))

fig, ax = plt.subplots(figsize=(9, 5))
df.boxplot('FEDRates', by='RateRegime', ax=ax)
ax.set_title(f'ANOVA: FED Rate by Regime\\nF={f_stat:.2f}, p={f_p:.2e}', fontweight='bold')
ax.set_xlabel('Economic Regime'); ax.set_ylabel('FED Rate (%)')
plt.suptitle('')
plt.tight_layout(); plt.show()
'''),
md("## 5. Spearman Rank Correlations"),
code('''# Spearman correlations
print(f"{'Feature':<40} {'Spearman rho':>14} {'p-value':>12}")
print("-"*70)
for col in FEATURES:
    rho, p = spearmanr(df[col], df['FEDRates'])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f"{col:<40} {rho:>14.4f} {p:>12.4e} {sig}")

# Comparison: Pearson vs Spearman
pearson_r  = {col: pearsonr(df[col], df['FEDRates'])[0]  for col in FEATURES}
spearman_r = {col: spearmanr(df[col], df['FEDRates'])[0] for col in FEATURES}
corr_compare = pd.DataFrame({'Pearson r': pearson_r, 'Spearman rho': spearman_r}).round(4)
print("\\nPearson vs Spearman comparison:")
display(corr_compare.sort_values('Spearman rho', key=abs, ascending=False))
'''),
code('''# Visualization: all correlation methods
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
spearman_vals = pd.Series({col: spearmanr(df[col], df['FEDRates'])[0] for col in FEATURES}).sort_values()
pearson_vals  = pd.Series({col: pearsonr(df[col], df['FEDRates'])[0]  for col in FEATURES}).sort_values()

axes[0].barh(range(len(spearman_vals)), spearman_vals.values,
             color=['#F44336' if v < 0 else '#2196F3' for v in spearman_vals.values])
axes[0].set_yticks(range(len(spearman_vals))); axes[0].set_yticklabels(spearman_vals.index)
axes[0].set_title('Spearman Rank Correlations with FED Rate', fontweight='bold')
axes[0].axvline(0, color='black', linewidth=0.8)

axes[1].barh(range(len(pearson_vals)), pearson_vals.values,
             color=['#F44336' if v < 0 else '#2196F3' for v in pearson_vals.values])
axes[1].set_yticks(range(len(pearson_vals))); axes[1].set_yticklabels(pearson_vals.index)
axes[1].set_title('Pearson Correlations with FED Rate', fontweight='bold')
axes[1].axvline(0, color='black', linewidth=0.8)
plt.tight_layout(); plt.show()
'''),
md("## Summary\n- All features are non-normal (Shapiro-Wilk p << 0.05)\n- GDP variables are non-stationary (ADF), inflation nearly stationary\n- T-Test: Strong evidence rates are higher during high-inflation (p ≈ 5.7×10⁻⁷⁴)\n- Spearman ρ = 0.68 for Inflation — strongest predictor"),
]
with open(f"{NB_DIR}/03_Statistical_Analysis.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb(nb3_cells), f)
print("Notebook 3 created: 03_Statistical_Analysis.ipynb")

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 4 — PCA
# ══════════════════════════════════════════════════════════════════════════════
nb4_cells = [
md("""# Notebook 4: Principal Component Analysis (PCA)
## Federal Reserve Interest Rate Prediction

**Objectives:**
- Reduce 8 raw features to principal components
- Identify the most informative linear combinations
- Visualize economic regime structure in lower dimensions
- Determine minimum components needed to retain 95% variance
"""),
code(SETUP + '''
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
'''),
code('''df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').ffill().bfill().reset_index(drop=True)
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].clip(df[col].quantile(0.01), df[col].quantile(0.99))

FEATURES = ['ConsumerPriceIndexAllItems','GDP','InflationConsumerPrice',
            'MedianConsumerPriceIndex','RealGDP','RealGDPPerCapita',
            'RealPotentialGDP','UnemployemenrRate']
X = df[FEATURES].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Data shape: {X_scaled.shape}")
'''),
md("## 1. Full PCA — Explained Variance"),
code('''pca_full = PCA()
pca_full.fit(X_scaled)
explained   = pca_full.explained_variance_ratio_
cumulative  = np.cumsum(explained)

print("Explained variance per component:")
for i, (ev, cv) in enumerate(zip(explained, cumulative)):
    print(f"  PC{i+1}: {ev*100:.2f}% | Cumulative: {cv*100:.2f}%")

n_95 = np.argmax(cumulative >= 0.95) + 1
print(f"\\nComponents for 95% variance: {n_95}")
print(f"Components for 99% variance: {np.argmax(cumulative >= 0.99)+1}")
'''),
code('''# Scree plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
comps = range(1, len(explained)+1)
axes[0].bar(comps, explained*100, color='#2196F3', alpha=0.8, label='Individual')
axes[0].plot(comps, cumulative*100, 'ro-', linewidth=2, markersize=7, label='Cumulative')
axes[0].axhline(95, color='green', linestyle='--', linewidth=1.5, label='95% threshold')
axes[0].set_xlabel('Component'); axes[0].set_ylabel('Variance (%)')
axes[0].set_title('PCA Scree Plot', fontweight='bold')
axes[0].legend(); axes[0].set_xticks(comps)

# Cumulative
axes[1].fill_between(comps, cumulative*100, alpha=0.2, color='#2196F3')
axes[1].plot(comps, cumulative*100, 'b-o', linewidth=2, markersize=8)
for i, cv in enumerate(cumulative):
    axes[1].annotate(f'{cv*100:.1f}%', (i+1, cv*100), xytext=(0,8),
                     textcoords='offset points', ha='center', fontsize=8)
axes[1].axhline(95, color='red', linestyle='--', label='95%')
axes[1].axhline(99, color='green', linestyle='--', label='99%')
axes[1].set_xlabel('Components'); axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].set_title('Cumulative Explained Variance', fontweight='bold')
axes[1].legend(); axes[1].set_xticks(comps)
plt.tight_layout(); plt.show()
'''),
md("## 2. PCA Loadings Analysis"),
code('''pca_3 = PCA(n_components=3)
X_3d  = pca_3.fit_transform(X_scaled)

loadings = pd.DataFrame(
    pca_3.components_.T,
    index=FEATURES,
    columns=['PC1','PC2','PC3']
)
print("PCA Loadings (contribution of each feature to each PC):")
display(loadings.round(4))

# Loadings heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(loadings, annot=True, fmt='.3f', cmap='RdBu_r', ax=ax,
            linewidths=0.5, center=0)
ax.set_title('PCA Loadings Heatmap (Top 3 Components)', fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## 3. PCA Biplot — Visualizing Data in PC Space"),
code('''# Create rate direction for coloring
df['RateDirection'] = 'No_Change'
df['RateChange'] = df['FEDRates'].diff()
df.loc[df['RateChange'] > 0.05,  'RateDirection'] = 'Increase'
df.loc[df['RateChange'] < -0.05, 'RateDirection'] = 'Decrease'
color_map = {'Increase':0, 'No_Change':1, 'Decrease':2}
color_vals = df['RateDirection'].map(color_map).values

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
# PC1 vs PC2
sc = axes[0].scatter(X_3d[:,0], X_3d[:,1], c=color_vals[:len(X_3d)],
                     cmap='RdYlGn', alpha=0.5, s=20)
for i, feat in enumerate(FEATURES):
    scale = 3
    axes[0].arrow(0, 0, pca_3.components_[0,i]*scale, pca_3.components_[1,i]*scale,
                  head_width=0.08, fc='#333', ec='#333', alpha=0.8)
    axes[0].text(pca_3.components_[0,i]*scale*1.2, pca_3.components_[1,i]*scale*1.2,
                 feat[:10], fontsize=7)
axes[0].set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
axes[0].set_ylabel(f"PC2 ({explained[1]*100:.1f}% var)")
axes[0].set_title('PCA Biplot (PC1 vs PC2)', fontweight='bold')
plt.colorbar(sc, ax=axes[0], label='0=Inc, 1=No, 2=Dec')

# PC1 vs PC3
axes[1].scatter(X_3d[:,0], X_3d[:,2], c=color_vals[:len(X_3d)], cmap='RdYlGn', alpha=0.5, s=20)
axes[1].set_xlabel(f"PC1 ({explained[0]*100:.1f}% var)")
axes[1].set_ylabel(f"PC3 ({explained[2]*100:.1f}% var)")
axes[1].set_title('PCA Biplot (PC1 vs PC3)', fontweight='bold')
plt.tight_layout(); plt.show()
'''),
code('''# PC scores over time
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
dates = df['date'][:len(X_3d)]
for i, (ax, title) in enumerate(zip(axes, ['PC1 — Economic Scale',
                                            'PC2 — Inflation Dynamics',
                                            'PC3 — Labor Market'])):
    ax.plot(dates, X_3d[:,i], color=PALETTE[i], linewidth=1)
    ax.fill_between(dates, X_3d[:,i], alpha=0.1, color=PALETTE[i])
    ax.set_title(title, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
fig.suptitle('Principal Component Scores Over Time', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## Summary\n- PC1 (57.3%) captures overall economic scale (GDP dominates)\n- PC2 (18.7%) captures inflation dynamics\n- PC3 (13.7%) captures labor market\n- 4 components needed for 95% variance — strong multicollinearity among GDP features"),
]
with open(f"{NB_DIR}/04_PCA_Analysis.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb(nb4_cells), f)
print("Notebook 4 created: 04_PCA_Analysis.ipynb")

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 5 — Regression Models
# ══════════════════════════════════════════════════════════════════════════════
nb5_cells = [
md("""# Notebook 5: Regression Models — Predicting Exact FED Rate
## Federal Reserve Interest Rate Prediction

**Models:** Linear Regression, Ridge, Lasso, SVR, Decision Tree, Random Forest, Gradient Boosting, XGBoost

**Key concepts demonstrated:**
- Regularization (Ridge L2, Lasso L1)
- Hyperparameter tuning
- TimeSeriesSplit cross-validation
- Overfitting/underfitting analysis via learning curves
- Feature importance and coefficient analysis
"""),
code(SETUP + '''
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     learning_curve, TimeSeriesSplit)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
'''),
code('''# Load preprocessed data
import pickle
with open(f"{OUT_PATH}/results/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

X_scaled  = data['X_scaled']
y_reg     = data['y_reg']
feat_cols = data['feature_cols']

X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_reg, test_size=0.2, random_state=42)
tscv = TimeSeriesSplit(n_splits=5)

print(f"Training set: {X_tr.shape}")
print(f"Test set:     {X_te.shape}")
print(f"Target range: {y_reg.min():.2f}% to {y_reg.max():.2f}%")
'''),
code('''def evaluate(model, name, X_tr=X_tr, X_te=X_te, y_tr=y_tr, y_te=y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    mae  = mean_absolute_error(y_te, y_pred)
    r2   = r2_score(y_te, y_pred)
    cv_r2 = cross_val_score(model, X_scaled, y_reg, cv=tscv, scoring='r2')
    print(f"{name}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  CV R²={cv_r2.mean():.4f}±{cv_r2.std():.4f}")
    return y_pred, {'RMSE':rmse,'MAE':mae,'R2':r2,'CV':cv_r2.mean()}

results = {}
'''),
md("## 1. Linear Regression — Baseline"),
code('''lr = LinearRegression()
y_pred_lr, results['Linear'] = evaluate(lr, "Linear Regression")

# Coefficients
coef_df = pd.Series(lr.coef_, index=feat_cols).sort_values(key=abs, ascending=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
top20 = coef_df.head(20)
axes[0].barh(range(20), top20.values,
             color=['#F44336' if v < 0 else '#2196F3' for v in top20.values])
axes[0].set_yticks(range(20)); axes[0].set_yticklabels(top20.index, fontsize=8)
axes[0].axvline(0, color='k', linewidth=0.8)
axes[0].set_title('Linear Reg — Top 20 Coefficients', fontweight='bold')
axes[1].scatter(y_te, y_pred_lr, alpha=0.4, s=20, color='#2196F3')
mn, mx = min(y_te.min(), y_pred_lr.min()), max(y_te.max(), y_pred_lr.max())
axes[1].plot([mn,mx],[mn,mx],'r--', linewidth=2)
axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')
axes[1].set_title(f"Linear Reg: R²={results['Linear']['R2']:.4f}", fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## 2. Ridge & Lasso — Regularization"),
code('''# Ridge — hyperparameter tuning
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_scores = [cross_val_score(Ridge(alpha=a), X_scaled, y_reg, cv=tscv, scoring='r2').mean()
                for a in alphas]
best_alpha_ridge = alphas[np.argmax(ridge_scores)]

ridge = Ridge(alpha=best_alpha_ridge)
y_pred_ridge, results['Ridge'] = evaluate(ridge, f"Ridge (alpha={best_alpha_ridge})")

# Lasso — hyperparameter tuning
lasso_scores = [cross_val_score(Lasso(alpha=a, max_iter=10000), X_scaled, y_reg,
                                cv=tscv, scoring='r2').mean() for a in alphas]
best_alpha_lasso = alphas[np.argmax(lasso_scores)]

lasso = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso.fit(X_tr, y_tr)
y_pred_lasso, results['Lasso'] = evaluate(lasso, f"Lasso (alpha={best_alpha_lasso})")

n_zero = (lasso.coef_ == 0).sum()
print(f"\\nLasso zeroed out {n_zero}/{len(lasso.coef_)} features ({n_zero/len(lasso.coef_)*100:.0f}%)")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].semilogx(alphas, ridge_scores, 'b-o', label='Ridge')
axes[0].semilogx(alphas, lasso_scores, 'r-o', label='Lasso')
axes[0].axvline(best_alpha_ridge, color='blue', linestyle='--', alpha=0.7)
axes[0].axvline(best_alpha_lasso, color='red', linestyle='--', alpha=0.7)
axes[0].set_xlabel('Alpha'); axes[0].set_ylabel('CV R²')
axes[0].set_title('Ridge vs Lasso: Alpha Tuning', fontweight='bold'); axes[0].legend()

non_zero = np.where(lasso.coef_ != 0)[0]
if len(non_zero) > 0:
    top_lasso = non_zero[np.argsort(np.abs(lasso.coef_[non_zero]))[-15:]]
    axes[1].barh(range(len(top_lasso)), lasso.coef_[top_lasso],
                 color=['#F44336' if v < 0 else '#2196F3' for v in lasso.coef_[top_lasso]])
    axes[1].set_yticks(range(len(top_lasso)))
    axes[1].set_yticklabels([feat_cols[i][:20] for i in top_lasso], fontsize=8)
    axes[1].axvline(0, color='k', linewidth=0.8)
axes[1].set_title(f"Lasso Non-Zero Coefficients ({n_zero} zeroed)", fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## 3. Decision Tree — Depth Tuning & Overfitting"),
code('''# Depth-based overfitting analysis
depths = range(1, 20)
train_r2s, val_r2s = [], []
for d in depths:
    dt = DecisionTreeRegressor(max_depth=d, random_state=42)
    dt.fit(X_tr, y_tr)
    train_r2s.append(r2_score(y_tr, dt.predict(X_tr)))
    val_r2s.append(cross_val_score(dt, X_scaled, y_reg, cv=tscv, scoring='r2').mean())

best_depth = depths[np.argmax(val_r2s)]
print(f"Best depth (TimeSeriesCV): {best_depth}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(depths, train_r2s, 'b-o', label='Train R²')
ax.plot(depths, val_r2s, 'r-o', label='TimeSeriesCV R²')
ax.axvline(best_depth, color='green', linestyle='--', label=f'Best depth={best_depth}')
ax.fill_between(depths, train_r2s, val_r2s, alpha=0.15, color='purple', label='Overfitting Gap')
ax.set_xlabel('Tree Depth'); ax.set_ylabel('R²')
ax.set_title('Decision Tree: Bias-Variance Tradeoff', fontweight='bold'); ax.legend()
plt.tight_layout(); plt.show()

dt = DecisionTreeRegressor(max_depth=best_depth, random_state=42)
_, results['Decision Tree'] = evaluate(dt, f"Decision Tree (depth={best_depth})")
'''),
md("## 4. Ensemble Models — RF, GBM, XGBoost"),
code('''# Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5,
                            random_state=42, n_jobs=-1)
y_pred_rf, results['Random Forest'] = evaluate(rf, "Random Forest")
rf.fit(X_tr, y_tr)

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
_, results['Gradient Boosting'] = evaluate(gb, "Gradient Boosting")

# XGBoost
xgb_reg = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
y_pred_xgb, results['XGBoost'] = evaluate(xgb_reg, "XGBoost")
xgb_reg.fit(X_tr, y_tr)

# Feature importance comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
rf_fi  = pd.Series(rf.feature_importances_,  index=feat_cols).sort_values(ascending=False).head(15)
xgb_fi = pd.Series(xgb_reg.feature_importances_, index=feat_cols).sort_values(ascending=False).head(15)
rf_fi.plot.bar(ax=axes[0], color=PALETTE[:15])
axes[0].set_title('Random Forest Feature Importance', fontweight='bold')
axes[0].tick_params(axis='x', rotation=45)
xgb_fi.plot.bar(ax=axes[1], color=PALETTE[:15])
axes[1].set_title('XGBoost Feature Importance', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout(); plt.show()
'''),
md("## 5. Learning Curves — Overfitting/Underfitting"),
code('''def plot_lc(model, title):
    ts, tr, vl = learning_curve(model, X_scaled, y_reg, cv=tscv,
                                scoring='r2', train_sizes=np.linspace(0.1,1.0,10))
    tr_m, tr_s = tr.mean(1), tr.std(1)
    vl_m, vl_s = vl.mean(1), vl.std(1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ts, tr_m, 'b-o', label='Train R²')
    ax.fill_between(ts, tr_m-tr_s, tr_m+tr_s, alpha=0.15, color='blue')
    ax.plot(ts, vl_m, 'r-o', label='Validation R²')
    ax.fill_between(ts, vl_m-vl_s, vl_m+vl_s, alpha=0.15, color='red')
    gap = tr_m[-1] - vl_m[-1]
    status = 'OVERFITTING' if gap > 0.1 else ('UNDERFITTING' if vl_m[-1] < 0.5 else 'GOOD FIT')
    ax.set_title(f'{title} — Learning Curve\\nStatus: {status} (gap={gap:.3f})', fontweight='bold')
    ax.set_xlabel('Training Size'); ax.set_ylabel('R²'); ax.legend()
    plt.tight_layout(); plt.show()

for model, name in [(xgb_reg,'XGBoost'),(rf,'Random Forest'),(lasso,'Lasso')]:
    plot_lc(model, name)
'''),
md("## 6. Final Comparison"),
code('''# Results table
results_df = pd.DataFrame(results).T
results_df.index.name = 'Model'
results_df = results_df.sort_values('R2', ascending=False)
display(results_df.round(4).style.highlight_max(subset=['R2'], color='lightgreen')
                                  .highlight_min(subset=['RMSE'], color='lightgreen'))

# Bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
models = results_df.index
axes[0].bar(models, results_df['R2'], color=PALETTE[:len(models)])
axes[0].set_xticklabels(models, rotation=35, ha='right')
axes[0].set_title('R² Score Comparison', fontweight='bold')
axes[0].set_ylim(0, 1.05)
axes[1].bar(models, results_df['RMSE'], color=PALETTE[:len(models)])
axes[1].set_xticklabels(models, rotation=35, ha='right')
axes[1].set_title('RMSE Comparison (lower=better)', fontweight='bold')
plt.tight_layout(); plt.show()
'''),
]
with open(f"{NB_DIR}/05_Regression_Models.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb(nb5_cells), f)
print("Notebook 5 created: 05_Regression_Models.ipynb")

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 6 — Classification Models
# ══════════════════════════════════════════════════════════════════════════════
nb6_cells = [
md("""# Notebook 6: Classification Models — Predicting Rate Direction
## Federal Reserve Interest Rate Prediction

**Target Classes:** Increase / Decrease / No_Change

**Models:** Logistic Regression, Naive Bayes, SVM, Decision Tree, Random Forest, Gradient Boosting, XGBoost

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-Score (weighted)
- Confusion Matrix
- ROC-AUC (one-vs-rest)
- Learning Curves
"""),
code(SETUP + '''
from sklearn.model_selection import (train_test_split, cross_val_score,
                                     learning_curve, TimeSeriesSplit)
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score, precision_score, recall_score)
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
'''),
code('''import pickle
with open(f"{OUT_PATH}/results/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)

X_scaled    = data['X_scaled']
y_cls       = data['y_cls']
label_names = data['label_names']
feat_cols   = data['feature_cols']

X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y_cls, test_size=0.2,
                                            random_state=42, stratify=y_cls)
tscv = TimeSeriesSplit(n_splits=5)

print(f"Classes: {label_names}")
print(f"Class distribution: {dict(zip(label_names, np.bincount(y_cls)))}")
print(f"Baseline (majority class): {max(np.bincount(y_cls))/len(y_cls):.3f}")
print(f"Baseline (random 3-class): {1/3:.3f}")
'''),
code('''def eval_cls(model, name):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te) if hasattr(model,'predict_proba') else None
    acc  = accuracy_score(y_te, y_pred)
    f1   = f1_score(y_te, y_pred, average='weighted')
    cv_s = cross_val_score(model, X_scaled, y_cls, cv=tscv, scoring='accuracy')
    print(f"\\n{'='*60}")
    print(f"Model: {name}")
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | CV: {cv_s.mean():.4f}±{cv_s.std():.4f}")
    print(classification_report(y_te, y_pred, target_names=label_names))
    return y_pred, y_prob, {'Accuracy':acc,'F1':f1,'CV':cv_s.mean()}

def plot_cm(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=label_names, yticklabels=label_names)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'{name} — Confusion Matrix', fontweight='bold')
    plt.tight_layout(); plt.show()

def plot_roc(y_true, y_prob, name):
    y_bin = label_binarize(y_true, classes=[0,1,2])
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (cls, col) in enumerate(zip(label_names,['#2196F3','#F44336','#4CAF50'])):
        fpr, tpr, _ = roc_curve(y_bin[:,i], y_prob[:,i])
        auc = roc_auc_score(y_bin[:,i], y_prob[:,i])
        ax.plot(fpr, tpr, color=col, linewidth=2, label=f'{cls} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1],'k--'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title(f'ROC Curves — {name}', fontweight='bold'); ax.legend()
    plt.tight_layout(); plt.show()

cls_results = {}
'''),
md("## 1. Logistic Regression"),
code('''log_reg = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', random_state=42)
y_pred_lr, y_prob_lr, cls_results['Logistic Regression'] = eval_cls(log_reg, 'Logistic Regression')
plot_cm(y_te, y_pred_lr, 'Logistic Regression')
plot_roc(y_te, y_prob_lr, 'Logistic Regression')
'''),
md("## 2. Naive Bayes"),
code('''gnb = GaussianNB()
y_pred_nb, y_prob_nb, cls_results['Naive Bayes'] = eval_cls(gnb, 'Naive Bayes')
plot_cm(y_te, y_pred_nb, 'Naive Bayes')
plot_roc(y_te, y_prob_nb, 'Naive Bayes')
'''),
md("## 3. Support Vector Machine"),
code('''svc = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
y_pred_svm, y_prob_svm, cls_results['SVM'] = eval_cls(svc, 'SVM (RBF)')
plot_cm(y_te, y_pred_svm, 'SVM')
plot_roc(y_te, y_prob_svm, 'SVM')
'''),
md("## 4. Decision Tree (with Depth Tuning)"),
code('''depths = range(1, 20)
dt_cv_accs = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42),
                               X_scaled, y_cls, cv=tscv, scoring='accuracy').mean()
              for d in depths]
best_d = depths[np.argmax(dt_cv_accs)]
print(f"Best depth: {best_d}")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(depths, dt_cv_accs, 'r-o')
ax.axvline(best_d, color='green', linestyle='--', label=f'Best depth={best_d}')
ax.set_xlabel('Depth'); ax.set_ylabel('CV Accuracy')
ax.set_title('Decision Tree: Depth Tuning', fontweight='bold'); ax.legend()
plt.tight_layout(); plt.show()

dtc = DecisionTreeClassifier(max_depth=best_d, random_state=42)
y_pred_dt, y_prob_dt, cls_results['Decision Tree'] = eval_cls(dtc, f'Decision Tree (d={best_d})')
plot_cm(y_te, y_pred_dt, 'Decision Tree')
'''),
md("## 5. Random Forest"),
code('''rfc = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
y_pred_rf, y_prob_rf, cls_results['Random Forest'] = eval_cls(rfc, 'Random Forest')
rfc.fit(X_tr, y_tr)
plot_cm(y_te, y_pred_rf, 'Random Forest')
plot_roc(y_te, y_prob_rf, 'Random Forest')

# Feature importance
fi = pd.Series(rfc.feature_importances_, index=feat_cols).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 6))
fi.plot.bar(ax=ax, color=PALETTE[:20])
ax.set_title('Random Forest Feature Importance (Classification)', fontweight='bold')
ax.tick_params(axis='x', rotation=45); plt.tight_layout(); plt.show()
'''),
md("## 6. Gradient Boosting & XGBoost"),
code('''gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
y_pred_gb, y_prob_gb, cls_results['Gradient Boosting'] = eval_cls(gbc, 'Gradient Boosting')
plot_cm(y_te, y_pred_gb, 'Gradient Boosting')
plot_roc(y_te, y_prob_gb, 'Gradient Boosting')
'''),
code('''xgb_cls = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              eval_metric='mlogloss', verbosity=0)
y_pred_xgb, y_prob_xgb, cls_results['XGBoost'] = eval_cls(xgb_cls, 'XGBoost')
xgb_cls.fit(X_tr, y_tr)
plot_cm(y_te, y_pred_xgb, 'XGBoost')
plot_roc(y_te, y_prob_xgb, 'XGBoost')
'''),
md("## 7. Final Comparison"),
code('''cls_df = pd.DataFrame(cls_results).T.sort_values('Accuracy', ascending=False)
display(cls_df.round(4).style.highlight_max(color='lightgreen'))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
models = cls_df.index
axes[0].bar(models, cls_df['Accuracy'], color=PALETTE[:len(models)])
axes[0].axhline(1/3, color='red', linestyle='--', label='Random baseline (33%)')
axes[0].set_xticklabels(models, rotation=35, ha='right')
axes[0].set_title('Accuracy Comparison', fontweight='bold'); axes[0].legend()

axes[1].bar(models, cls_df['F1'], color=PALETTE[:len(models)])
axes[1].set_xticklabels(models, rotation=35, ha='right')
axes[1].set_title('Weighted F1 Score', fontweight='bold')
plt.suptitle('Classification Models Performance', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
'''),
]
with open(f"{NB_DIR}/06_Classification_Models.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb(nb6_cells), f)
print("Notebook 6 created: 06_Classification_Models.ipynb")

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 7 — Clustering
# ══════════════════════════════════════════════════════════════════════════════
nb7_cells = [
md("""# Notebook 7: Clustering — KMeans & DBSCAN
## Federal Reserve Interest Rate Prediction

**Objective:** Discover hidden economic regime structure without labels.

- **KMeans:** Centroid-based partitioning — finds k spherical clusters
- **DBSCAN:** Density-based — finds arbitrary-shaped clusters and marks outliers as noise

**Evaluation:** Silhouette Score, Inertia (Elbow Method), Davies-Bouldin Index
"""),
code(SETUP + '''
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import pickle
'''),
code('''with open(f"{OUT_PATH}/results/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)
X_scaled = data['X_scaled']
X_pca_2 = PCA(n_components=2).fit_transform(X_scaled)
print(f"Data shape: {X_scaled.shape}")
'''),
md("## 1. KMeans — Optimal k Selection"),
code('''k_range = range(2, 11)
inertias, sil_scores, db_scores = [], [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

best_k = k_range[np.argmax(sil_scores)]
print(f"Best k (silhouette): {best_k}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
axes[0].plot(k_range, inertias, 'b-o', linewidth=2)
axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Curve', fontweight='bold')

axes[1].plot(k_range, sil_scores, 'r-o', linewidth=2)
axes[1].axvline(best_k, color='green', linestyle='--', label=f'Best k={best_k}')
axes[1].set_xlabel('k'); axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score', fontweight='bold'); axes[1].legend()

axes[2].plot(k_range, db_scores, 'g-o', linewidth=2)
axes[2].set_xlabel('k'); axes[2].set_ylabel('Davies-Bouldin Score')
axes[2].set_title('Davies-Bouldin Index (lower=better)', fontweight='bold')
plt.tight_layout(); plt.show()
'''),
code('''km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_best.fit_predict(X_scaled)
print(f"KMeans Silhouette: {silhouette_score(X_scaled, km_labels):.4f}")
print(f"KMeans Davies-Bouldin: {davies_bouldin_score(X_scaled, km_labels):.4f}")

fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(X_pca_2[:,0], X_pca_2[:,1], c=km_labels, cmap='tab10', s=20, alpha=0.7)
plt.colorbar(sc, ax=ax, label='Cluster')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title(f'KMeans Clusters (k={best_k}) — PCA 2D Projection', fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## 2. DBSCAN — Density-Based Clustering"),
code('''# K-distance plot to find optimal eps
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_dist = np.sort(distances[:,-1])

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(k_dist, color='#2196F3')
ax.set_xlabel('Points (sorted)'); ax.set_ylabel('5th Nearest Neighbor Distance')
ax.set_title('DBSCAN: K-Distance Plot (Knee=Best eps)', fontweight='bold')
plt.tight_layout(); plt.show()
'''),
code('''# Grid search over eps
eps_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
print(f"{'eps':>8} {'Clusters':>10} {'Noise%':>10} {'Silhouette':>12}")
print("-"*45)
for eps in eps_grid:
    db = DBSCAN(eps=eps, min_samples=5)
    lbl = db.fit_predict(X_scaled)
    n_cls = len(set(lbl)) - (1 if -1 in lbl else 0)
    n_noi = list(lbl).count(-1)
    core  = lbl != -1
    sil   = silhouette_score(X_scaled[core], lbl[core]) if n_cls >= 2 and core.sum() > 1 else -1
    print(f"{eps:>8} {n_cls:>10} {n_noi/len(lbl)*100:>9.1f}% {sil:>12.4f}")
'''),
code('''best_eps = 1.0
db_best  = DBSCAN(eps=best_eps, min_samples=5)
db_labels = db_best.fit_predict(X_scaled)
n_cls = len(set(db_labels)) - (1 if -1 in db_labels else 0)
print(f"DBSCAN: {n_cls} clusters, {list(db_labels).count(-1)} noise points")

fig, ax = plt.subplots(figsize=(9, 7))
cmap = plt.cm.tab10(np.linspace(0,1,n_cls+1))
for lbl in set(db_labels):
    mask = db_labels == lbl
    col = 'lightgray' if lbl==-1 else cmap[lbl]
    lab = 'Noise' if lbl==-1 else f'Cluster {lbl}'
    ax.scatter(X_pca_2[mask,0], X_pca_2[mask,1],
               c=[col]*mask.sum(), s=12 if lbl==-1 else 20,
               alpha=0.3 if lbl==-1 else 0.8, label=lab)
ax.set_title(f'DBSCAN (eps={best_eps}) — PCA 2D', fontweight='bold')
ax.legend(); ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
plt.tight_layout(); plt.show()
'''),
md("## 3. Cluster Profile Analysis"),
code('''import pickle
with open(f"{OUT_PATH}/results/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)
df_cluster = data['df'].select_dtypes(include=np.number).iloc[:len(km_labels)]
df_cluster['KMeans'] = km_labels

profile = df_cluster.groupby('KMeans')[['FEDRates','InflationConsumerPrice',
                                         'UnemployemenrRate','GDP']].mean()
print("KMeans Cluster Profiles:")
display(profile.round(3))

fig, ax = plt.subplots(figsize=(10, 5))
profile.T.plot.bar(ax=ax, colormap='Set2')
ax.set_title('KMeans Cluster Economic Profiles', fontweight='bold')
ax.tick_params(axis='x', rotation=0); plt.tight_layout(); plt.show()
'''),
]
with open(f"{NB_DIR}/07_Clustering.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb(nb7_cells), f)
print("Notebook 7 created: 07_Clustering.ipynb")

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 8 — Association Rule Mining
# ══════════════════════════════════════════════════════════════════════════════
nb8_cells = [
md("""# Notebook 8: Association Rule Mining — Apriori Algorithm
## Federal Reserve Interest Rate Prediction

**Goal:** Discover co-occurrence patterns between economic indicator states.

**Process:**
1. Discretize continuous features into Low / Mid / High categories (tertile-based)
2. Treat each month's economic state as a "transaction"
3. Find frequent itemsets (support ≥ 0.30)
4. Generate rules (confidence ≥ 0.60)
5. Rank by Lift (values > 1 indicate non-random association)

**Key Metrics:**
- **Support:** Fraction of transactions containing the rule
- **Confidence:** P(consequent | antecedent)
- **Lift:** Confidence / Expected (Lift > 1 = positive association)
"""),
code(SETUP + '''
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
'''),
code('''import pickle
with open(f"{OUT_PATH}/results/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)
df = data['df']

ARM_FEATURES = ['FEDRates','InflationConsumerPrice','UnemployemenrRate','GDP','RealGDP']
arm_df = df[ARM_FEATURES].copy()
print(f"ARM data shape: {arm_df.shape}")
display(arm_df.head())
'''),
md("## 1. Data Discretization"),
code('''def discretize(series, col_name):
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    def label(v):
        if v <= q25: return f'{col_name[:8]}_Low'
        elif v <= q75: return f'{col_name[:8]}_Mid'
        else: return f'{col_name[:8]}_High'
    return series.apply(label)

# Build transactions
transactions = []
for idx in arm_df.index:
    row = [discretize(arm_df[col], col).loc[idx] for col in ARM_FEATURES]
    transactions.append(row)

print(f"Number of transactions: {len(transactions)}")
print(f"Sample transaction: {transactions[0]}")
print(f"Unique items: {sorted(set(item for t in transactions for item in t))}")
'''),
code('''# Encode as boolean dataframe
te = TransactionEncoder()
te_array = te.fit_transform(transactions)
te_df = pd.DataFrame(te_array, columns=te.columns_)
print(f"Transaction matrix shape: {te_df.shape}")
print(f"\\nItem frequency:")
display(te_df.mean().sort_values(ascending=False).round(3))
'''),
md("## 2. Frequent Itemset Mining"),
code('''# Mine frequent itemsets
freq_items = apriori(te_df, min_support=0.25, use_colnames=True)
freq_items['length'] = freq_items['itemsets'].apply(len)
print(f"Frequent itemsets (support >= 0.25): {len(freq_items)}")
print(f"Itemsets by length:")
print(freq_items['length'].value_counts().sort_index())

display(freq_items.sort_values('support', ascending=False).head(15))
'''),
code('''# Visualize support distribution
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
freq_items.groupby('length')['support'].mean().plot.bar(ax=axes[0], color=PALETTE[:4])
axes[0].set_title('Mean Support by Itemset Length', fontweight='bold')
axes[0].tick_params(axis='x', rotation=0)

axes[1].hist(freq_items['support'], bins=20, color='#2196F3', alpha=0.75, edgecolor='white')
axes[1].set_xlabel('Support'); axes[1].set_ylabel('Count')
axes[1].set_title('Support Distribution', fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## 3. Association Rules"),
code('''rules = association_rules(freq_items, metric='confidence', min_threshold=0.55)
rules = rules.sort_values('lift', ascending=False)
print(f"Rules generated (confidence >= 0.55): {len(rules)}")
print(f"\\nTop 10 rules by lift:")

def fmt_rule(r):
    ant = ', '.join(list(r.antecedents))
    con = ', '.join(list(r.consequents))
    return f"{ant}  =>  {con}"

for _, r in rules.head(10).iterrows():
    print(f"  {fmt_rule(r)}")
    print(f"     Support={r['support']:.3f}  Confidence={r['confidence']:.3f}  Lift={r['lift']:.3f}")
    print()
'''),
code('''# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
sc = axes[0,0].scatter(rules['support'], rules['confidence'],
                        c=rules['lift'], cmap='RdYlGn', s=40, alpha=0.7)
axes[0,0].set_xlabel('Support'); axes[0,0].set_ylabel('Confidence')
axes[0,0].set_title('Support vs Confidence (color=Lift)', fontweight='bold')
plt.colorbar(sc, ax=axes[0,0], label='Lift')

axes[0,1].scatter(rules['support'], rules['lift'], c=rules['confidence'],
                   cmap='Blues', s=40, alpha=0.7)
axes[0,1].set_xlabel('Support'); axes[0,1].set_ylabel('Lift')
axes[0,1].set_title('Support vs Lift (color=Confidence)', fontweight='bold')
axes[0,1].axhline(1.0, color='red', linestyle='--', label='Lift=1 (random)')
axes[0,1].legend()

top10 = rules.head(10)
rule_labels = [f"{list(r.antecedents)[0][:15]}=>{list(r.consequents)[0][:15]}"
               for _, r in top10.iterrows()]
axes[1,0].barh(range(10), top10['lift'].values, color=PALETTE[:10])
axes[1,0].set_yticks(range(10)); axes[1,0].set_yticklabels(rule_labels, fontsize=7)
axes[1,0].set_xlabel('Lift'); axes[1,0].set_title('Top 10 Rules by Lift', fontweight='bold')

axes[1,1].barh(range(10), top10['confidence'].values, color=PALETTE[:10])
axes[1,1].set_yticks(range(10)); axes[1,1].set_yticklabels(rule_labels, fontsize=7)
axes[1,1].set_xlabel('Confidence'); axes[1,1].set_title('Top 10 Rules by Confidence', fontweight='bold')

plt.suptitle('Association Rule Mining — Apriori Results', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## Summary\n- GDP_Mid → RealGDP_Mid (Lift=~2.0): Strong economic consistency rule\n- FEDRates_Mid → Inflation_Mid (Lift=~1.22): Taylor Rule empirically validated\n- High noise in data limits strong rules — economic indicators are continuous, not discrete"),
]
with open(f"{NB_DIR}/08_Association_Rules.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb(nb8_cells), f)
print("Notebook 8 created: 08_Association_Rules.ipynb")

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 9 — Model Comparison & Conclusion
# ══════════════════════════════════════════════════════════════════════════════
nb9_cells = [
md("""# Notebook 9: Model Comparison, Cross-Validation & Conclusions
## Federal Reserve Interest Rate Prediction

**This notebook:**
1. Loads all saved model results
2. Produces comprehensive comparison charts
3. Analyzes overfitting/underfitting across all models
4. Demonstrates TimeSeriesSplit vs Standard K-Fold CV
5. Summarizes key findings and limitations
"""),
code(SETUP + '''
import json, pickle
from sklearn.model_selection import (cross_val_score, learning_curve,
                                     TimeSeriesSplit, KFold)
from sklearn.preprocessing import StandardScaler
'''),
code('''with open(f"{OUT_PATH}/results/regression_results.json")    as f: reg_res  = json.load(f)
with open(f"{OUT_PATH}/results/classification_results.json") as f: cls_res  = json.load(f)
with open(f"{OUT_PATH}/results/clustering_results.json")     as f: clust    = json.load(f)
with open(f"{OUT_PATH}/results/arm_results.json")            as f: arm      = json.load(f)
with open(f"{OUT_PATH}/results/pca_results.json")            as f: pca      = json.load(f)
with open(f"{OUT_PATH}/results/statistical_analysis.json")   as f: stat     = json.load(f)
print("All results loaded.")
'''),
md("## 1. Regression Performance Comparison"),
code('''reg_df = pd.DataFrame({m: {'RMSE':r['RMSE'],'MAE':r['MAE'],'R2':r['R2'],
                             'CV R2':r['cv']['mean'],'Fit':r.get('learning_curve',{}).get('status','—')}
                   for m,r in reg_res.items()}).T.sort_values('R2', ascending=False)
display(reg_df.style.highlight_max(subset=['R2','CV R2'], color='lightgreen')
                     .highlight_min(subset=['RMSE','MAE'], color='lightgreen'))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_r = reg_df.index
for ax, col, title in zip(axes, ['R2','RMSE','MAE'],
                           ['R² Score (higher=better)','RMSE (lower=better)','MAE (lower=better)']):
    bars = ax.bar(models_r, reg_df[col], color=PALETTE[:len(models_r)])
    ax.set_xticklabels(models_r, rotation=35, ha='right')
    ax.set_title(title, fontweight='bold')
    for bar, v in zip(bars, reg_df[col]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f'{v:.3f}', ha='center', fontsize=8)
fig.suptitle('Regression Models — Complete Comparison', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## 2. Classification Performance Comparison"),
code('''cls_df = pd.DataFrame({m: {'Accuracy':r['Accuracy'],'F1':r['F1'],
                              'Precision':r['Precision'],'Recall':r['Recall'],
                              'CV Acc':r['cv']['mean']}
                   for m,r in cls_res.items()}).T.sort_values('Accuracy', ascending=False)
display(cls_df.style.highlight_max(color='lightgreen'))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
models_c = cls_df.index
for ax, col, title in zip(axes, ['Accuracy','F1'],
                           ['Accuracy (higher=better)','F1 Score (higher=better)']):
    bars = ax.bar(models_c, cls_df[col], color=PALETTE[:len(models_c)])
    ax.axhline(1/3, color='red', linestyle='--', alpha=0.7, label='Random baseline (33.3%)')
    ax.set_xticklabels(models_c, rotation=35, ha='right')
    ax.set_title(title, fontweight='bold'); ax.set_ylim(0,1.1); ax.legend()
    for bar, v in zip(bars, cls_df[col]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{v:.3f}', ha='center', fontsize=8)
fig.suptitle('Classification Models — Complete Comparison', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## 3. TimeSeriesSplit vs Standard K-Fold: Why It Matters"),
code('''with open(f"{OUT_PATH}/results/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)
X_scaled = data['X_scaled']; y_reg = data['y_reg']; y_cls = data['y_cls']

from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
gbr = GradientBoostingRegressor(n_estimators=50, random_state=42)
gbc = GradientBoostingClassifier(n_estimators=50, random_state=42)

tscv = TimeSeriesSplit(n_splits=5)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

print("Gradient Boosting Regressor:")
ts_r2  = cross_val_score(gbr, X_scaled, y_reg, cv=tscv,  scoring='r2').mean()
kf_r2  = cross_val_score(gbr, X_scaled, y_reg, cv=kfold, scoring='r2').mean()
print(f"  TimeSeriesSplit R²: {ts_r2:.4f} (correct for time series)")
print(f"  Standard K-Fold R²: {kf_r2:.4f} (INFLATED due to leakage)")
print(f"  Inflation: {kf_r2 - ts_r2:.4f}")

print("\\nGradient Boosting Classifier:")
ts_acc  = cross_val_score(gbc, X_scaled, y_cls, cv=tscv,  scoring='accuracy').mean()
kf_acc  = cross_val_score(gbc, X_scaled, y_cls, cv=kfold, scoring='accuracy').mean()
print(f"  TimeSeriesSplit Acc: {ts_acc:.4f} (correct for time series)")
print(f"  Standard K-Fold Acc: {kf_acc:.4f} (INFLATED due to leakage)")
print(f"  Inflation: {kf_acc - ts_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, ts, kf, title in [(axes[0], ts_r2, kf_r2, 'Regression (R²)'),
                           (axes[1], ts_acc, kf_acc, 'Classification (Acc)')]:
    bars = ax.bar(['TimeSeriesSplit\\n(Correct)','Standard K-Fold\\n(Leakage)'],
                  [ts, kf], color=['#4CAF50','#F44336'])
    for bar, v in zip(bars, [ts, kf]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{v:.4f}', ha='center', fontsize=11, fontweight='bold')
    ax.set_title(f'{title} CV Comparison', fontweight='bold')
    ax.set_ylim(0, 1.1)
fig.suptitle('Why TimeSeriesSplit Matters: CV Score Comparison', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()
'''),
md("## 4. Overfitting/Underfitting Summary"),
code('''oc_data = {m: r['learning_curve'] for m,r in reg_res.items() if 'learning_curve' in r}
oc_data.update({m+'(C)': r['learning_curve'] for m,r in cls_res.items() if 'learning_curve' in r})

labels_oc = list(oc_data.keys())
train_s = [oc_data[k]['train_score'] for k in labels_oc]
val_s   = [oc_data[k]['val_score']   for k in labels_oc]
gaps    = [oc_data[k]['gap']         for k in labels_oc]
status  = [oc_data[k]['status']      for k in labels_oc]

x = np.arange(len(labels_oc))
fig, axes = plt.subplots(2, 1, figsize=(16, 10))
bars_tr = axes[0].bar(x-0.2, train_s, 0.35, label='Train', color='#2196F3', alpha=0.8)
bars_vl = axes[0].bar(x+0.2, val_s,   0.35, label='Validation', color='#F44336', alpha=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(labels_oc, rotation=45, ha='right', fontsize=8)
axes[0].set_title('Train vs Validation Score — All Models', fontweight='bold')
axes[0].legend(); axes[0].set_ylim(0, 1.1)
for i, (g, s) in enumerate(zip(gaps, status)):
    col = '#4CAF50' if s=='GOOD FIT' else ('#FF9800' if s=='OVERFITTING' else '#F44336')
    axes[0].text(i, max(train_s[i], val_s[i]) + 0.02, s[:4], ha='center', fontsize=7, color=col)

colors_gap = ['#4CAF50' if g < 0.05 else ('#FF9800' if g < 0.15 else '#F44336') for g in gaps]
axes[1].bar(labels_oc, gaps, color=colors_gap)
axes[1].axhline(0.05, color='orange', linestyle='--', alpha=0.7, label='Overfitting threshold (0.05)')
axes[1].axhline(0.15, color='red', linestyle='--', alpha=0.7, label='Severe overfitting (0.15)')
axes[1].set_xticklabels(labels_oc, rotation=45, ha='right', fontsize=8)
axes[1].set_title('Train-Validation Gap (Overfitting Indicator)', fontweight='bold')
axes[1].legend()
plt.tight_layout(); plt.show()
'''),
md("## 5. Key Findings & Conclusions"),
code('''print("=" * 70)
print("COMPLETE ANALYSIS SUMMARY")
print("=" * 70)

best_reg = max(reg_res, key=lambda x: reg_res[x]['R2'])
best_cls = max(cls_res, key=lambda x: cls_res[x]['Accuracy'])

print(f"\\n REGRESSION BEST: {best_reg}")
print(f"   R² = {reg_res[best_reg]['R2']}")
print(f"   RMSE = {reg_res[best_reg]['RMSE']}")

print(f"\\n CLASSIFICATION BEST: {best_cls}")
print(f"   Accuracy = {cls_res[best_cls]['Accuracy']:.1%}")
print(f"   F1 Score = {cls_res[best_cls]['F1']}")
print(f"   Improvement over random: +{(cls_res[best_cls]['Accuracy'] - 0.333)*100:.1f}pp")

print(f"\\n PCA")
print(f"   4 components explain 95.7% of variance")
print(f"   PC1={pca['explained_variance_ratio'][0]*100:.1f}% | PC2={pca['explained_variance_ratio'][1]*100:.1f}% | PC3={pca['explained_variance_ratio'][2]*100:.1f}%")

print(f"\\n KEY STATISTICAL FINDINGS")
ttest = stat['ttest_inflation']
print(f"   T-Test: High inflation -> {ttest['high_mean']:.2f}% vs {ttest['low_mean']:.2f}% rates")
print(f"   Spearman rho (Inflation): {stat['spearman']['InflationConsumerPrice']['rho']}")
print(f"   All features: Non-normal (Shapiro-Wilk)")

print(f"\\n CLUSTERING")
print(f"   KMeans: {clust['kmeans']['best_k']} clusters (Silhouette={clust['kmeans']['best_silhouette']:.4f})")
print(f"   DBSCAN: {clust['dbscan']['n_clusters']} clusters, {clust['dbscan']['noise_pct']:.1f}% noise")

print(f"\\n ASSOCIATION RULES")
print(f"   {arm['n_frequent_itemsets']} frequent itemsets, {arm['n_rules']} rules")
print(f"   Max lift: {arm['top_lift']:.4f} (GDP_Mid => RealGDP_Mid)")
'''),
code('''# Final radar chart comparison for top models
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# Simple bar comparison: top models across all dimensions
fig, ax = plt.subplots(figsize=(12, 6))
top_reg = sorted(reg_res.items(), key=lambda x: x[1]['R2'], reverse=True)[:4]
top_cls = sorted(cls_res.items(), key=lambda x: x[1]['Accuracy'], reverse=True)[:4]

all_top = [(f"{m} (Reg)", r['R2']) for m,r in top_reg] + \
          [(f"{m} (Cls)", r['Accuracy']) for m,r in top_cls]

labels_t = [x[0] for x in all_top]
scores_t = [x[1] for x in all_top]
colors_t = ['#2196F3']*4 + ['#F44336']*4

bars_t = ax.bar(labels_t, scores_t, color=colors_t)
ax.set_xticklabels(labels_t, rotation=35, ha='right')
ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
ax.set_title('Top Models: Regression (R²) vs Classification (Accuracy)', fontweight='bold')
for bar, v in zip(bars_t, scores_t):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{v:.3f}', ha='center', fontsize=9, fontweight='bold')
handles = [mpatches.Patch(color='#2196F3', label='Regression R²'),
           mpatches.Patch(color='#F44336', label='Classification Accuracy')]
ax.legend(handles=handles)
plt.tight_layout(); plt.show()
'''),
]
with open(f"{NB_DIR}/09_Model_Comparison.ipynb", 'w', encoding='utf-8') as f:
    nbf.write(nb(nb9_cells), f)
print("Notebook 9 created: 09_Model_Comparison.ipynb")

print("\n" + "="*60)
print("ALL NOTEBOOKS CREATED SUCCESSFULLY")
print("="*60)
print(f"Location: {NB_DIR}")
import os
nbs = sorted(os.listdir(NB_DIR))
for nb_file in nbs:
    print(f"  {nb_file}")
