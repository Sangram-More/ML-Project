"""
PART 2: All ML Models — Regression, Classification, Clustering, ARM
Federal Reserve Interest Rate Prediction — Professional ML Pipeline
"""

import warnings
warnings.filterwarnings('ignore')
import pickle, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score,
    learning_curve, GridSearchCV, StratifiedKFold, KFold, TimeSeriesSplit)
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, precision_score, recall_score)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
import xgboost as xgb
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

BASE   = "d:/Projects/ML website/ML-Project"
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

# ── Load preprocessed data ────────────────────────────────────────────────────
with open(f"{RES}/preprocessed_data.pkl", 'rb') as f:
    data = pickle.load(f)

X         = data['X']
X_scaled  = data['X_scaled']
y_reg     = data['y_reg']
y_cls     = data['y_cls']
feat_cols = data['feature_cols']
label_names = data['label_names']   # ['Decrease','Increase','No_Change']
scaler    = data['scaler']
le        = data['le']
df        = data['df']
X_pca_3   = data['X_pca_3']

CLASS_COLORS = {'Decrease':'#F44336','Increase':'#4CAF50','No_Change':'#FF9800'}
ALL_RESULTS  = {}   # accumulate final metrics here

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def reg_metrics(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  {name}: RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {'RMSE': round(rmse,4), 'MAE': round(mae,4), 'R2': round(r2,4)}

def cls_metrics(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"  {name}: Acc={acc:.4f}  F1={f1:.4f}  Prec={prec:.4f}  Rec={rec:.4f}")
    return {'Accuracy': round(acc,4), 'F1': round(f1,4),
            'Precision': round(prec,4), 'Recall': round(rec,4)}

def plot_confusion_matrix(y_true, y_pred, labels, title, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, annot_kws={'size':13})
    ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')
    savefig(path)

def plot_learning_curve(estimator, X, y, title, path, cv=None, scoring='r2',
                        is_classifier=False):
    if is_classifier:
        scoring = 'accuracy'
    if cv is None:
        cv = TimeSeriesSplit(n_splits=5)
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(train_sizes, train_mean, 'b-o', label='Training Score')
    ax.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15, color='blue')
    ax.plot(train_sizes, val_mean, 'r-o', label='Validation Score')
    ax.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color='red')

    # Detect over/underfitting
    gap = train_mean[-1] - val_mean[-1]
    if val_mean[-1] < 0.6:
        status = 'UNDERFITTING'
    elif gap > 0.1:
        status = 'OVERFITTING'
    else:
        status = 'GOOD FIT'
    ax.set_title(f'Learning Curve — {title}\nStatus: {status}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Set Size'); ax.set_ylabel(scoring.upper())
    ax.legend(); ax.grid(True)
    ax.text(0.98, 0.05, f'Gap: {gap:.3f}', transform=ax.transAxes,
            ha='right', fontsize=10, color='purple',
            bbox=dict(boxstyle='round', facecolor='lightyellow'))
    savefig(path)
    return {'train_score': float(train_mean[-1]), 'val_score': float(val_mean[-1]),
            'gap': float(gap), 'status': status}

def plot_actual_vs_predicted(y_true, y_pred, title, path, sample=None):
    if sample and len(y_true) > sample:
        idx = np.random.choice(len(y_true), sample, replace=False)
        idx = np.sort(idx)
        y_true = y_true[idx]; y_pred = y_pred[idx]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_true, y_pred, alpha=0.4, s=20, color='#2196F3')
    mn = min(y_true.min(), y_pred.min()); mx = max(y_true.max(), y_pred.max())
    axes[0].plot([mn,mx],[mn,mx],'r--', linewidth=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'{title}\nActual vs Predicted', fontsize=11, fontweight='bold')
    axes[0].legend()

    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.4, s=20, color='#FF9800')
    axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Residuals')
    axes[1].set_title(f'{title}\nResidual Plot', fontsize=11, fontweight='bold')
    savefig(path)

def plot_roc_multiclass(y_true, y_prob, labels, title, path):
    n_classes = len(labels)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ['#2196F3','#F44336','#4CAF50']
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        auc = roc_auc_score(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=col, linewidth=2, label=f'{lbl} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1],'k--', linewidth=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves — {title}', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right')
    savefig(path)

def cv_scores(model, X, y, cv=None, scoring='r2', name=''):
    # Use TimeSeriesSplit to respect temporal order and avoid data leakage
    if cv is None:
        cv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"  TimeSeriesCV {scoring}: {scores.mean():.4f} +/- {scores.std():.4f}")
    return {'mean': float(scores.mean()), 'std': float(scores.std()),
            'scores': [float(s) for s in scores]}

# ══════════════════════════════════════════════════════════════════════════════
# 8. REGRESSION MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 8 — REGRESSION MODELS")
print("="*70)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y_reg, test_size=0.2, random_state=42)

reg_results = {}

# ── 8.1 Linear Regression ──────────────────────────────────────────────────
print("\n[8.1] Linear Regression")
lr = LinearRegression()
lr.fit(X_tr, y_tr)
y_pred_lr = lr.predict(X_te)
reg_results['Linear Regression'] = reg_metrics(y_te, y_pred_lr, 'LinearReg')
reg_results['Linear Regression']['cv'] = cv_scores(lr, X_scaled, y_reg, name='LinearReg')
lc = plot_learning_curve(lr, X_scaled, y_reg, 'Linear Regression',
                         f"{CHARTS}/regression/01_lr_learning_curve.png")
reg_results['Linear Regression']['learning_curve'] = lc
plot_actual_vs_predicted(y_te, y_pred_lr, 'Linear Regression',
                         f"{CHARTS}/regression/02_lr_actual_vs_pred.png")

# Coefficients
coef_df = pd.DataFrame({'Feature': feat_cols, 'Coefficient': lr.coef_})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=False).index)
fig, ax = plt.subplots(figsize=(10, 6))
top = coef_df.head(20)
colors_c = ['#F44336' if v < 0 else '#2196F3' for v in top['Coefficient']]
ax.barh(range(len(top)), top['Coefficient'], color=colors_c)
ax.set_yticks(range(len(top))); ax.set_yticklabels(top['Feature'], fontsize=8)
ax.set_title('Linear Regression — Top 20 Feature Coefficients', fontsize=12, fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
savefig(f"{CHARTS}/regression/03_lr_coefficients.png")

# ── 8.2 Ridge Regression ───────────────────────────────────────────────────
print("\n[8.2] Ridge Regression")
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
ridge_cv_r2 = []
for a in alphas:
    r = Ridge(alpha=a)
    s = cross_val_score(r, X_scaled, y_reg, cv=5, scoring='r2').mean()
    ridge_cv_r2.append(s)
best_alpha_ridge = alphas[np.argmax(ridge_cv_r2)]
print(f"  Best Ridge alpha: {best_alpha_ridge}")

ridge = Ridge(alpha=best_alpha_ridge)
ridge.fit(X_tr, y_tr)
y_pred_ridge = ridge.predict(X_te)
reg_results['Ridge Regression'] = reg_metrics(y_te, y_pred_ridge, 'Ridge')
reg_results['Ridge Regression']['cv'] = cv_scores(ridge, X_scaled, y_reg)
lc = plot_learning_curve(ridge, X_scaled, y_reg, 'Ridge Regression',
                         f"{CHARTS}/regression/04_ridge_learning_curve.png")
reg_results['Ridge Regression']['learning_curve'] = lc
plot_actual_vs_predicted(y_te, y_pred_ridge, 'Ridge Regression',
                         f"{CHARTS}/regression/05_ridge_actual_vs_pred.png")

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogx(alphas, ridge_cv_r2, 'b-o', linewidth=2, markersize=7)
ax.axvline(best_alpha_ridge, color='red', linestyle='--', linewidth=1.5,
           label=f'Best alpha={best_alpha_ridge}')
ax.set_xlabel('Alpha (Regularization)'); ax.set_ylabel('CV R² Score')
ax.set_title('Ridge Regression — Alpha Tuning', fontsize=12, fontweight='bold')
ax.legend()
savefig(f"{CHARTS}/regression/06_ridge_alpha_tuning.png")

# ── 8.3 Lasso Regression ───────────────────────────────────────────────────
print("\n[8.3] Lasso Regression")
lasso_cv_r2 = []
for a in alphas:
    r = Lasso(alpha=a, max_iter=10000)
    s = cross_val_score(r, X_scaled, y_reg, cv=5, scoring='r2').mean()
    lasso_cv_r2.append(s)
best_alpha_lasso = alphas[np.argmax(lasso_cv_r2)]
print(f"  Best Lasso alpha: {best_alpha_lasso}")

lasso = Lasso(alpha=best_alpha_lasso, max_iter=10000)
lasso.fit(X_tr, y_tr)
y_pred_lasso = lasso.predict(X_te)
reg_results['Lasso Regression'] = reg_metrics(y_te, y_pred_lasso, 'Lasso')
reg_results['Lasso Regression']['cv'] = cv_scores(lasso, X_scaled, y_reg)
lc = plot_learning_curve(lasso, X_scaled, y_reg, 'Lasso Regression',
                         f"{CHARTS}/regression/07_lasso_learning_curve.png")
reg_results['Lasso Regression']['learning_curve'] = lc
plot_actual_vs_predicted(y_te, y_pred_lasso, 'Lasso Regression',
                         f"{CHARTS}/regression/08_lasso_actual_vs_pred.png")

# Lasso zero coefficients
n_zero = np.sum(lasso.coef_ == 0)
print(f"  Lasso zeroed out {n_zero}/{len(lasso.coef_)} features (feature selection)")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].semilogx(alphas, lasso_cv_r2, 'g-o', linewidth=2, markersize=7)
axes[0].axvline(best_alpha_lasso, color='red', linestyle='--', linewidth=1.5,
                label=f'Best alpha={best_alpha_lasso}')
axes[0].set_xlabel('Alpha'); axes[0].set_ylabel('CV R² Score')
axes[0].set_title('Lasso — Alpha Tuning', fontsize=12, fontweight='bold')
axes[0].legend()

non_zero = np.where(lasso.coef_ != 0)[0]
if len(non_zero) > 0:
    top_n = min(20, len(non_zero))
    top_idx = non_zero[np.argsort(np.abs(lasso.coef_[non_zero]))[-top_n:]]
    colors_l = ['#F44336' if v < 0 else '#2196F3' for v in lasso.coef_[top_idx]]
    axes[1].barh(range(len(top_idx)), lasso.coef_[top_idx], color=colors_l)
    axes[1].set_yticks(range(len(top_idx)))
    axes[1].set_yticklabels([feat_cols[i][:20] for i in top_idx], fontsize=8)
    axes[1].axvline(0, color='black', linewidth=0.8)
axes[1].set_title(f'Lasso Non-Zero Coefficients\n({n_zero} features zeroed out)',
                  fontsize=12, fontweight='bold')
savefig(f"{CHARTS}/regression/09_lasso_feature_selection.png")

# ── 8.4 Support Vector Regression (SVR) ────────────────────────────────────
print("\n[8.4] SVR — Support Vector Regression")
svr = SVR(kernel='rbf', C=10, epsilon=0.1, gamma='scale')
svr.fit(X_tr, y_tr)
y_pred_svr = svr.predict(X_te)
reg_results['SVR'] = reg_metrics(y_te, y_pred_svr, 'SVR')
reg_results['SVR']['cv'] = cv_scores(svr, X_scaled, y_reg)
lc = plot_learning_curve(svr, X_scaled, y_reg, 'SVR',
                         f"{CHARTS}/regression/10_svr_learning_curve.png")
reg_results['SVR']['learning_curve'] = lc
plot_actual_vs_predicted(y_te, y_pred_svr, 'SVR',
                         f"{CHARTS}/regression/11_svr_actual_vs_pred.png")

# ── 8.5 Decision Tree Regressor ─────────────────────────────────────────────
print("\n[8.5] Decision Tree Regressor (with depth tuning)")
depths = range(1, 20)
dt_train_r2, dt_val_r2 = [], []
tscv = TimeSeriesSplit(n_splits=5)
for d in depths:
    dt_tmp = DecisionTreeRegressor(max_depth=d, random_state=42)
    tr_s = cross_val_score(dt_tmp, X_scaled, y_reg, cv=tscv, scoring='r2').mean()
    dt_val_r2.append(tr_s)
best_depth_dt = depths[np.argmax(dt_val_r2)]
print(f"  Best Decision Tree depth: {best_depth_dt}")

dt_reg = DecisionTreeRegressor(max_depth=best_depth_dt, random_state=42)
dt_reg.fit(X_tr, y_tr)
y_pred_dtr = dt_reg.predict(X_te)
reg_results['Decision Tree'] = reg_metrics(y_te, y_pred_dtr, 'DecisionTree')
reg_results['Decision Tree']['cv'] = cv_scores(dt_reg, X_scaled, y_reg)
lc = plot_learning_curve(dt_reg, X_scaled, y_reg, 'Decision Tree Regressor',
                         f"{CHARTS}/regression/12_dt_learning_curve.png")
reg_results['Decision Tree']['learning_curve'] = lc
plot_actual_vs_predicted(y_te, y_pred_dtr, 'Decision Tree Regressor',
                         f"{CHARTS}/regression/13_dt_actual_vs_pred.png")

# Overfitting analysis: depth vs R²
dt_train_scores, dt_val_scores_dep = [], []
for d in depths:
    dt_tmp = DecisionTreeRegressor(max_depth=d, random_state=42)
    dt_tmp.fit(X_tr, y_tr)
    dt_train_scores.append(r2_score(y_tr, dt_tmp.predict(X_tr)))
    dt_val_scores_dep.append(r2_score(y_te, dt_tmp.predict(X_te)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(depths, dt_train_scores, 'b-o', label='Train R²', linewidth=2)
ax.plot(depths, dt_val_scores_dep, 'r-o', label='Test R²', linewidth=2)
ax.axvline(best_depth_dt, color='green', linestyle='--', linewidth=1.5,
           label=f'Best Depth={best_depth_dt}')
ax.fill_between(depths, dt_train_scores, dt_val_scores_dep, alpha=0.15,
                color='purple', label='Overfitting Gap')
ax.set_xlabel('Tree Depth'); ax.set_ylabel('R² Score')
ax.set_title('Decision Tree — Overfitting Analysis (Depth vs R²)', fontsize=12, fontweight='bold')
ax.legend()
savefig(f"{CHARTS}/regression/14_dt_depth_overfitting.png")

# ── 8.6 Random Forest Regressor ─────────────────────────────────────────────
print("\n[8.6] Random Forest Regressor")
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5,
                                random_state=42, n_jobs=-1)
rf_reg.fit(X_tr, y_tr)
y_pred_rfr = rf_reg.predict(X_te)
reg_results['Random Forest'] = reg_metrics(y_te, y_pred_rfr, 'RandomForest')
reg_results['Random Forest']['cv'] = cv_scores(rf_reg, X_scaled, y_reg)
lc = plot_learning_curve(rf_reg, X_scaled, y_reg, 'Random Forest Regressor',
                         f"{CHARTS}/regression/15_rf_learning_curve.png")
reg_results['Random Forest']['learning_curve'] = lc
plot_actual_vs_predicted(y_te, y_pred_rfr, 'Random Forest Regressor',
                         f"{CHARTS}/regression/16_rf_actual_vs_pred.png")

fi = pd.Series(rf_reg.feature_importances_, index=feat_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
fi.head(20).plot.bar(ax=ax, color=PALETTE[:20])
ax.set_title('Random Forest — Top 20 Feature Importances', fontsize=12, fontweight='bold')
ax.set_xlabel('Feature'); ax.set_ylabel('Importance')
ax.tick_params(axis='x', rotation=45)
savefig(f"{CHARTS}/regression/17_rf_feature_importance.png")

# ── 8.7 Gradient Boosting Regressor ─────────────────────────────────────────
print("\n[8.7] Gradient Boosting Regressor")
gb_reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                    max_depth=4, random_state=42)
gb_reg.fit(X_tr, y_tr)
y_pred_gbr = gb_reg.predict(X_te)
reg_results['Gradient Boosting'] = reg_metrics(y_te, y_pred_gbr, 'GradientBoosting')
reg_results['Gradient Boosting']['cv'] = cv_scores(gb_reg, X_scaled, y_reg)
lc = plot_learning_curve(gb_reg, X_scaled, y_reg, 'Gradient Boosting',
                         f"{CHARTS}/regression/18_gb_learning_curve.png")
reg_results['Gradient Boosting']['learning_curve'] = lc
plot_actual_vs_predicted(y_te, y_pred_gbr, 'Gradient Boosting',
                         f"{CHARTS}/regression/19_gb_actual_vs_pred.png")

# ── 8.8 XGBoost Regressor ───────────────────────────────────────────────────
print("\n[8.8] XGBoost Regressor")
xgb_reg = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             verbosity=0)
xgb_reg.fit(X_tr, y_tr)
y_pred_xgbr = xgb_reg.predict(X_te)
reg_results['XGBoost'] = reg_metrics(y_te, y_pred_xgbr, 'XGBoost')
reg_results['XGBoost']['cv'] = cv_scores(xgb_reg, X_scaled, y_reg)
lc = plot_learning_curve(xgb_reg, X_scaled, y_reg, 'XGBoost Regressor',
                         f"{CHARTS}/regression/20_xgb_learning_curve.png")
reg_results['XGBoost']['learning_curve'] = lc
plot_actual_vs_predicted(y_te, y_pred_xgbr, 'XGBoost Regressor',
                         f"{CHARTS}/regression/21_xgb_actual_vs_pred.png")

# ── Regression Comparison Chart ──────────────────────────────────────────────
print("\nRegression Model Summary:")
reg_names = list(reg_results.keys())
rmse_vals = [reg_results[m]['RMSE'] for m in reg_names]
r2_vals   = [reg_results[m]['R2'] for m in reg_names]
cv_vals   = [reg_results[m]['cv']['mean'] for m in reg_names]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors_r = PALETTE[:len(reg_names)]
bars0 = axes[0].bar(reg_names, rmse_vals, color=colors_r)
axes[0].set_title('RMSE (lower is better)', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(reg_names, rotation=35, ha='right')
for bar, v in zip(bars0, rmse_vals):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{v:.3f}', ha='center', fontsize=9)

bars1 = axes[1].bar(reg_names, r2_vals, color=colors_r)
axes[1].set_title('R² Score (higher is better)', fontsize=12, fontweight='bold')
axes[1].set_xticklabels(reg_names, rotation=35, ha='right')
axes[1].set_ylim(0, 1.05)
for bar, v in zip(bars1, r2_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{v:.3f}', ha='center', fontsize=9)

bars2 = axes[2].bar(reg_names, cv_vals, color=colors_r)
axes[2].set_title('5-Fold CV R² Score', fontsize=12, fontweight='bold')
axes[2].set_xticklabels(reg_names, rotation=35, ha='right')
axes[2].set_ylim(0, 1.05)
for bar, v in zip(bars2, cv_vals):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                 f'{v:.3f}', ha='center', fontsize=9)

fig.suptitle('Regression Models — Performance Comparison', fontsize=14, fontweight='bold')
savefig(f"{CHARTS}/comparison/reg_model_comparison.png")

with open(f"{RES}/regression_results.json", 'w') as f:
    json.dump(reg_results, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# 9. CLASSIFICATION MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 9 — CLASSIFICATION MODELS")
print("="*70)

X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
    X_scaled, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

cls_results = {}

# ── 9.1 Logistic Regression ─────────────────────────────────────────────────
print("\n[9.1] Logistic Regression")
log_reg = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', random_state=42)
log_reg.fit(X_tr_c, y_tr_c)
y_pred_log = log_reg.predict(X_te_c)
y_prob_log = log_reg.predict_proba(X_te_c)
cls_results['Logistic Regression'] = cls_metrics(y_te_c, y_pred_log, 'LogReg')
cls_results['Logistic Regression']['cv'] = cv_scores(log_reg, X_scaled, y_cls, scoring='accuracy')
lc = plot_learning_curve(log_reg, X_scaled, y_cls, 'Logistic Regression',
                         f"{CHARTS}/classification/01_logreg_lc.png", is_classifier=True)
cls_results['Logistic Regression']['learning_curve'] = lc
plot_confusion_matrix(y_te_c, y_pred_log, label_names, 'Logistic Regression Confusion Matrix',
                      f"{CHARTS}/classification/02_logreg_cm.png")
plot_roc_multiclass(y_te_c, y_prob_log, label_names, 'Logistic Regression',
                    f"{CHARTS}/classification/03_logreg_roc.png")
print(classification_report(y_te_c, y_pred_log, target_names=label_names))

# ── 9.2 Naive Bayes ─────────────────────────────────────────────────────────
print("\n[9.2] Gaussian Naive Bayes")
gnb = GaussianNB()
gnb.fit(X_tr_c, y_tr_c)
y_pred_gnb = gnb.predict(X_te_c)
y_prob_gnb = gnb.predict_proba(X_te_c)
cls_results['Naive Bayes'] = cls_metrics(y_te_c, y_pred_gnb, 'GaussianNB')
cls_results['Naive Bayes']['cv'] = cv_scores(gnb, X_scaled, y_cls, scoring='accuracy')
lc = plot_learning_curve(gnb, X_scaled, y_cls, 'Naive Bayes',
                         f"{CHARTS}/classification/04_nb_lc.png", is_classifier=True)
cls_results['Naive Bayes']['learning_curve'] = lc
plot_confusion_matrix(y_te_c, y_pred_gnb, label_names, 'Naive Bayes Confusion Matrix',
                      f"{CHARTS}/classification/05_nb_cm.png")
plot_roc_multiclass(y_te_c, y_prob_gnb, label_names, 'Naive Bayes',
                    f"{CHARTS}/classification/06_nb_roc.png")
print(classification_report(y_te_c, y_pred_gnb, target_names=label_names))

# ── 9.3 SVM Classifier ──────────────────────────────────────────────────────
print("\n[9.3] SVM Classifier (RBF kernel)")
svc = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
svc.fit(X_tr_c, y_tr_c)
y_pred_svc = svc.predict(X_te_c)
y_prob_svc = svc.predict_proba(X_te_c)
cls_results['SVM'] = cls_metrics(y_te_c, y_pred_svc, 'SVC')
cls_results['SVM']['cv'] = cv_scores(svc, X_scaled, y_cls, scoring='accuracy')
lc = plot_learning_curve(svc, X_scaled, y_cls, 'SVM',
                         f"{CHARTS}/classification/07_svm_lc.png", is_classifier=True)
cls_results['SVM']['learning_curve'] = lc
plot_confusion_matrix(y_te_c, y_pred_svc, label_names, 'SVM Confusion Matrix',
                      f"{CHARTS}/classification/08_svm_cm.png")
plot_roc_multiclass(y_te_c, y_prob_svc, label_names, 'SVM',
                    f"{CHARTS}/classification/09_svm_roc.png")
print(classification_report(y_te_c, y_pred_svc, target_names=label_names))

# ── 9.4 Decision Tree Classifier ────────────────────────────────────────────
print("\n[9.4] Decision Tree Classifier")
depths_c = range(1, 20)
dt_val_acc = []
tscv_cls = TimeSeriesSplit(n_splits=5)
for d in depths_c:
    dt_tmp = DecisionTreeClassifier(max_depth=d, random_state=42)
    s = cross_val_score(dt_tmp, X_scaled, y_cls, cv=tscv_cls, scoring='accuracy').mean()
    dt_val_acc.append(s)
best_depth_dtc = depths_c[np.argmax(dt_val_acc)]
print(f"  Best DT Classifier depth: {best_depth_dtc}")

dtc = DecisionTreeClassifier(max_depth=best_depth_dtc, random_state=42)
dtc.fit(X_tr_c, y_tr_c)
y_pred_dtc = dtc.predict(X_te_c)
y_prob_dtc = dtc.predict_proba(X_te_c)
cls_results['Decision Tree'] = cls_metrics(y_te_c, y_pred_dtc, 'DecTree')
cls_results['Decision Tree']['cv'] = cv_scores(dtc, X_scaled, y_cls, scoring='accuracy')
lc = plot_learning_curve(dtc, X_scaled, y_cls, 'Decision Tree Classifier',
                         f"{CHARTS}/classification/10_dt_lc.png", is_classifier=True)
cls_results['Decision Tree']['learning_curve'] = lc
plot_confusion_matrix(y_te_c, y_pred_dtc, label_names, 'Decision Tree Confusion Matrix',
                      f"{CHARTS}/classification/11_dt_cm.png")
plot_roc_multiclass(y_te_c, y_prob_dtc, label_names, 'Decision Tree',
                    f"{CHARTS}/classification/12_dt_roc.png")

# Depth vs accuracy for overfitting
dt_train_acc_depth = []
for d in depths_c:
    dt_tmp = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt_tmp.fit(X_tr_c, y_tr_c)
    dt_train_acc_depth.append(accuracy_score(y_tr_c, dt_tmp.predict(X_tr_c)))

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(depths_c, dt_train_acc_depth, 'b-o', label='Train Accuracy')
ax.plot(depths_c, dt_val_acc, 'r-o', label='CV Accuracy')
ax.axvline(best_depth_dtc, color='green', linestyle='--',
           label=f'Best Depth={best_depth_dtc}')
ax.set_xlabel('Tree Depth'); ax.set_ylabel('Accuracy')
ax.set_title('Decision Tree Classifier — Overfitting Analysis', fontsize=12, fontweight='bold')
ax.legend()
savefig(f"{CHARTS}/classification/13_dt_depth_overfitting.png")

# ── 9.5 Random Forest Classifier ────────────────────────────────────────────
print("\n[9.5] Random Forest Classifier")
rfc = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=5,
                               random_state=42, n_jobs=-1)
rfc.fit(X_tr_c, y_tr_c)
y_pred_rfc = rfc.predict(X_te_c)
y_prob_rfc = rfc.predict_proba(X_te_c)
cls_results['Random Forest'] = cls_metrics(y_te_c, y_pred_rfc, 'RandomForest')
cls_results['Random Forest']['cv'] = cv_scores(rfc, X_scaled, y_cls, scoring='accuracy')
lc = plot_learning_curve(rfc, X_scaled, y_cls, 'Random Forest Classifier',
                         f"{CHARTS}/classification/14_rf_lc.png", is_classifier=True)
cls_results['Random Forest']['learning_curve'] = lc
plot_confusion_matrix(y_te_c, y_pred_rfc, label_names, 'Random Forest Confusion Matrix',
                      f"{CHARTS}/classification/15_rf_cm.png")
plot_roc_multiclass(y_te_c, y_prob_rfc, label_names, 'Random Forest',
                    f"{CHARTS}/classification/16_rf_roc.png")
print(classification_report(y_te_c, y_pred_rfc, target_names=label_names))

fi_rfc = pd.Series(rfc.feature_importances_, index=feat_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
fi_rfc.head(20).plot.bar(ax=ax, color=PALETTE[:20])
ax.set_title('Random Forest Classifier — Top 20 Feature Importances', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
savefig(f"{CHARTS}/classification/17_rf_feature_importance.png")

# ── 9.6 Gradient Boosting Classifier ────────────────────────────────────────
print("\n[9.6] Gradient Boosting Classifier")
gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                  max_depth=4, random_state=42)
gbc.fit(X_tr_c, y_tr_c)
y_pred_gbc = gbc.predict(X_te_c)
y_prob_gbc = gbc.predict_proba(X_te_c)
cls_results['Gradient Boosting'] = cls_metrics(y_te_c, y_pred_gbc, 'GradBoost')
cls_results['Gradient Boosting']['cv'] = cv_scores(gbc, X_scaled, y_cls, scoring='accuracy')
lc = plot_learning_curve(gbc, X_scaled, y_cls, 'Gradient Boosting Classifier',
                         f"{CHARTS}/classification/18_gb_lc.png", is_classifier=True)
cls_results['Gradient Boosting']['learning_curve'] = lc
plot_confusion_matrix(y_te_c, y_pred_gbc, label_names, 'Gradient Boosting Confusion Matrix',
                      f"{CHARTS}/classification/19_gb_cm.png")
plot_roc_multiclass(y_te_c, y_prob_gbc, label_names, 'Gradient Boosting',
                    f"{CHARTS}/classification/20_gb_roc.png")
print(classification_report(y_te_c, y_pred_gbc, target_names=label_names))

# ── 9.7 XGBoost Classifier ──────────────────────────────────────────────────
print("\n[9.7] XGBoost Classifier")
xgb_cls = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                              subsample=0.8, colsample_bytree=0.8, random_state=42,
                              eval_metric='mlogloss', verbosity=0)
xgb_cls.fit(X_tr_c, y_tr_c)
y_pred_xgbc = xgb_cls.predict(X_te_c)
y_prob_xgbc = xgb_cls.predict_proba(X_te_c)
cls_results['XGBoost'] = cls_metrics(y_te_c, y_pred_xgbc, 'XGBoost')
cls_results['XGBoost']['cv'] = cv_scores(xgb_cls, X_scaled, y_cls, scoring='accuracy')
lc = plot_learning_curve(xgb_cls, X_scaled, y_cls, 'XGBoost Classifier',
                         f"{CHARTS}/classification/21_xgb_lc.png", is_classifier=True)
cls_results['XGBoost']['learning_curve'] = lc
plot_confusion_matrix(y_te_c, y_pred_xgbc, label_names, 'XGBoost Confusion Matrix',
                      f"{CHARTS}/classification/22_xgb_cm.png")
plot_roc_multiclass(y_te_c, y_prob_xgbc, label_names, 'XGBoost',
                    f"{CHARTS}/classification/23_xgb_roc.png")
print(classification_report(y_te_c, y_pred_xgbc, target_names=label_names))

fi_xgb = pd.Series(xgb_cls.feature_importances_, index=feat_cols).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
fi_xgb.head(20).plot.bar(ax=ax, color=PALETTE[:20])
ax.set_title('XGBoost — Top 20 Feature Importances', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
savefig(f"{CHARTS}/classification/24_xgb_feature_importance.png")

# ── Classification Comparison ────────────────────────────────────────────────
cls_names  = list(cls_results.keys())
acc_vals   = [cls_results[m]['Accuracy'] for m in cls_names]
f1_vals    = [cls_results[m]['F1'] for m in cls_names]
cv_acc     = [cls_results[m]['cv']['mean'] for m in cls_names]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
colors_c2 = PALETTE[:len(cls_names)]
for ax, vals, title in zip(axes, [acc_vals, f1_vals, cv_acc],
                            ['Test Accuracy','Weighted F1-Score','5-Fold CV Accuracy']):
    bars = ax.bar(cls_names, vals, color=colors_c2)
    ax.set_title(title + ' (higher is better)', fontsize=11, fontweight='bold')
    ax.set_xticklabels(cls_names, rotation=35, ha='right')
    ax.set_ylim(0, 1.1)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{v:.3f}', ha='center', fontsize=9)
fig.suptitle('Classification Models — Performance Comparison', fontsize=14, fontweight='bold')
savefig(f"{CHARTS}/comparison/cls_model_comparison.png")

with open(f"{RES}/classification_results.json", 'w') as f:
    json.dump(cls_results, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# 10. CLUSTERING — DBSCAN + KMeans
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 10 — CLUSTERING (DBSCAN + KMeans)")
print("="*70)

# Use PCA-reduced data for visualization
pca_2 = PCA(n_components=2)
X_2d = pca_2.fit_transform(X_scaled)

# ── 10.1 KMeans (for comparison) ─────────────────────────────────────────
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

k_range = range(2, 10)
silhouette_scores, inertias = [], []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_k = km.fit_predict(X_scaled)
    silhouette_scores.append(silhouette_score(X_scaled, labels_k))
    inertias.append(km.inertia_)

best_k = k_range[np.argmax(silhouette_scores)]
km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_best.fit_predict(X_scaled)
print(f"KMeans best k={best_k}  Silhouette={max(silhouette_scores):.4f}")

# ── 10.2 DBSCAN ──────────────────────────────────────────────────────────
# Parameter search: eps
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
k_dist = np.sort(distances[:, -1])

eps_candidates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
dbscan_results_grid = []
for eps in eps_candidates:
    db = DBSCAN(eps=eps, min_samples=5)
    lbl = db.fit_predict(X_scaled)
    n_clusters = len(set(lbl)) - (1 if -1 in lbl else 0)
    n_noise    = list(lbl).count(-1)
    noise_pct  = n_noise / len(lbl) * 100
    if n_clusters >= 2:
        core_mask = lbl != -1
        if core_mask.sum() > 0:
            sil = silhouette_score(X_scaled[core_mask], lbl[core_mask])
        else:
            sil = -1
    else:
        sil = -1
    dbscan_results_grid.append({'eps': eps, 'n_clusters': n_clusters,
                                 'noise_pct': noise_pct, 'silhouette': sil})
    print(f"  DBSCAN eps={eps}: clusters={n_clusters}, noise={noise_pct:.1f}%, sil={sil:.4f}")

best_eps = max((r for r in dbscan_results_grid if r['n_clusters'] >= 2),
               key=lambda r: r['silhouette'], default={'eps': 1.5})['eps']
print(f"  Best DBSCAN eps: {best_eps}")

db_best = DBSCAN(eps=best_eps, min_samples=5)
db_labels = db_best.fit_predict(X_scaled)
n_db_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_db_noise    = list(db_labels).count(-1)
print(f"  DBSCAN best: {n_db_clusters} clusters, {n_db_noise} noise points ({n_db_noise/len(db_labels)*100:.1f}%)")

# Cluster visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# KMeans elbow
axes[0, 0].plot(k_range, inertias, 'b-o', linewidth=2, markersize=7)
axes[0, 0].set_xlabel('k'); axes[0, 0].set_ylabel('Inertia')
axes[0, 0].set_title('KMeans Elbow Curve', fontsize=12, fontweight='bold')

axes[0, 1].plot(k_range, silhouette_scores, 'r-o', linewidth=2, markersize=7)
axes[0, 1].axvline(best_k, color='green', linestyle='--', label=f'Best k={best_k}')
axes[0, 1].set_xlabel('k'); axes[0, 1].set_ylabel('Silhouette Score')
axes[0, 1].set_title('KMeans Silhouette Score', fontsize=12, fontweight='bold')
axes[0, 1].legend()

sc0 = axes[0, 2].scatter(X_2d[:, 0], X_2d[:, 1], c=km_labels, cmap='tab10', s=15, alpha=0.6)
axes[0, 2].set_title(f'KMeans Clusters (k={best_k}) — PCA 2D', fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('PC1'); axes[0, 2].set_ylabel('PC2')
plt.colorbar(sc0, ax=axes[0, 2])

axes[1, 0].plot(range(len(k_dist)), k_dist, color='#2196F3', linewidth=1)
axes[1, 0].axhline(best_eps, color='red', linestyle='--', linewidth=1.5, label=f'eps={best_eps}')
axes[1, 0].set_xlabel('Points'); axes[1, 0].set_ylabel('5th Nearest Neighbor Distance')
axes[1, 0].set_title('DBSCAN k-Distance Plot (Knee=eps)', fontsize=12, fontweight='bold')
axes[1, 0].legend()

eps_arr = [r['eps'] for r in dbscan_results_grid]
sil_arr = [r['silhouette'] for r in dbscan_results_grid]
axes[1, 1].plot(eps_arr, sil_arr, 'g-o', linewidth=2, markersize=7)
axes[1, 1].axvline(best_eps, color='red', linestyle='--', label=f'Best eps={best_eps}')
axes[1, 1].set_xlabel('eps'); axes[1, 1].set_ylabel('Silhouette Score')
axes[1, 1].set_title('DBSCAN eps Tuning', fontsize=12, fontweight='bold')
axes[1, 1].legend()

colors_db = plt.cm.tab10(np.linspace(0, 1, n_db_clusters + 1))
for lbl in set(db_labels):
    mask = db_labels == lbl
    color = 'lightgray' if lbl == -1 else colors_db[lbl]
    label_str = 'Noise' if lbl == -1 else f'Cluster {lbl}'
    axes[1, 2].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[color]*mask.sum(),
                       s=12 if lbl == -1 else 20, alpha=0.4 if lbl == -1 else 0.7,
                       label=label_str)
axes[1, 2].set_title(f'DBSCAN Clusters (eps={best_eps}) — PCA 2D', fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('PC1'); axes[1, 2].set_ylabel('PC2')
axes[1, 2].legend(markerscale=2, fontsize=8)
fig.suptitle('Clustering Analysis — KMeans vs DBSCAN', fontsize=14, fontweight='bold')
savefig(f"{CHARTS}/clustering/01_clustering_overview.png")

# Cluster characteristics
cluster_df = df.select_dtypes(include=[np.number]).copy()
cluster_df = cluster_df[['FEDRates','InflationConsumerPrice','UnemployemenrRate',
                          'GDP','RealGDP']].iloc[:len(km_labels)]
cluster_df['KMeans_Cluster'] = km_labels
cluster_df['DBSCAN_Cluster'] = db_labels

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cluster_df.groupby('KMeans_Cluster')[['FEDRates','InflationConsumerPrice','UnemployemenrRate']].mean().plot.bar(ax=axes[0])
axes[0].set_title('KMeans Cluster Profiles', fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=0)

dbscan_valid = cluster_df[cluster_df['DBSCAN_Cluster'] != -1]
if len(dbscan_valid['DBSCAN_Cluster'].unique()) > 1:
    dbscan_valid.groupby('DBSCAN_Cluster')[['FEDRates','InflationConsumerPrice','UnemployemenrRate']].mean().plot.bar(ax=axes[1])
axes[1].set_title('DBSCAN Cluster Profiles', fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=0)
savefig(f"{CHARTS}/clustering/02_cluster_profiles.png")

clustering_results = {
    'kmeans': {'best_k': int(best_k), 'best_silhouette': float(max(silhouette_scores)),
               'inertia': float(km_best.inertia_)},
    'dbscan': {'best_eps': float(best_eps), 'n_clusters': int(n_db_clusters),
               'n_noise': int(n_db_noise), 'noise_pct': float(n_db_noise/len(db_labels)*100)},
    'dbscan_grid': dbscan_results_grid
}
with open(f"{RES}/clustering_results.json", 'w') as f:
    json.dump(clustering_results, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# 11. ASSOCIATION RULE MINING — APRIORI
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 11 — ASSOCIATION RULE MINING (APRIORI)")
print("="*70)

# Discretize economic indicators into bins
arm_df = df[['FEDRates','InflationConsumerPrice','UnemployemenrRate',
             'GDP','RealGDP']].copy()
arm_df = arm_df.iloc[:len(X_scaled)]  # align to processed set

def discretize(series, col_name):
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    def label(v):
        if v <= q25: return f'{col_name}_Low'
        elif v <= q75: return f'{col_name}_Mid'
        else: return f'{col_name}_High'
    return series.apply(label)

arm_transactions = []
for idx in arm_df.index:
    row = []
    for col in arm_df.columns:
        row.append(discretize(arm_df[col], col[:8]).loc[idx])
    arm_transactions.append(row)

te = TransactionEncoder()
te_array = te.fit_transform(arm_transactions)
te_df = pd.DataFrame(te_array, columns=te.columns_)

print(f"Transaction matrix shape: {te_df.shape}")
freq_items = apriori(te_df, min_support=0.3, use_colnames=True)
print(f"Frequent itemsets (support >= 0.3): {len(freq_items)}")

rules = association_rules(freq_items, metric='confidence', min_threshold=0.6)
rules = rules.sort_values('lift', ascending=False)
print(f"Association rules (confidence >= 0.6): {len(rules)}")
print("\nTop 10 rules by lift:")
print(rules[['antecedents','consequents','support','confidence','lift']].head(10).to_string())

rules_serializable = rules.copy()
rules_serializable['antecedents'] = rules_serializable['antecedents'].apply(lambda x: list(x))
rules_serializable['consequents'] = rules_serializable['consequents'].apply(lambda x: list(x))
rules_serializable.head(100).to_json(f"{RES}/top_arm_rules.json", orient='records', indent=2)

# ARM Charts
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].scatter(rules['support'], rules['confidence'],
                   c=rules['lift'], cmap='RdYlGn', s=30, alpha=0.6)
axes[0, 0].set_xlabel('Support'); axes[0, 0].set_ylabel('Confidence')
axes[0, 0].set_title('Support vs Confidence (color=Lift)', fontsize=12, fontweight='bold')
plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), ax=axes[0, 0], label='Lift')

axes[0, 1].scatter(rules['support'], rules['lift'],
                   c=rules['confidence'], cmap='Blues', s=30, alpha=0.6)
axes[0, 1].set_xlabel('Support'); axes[0, 1].set_ylabel('Lift')
axes[0, 1].set_title('Support vs Lift (color=Confidence)', fontsize=12, fontweight='bold')

top_rules = rules.head(15)
rule_labels = [f"{list(r.antecedents)[0][:12]} => {list(r.consequents)[0][:12]}"
               for _, r in top_rules.iterrows()]
axes[1, 0].barh(range(len(top_rules)), top_rules['lift'].values, color=PALETTE[:len(top_rules)])
axes[1, 0].set_yticks(range(len(top_rules)))
axes[1, 0].set_yticklabels(rule_labels, fontsize=7)
axes[1, 0].set_xlabel('Lift'); axes[1, 0].set_title('Top 15 Rules by Lift', fontsize=12, fontweight='bold')

axes[1, 1].barh(range(len(top_rules)), top_rules['confidence'].values, color=PALETTE[:len(top_rules)])
axes[1, 1].set_yticks(range(len(top_rules)))
axes[1, 1].set_yticklabels(rule_labels, fontsize=7)
axes[1, 1].set_xlabel('Confidence'); axes[1, 1].set_title('Top 15 Rules by Confidence', fontsize=12, fontweight='bold')

fig.suptitle('Association Rule Mining — Apriori Results', fontsize=14, fontweight='bold')
savefig(f"{CHARTS}/arm/01_arm_overview.png")

# ARM heatmap of top rules
if len(rules) >= 10:
    top20 = rules.head(20).copy()
    pivot_data = pd.DataFrame({
        'Rule': [f"R{i+1}" for i in range(len(top20))],
        'Lift': top20['lift'].values,
        'Confidence': top20['confidence'].values,
        'Support': top20['support'].values,
    }).set_index('Rule')
    fig, ax = plt.subplots(figsize=(8, 8))
    norm_pivot = (pivot_data - pivot_data.min()) / (pivot_data.max() - pivot_data.min() + 1e-9)
    sns.heatmap(norm_pivot, annot=pivot_data.round(3), fmt='.3f', cmap='YlOrRd',
                ax=ax, linewidths=0.5)
    ax.set_title('Top 20 Association Rules — Metrics Heatmap', fontsize=12, fontweight='bold')
    savefig(f"{CHARTS}/arm/02_arm_heatmap.png")

arm_results = {
    'n_frequent_itemsets': int(len(freq_items)),
    'n_rules': int(len(rules)),
    'top_lift': float(rules['lift'].max()),
    'top_confidence': float(rules['confidence'].max()),
    'top_support': float(rules['support'].max()),
}
with open(f"{RES}/arm_results.json", 'w') as f:
    json.dump(arm_results, f, indent=2)

# ══════════════════════════════════════════════════════════════════════════════
# 12. OVERALL MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print(" STEP 12 — OVERALL MODEL COMPARISON")
print("="*70)

# Regression summary table
print("\nRegression Summary:")
reg_summary = pd.DataFrame(reg_results).T[['RMSE','MAE','R2']]
print(reg_summary.sort_values('R2', ascending=False).to_string())

print("\nClassification Summary:")
cls_summary = pd.DataFrame(cls_results).T[['Accuracy','F1','Precision','Recall']]
print(cls_summary.sort_values('Accuracy', ascending=False).to_string())

# ── Master comparison chart ──────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
reg_sorted = reg_summary.sort_values('R2', ascending=True)
colors_rs = [PALETTE[i % len(PALETTE)] for i in range(len(reg_sorted))]
ax1.barh(reg_sorted.index, reg_sorted['R2'], color=colors_rs)
ax1.set_xlabel('R² Score'); ax1.set_title('Regression — R² Score', fontsize=12, fontweight='bold')
ax1.axvline(0.9, color='green', linestyle='--', alpha=0.7, label='0.9')
ax1.legend()
for i, v in enumerate(reg_sorted['R2']):
    ax1.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

ax2 = fig.add_subplot(gs[0, 1])
reg_sorted2 = reg_summary.sort_values('RMSE', ascending=False)
ax2.barh(reg_sorted2.index, reg_sorted2['RMSE'],
         color=[PALETTE[i % len(PALETTE)] for i in range(len(reg_sorted2))])
ax2.set_xlabel('RMSE'); ax2.set_title('Regression — RMSE (lower=better)', fontsize=12, fontweight='bold')
for i, v in enumerate(reg_sorted2['RMSE']):
    ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

ax3 = fig.add_subplot(gs[1, 0])
cls_sorted = cls_summary.sort_values('Accuracy', ascending=True)
ax3.barh(cls_sorted.index, cls_sorted['Accuracy'],
         color=[PALETTE[i % len(PALETTE)] for i in range(len(cls_sorted))])
ax3.set_xlabel('Accuracy'); ax3.set_title('Classification — Accuracy', fontsize=12, fontweight='bold')
ax3.set_xlim(0, 1.05)
for i, v in enumerate(cls_sorted['Accuracy']):
    ax3.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

ax4 = fig.add_subplot(gs[1, 1])
cls_sorted2 = cls_summary.sort_values('F1', ascending=True)
ax4.barh(cls_sorted2.index, cls_sorted2['F1'],
         color=[PALETTE[i % len(PALETTE)] for i in range(len(cls_sorted2))])
ax4.set_xlabel('F1 Score'); ax4.set_title('Classification — Weighted F1', fontsize=12, fontweight='bold')
ax4.set_xlim(0, 1.05)
for i, v in enumerate(cls_sorted2['F1']):
    ax4.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

fig.suptitle('Complete Model Performance Comparison', fontsize=16, fontweight='bold')
savefig(f"{CHARTS}/comparison/master_comparison.png")

# Cross-validation comparison chart
fig, ax = plt.subplots(figsize=(14, 6))
all_cv_models = list(reg_names) + list(cls_names)
all_cv_scores = [reg_results[m]['cv']['mean'] for m in reg_names] + \
                [cls_results[m]['cv']['mean'] for m in cls_names]
all_cv_stds   = [reg_results[m]['cv']['std'] for m in reg_names] + \
                [cls_results[m]['cv']['std'] for m in cls_names]
all_types     = ['Regression']*len(reg_names) + ['Classification']*len(cls_names)
colors_cv = ['#2196F3' if t == 'Regression' else '#F44336' for t in all_types]
bars_cv = ax.bar(all_cv_models, all_cv_scores, color=colors_cv,
                 yerr=all_cv_stds, capsize=4)
ax.set_xticklabels(all_cv_models, rotation=40, ha='right')
ax.set_ylabel('5-Fold CV Score')
ax.set_title('Cross-Validation Scores — All Models\n(Blue=Regression R², Red=Classification Accuracy)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.1)
for bar, v in zip(bars_cv, all_cv_scores):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.015,
            f'{v:.3f}', ha='center', fontsize=8)
handles = [plt.Rectangle((0,0),1,1,color='#2196F3'), plt.Rectangle((0,0),1,1,color='#F44336')]
ax.legend(handles, ['Regression (R²)', 'Classification (Accuracy)'])
savefig(f"{CHARTS}/comparison/cv_all_models.png")

# Overfitting/underfitting summary
oc_data = {}
for m in reg_names:
    if 'learning_curve' in reg_results[m]:
        oc_data[f'{m} (Reg)'] = reg_results[m]['learning_curve']
for m in cls_names:
    if 'learning_curve' in cls_results[m]:
        oc_data[f'{m} (Cls)'] = cls_results[m]['learning_curve']

fig, ax = plt.subplots(figsize=(15, 6))
labels_oc = list(oc_data.keys())
train_oc  = [oc_data[k]['train_score'] for k in labels_oc]
val_oc    = [oc_data[k]['val_score'] for k in labels_oc]
x_oc = np.arange(len(labels_oc))
bars_tr = ax.bar(x_oc - 0.2, train_oc, 0.35, label='Train Score', color='#2196F3', alpha=0.8)
bars_vl = ax.bar(x_oc + 0.2, val_oc, 0.35, label='Val Score', color='#F44336', alpha=0.8)
ax.set_xticks(x_oc)
ax.set_xticklabels(labels_oc, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Score')
ax.set_title('Overfitting/Underfitting Analysis — Train vs Validation Score',
             fontsize=13, fontweight='bold')
ax.legend()
ax.set_ylim(0, 1.1)

for bar_tr, bar_vl, t, v in zip(bars_tr, bars_vl, train_oc, val_oc):
    gap = t - v
    color = '#FF9800' if gap > 0.1 else ('#4CAF50' if v >= 0.6 else '#F44336')
    ax.text(bar_tr.get_x() + bar_tr.get_width()/2,
            max(t, v) + 0.02, f'Gap:{gap:.2f}',
            ha='center', fontsize=7, color=color, fontweight='bold')
savefig(f"{CHARTS}/comparison/overfitting_analysis.png")

# Save final summary
summary = {
    'regression': reg_results,
    'classification': cls_results,
    'clustering': clustering_results,
    'arm': arm_results
}
with open(f"{RES}/complete_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print("\nPart 2 complete. All ML models trained and evaluated.")
print(f"All charts saved to: {CHARTS}")
print(f"All results saved to: {RES}")
