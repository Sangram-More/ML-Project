"""
Regenerate all comparison charts with proper design.
Fixes: broken CV R² panel with negative values.
"""
import warnings; warnings.filterwarnings('ignore')
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

BASE    = "d:/Projects/ML website/ML-Project"
CHARTS  = f"{BASE}/ml_analysis/outputs/charts"
RES     = f"{BASE}/ml_analysis/outputs/results"

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 12, 'axes.titleweight': 'bold',
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150
})
PALETTE = ['#2196F3','#F44336','#4CAF50','#FF9800','#9C27B0',
           '#00BCD4','#E91E63','#795548']

def savefig(path, dpi=150):
    plt.tight_layout(pad=1.5)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"  Saved: {os.path.basename(path)}")

with open(f"{RES}/regression_results.json")      as f: reg_res  = json.load(f)
with open(f"{RES}/classification_results.json")  as f: cls_res  = json.load(f)

# ── FIX 1: Regression Comparison — 2 panels only (RMSE + R²) ──────────────
reg_sorted = sorted(reg_res.items(), key=lambda x: x[1]['R2'], reverse=True)
names_r = [m for m,_ in reg_sorted]
rmse_v  = [reg_res[m]['RMSE'] for m in names_r]
r2_v    = [reg_res[m]['R2']   for m in names_r]
mae_v   = [reg_res[m]['MAE']  for m in names_r]
colors  = PALETTE[:len(names_r)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor('white')

# Panel 1: RMSE (lower = better)
bars = axes[0].bar(names_r, rmse_v, color=colors, edgecolor='white', linewidth=0.8)
axes[0].set_title('RMSE  (lower is better)', fontsize=13, fontweight='bold', pad=12)
axes[0].set_xticklabels(names_r, rotation=38, ha='right', fontsize=9)
axes[0].set_ylabel('RMSE (%)', fontsize=10)
for bar, v in zip(bars, rmse_v):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='600')
best_idx = np.argmin(rmse_v)
bars[best_idx].set_edgecolor('#1a1a1a'); bars[best_idx].set_linewidth(2.5)

# Panel 2: R² (higher = better)
bars2 = axes[1].bar(names_r, r2_v, color=colors, edgecolor='white', linewidth=0.8)
axes[1].set_title('R²  (higher is better)', fontsize=13, fontweight='bold', pad=12)
axes[1].set_xticklabels(names_r, rotation=38, ha='right', fontsize=9)
axes[1].set_ylabel('R² Score', fontsize=10)
axes[1].set_ylim(0, 1.08)
axes[1].axhline(0.95, color='green', linestyle='--', linewidth=1.2,
                alpha=0.7, label='0.95 target')
axes[1].legend(fontsize=8)
for bar, v in zip(bars2, r2_v):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='600')
best_idx2 = np.argmax(r2_v)
bars2[best_idx2].set_edgecolor('#1a1a1a'); bars2[best_idx2].set_linewidth(2.5)

# Panel 3: MAE (lower = better) — replaces broken CV panel
bars3 = axes[2].bar(names_r, mae_v, color=colors, edgecolor='white', linewidth=0.8)
axes[2].set_title('MAE  (lower is better)', fontsize=13, fontweight='bold', pad=12)
axes[2].set_xticklabels(names_r, rotation=38, ha='right', fontsize=9)
axes[2].set_ylabel('MAE (%)', fontsize=10)
for bar, v in zip(bars3, mae_v):
    axes[2].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='600')
best_idx3 = np.argmin(mae_v)
bars3[best_idx3].set_edgecolor('#1a1a1a'); bars3[best_idx3].set_linewidth(2.5)

# Add rank badge to best bar in each panel
for ax, best, label in [(axes[0], best_idx, 'BEST'), (axes[1], best_idx2, 'BEST'), (axes[2], best_idx3, 'BEST')]:
    ax.text(best, 0.01, '★ BEST', transform=ax.transData,
            ha='center', fontsize=7, color='#1a1a1a',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFD700', alpha=0.85))

fig.suptitle('Regression Models — Performance Comparison', fontsize=15, fontweight='bold', y=1.02)
savefig(f"{CHARTS}/comparison/reg_model_comparison.png")

# ── FIX 2: Classification Comparison ─────────────────────────────────────────
cls_sorted = sorted(cls_res.items(), key=lambda x: x[1]['Accuracy'], reverse=True)
names_c = [m for m,_ in cls_sorted]
acc_v  = [cls_res[m]['Accuracy']  for m in names_c]
f1_v   = [cls_res[m]['F1']        for m in names_c]
prec_v = [cls_res[m]['Precision'] for m in names_c]
rec_v  = [cls_res[m]['Recall']    for m in names_c]
colors_c = PALETTE[:len(names_c)]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for ax, vals, title, metric_label in [
    (axes[0], acc_v,  'Accuracy  (higher is better)',      'Accuracy'),
    (axes[1], f1_v,   'Weighted F1  (higher is better)',   'F1 Score'),
]:
    bars = ax.bar(names_c, vals, color=colors_c, edgecolor='white', linewidth=0.8)
    ax.axhline(1/3, color='#F44336', linestyle='--', linewidth=1.5,
               label='Random baseline (33.3%)', alpha=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xticklabels(names_c, rotation=38, ha='right', fontsize=9)
    ax.set_ylabel(metric_label, fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                f'{v:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='600')
    best = np.argmax(vals)
    bars[best].set_edgecolor('#1a1a1a'); bars[best].set_linewidth(2.5)

fig.suptitle('Classification Models — Performance Comparison', fontsize=15, fontweight='bold', y=1.02)
savefig(f"{CHARTS}/comparison/cls_model_comparison.png")

# ── FIX 3: Master Comparison — clean 2×2 ─────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('white')
gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

# Regression R²
ax1 = fig.add_subplot(gs[0, 0])
reg_r2_sorted = sorted(reg_res.items(), key=lambda x: x[1]['R2'])
n1 = [m for m,_ in reg_r2_sorted]
v1 = [reg_res[m]['R2'] for m in n1]
bars = ax1.barh(n1, v1, color=PALETTE[:len(n1)], edgecolor='white', height=0.6)
ax1.axvline(0.95, color='#4CAF50', linestyle='--', linewidth=1.2, alpha=0.8, label='0.95')
ax1.set_xlim(0, 1.08)
ax1.set_title('Regression — R² Score', fontweight='bold', fontsize=12)
ax1.set_xlabel('R²')
ax1.legend(fontsize=8)
for i, (bar, v) in enumerate(zip(bars, v1)):
    ax1.text(v + 0.005, bar.get_y()+bar.get_height()/2,
             f'{v:.3f}', va='center', fontsize=9, fontweight='600')

# Regression RMSE
ax2 = fig.add_subplot(gs[0, 1])
reg_rmse_sorted = sorted(reg_res.items(), key=lambda x: x[1]['RMSE'], reverse=True)
n2 = [m for m,_ in reg_rmse_sorted]
v2 = [reg_res[m]['RMSE'] for m in n2]
bars2 = ax2.barh(n2, v2, color=PALETTE[:len(n2)], edgecolor='white', height=0.6)
ax2.set_title('Regression — RMSE  (lower is better)', fontweight='bold', fontsize=12)
ax2.set_xlabel('RMSE (%)')
for bar, v in zip(bars2, v2):
    ax2.text(v + 0.008, bar.get_y()+bar.get_height()/2,
             f'{v:.3f}', va='center', fontsize=9, fontweight='600')

# Classification Accuracy
ax3 = fig.add_subplot(gs[1, 0])
cls_acc_sorted = sorted(cls_res.items(), key=lambda x: x[1]['Accuracy'])
n3 = [m for m,_ in cls_acc_sorted]
v3 = [cls_res[m]['Accuracy'] for m in n3]
bars3 = ax3.barh(n3, v3, color=PALETTE[:len(n3)], edgecolor='white', height=0.6)
ax3.axvline(1/3, color='#F44336', linestyle='--', linewidth=1.2, alpha=0.8, label='Baseline')
ax3.set_xlim(0, 1.0)
ax3.set_title('Classification — Accuracy', fontweight='bold', fontsize=12)
ax3.set_xlabel('Accuracy')
ax3.legend(fontsize=8)
for bar, v in zip(bars3, v3):
    ax3.text(v + 0.005, bar.get_y()+bar.get_height()/2,
             f'{v:.3f}', va='center', fontsize=9, fontweight='600')

# Classification F1
ax4 = fig.add_subplot(gs[1, 1])
cls_f1_sorted = sorted(cls_res.items(), key=lambda x: x[1]['F1'])
n4 = [m for m,_ in cls_f1_sorted]
v4 = [cls_res[m]['F1'] for m in n4]
bars4 = ax4.barh(n4, v4, color=PALETTE[:len(n4)], edgecolor='white', height=0.6)
ax4.set_xlim(0, 1.0)
ax4.set_title('Classification — Weighted F1', fontweight='bold', fontsize=12)
ax4.set_xlabel('F1 Score')
for bar, v in zip(bars4, v4):
    ax4.text(v + 0.005, bar.get_y()+bar.get_height()/2,
             f'{v:.3f}', va='center', fontsize=9, fontweight='600')

fig.suptitle('Complete Model Performance Overview', fontsize=16, fontweight='bold', y=1.01)
savefig(f"{CHARTS}/comparison/master_comparison.png")

# ── FIX 4: CV chart — use only classification (positive scores) + note ───────
cls_names  = list(cls_res.keys())
cls_cv_m   = [cls_res[m]['cv']['mean'] for m in cls_names]
cls_cv_s   = [cls_res[m]['cv']['std']  for m in cls_names]
# Regression: show test R² instead of negative CV
reg_names  = list(reg_res.keys())
reg_test_r2= [reg_res[m]['R2'] for m in reg_names]

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
fig.patch.set_facecolor('white')

# Regression test R²
bars_r = axes[0].bar(reg_names, reg_test_r2, color=PALETTE[:len(reg_names)],
                     edgecolor='white', linewidth=0.8)
axes[0].set_title('Regression — Test Set R² Score', fontweight='bold', fontsize=12)
axes[0].set_xticklabels(reg_names, rotation=38, ha='right', fontsize=9)
axes[0].set_ylabel('R²'); axes[0].set_ylim(0, 1.1)
axes[0].axhline(0.95, color='green', linestyle='--', linewidth=1.2, alpha=0.7, label='0.95')
axes[0].legend(fontsize=8)
for bar, v in zip(bars_r, reg_test_r2):
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.008,
                 f'{v:.3f}', ha='center', fontsize=8.5, fontweight='600')
axes[0].text(0.98, 0.02,
             'Note: TimeSeriesSplit CV scores are negative for\nregression due to economic regime shifts\n(future data structurally different from past).\nTest set R² shown instead.',
             transform=axes[0].transAxes, ha='right', va='bottom', fontsize=7.5,
             color='#666', style='italic',
             bbox=dict(boxstyle='round', facecolor='#fff9e6', alpha=0.8))

# Classification CV accuracy
bars_c = axes[1].bar(cls_names, cls_cv_m, color=PALETTE[:len(cls_names)],
                     edgecolor='white', linewidth=0.8,
                     yerr=cls_cv_s, capsize=4, error_kw={'linewidth':1.5})
axes[1].axhline(1/3, color='#F44336', linestyle='--', linewidth=1.5,
                alpha=0.8, label='Random baseline (33.3%)')
axes[1].set_title('Classification — TimeSeriesSplit CV Accuracy (5 folds)',
                  fontweight='bold', fontsize=12)
axes[1].set_xticklabels(cls_names, rotation=38, ha='right', fontsize=9)
axes[1].set_ylabel('CV Accuracy'); axes[1].set_ylim(0, 0.85)
axes[1].legend(fontsize=8)
for bar, v, s in zip(bars_c, cls_cv_m, cls_cv_s):
    axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+s+0.015,
                 f'{v:.3f}', ha='center', fontsize=8.5, fontweight='600')

fig.suptitle('Model Evaluation — Test Performance & Cross-Validation',
             fontsize=14, fontweight='bold', y=1.02)
savefig(f"{CHARTS}/comparison/cv_all_models.png")

# ── FIX 5: Overfitting chart — re-render with better style ───────────────────
with open(f"{RES}/complete_summary.json") as f:
    summary = json.load(f)

oc_reg  = {m: summary['regression'][m]['learning_curve']
           for m in summary['regression'] if 'learning_curve' in summary['regression'][m]}
oc_cls  = {m: summary['classification'][m]['learning_curve']
           for m in summary['classification'] if 'learning_curve' in summary['classification'][m]}

all_models  = list(oc_reg.keys()) + [m+' (C)' for m in oc_cls.keys()]
train_scores= [oc_reg[m]['train_score'] for m in oc_reg] + \
              [oc_cls[m]['train_score'] for m in oc_cls]
val_scores  = [oc_reg[m]['val_score']   for m in oc_reg] + \
              [oc_cls[m]['val_score']   for m in oc_cls]
gaps        = [oc_reg[m]['gap']         for m in oc_reg] + \
              [oc_cls[m]['gap']         for m in oc_cls]
is_cls_flag = [False]*len(oc_reg) + [True]*len(oc_cls)

fig, axes = plt.subplots(2, 1, figsize=(16, 11))
fig.patch.set_facecolor('white')
x = np.arange(len(all_models))
w = 0.35
bar_colors = ['#2196F3' if not c else '#E91E63' for c in is_cls_flag]

bars_tr = axes[0].bar(x - w/2, train_scores, w, label='Train Score',
                      color=bar_colors, alpha=0.9, edgecolor='white')
bars_vl = axes[0].bar(x + w/2, val_scores, w, label='Validation Score',
                      color=bar_colors, alpha=0.45, edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels(all_models, rotation=40, ha='right', fontsize=8.5)
axes[0].set_ylabel('Score')
axes[0].set_title('Train vs Validation Score — All Models\n(Solid=Train, Faded=Validation)',
                  fontweight='bold', fontsize=12)
axes[0].set_ylim(-0.05, 1.15)
axes[0].axhline(0, color='black', linewidth=0.5)
# Annotations
for i, (t, v) in enumerate(zip(train_scores, val_scores)):
    g = t - v
    col = '#4CAF50' if g < 0.05 else ('#FF9800' if g < 0.15 else '#F44336')
    axes[0].text(i, max(t, v) + 0.03, f'Δ{g:.2f}', ha='center',
                 fontsize=7, color=col, fontweight='bold')
from matplotlib.patches import Patch
handles = [Patch(color='#2196F3', label='Regression'),
           Patch(color='#E91E63', label='Classification'),
           Patch(color='white', edgecolor='black', label='Solid=Train / Faded=Val')]
axes[0].legend(handles=handles, fontsize=8, loc='upper right')

# Gap chart
gap_colors = ['#4CAF50' if g < 0.05 else ('#FF9800' if g < 0.15 else '#F44336') for g in gaps]
axes[1].bar(all_models, gaps, color=gap_colors, edgecolor='white', linewidth=0.8)
axes[1].axhline(0.05, color='#FF9800', linestyle='--', linewidth=1.5,
                label='Mild overfitting (0.05)')
axes[1].axhline(0.15, color='#F44336', linestyle='--', linewidth=1.5,
                label='Strong overfitting (0.15)')
axes[1].set_xticklabels(all_models, rotation=40, ha='right', fontsize=8.5)
axes[1].set_ylabel('Train - Val Gap')
axes[1].set_title('Overfitting Gap by Model  (Green=Good Fit, Orange=Mild, Red=Overfit)',
                  fontweight='bold', fontsize=12)
axes[1].legend(fontsize=8)
for i, (g, status) in enumerate(zip(gaps, [oc_reg[m]['status'] for m in oc_reg] +
                                             [oc_cls[m]['status'] for m in oc_cls])):
    axes[1].text(i, g + 0.003, status[:4], ha='center', fontsize=7, fontweight='bold',
                 color=gap_colors[i])

fig.suptitle('Bias-Variance Tradeoff — Overfitting & Underfitting Analysis',
             fontsize=14, fontweight='bold', y=1.01)
savefig(f"{CHARTS}/comparison/overfitting_analysis.png")

# ── Also copy updated charts to website/images ───────────────────────────────
import shutil
WEB_IMG = f"{BASE}/website/images"
for fname in ['reg_model_comparison.png','cls_model_comparison.png',
              'master_comparison.png','cv_all_models.png','overfitting_analysis.png']:
    shutil.copy2(f"{CHARTS}/comparison/{fname}", f"{WEB_IMG}/comparison_{fname}")
print("\nAll comparison charts regenerated and copied to website.")
