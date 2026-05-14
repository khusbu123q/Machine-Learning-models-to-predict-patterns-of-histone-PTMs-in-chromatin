import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, f1_score)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")
import os

TSV_FILE = "final_results_bmal1.tsv"
OUT_DIR  = "classifier_output"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  CYCLIC GENE CLASSIFIER  (Imbalance-corrected)")
print("  Random Forest | SVM | Logistic Regression | XGBoost")
print("=" * 60)

print("\n[1/6] Loading data...")
df = pd.read_csv(TSV_FILE, sep="\t")
print(f"  Rows  : {len(df):,}")
print(f"  Genes : {df['gene_id'].nunique():,}")

print("\n[2/6] Reshaping — one row per gene...")
gene_matrix = df.pivot_table(
    index=["gene_id", "is_BMAL1_target"],
    columns="position",
    values="avg_score",
    aggfunc="mean"
).reset_index()

gene_matrix.columns.name = None
feature_cols = sorted([c for c in gene_matrix.columns if isinstance(c, (int, float))])

X         = gene_matrix[feature_cols].fillna(0).values
y         = gene_matrix["is_BMAL1_target"].values
gene_ids  = gene_matrix["gene_id"].values
positions = np.array(feature_cols)

n_cyclic    = int(y.sum())
n_noncyclic = int((1 - y).sum())
imb_ratio   = n_noncyclic / n_cyclic
scale_pos   = int(imb_ratio)   # for XGBoost

print(f"  Samples        : {len(X):,}")
print(f"  Features       : {len(feature_cols)} bins")
print(f"  Cyclic (1)     : {n_cyclic:,}  ({y.mean()*100:.1f}%)")
print(f"  Non-cyclic (0) : {n_noncyclic:,}  ({(1-y.mean())*100:.1f}%)")
print(f"  Imbalance ratio: {imb_ratio:.1f}:1  → SMOTE + class_weight + scale_pos_weight")

print("\n[3/6] Defining models...")

models = {
    "Random Forest": ImbPipeline([
        ("smote",  SMOTE(random_state=42, k_neighbors=5)),
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(
            n_estimators=500, max_depth=15, min_samples_leaf=2,
            max_features=0.7, class_weight="balanced",
            random_state=42, n_jobs=-1))
    ]),
    "SVM (RBF)": ImbPipeline([
        ("smote",  SMOTE(random_state=42, k_neighbors=5)),
        ("scaler", StandardScaler()),
        ("model",  SVC(
            kernel="rbf", C=10.0, gamma="scale",
            class_weight="balanced", probability=True, random_state=42))
    ]),
    "Logistic Regression": ImbPipeline([
        ("smote",  SMOTE(random_state=42, k_neighbors=5)),
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=1000,
            solver="lbfgs", random_state=42, n_jobs=-1))
    ]),
    "XGBoost": ImbPipeline([
        ("smote",  SMOTE(random_state=42, k_neighbors=5)),
        ("scaler", StandardScaler()),
        ("model",  XGBClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos,   # handles imbalance natively
            eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0))
    ])
}

colors = {
    "Random Forest"      : "#E74C3C",
    "SVM (RBF)"          : "#3498DB",
    "Logistic Regression": "#2ECC71",
    "XGBoost"            : "#F39C12",
    "Ensemble"           : "#9B59B6"
}

print("\n[4/6] Running 5-fold stratified cross-validation...")
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    print(f"\n  {name}:")
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc",           n_jobs=-1)
    f1_scores  = cross_val_score(model, X, y, cv=cv, scoring="f1",                n_jobs=-1)
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy",          n_jobs=-1)
    pr_scores  = cross_val_score(model, X, y, cv=cv, scoring="average_precision", n_jobs=-1)
    results[name] = {"AUC": auc_scores, "F1": f1_scores,
                     "Accuracy": acc_scores, "Avg Prec": pr_scores}
    print(f"    ROC-AUC  : {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    print(f"    F1       : {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
    print(f"    Accuracy : {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")
    print(f"    Avg Prec : {pr_scores.mean():.3f} ± {pr_scores.std():.3f}")

print("\n[5/6] Fitting final models + optimising decision threshold...")
fitted_models      = {}
all_probs          = {}
all_preds          = {}
optimal_thresholds = {}

for name, model in models.items():
    model.fit(X, y)
    fitted_models[name] = model
    all_probs[name]     = model.predict_proba(X)[:, 1]

    # Find threshold that maximises F1
    prec, rec, thresholds = precision_recall_curve(y, all_probs[name])
    f1_vals   = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx  = np.argmax(f1_vals)
    opt_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    optimal_thresholds[name] = opt_thresh
    all_preds[name] = (all_probs[name] >= opt_thresh).astype(int)
    print(f"  {name} fitted | optimal threshold = {opt_thresh:.3f}  "
          f"(F1={f1_vals[best_idx]:.3f})")

# Ensemble
ensemble_probs = np.mean([all_probs[n] for n in models], axis=0)
prec_e, rec_e, thresh_e = precision_recall_curve(y, ensemble_probs)
f1_e       = 2 * prec_e * rec_e / (prec_e + rec_e + 1e-8)
best_e     = np.argmax(f1_e)
ens_thresh = thresh_e[best_e] if best_e < len(thresh_e) else 0.5
ensemble_preds = (ensemble_probs >= ens_thresh).astype(int)
print(f"  Ensemble | optimal threshold = {ens_thresh:.3f}  "
      f"(F1={f1_e[best_e]:.3f})")

# Save predictions
pred_df = pd.DataFrame({"gene_id": gene_ids, "true_label": y})
for name in models:
    safe = name.lower().replace(" ", "_").replace("(","").replace(")","")
    pred_df[f"prob_{safe}"] = all_probs[name]
    pred_df[f"pred_{safe}"] = all_preds[name]
pred_df["prob_ensemble"] = ensemble_probs
pred_df["pred_ensemble"] = ensemble_preds
pred_df["cyclic_label"]  = pred_df["pred_ensemble"].map({1: "Cyclic", 0: "Non-Cyclic"})
pred_df.to_csv(f"{OUT_DIR}/gene_predictions.csv", index=False)
print(f"  Saved: {OUT_DIR}/gene_predictions.csv")

print("\n[6/6] Generating plots...")

fig = plt.figure(figsize=(22, 24))
fig.suptitle("Cyclic Gene Classifier — Performance Report\n"
             "(SMOTE + class_weight + optimal threshold)",
             fontsize=15, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.50, wspace=0.35)

# --- Row 0: CV bar chart + ROC ---
ax1 = fig.add_subplot(gs[0, :2])
metrics   = ["AUC", "F1", "Accuracy", "Avg Prec"]
x         = np.arange(len(metrics))
n_models  = len(models)
bar_width = 0.18
offsets   = np.linspace(-(n_models-1)/2 * bar_width,
                         (n_models-1)/2 * bar_width, n_models)
for i, (name, color) in enumerate(zip(models.keys(), list(colors.values())[:n_models])):
    means = [results[name][m].mean() for m in metrics]
    stds  = [results[name][m].std()  for m in metrics]
    ax1.bar(x + offsets[i], means, bar_width, yerr=stds,
            label=name, color=color, alpha=0.85, capsize=4)
ax1.set_xticks(x); ax1.set_xticklabels(metrics)
ax1.set_ylabel("Score")
ax1.set_title("5-Fold Cross-Validation Performance (SMOTE + balanced weights)")
ax1.set_ylim(0, 1.2); ax1.legend(fontsize=8)
ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.4)
ax1.grid(axis="y", alpha=0.3)

ax2 = fig.add_subplot(gs[0, 2])
for name, color in colors.items():
    probs   = ensemble_probs if name == "Ensemble" else all_probs.get(name)
    if probs is None:
        continue
    fpr, tpr, _ = roc_curve(y, probs)
    auc_val = roc_auc_score(y, probs)
    lw = 2.5 if name == "Ensemble" else 1.8
    ax2.plot(fpr, tpr, color=color, lw=lw, label=f"{name} ({auc_val:.3f})")
ax2.plot([0,1],[0,1],"k--",alpha=0.4)
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.set_title("ROC Curves"); ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

ax3 = fig.add_subplot(gs[1, 0])
baseline = y.mean()
for name, color in colors.items():
    probs = ensemble_probs if name == "Ensemble" else all_probs.get(name)
    if probs is None:
        continue
    prec, rec, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)
    lw = 2.5 if name == "Ensemble" else 1.8
    ax3.plot(rec, prec, color=color, lw=lw, label=f"{name} (AP={ap:.3f})")
ax3.axhline(baseline, color="gray", linestyle="--", alpha=0.5,
            label=f"Baseline ({baseline:.3f})")
ax3.set_xlabel("Recall"); ax3.set_ylabel("Precision")
ax3.set_title("Precision-Recall Curves"); ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

for col_idx, name in enumerate(["Random Forest", "SVM (RBF)"]):
    ax = fig.add_subplot(gs[1, col_idx + 1])
    cm = confusion_matrix(y, all_preds[name])
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-Cyc","Cyclic"], fontsize=8)
    ax.set_yticklabels(["Non-Cyc","Cyclic"], fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name}\nConfusion Matrix (t={optimal_thresholds[name]:.2f})", fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=11, fontweight="bold")

for col_idx, name in enumerate(["Logistic Regression", "XGBoost", "Ensemble"]):
    ax = fig.add_subplot(gs[2, col_idx])
    preds = ensemble_preds if name == "Ensemble" else all_preds[name]
    cm    = confusion_matrix(y, preds)
    ax.imshow(cm, cmap="Purples" if name == "Ensemble" else "Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-Cyc","Cyclic"], fontsize=8)
    ax.set_yticklabels(["Non-Cyc","Cyclic"], fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    thresh_label = f"t={ens_thresh:.2f}" if name == "Ensemble" else f"t={optimal_thresholds[name]:.2f}"
    ax.set_title(f"{name}\nConfusion Matrix ({thresh_label})", fontsize=8)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black",
                    fontsize=11, fontweight="bold")

ax7 = fig.add_subplot(gs[3, 0])
rf_model    = fitted_models["Random Forest"].named_steps["model"]
importances = rf_model.feature_importances_
top20_idx   = np.argsort(importances)[-20:]
top20_pos   = positions[top20_idx]
top20_imp   = importances[top20_idx]
bar_col     = ["#E74C3C" if p >= 0 else "#3498DB" for p in top20_pos]
ax7.barh(range(20), top20_imp, color=bar_col, alpha=0.85)
ax7.set_yticks(range(20))
ax7.set_yticklabels([f"{int(p)}bp" for p in top20_pos], fontsize=7)
ax7.set_xlabel("Importance")
ax7.set_title("Top 20 Bins — Random Forest\nRed=downstream  Blue=upstream", fontsize=8)
ax7.grid(axis="x", alpha=0.3)

ax8 = fig.add_subplot(gs[3, 1])
xgb_model    = fitted_models["XGBoost"].named_steps["model"]
xgb_imp      = xgb_model.feature_importances_
top20_xgb    = np.argsort(xgb_imp)[-20:]
top20_pos_xgb = positions[top20_xgb]
top20_imp_xgb = xgb_imp[top20_xgb]
bar_col_xgb  = ["#F39C12" if p >= 0 else "#8E44AD" for p in top20_pos_xgb]
ax8.barh(range(20), top20_imp_xgb, color=bar_col_xgb, alpha=0.85)
ax8.set_yticks(range(20))
ax8.set_yticklabels([f"{int(p)}bp" for p in top20_pos_xgb], fontsize=7)
ax8.set_xlabel("Importance")
ax8.set_title("Top 20 Bins — XGBoost\nOrange=downstream  Purple=upstream", fontsize=8)
ax8.grid(axis="x", alpha=0.3)

ax9 = fig.add_subplot(gs[3, 2])
ax9.hist(pred_df[pred_df["true_label"]==0]["prob_ensemble"],
         bins=50, alpha=0.6, color="#3498DB", label="Non-Cyclic", density=True)
ax9.hist(pred_df[pred_df["true_label"]==1]["prob_ensemble"],
         bins=50, alpha=0.6, color="#E74C3C", label="Cyclic",     density=True)
ax9.axvline(ens_thresh, color="black", linestyle="--",
            label=f"Optimal t={ens_thresh:.2f}")
ax9.axvline(0.5, color="gray", linestyle=":", alpha=0.5, label="Default=0.5")
ax9.set_xlabel("Ensemble Probability"); ax9.set_ylabel("Density")
ax9.set_title("Prediction Probability Distribution\n(Ensemble)")
ax9.legend(fontsize=8); ax9.grid(alpha=0.3)

plt.savefig(f"{OUT_DIR}/classifier_report.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {OUT_DIR}/classifier_report.png")

print("\n" + "=" * 70)
print("  FINAL CLASSIFICATION REPORT (Ensemble — optimal threshold)")
print("=" * 70)
print(classification_report(y, ensemble_preds,
      target_names=["Non-Cyclic", "Cyclic"]))

print(f"  {'Model':<22} {'AUC':>8} {'CV F1':>8} {'Accuracy':>10} "
      f"{'Avg Prec':>10} {'Threshold':>10}")
print("  " + "-" * 72)
for name in models:
    r = results[name]
    print(f"  {name:<22} {r['AUC'].mean():>8.3f} {r['F1'].mean():>8.3f} "
          f"{r['Accuracy'].mean():>10.3f} {r['Avg Prec'].mean():>10.3f} "
          f"{optimal_thresholds[name]:>10.3f}")
ens_auc = roc_auc_score(y, ensemble_probs)
ens_f1  = f1_score(y, ensemble_preds)
print(f"  {'Ensemble':<22} {ens_auc:>8.3f} {ens_f1:>8.3f} "
      f"{'':>10} {'':>10} {ens_thresh:>10.3f}")

print(f"\n  Files saved to: {OUT_DIR}/")
print(f"    gene_predictions.csv  — per-gene predictions + probabilities")
print(f"    classifier_report.png — full performance plots")
print("=" * 70)
