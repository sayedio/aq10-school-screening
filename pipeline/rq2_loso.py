

import os, json, time, random, warnings
from datetime import datetime

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import chi2 as chi2_dist

np.random.seed(SEED)

# from google.colab import drive  # Colab-only; locally, set the AQ10_BASE environment variable instead
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix, ConfusionMatrixDisplay,
)

# ══════════════════════════════════════════════════════════════════════════════
# 0. PATHS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 78)
print("RQ2 — LOSO INSTITUTIONAL GENERALIZATION  |  MONO-CLASS SCHOOLS")
print("FINAL-MONO-10.1-RQ2")
print("=" * 78)

# drive.mount("/content/drive", force_remount=True)

BASE    = os.environ.get("AQ10_BASE", "/content/drive/MyDrive/Autism")
PROJECT = os.path.join(BASE, "aq_10_v2")

RQ1_MODELS_DIR = os.path.join(PROJECT, "rq1_baseline", "models")
RQ1_FLAG       = os.path.join(PROJECT, "logs", "rq1_done.flag")
BD_CLEAN_PATH  = os.path.join(PROJECT, "data", "processed", "bd_clean.csv")
BEST_CONFIG    = os.path.join(RQ1_MODELS_DIR, "best_model_config.json")

RQ2_ROOT = os.path.join(PROJECT, "rq2_loso")
DIRS = {
    "root": RQ2_ROOT, "tables": os.path.join(RQ2_ROOT, "tables"),
    "paper_ready": os.path.join(RQ2_ROOT, "tables", "paper_ready"),
    "figures": os.path.join(RQ2_ROOT, "figures"),
    "results": os.path.join(RQ2_ROOT, "results"),
    "predictions": os.path.join(RQ2_ROOT, "predictions"),
    "logs": os.path.join(RQ2_ROOT, "logs"),
    "config": os.path.join(RQ2_ROOT, "config"),
    "paper_tables": os.path.join(PROJECT, "paper_results", "rq2", "tables"),
    "paper_figures": os.path.join(PROJECT, "paper_results", "rq2", "figures"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS & HELPERS
# ══════════════════════════════════════════════════════════════════════════════

S0_COLS     = [f"A_Q{i}" for i in range(1, 11)]
THRESHOLD   = 0.50
N_BOOTSTRAP = 2000

METRIC_KEYS = [
    "AUROC", "AUPRC", "Accuracy", "Precision_PPV",
    "Sensitivity", "Specificity", "F1_Score", "Brier", "ECE",
]

LOCKED_PARAMS = {
    "C": 1.0, "kernel": "rbf", "gamma": "scale",
    "probability": True, "random_state": SEED,
}

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    n = len(y_true)
    if n == 0: return np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & ((y_prob <= hi) if i == n_bins - 1 else (y_prob < hi))
        if mask.sum() == 0: continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)

def metric_bundle(y_true, y_prob, y_pred):
    """Full metrics — requires both classes."""
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    y_pred = np.asarray(y_pred, int)
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in METRIC_KEYS} | {"TP":0,"TN":0,"FP":0,"FN":0}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "AUROC": float(roc_auc_score(y_true, y_prob)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision_PPV": float(precision_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn/(tn+fp)) if (tn+fp)>0 else 0.0,
        "F1_Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "Brier": float(brier_score_loss(y_true, y_prob)),
        "ECE": float(expected_calibration_error(y_true, y_prob)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

def per_school_mono_metrics(y_true, y_prob, y_pred):
    """
    Mono-class safe per-school metrics.
    ASD-only school  -> reports Sensitivity (flagging rate)
    Normal-only school -> reports Specificity (1 - false positive rate)
    """
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    y_prob = np.asarray(y_prob, float)

    n = len(y_true)
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    flagging_rate = float(y_pred.mean())
    mean_prob = float(y_prob.mean())

    result = {
        "n": n, "n_pos": n_pos, "n_neg": n_neg,
        "mean_prob": round(mean_prob, 4),
        "flagging_rate": round(flagging_rate, 4),
        "AUROC": np.nan, "Sensitivity": np.nan, "Specificity": np.nan,
        "PPV": np.nan, "NPV": np.nan, "F1": np.nan,
        "TP": int((y_pred==1).sum()), "TN": int((y_pred==0).sum()),
        "FP": int(((y_pred==1)&(y_true==0)).sum()),
        "FN": int(((y_pred==0)&(y_true==1)).sum()),
    }

    if n_pos > 0 and n_neg > 0:
        result["AUROC"] = float(roc_auc_score(y_true, y_prob))
        result["Sensitivity"] = float(recall_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        result["Specificity"] = float(tn/(tn+fp))
        result["PPV"] = float(tp/(tp+fp)) if (tp+fp)>0 else np.nan
        result["NPV"] = float(tn/(tn+fn)) if (tn+fn)>0 else np.nan
        result["F1"] = float(f1_score(y_true, y_pred, zero_division=0))
    elif n_pos > 0:
        result["Sensitivity"] = round(flagging_rate, 4)
        result["PPV"] = 1.0 if flagging_rate > 0 else np.nan
    else:
        result["Specificity"] = round(1.0 - flagging_rate, 4)
        result["NPV"] = 1.0 if flagging_rate < 1.0 else np.nan

    return result

def mcnemar_test(pred_a, pred_b):
    """Continuity-corrected McNemar."""
    pred_a = np.asarray(pred_a, int)
    pred_b = np.asarray(pred_b, int)
    b = int(((pred_a==1)&(pred_b==0)).sum())
    c = int(((pred_a==0)&(pred_b==1)).sum())
    n_disc = b + c
    agree_pp = int(((pred_a==1)&(pred_b==1)).sum())
    agree_nn = int(((pred_a==0)&(pred_b==0)).sum())
    if n_disc == 0:
        return b, c, n_disc, agree_pp, agree_nn, 0.0, 1.0
    chi2_stat = (abs(b - c) - 1.0)**2 / n_disc
    p = float(1.0 - chi2_dist.cdf(chi2_stat, df=1))
    return b, c, n_disc, agree_pp, agree_nn, float(chi2_stat), p

def stratified_bootstrap_indices(y, n_boot, seed):
    y = np.asarray(y, int)
    rng = np.random.RandomState(seed)
    pos = np.where(y==1)[0]
    neg = np.where(y==0)[0]
    return [np.concatenate([
        rng.choice(pos, len(pos), replace=True),
        rng.choice(neg, len(neg), replace=True)
    ]) for _ in range(n_boot)]

def percentile_ci(boot_vals, alpha=0.05):
    boot_vals = np.asarray(boot_vals, float)
    boot_vals = boot_vals[~np.isnan(boot_vals)]
    if len(boot_vals) < 50: return np.nan, np.nan
    return float(np.percentile(boot_vals, 100*alpha/2)), \
           float(np.percentile(boot_vals, 100*(1-alpha/2)))

def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(**LOCKED_PARAMS)),
    ])

def save_tbl(df, name):
    for d in [DIRS["tables"], DIRS["paper_tables"]]:
        df.to_csv(os.path.join(d, name), index=False)

def save_pr(df, name):
    for d in [DIRS["paper_ready"], DIRS["paper_tables"]]:
        df.to_csv(os.path.join(d, name), index=False)

def save_fig(fig, name):
    for d in [DIRS["figures"], DIRS["paper_figures"]]:
        fig.savefig(os.path.join(d, name), dpi=300, bbox_inches="tight")
    plt.close(fig)

def reliability_curve(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins+1)
    cx, cy = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & ((y_prob <= hi) if i==n_bins-1 else (y_prob < hi))
        if mask.sum() == 0: continue
        cx.append(float(y_prob[mask].mean()))
        cy.append(float(y_true[mask].mean()))
    return np.array(cx), np.array(cy)

# ══════════════════════════════════════════════════════════════════════════════
# 2. LOCK VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 1] Verifying upstream locks...")

if not os.path.exists(RQ1_FLAG):
    raise FileNotFoundError(f"Missing RQ1 flag: {RQ1_FLAG}")
if not os.path.exists(BD_CLEAN_PATH):
    raise FileNotFoundError(f"Missing bd_clean.csv: {BD_CLEAN_PATH}")
if not os.path.exists(BEST_CONFIG):
    raise FileNotFoundError(f"Missing best_model_config.json: {BEST_CONFIG}")

with open(BEST_CONFIG) as f:
    best_cfg = json.load(f)
assert best_cfg["selected_algorithm"] == "Support Vector Machine"
assert best_cfg["hyperparameters"]["C"] == 1.0
print(f"  ✓ M* locked: SVM(C=1.0, rbf, gamma=scale)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 2] Loading bd_clean.csv...")

bd_clean = pd.read_csv(BD_CLEAN_PATH).reset_index(drop=True)

# Identify available columns (robust to schema variations)
available_cols = list(bd_clean.columns)
print(f"  Available columns: {available_cols}")

for c in S0_COLS:
    bd_clean[c] = bd_clean[c].astype(int)
bd_clean["label"] = bd_clean["label"].astype(int)

# AQ-10 score column might be named AQ_score_legacy or AQ10_score
aq_score_col = "AQ_score_legacy" if "AQ_score_legacy" in available_cols else "AQ10_score"
aq_rule_col  = "AQ_rule_pred" if "AQ_rule_pred" in available_cols else "AQ_rule_pred"
if aq_score_col not in available_cols:
    # Compute it if missing
    bd_clean["AQ_score_legacy"] = (bd_clean[S0_COLS] >= 2).sum(axis=1)
    aq_score_col = "AQ_score_legacy"

bd_clean[aq_score_col] = bd_clean[aq_score_col].astype(int)
bd_clean[aq_rule_col]  = bd_clean[aq_rule_col].astype(int)

assert bd_clean["ID"].is_unique
assert bd_clean[S0_COLS].isin(range(4)).all().all()

N_TOTAL = len(bd_clean)
SCHOOLS = sorted(bd_clean["school_id"].unique())
N_SCHOOLS = len(SCHOOLS)

# Verify mono-class assumption
school_class_counts = bd_clean.groupby("school_id")["label"].nunique()
mono_class_schools = school_class_counts[school_class_counts == 1].index.tolist()
mixed_class_schools = school_class_counts[school_class_counts > 1].index.tolist()

print(f"  Total rows    : {N_TOTAL}")
print(f"  Schools       : {N_SCHOOLS}")
print(f"  Mono-class    : {len(mono_class_schools)}")
print(f"  Mixed-class   : {len(mixed_class_schools)}")

# CRITICAL: Verify ≥2 schools per class for valid LOSO training
asd_schools  = bd_clean[bd_clean["label"]==1]["school_id"].unique()
norm_schools = bd_clean[bd_clean["label"]==0]["school_id"].unique()
print(f"  ASD schools   : {len(asd_schools)}  -> {list(asd_schools)}")
print(f"  Normal schools: {len(norm_schools)} -> {list(norm_schools)}")

if len(asd_schools) < 2 or len(norm_schools) < 2:
    raise ValueError(
        f"LOSO requires ≥2 schools per class. "
        f"ASD schools={len(asd_schools)}, Normal schools={len(norm_schools)}."
    )

# ══════════════════════════════════════════════════════════════════════════════
# 4. LOSO CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("[Step 3] LOSO cross-validation (mono-class schools)")
print(f"         {N_SCHOOLS} folds — one per school")
print("         Training-set mono-class guard: ACTIVE")
print("=" * 78)

oof_rows = []
train_mono_warnings = []

t0 = time.time()
for fold_idx, sid in enumerate(SCHOOLS, 1):
    held_mask  = bd_clean["school_id"] == sid
    train_mask = ~held_mask
    held_df  = bd_clean[held_mask].copy()
    train_df = bd_clean[train_mask].copy()

    X_held, y_held = held_df[S0_COLS].values, held_df["label"].values
    X_train, y_train = train_df[S0_COLS].values, train_df["label"].values

    # GUARD: training must contain both classes
    if len(np.unique(y_train)) < 2:
        train_mono_warnings.append(sid)
        print(f"  [SKIP] Fold {fold_idx}/{N_SCHOOLS} {sid}: "
              f"training mono-class (only {np.unique(y_train)}).")
        continue

    model = build_model()
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_held)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)

    for i in range(len(held_df)):
        oof_rows.append({
            "ID": held_df["ID"].iloc[i],
            "school_id": sid,
            "label": int(y_held[i]),
            "y_prob": float(prob[i]),
            "y_pred": int(pred[i]),
        })

    n_pos = int(y_held.sum())
    n_neg = len(y_held) - n_pos
    print(f"  Fold {fold_idx:2d}/{N_SCHOOLS}  {sid}  "
          f"held={len(held_df):3d}  train={len(train_df):3d}  "
          f"pos={n_pos}  neg={n_neg}")

print(f"\n  LOSO complete in {time.time()-t0:.1f}s")

if train_mono_warnings:
    raise RuntimeError(f"Training mono-class in folds: {train_mono_warnings}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILD OOF TABLE
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 4] Building OOF table...")

oof_df = pd.DataFrame(oof_rows)
assert len(oof_df) == N_TOTAL, f"OOF size mismatch: {len(oof_df)} != {N_TOTAL}"
assert oof_df["ID"].is_unique, "Duplicate IDs in OOF"

# Merge AQ-10 rule data from bd_clean — only columns that exist
merge_cols = ["ID", aq_score_col, aq_rule_col]
optional_cols = ["Age", "Gender", "school_name"]
for oc in optional_cols:
    if oc in bd_clean.columns:
        merge_cols.append(oc)

oof_full = oof_df.merge(bd_clean[merge_cols], on="ID", how="left")

# Add rule probability (raw score / 10) and rule prediction
oof_full["rule_prob"] = oof_full[aq_score_col].astype(float) / 10.0
oof_full["rule_pred"] = oof_full[aq_rule_col].astype(int)

print(f"  OOF rows   : {len(oof_full)}")
print(f"  Columns    : {list(oof_full.columns)}")
print(f"  ASD/Normal : {(oof_full['label']==1).sum()} / {(oof_full['label']==0).sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. POOLED GLOBAL METRICS (Primary)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 5] Pooled global metrics (pool first, compute once)...")

y_all     = oof_full["label"].values
prob_svm  = oof_full["y_prob"].values
pred_svm  = oof_full["y_pred"].values
pred_rule = oof_full["rule_pred"].values

pooled_svm  = metric_bundle(y_all, prob_svm, pred_svm)

# For AQ-10 Rule: use raw score/10 for Brier/ECE context only
prob_rule_approx = oof_full["rule_prob"].values
pooled_rule = metric_bundle(y_all, prob_rule_approx, pred_rule)

print(f"\n  {'Metric':<18} {'SVM-LOSO':>12} {'AQ-10 Rule':>12}")
print("  " + "-" * 45)
for k in ["AUROC", "Sensitivity", "Specificity", "Accuracy", "F1_Score", "Brier"]:
    print(f"  {k:<18} {pooled_svm[k]:>12.4f} {pooled_rule[k]:>12.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. PRIMARY TEST: McNemar (SVM binary vs AQ-10 ≥6 rule)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("[Step 6] PRIMARY TEST: McNemar (SVM binary vs AQ-10 ≥6 rule)")
print("         Continuity-corrected  |  alpha = 0.05")
print("=" * 78)

b_mc, c_mc, n_disc, agree_pp, agree_nn, chi2_mc, p_mc = mcnemar_test(pred_svm, pred_rule)
sig_mcnemar = p_mc < 0.05

print(f"\n  Discordant: b={b_mc} (SVM+/Rule-), c={c_mc} (SVM-/Rule+)")
print(f"  Concordant: both+={agree_pp}, both-={agree_nn}")
print(f"  chi²      : {chi2_mc:.4f}")
print(f"  p-value   : {p_mc:.6g}")
print(f"  Significant (α=0.05): {'YES' if sig_mcnemar else 'NO'}")

mcnemar_result = {
    "comparison": "SVM-LOSO binary (t=0.50) vs AQ-10 >=6 rule",
    "b_SVM1_Rule0": b_mc, "c_SVM0_Rule1": c_mc,
    "n_discordant": n_disc,
    "n_concordant_both_pos": agree_pp, "n_concordant_both_neg": agree_nn,
    "chi2_continuity_corrected": round(chi2_mc, 4),
    "p_value": round(p_mc, 6),
    "significant_0.05": bool(sig_mcnemar),
}

mcnemar_df = pd.DataFrame([mcnemar_result])
save_tbl(mcnemar_df, "T_rq2_primary_mcnemar.csv")
save_pr(mcnemar_df, "T_rq2_paper_primary_mcnemar.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 8. BOOTSTRAP CI ON POOLED METRICS
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[Step 7] Stratified bootstrap CI (B={N_BOOTSTRAP})...")

boot_idx = stratified_bootstrap_indices(y_all, N_BOOTSTRAP, SEED)
boot_store = {"SVM": {k: [] for k in METRIC_KEYS}, "Rule": {k: [] for k in METRIC_KEYS}}

for b, idx in enumerate(boot_idx):
    y_b = y_all[idx]
    if len(np.unique(y_b)) < 2: continue
    mb_s = metric_bundle(y_b, prob_svm[idx], pred_svm[idx])
    mb_r = metric_bundle(y_b, prob_rule_approx[idx], pred_rule[idx])
    for k in METRIC_KEYS:
        boot_store["SVM"][k].append(mb_s[k])
        boot_store["Rule"][k].append(mb_r[k])
    if (b+1) % 500 == 0:
        print(f"    bootstrap {b+1}/{N_BOOTSTRAP}")

ci_rows = []
for model_name, pt_est, store in [
    ("SVM-LOSO (S0)", pooled_svm, boot_store["SVM"]),
    ("AQ-10 Clinical Rule", pooled_rule, boot_store["Rule"]),
]:
    row = {"Model": model_name}
    for k in METRIC_KEYS:
        pt = pt_est[k]
        lo, hi = percentile_ci(store[k])
        row[f"{k}_Point"] = round(float(pt), 4)
        row[f"{k}_CI_Lo"] = round(lo, 4) if not np.isnan(lo) else None
        row[f"{k}_CI_Hi"] = round(hi, 4) if not np.isnan(hi) else None
        row[f"{k}_Formatted"] = (
            f"{pt:.4f} [{lo:.4f}, {hi:.4f}]"
            if not (np.isnan(lo) or np.isnan(hi)) else f"{pt:.4f} [NA]"
        )
    for k in ["TP", "TN", "FP", "FN"]:
        row[k] = pt_est[k]
    ci_rows.append(row)

ci_df = pd.DataFrame(ci_rows)
save_tbl(ci_df, "T_rq2_bootstrap_CI.csv")

paper_ci = ci_df[["Model"] + [f"{k}_Formatted" for k in METRIC_KEYS]].copy()
paper_ci.columns = ["Model"] + METRIC_KEYS
save_pr(paper_ci, "T_rq2_paper_pooled_CI.csv")

print("\n  Pooled 95% CI:")
print(f"  {'Metric':<16} {'SVM-LOSO':<28} {'AQ-10 Rule':<28}")
print("  " + "-" * 75)
for k in ["AUROC", "Sensitivity", "Specificity", "F1_Score", "Accuracy", "Brier"]:
    s_row = ci_df[ci_df["Model"].str.startswith("SVM")].iloc[0]
    r_row = ci_df[ci_df["Model"].str.startswith("AQ-10")].iloc[0]
    print(f"  {k:<16} {s_row[f'{k}_Formatted']:<28} {r_row[f'{k}_Formatted']:<28}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. PER-SCHOOL BREAKDOWN (Mono-Class Partial Metrics)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 8] Per-school breakdown (mono-class partial metrics)...")

school_records = []
for sid in SCHOOLS:
    mask = oof_full["school_id"] == sid
    sub = oof_full[mask]
    y_s = sub["label"].values
    prob_s = sub["y_prob"].values
    pred_s = sub["y_pred"].values

    sm = per_school_mono_metrics(y_s, prob_s, pred_s)
    school_type = "ASD" if sm["n_pos"] > 0 else "Normal"

    # Get school name if available, else use school_id
    school_name_val = sub["school_name"].iloc[0] if "school_name" in sub.columns else sid

    school_records.append({
        "school_id": sid,
        "school_name": school_name_val,
        "school_type": school_type,
        "n": sm["n"], "n_ASD": sm["n_pos"], "n_Normal": sm["n_neg"],
        "mean_prob": sm["mean_prob"],
        "flagging_rate": sm["flagging_rate"],
        "AUROC": sm["AUROC"],
        "Sensitivity": sm["Sensitivity"],
        "Specificity": sm["Specificity"],
        "PPV": sm["PPV"], "NPV": sm["NPV"], "F1": sm["F1"],
        "TP": sm["TP"], "TN": sm["TN"], "FP": sm["FP"], "FN": sm["FN"],
    })

school_df = pd.DataFrame(school_records)
save_tbl(school_df, "T_rq2_per_school_breakdown.csv")
save_pr(school_df, "T_rq2_paper_per_school.csv")

print(f"\n  {'School':<12} {'Type':<8} {'n':>4} {'Sens':>8} {'Spec':>8} {'FlagRate':>10}")
print("  " + "-" * 60)
for _, r in school_df.iterrows():
    sens = f"{r['Sensitivity']:.3f}" if not pd.isna(r['Sensitivity']) else "N/A"
    spec = f"{r['Specificity']:.3f}" if not pd.isna(r['Specificity']) else "N/A"
    print(f"  {r['school_id']:<12} {r['school_type']:<8} {r['n']:>4} "
          f"{sens:>8} {spec:>8} {r['flagging_rate']:>10.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE OOF PREDICTIONS (Robust column selection)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 9] Saving OOF predictions...")

# Build save columns dynamically based on what exists
base_save_cols = ["ID", "school_id", "label", "y_prob", "y_pred",
                  "rule_prob", "rule_pred", aq_score_col]
optional_save = ["school_name", "Age", "Gender"]
save_cols = [c for c in base_save_cols if c in oof_full.columns]
for c in optional_save:
    if c in oof_full.columns:
        save_cols.append(c)

oof_save = oof_full[save_cols].copy()
OOF_PATH = os.path.join(DIRS["predictions"], "rq2_loso_s0_predictions.csv")
oof_save.to_csv(OOF_PATH, index=False)
print(f"  ✓ Saved: {OOF_PATH}")
print(f"    Columns: {save_cols}")

# ══════════════════════════════════════════════════════════════════════════════
# 11. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 10] Generating figures...")

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

COLOR_SVM = "#ff7f0e"
COLOR_RULE = "#333333"
COLOR_ASD_SCHOOL = "#d62728"
COLOR_NORM_SCHOOL = "#1f77b4"

# ── Figure 1: Pooled ROC (SVM only) ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fpr_svm, tpr_svm, _ = roc_curve(y_all, prob_svm)
ax.plot(fpr_svm, tpr_svm, lw=2.5, color=COLOR_SVM,
        label=f"SVM-LOSO  AUC = {pooled_svm['AUROC']:.4f}")

# Faint context: AQ-10 total score (NOT the rule)
fpr_aq, tpr_aq, _ = roc_curve(y_all, oof_full[aq_score_col].values)
auc_aq = roc_auc_score(y_all, oof_full[aq_score_col].values)
ax.plot(fpr_aq, tpr_aq, "--", lw=1.5, color="#bbbbbb", alpha=0.7,
        label=f"AQ-10 total score (context)  AUC = {auc_aq:.4f}")

# Operating point marker
ax.scatter(1 - pooled_svm["Specificity"], pooled_svm["Sensitivity"],
           marker="D", s=100, color=COLOR_SVM, zorder=5, edgecolor="white",
           label=f"SVM @ t=0.50  (Sens={pooled_svm['Sensitivity']:.3f}, "
                 f"Spec={pooled_svm['Specificity']:.3f})")

ax.plot([0, 1], [0, 1], ":", color="gray", alpha=0.5)
ax.set_xlabel("1 - Specificity (FPR)")
ax.set_ylabel("Sensitivity (TPR)")
ax.set_title("RQ2 — Pooled OOF ROC (LOSO, S0)\n"
             "AQ-10 total score shown for context only")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, linestyle=":", alpha=0.4)
save_fig(fig, "F_rq2_1_pooled_roc.png")
print("  ✓ F1: Pooled ROC")

# ── Figure 2: Per-School Partial Metrics ────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(10, N_SCHOOLS*0.8), 5))

x = np.arange(N_SCHOOLS)
bar_w = 0.5
colors = [COLOR_ASD_SCHOOL if t=="ASD" else COLOR_NORM_SCHOOL
          for t in school_df["school_type"]]

heights = []
labels = []
for _, r in school_df.iterrows():
    if r["school_type"] == "ASD":
        heights.append(r["Sensitivity"] if not pd.isna(r["Sensitivity"]) else 0)
        labels.append("Sens")
    else:
        heights.append(r["Specificity"] if not pd.isna(r["Specificity"]) else 0)
        labels.append("Spec")

bars = ax.bar(x, heights, bar_w, color=colors, alpha=0.85, edgecolor="white")

for bar, h, lbl in zip(bars, heights, labels):
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
            f"{h:.2f}\n({lbl})", ha="center", va="bottom", fontsize=8)

ax.axhline(pooled_svm["Sensitivity"], color=COLOR_ASD_SCHOOL, ls="--", lw=1.5,
           alpha=0.6, label=f"Global Sensitivity = {pooled_svm['Sensitivity']:.3f}")
ax.axhline(pooled_svm["Specificity"], color=COLOR_NORM_SCHOOL, ls="--", lw=1.5,
           alpha=0.6, label=f"Global Specificity = {pooled_svm['Specificity']:.3f}")

ax.set_xticks(x)
ax.set_xticklabels(school_df["school_id"], rotation=45, ha="right", fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Metric value")
ax.set_title("RQ2 — Per-School Partial Metrics (Mono-Class Schools)\n"
             "ASD schools: Sensitivity  |  Normal schools: Specificity")
legend_handles = [
    mpatches.Patch(color=COLOR_ASD_SCHOOL, label="ASD school (Sensitivity)"),
    mpatches.Patch(color=COLOR_NORM_SCHOOL, label="Normal school (Specificity)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
ax.grid(True, axis="y", linestyle=":", alpha=0.4)
save_fig(fig, "F_rq2_2_per_school_partial_metrics.png")
print("  ✓ F2: Per-school partial metrics")

# ── Figure 3: Pooled Confusion Matrices ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax_, prds, title in [
    (axes[0], pred_svm, f"SVM-LOSO (S0)\nAUROC={pooled_svm['AUROC']:.4f}"),
    (axes[1], pred_rule, f"AQ-10 Clinical Rule\nscore ≥ 6"),
]:
    cm = confusion_matrix(y_all, prds, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "ASD"]).plot(
        ax=ax_, colorbar=False, cmap="Blues")
    ax_.set_title(title, fontsize=11)
fig.suptitle(f"RQ2 — Pooled OOF Confusion Matrices  (N={N_TOTAL})", y=1.04)
fig.tight_layout()
save_fig(fig, "F_rq2_3_pooled_confusion.png")
print("  ✓ F3: Pooled confusion matrices")

# ── Figure 4: Calibration (SVM only) ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Perfect calibration")
cx, cy = reliability_curve(y_all, prob_svm)
ax.plot(cx, cy, "o-", color=COLOR_SVM, lw=2, ms=6,
        label=f"SVM-LOSO  ECE={pooled_svm['ECE']:.4f}, Brier={pooled_svm['Brier']:.4f}")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed fraction positive (ASD)")
ax.set_title(f"RQ2 — Calibration (Pooled OOF, N={N_TOTAL})")
ax.legend(loc="upper left")
ax.grid(True, linestyle=":", alpha=0.4)
save_fig(fig, "F_rq2_4_calibration.png")
print("  ✓ F4: Calibration")

# ── Figure 5: McNemar Disagreement ──────────────────────────────────────────
fig, axes5 = plt.subplots(1, 2, figsize=(12, 5))

ax5l = axes5[0]
patterns = ["Both flag\n(agree +)", "SVM flags\nRule misses",
            "Rule flags\nSVM misses", "Both miss\n(agree -)"]
vals = [agree_pp, b_mc, c_mc, agree_nn]
clrs = ["#2ca02c", COLOR_SVM, "#d62728", "#aaaaaa"]
bars5 = ax5l.bar(patterns, vals, color=clrs, alpha=0.85, edgecolor="white")
for b5 in bars5:
    h = b5.get_height()
    ax5l.text(b5.get_x() + b5.get_width()/2, h + 0.3,
              f"{h}\n({100*h/len(y_all):.1f}%)", ha="center", va="bottom", fontsize=9)
ax5l.set_ylabel("Number of participants")
ax5l.set_title(f"Prediction Agreement  (N={len(y_all)})")
ax5l.grid(True, axis="y", linestyle=":", alpha=0.4)

ax5r = axes5[1]
if n_disc > 0:
    disc_vals = [b_mc, c_mc]
    disc_lbls = [f"SVM flags, Rule misses\n(b={b_mc})",
                 f"Rule flags, SVM misses\n(c={c_mc})"]
    ax5r.pie(disc_vals, labels=disc_lbls, colors=[COLOR_SVM, "#d62728"],
             autopct="%1.1f%%", startangle=90,
             wedgeprops={"linewidth": 1.5, "edgecolor": "white"})
    sig_txt = "SIGNIFICANT" if sig_mcnemar else "not significant"
    ax5r.set_title(f"McNemar Discordant Pairs  (n={n_disc})\n"
                   f"χ²={chi2_mc:.3f}, p={p_mc:.4f}  [{sig_txt}]")
else:
    ax5r.text(0.5, 0.5, "No discordant pairs", ha="center", va="center",
              transform=ax5r.transAxes, fontsize=12)
    ax5r.set_title("McNemar Discordant Pairs")
    ax5r.axis("off")

fig.suptitle("RQ2 — McNemar Test: SVM-LOSO vs AQ-10 Clinical Rule", y=1.02)
fig.tight_layout()
save_fig(fig, "F_rq2_5_mcnemar_disagreement.png")
print("  ✓ F5: McNemar disagreement")

# ══════════════════════════════════════════════════════════════════════════════
# 12. MASTER SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 11] Writing master summary...")

rq2_summary = {
    "module": "RQ2_Institutional_Generalization_LOSO",
    "version": "FINAL-MONO-10.1-RQ2",
    "status": "COMPLETE",
    "timestamp": datetime.now().isoformat(),
    "locked_algorithm": "Support Vector Machine",
    "locked_hyperparameters": LOCKED_PARAMS,
    "feature_set": "S0",
    "n_schools": N_SCHOOLS,
    "school_ids": SCHOOLS,
    "mono_class_design": True,
    "n_asd_schools": int(len(asd_schools)),
    "n_normal_schools": int(len(norm_schools)),
    "pooled_oof_n": N_TOTAL,
    "pooled_svm_metrics": {k: float(pooled_svm[k]) for k in METRIC_KEYS},
    "pooled_rule_metrics": {k: float(pooled_rule[k]) for k in METRIC_KEYS},
    "primary_test_mcnemar": mcnemar_result,
    "bootstrap_n": N_BOOTSTRAP,
    "limitation_note": (
        "Schools were mono-class by design (ASD schools contained only "
        "children with ASD; mainstream schools contained only typically "
        "developing children). Per-school AUROC/sensitivity/specificity "
        "were therefore undefined for held-out folds. We report pooled "
        "out-of-fold metrics and school-type-specific partial metrics "
        "(sensitivity for ASD-only schools; specificity for normal-only "
        "schools) to characterize institutional variation."
    ),
}

for path in [os.path.join(DIRS["results"], "RQ2_Master_Summary.json"),
             os.path.join(DIRS["config"], "RQ2_Master_Summary.json")]:
    with open(path, "w") as f:
        json.dump(rq2_summary, f, indent=4, default=str)

# Done flag
flag_content = (
    f"RQ2 FINAL-MONO-10.1-RQ2 COMPLETE\n"
    f"timestamp={datetime.now().isoformat()}\n"
    f"locked_algorithm=Support Vector Machine\n"
    f"feature_set=S0\n"
    f"n_schools={N_SCHOOLS}\n"
    f"n_asd_schools={len(asd_schools)}\n"
    f"n_normal_schools={len(norm_schools)}\n"
    f"pooled_oof_n={N_TOTAL}\n"
    f"pooled_auroc={pooled_svm['AUROC']:.6f}\n"
    f"pooled_sensitivity={pooled_svm['Sensitivity']:.6f}\n"
    f"pooled_specificity={pooled_svm['Specificity']:.6f}\n"
    f"mcnemar_chi2={chi2_mc:.6f}\n"
    f"mcnemar_p={p_mc:.6g}\n"
    f"mcnemar_significant={sig_mcnemar}\n"
    f"oof_path={OOF_PATH}\n"
)
with open(os.path.join(DIRS["logs"], "rq2_done.flag"), "w") as f:
    f.write(flag_content)
with open(os.path.join(PROJECT, "logs", "rq2_done.flag"), "w") as f:
    f.write(flag_content)

# ══════════════════════════════════════════════════════════════════════════════
# 13. FINAL CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("RQ2 SUMMARY — LOSO INSTITUTIONAL GENERALIZATION (MONO-CLASS SCHOOLS)")
print("=" * 78)

print(f"\nDESIGN")
print(f"  Schools total      : {N_SCHOOLS}")
print(f"  ASD schools        : {len(asd_schools)}")
print(f"  Normal schools     : {len(norm_schools)}")
print(f"  Mono-class design  : YES (all schools mono-class)")

print(f"\nPOOLED OOF METRICS")
print(f"  {'Metric':<16} {'SVM-LOSO':>14} {'AQ-10 Rule':>14}")
print("  " + "-" * 48)
for k in ["AUROC", "Sensitivity", "Specificity", "Accuracy", "F1_Score", "Brier"]:
    print(f"  {k:<16} {pooled_svm[k]:>14.4f} {pooled_rule[k]:>14.4f}")

print(f"\nPRIMARY TEST: McNemar")
print(f"  Discordant pairs   : b={b_mc}, c={c_mc} (n={n_disc})")
print(f"  χ² (continuity-corr): {chi2_mc:.4f}")
print(f"  p-value            : {p_mc:.6g}")
print(f"  Significant (α=0.05): {'YES' if sig_mcnemar else 'NO'}")

print(f"\nPER-SCHOOL PARTIAL METRICS")
for _, r in school_df.iterrows():
    metric = f"Sens={r['Sensitivity']:.3f}" if r['school_type']=='ASD' else f"Spec={r['Specificity']:.3f}"
    print(f"  {r['school_id']:<12} ({r['school_type']:<6})  n={r['n']:>3}  {metric}  "
          f"flag_rate={r['flagging_rate']:.3f}")

print(f"\nOUTPUTS")
print(f"  OOF predictions : {OOF_PATH}")
print(f"  Tables          : {DIRS['tables']}")
print(f"  Figures         : {DIRS['figures']}")

print("\n" + "=" * 78)
print("STATUS: RQ2 FINAL-MONO-10.1-RQ2 COMPLETE")
print("=" * 78)

# from google.colab import drive  # Colab-only; locally, set the AQ10_BASE environment variable instead
# drive.mount('/content/drive')

