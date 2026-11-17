"""RQ2 - Leave-one-school-out validation of the locked SVM.

Run after rq1_baseline.py. Refits the locked SVM twelve times, each time with one school left
out, and compares the pooled out-of-fold decisions with the AQ-10 rule. Writes tables and the
out-of-fold predictions to <work>/aq_10/rq2_loso/, and three figures to <work>/aq_10/figures/:
the AQ-10 total score by group with the share flagged by the SVM at each score, the pooled ROC
curves with calibration, and the SVM probability of every child by left-out school.
"""
import os, json, time, random, warnings
from datetime import datetime

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import chi2 as chi2_dist

np.random.seed(SEED)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    accuracy_score, precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix,
)

# ══════════════════════════════════════════════════════════════════════════════
# 0. PATHS
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 78)
print("RQ2 — LOSO INSTITUTIONAL GENERALIZATION  |  MONO-CLASS SCHOOLS")
print("FINAL-MONO-10.1-RQ2")
print("=" * 78)

BASE    = os.environ.get("AQ10_BASE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work"))
PROJECT = os.path.join(BASE, "aq_10")

RQ1_MODELS_DIR = os.path.join(PROJECT, "rq1_baseline", "models")
RQ1_FLAG       = os.path.join(PROJECT, "logs", "rq1_done.flag")
BD_CLEAN_PATH  = os.path.join(PROJECT, "data", "processed", "bd_clean.csv")
BEST_CONFIG    = os.path.join(RQ1_MODELS_DIR, "best_model_config.json")

RQ2_ROOT = os.path.join(PROJECT, "rq2_loso")
DIRS = {
    "root": RQ2_ROOT, "tables": os.path.join(RQ2_ROOT, "tables"),
    "paper_ready": os.path.join(RQ2_ROOT, "tables", "paper_ready"),
    "results": os.path.join(RQ2_ROOT, "results"),
    "predictions": os.path.join(RQ2_ROOT, "predictions"),
    "logs": os.path.join(RQ2_ROOT, "logs"),
    "config": os.path.join(RQ2_ROOT, "config"),
    "paper_tables": os.path.join(PROJECT, "paper_results", "rq2", "tables"),
    "figures": os.path.join(PROJECT, "figures"),
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

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Arial is often missing on Linux; no warning flood
FIG_STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
             "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
             "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False}
FIG_DIR = DIRS["figures"]
os.makedirs(FIG_DIR, exist_ok=True)
FIGURES_SAVED = []


def save_figure(fig, name):
    """Save a figure at 600 dpi in <work>/aq_10/figures/ and close it."""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=600)
    plt.close(fig)
    FIGURES_SAVED.append(name)
    print(f"  [figure] {path}")



def reliability_bins(y_true, prob, n_bins=10):
    """Ten equal-width bins, as in the ECE: bin mean probability, observed share and size."""
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    keep = [k for k in range(n_bins) if (b == k).any()]
    return (np.array([prob[b == k].mean() for k in keep]), np.array([y_true[b == k].mean() for k in keep]),
            np.array([(b == k).sum() for k in keep]))


print("\n[Step 10] Figures...")
MIN_N_PER_SCORE = 5                    # scores held by fewer children of a group are not plotted in panel (b)
score_f2 = oof_full[aq_score_col].values.astype(int)
lab_f2, svm_f2 = oof_full["label"].values, oof_full["y_pred"].values
S_f2 = np.arange(11)
ASD_C, MN_C, RULE_C, INK = "#CC79A7", "#56B4E9", "#D55E00", "#333333"
with plt.rc_context(FIG_STYLE):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(3.5, 2.0))
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.185, top=0.8, wspace=0.42)
    # (a) distribution of the total score in each group
    top_a = 0.0
    for k, (g, c, name) in enumerate(((1, ASD_C, "ASD schools"), (0, MN_C, "Mainstream"))):
        pct = np.array([100 * np.mean(score_f2[lab_f2 == g] == s) for s in S_f2])
        ax_a.bar(S_f2 + (k - 0.5) * 0.4, pct, 0.4, color=c, label=name, lw=0)
        top_a = max(top_a, pct.max())
    top_a = max(40.0, 5 * np.ceil(1.05 * top_a / 5))
    ax_a.axvline(5.5, color=INK, ls="--", lw=0.7)
    ax_a.text(5.65, 0.95 * top_a, "cut-off", fontsize=6, color=INK, va="top")
    ax_a.set_xlabel("AQ-10 total score"); ax_a.set_ylabel("Children in group (%)")
    ax_a.set_xticks(S_f2[::2]); ax_a.set_xlim(-0.6, 10.6); ax_a.set_ylim(0, top_a)
    # (b) flagged by the SVM when the school was unseen, at each total score
    for g, c, mk, name in ((1, ASD_C, "o", "SVM, ASD schools"), (0, MN_C, "s", "SVM, mainstream")):
        n_s = np.array([np.sum((lab_f2 == g) & (score_f2 == s)) for s in S_f2])
        f_s = np.array([svm_f2[(lab_f2 == g) & (score_f2 == s)].sum() for s in S_f2])
        keep = n_s >= MIN_N_PER_SCORE
        ax_b.plot(S_f2[keep], 100 * f_s[keep] / n_s[keep], marker=mk, ms=3, color=c, lw=1.1, label=name)
    ax_b.step(S_f2, 100 * (S_f2 >= 6), where="mid", color=RULE_C, ls="--", lw=0.9, label="AQ-10 rule")
    ax_b.set_xlabel("AQ-10 total score"); ax_b.set_ylabel("Flagged by the SVM (%)")
    ax_b.set_xticks(S_f2[::2]); ax_b.set_xlim(-0.4, 10.4); ax_b.set_ylim(-3, 103)
    for ax_, letter in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax_.text(-0.3, 1.06, letter, transform=ax_.transAxes, fontsize=8, fontweight="bold", va="bottom")
    fig.legend(handles=[Patch(color=ASD_C, label="ASD schools"), Patch(color=MN_C, label="Mainstream schools"),
                        Line2D([], [], color=RULE_C, ls="--", lw=0.9, label="AQ-10 rule")],
               loc="upper center", ncol=3, frameon=False, fontsize=6.5, handlelength=1.4, columnspacing=1.4,
               bbox_to_anchor=(0.53, 1.0))
    save_figure(fig, "rq2_aq10_score_and_svm_flags.png")

# Pooled ROC curves of the SVM and the AQ-10 total score, and calibration of the SVM
prob_f2 = oof_full["y_prob"].values
with plt.rc_context(FIG_STYLE):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.16, 2.7))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.9, wspace=0.28)
    for sc, col, ls, name in ((prob_f2, "#0072B2", "-", "SVM"), (score_f2, "#7F7F7F", (0, (4, 2)), "AQ-10 total score")):
        fpr_, tpr_, _ = roc_curve(lab_f2, sc)
        ax_a.plot(fpr_, tpr_, color=col, ls=ls, lw=1.2, label=f"{name} (AUROC {roc_auc_score(lab_f2, sc):.3f})")
    ax_a.plot(1 - pooled_svm["Specificity"], pooled_svm["Sensitivity"], "o", color="#0072B2", ms=4, label="SVM, threshold 0.50")
    r6_ = score_f2 >= 6
    ax_a.plot(r6_[lab_f2 == 0].mean(), r6_[lab_f2 == 1].mean(), "s", color=RULE_C, ms=4, label="AQ-10 rule (score \u2265 6)")
    ax_a.plot([0, 1], [0, 1], ":", color="#BBBBBB", lw=0.6)
    ax_a.set_xlabel("False-positive rate"); ax_a.set_ylabel("Sensitivity")
    ax_a.legend(title=f"{N_TOTAL} children, school left out", loc="lower right", frameon=False, fontsize=6.2, title_fontsize=6.2)
    mp, ob, nb_ = reliability_bins(lab_f2, prob_f2)
    ax_b.plot([0, 1], [0, 1], ":", color="#999999", lw=0.7)
    ax_b.plot(mp, ob, color="#0072B2", lw=0.8)
    ax_b.scatter(mp, ob, s=6 + 40 * nb_ / nb_.max(), color="#0072B2", zorder=3)
    ax_b.text(0.04, 0.96, f"SVM\nBrier {pooled_svm['Brier']:.3f}\nECE {pooled_svm['ECE']:.3f}", transform=ax_b.transAxes,
              va="top", fontsize=6.5)
    ax_b.set_xlabel("Predicted probability (mean of ten equal bins; point size = children)")
    ax_b.set_ylabel("Observed share of ASD-school children")
    ax_b.set_xlim(-0.02, 1.02); ax_b.set_ylim(-0.02, 1.02)
    for ax_, letter in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax_.text(-0.17, 1.02, letter, transform=ax_.transAxes, fontsize=8, fontweight="bold", va="bottom")
    save_figure(fig, "rq2_pooled_roc_and_calibration.png")

# SVM probability of every child, by left-out school
asd_f2 = sorted(oof_full.loc[oof_full["label"] == 1, "school_id"].unique())
mn_f2 = sorted(oof_full.loc[oof_full["label"] == 0, "school_id"].unique())
order_f2 = asd_f2 + mn_f2
with plt.rc_context(FIG_STYLE):
    fig, ax = plt.subplots(figsize=(7.16, 2.4))
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.17, top=0.78)
    jit = np.random.default_rng(2)                    # jitter only; no effect on any number
    for i, s in enumerate(order_f2):
        v = oof_full.loc[oof_full["school_id"] == s, "y_prob"].values
        ax.scatter(i + jit.uniform(-0.27, 0.27, len(v)), v, s=2.5, lw=0, alpha=0.65,
                   color=ASD_C if s in asd_f2 else MN_C)
        ax.plot([i - 0.32, i + 0.32], [np.median(v)] * 2, color="#222222", lw=1.2)
        ax.text(i, 1.06, f"{int((v >= THRESHOLD).sum())} / {len(v)}", ha="center", va="bottom", fontsize=6)
    ax.axhline(THRESHOLD, color=RULE_C, ls="--", lw=0.7)
    ax.axvline(len(asd_f2) - 0.5, color="#999999", ls=":", lw=0.8)
    ax.set_xticks(range(len(order_f2))); ax.set_xticklabels([f"S{s[-2:]}" for s in order_f2])
    ax.set_xlim(-0.6, len(order_f2) - 0.4); ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("SVM probability (school left out)")
    ax.text((len(asd_f2) - 1) / 2, 1.17, "ASD schools", ha="center", fontsize=7, transform=ax.get_xaxis_transform())
    ax.text(len(asd_f2) + (len(mn_f2) - 1) / 2, 1.17, "Mainstream schools", ha="center", fontsize=7,
            transform=ax.get_xaxis_transform())
    fig.text(0.99, 0.01, "Above each school: children flagged / children. Black bar: median. Dashed line: threshold 0.50.",
             ha="right", va="bottom", fontsize=6, color="#555555")
    save_figure(fig, "rq2_svm_probability_by_school.png")


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

