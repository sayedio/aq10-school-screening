

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
from scipy.stats import norm, chi2 as chi2_dist

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
# 0. PATHS & SETUP
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 78)
print("RQ3 — PAIRED LOSO: S0 vs S1  |  MONO-CLASS SCHOOLS")
print("FINAL-MONO-11.1-RQ3")
print("=" * 78)

# drive.mount("/content/drive", force_remount=True)

BASE    = os.environ.get("AQ10_BASE", "/content/drive/MyDrive/Autism")
PROJECT = os.path.join(BASE, "aq_10")

RQ2_PRED_DIR = os.path.join(PROJECT, "rq2_loso", "predictions")
RQ2_FLAG     = os.path.join(PROJECT, "rq2_loso", "logs", "rq2_done.flag")
BD_CLEAN_PATH = os.path.join(PROJECT, "data", "processed", "bd_clean.csv")

RQ3_ROOT = os.path.join(PROJECT, "rq3_augmentation")
DIRS = {
    "root": RQ3_ROOT, "tables": os.path.join(RQ3_ROOT, "tables"),
    "paper_ready": os.path.join(RQ3_ROOT, "tables", "paper_ready"),
    "figures": os.path.join(RQ3_ROOT, "figures"),
    "results": os.path.join(RQ3_ROOT, "results"),
    "predictions": os.path.join(RQ3_ROOT, "predictions"),
    "logs": os.path.join(RQ3_ROOT, "logs"),
    "config": os.path.join(RQ3_ROOT, "config"),
    "paper_tables": os.path.join(PROJECT, "paper_results", "rq3", "tables"),
    "paper_figures": os.path.join(PROJECT, "paper_results", "rq3", "figures"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# 1. CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

S0_COLS = [f"A_Q{i}" for i in range(1, 11)]
S1_COLS = S0_COLS + [f"A_SQ{i}" for i in range(1, 5)]

THRESHOLD   = 0.50
N_BOOTSTRAP = 2000
ALPHA       = 0.05

METRIC_KEYS = [
    "AUROC", "AUPRC", "Accuracy", "Precision_PPV",
    "Sensitivity", "Specificity", "F1_Score", "Brier", "ECE",
]

LOCKED_PARAMS = {
    "C": 1.0, "kernel": "rbf", "gamma": "scale",
    "probability": True, "random_state": SEED,
}

# Pre-specified clinical safety limits for F* decision
MAX_FP_INFLATION = 5      # S1 may add at most 5 extra false positives
MAX_FN_INCREASE  = 0      # S1 must NOT increase false negatives

# ══════════════════════════════════════════════════════════════════════════════
# 2. HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    n = len(y_true)
    if n == 0: return np.nan
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & ((y_prob <= hi) if i == n_bins-1 else (y_prob < hi))
        if mask.sum() == 0: continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)

def metric_bundle(y_true, y_prob, y_pred):
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

def delong_paired_test(y_true, score_a, score_b):
    """Sun & Xu 2014 placement-value paired DeLong."""
    y_true  = np.asarray(y_true, int)
    score_a = np.asarray(score_a, float)
    score_b = np.asarray(score_b, float)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n1, n0  = len(pos_idx), len(neg_idx)
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan, np.nan, np.nan

    def _kernel(s_pos, s_neg):
        diff = s_pos[:, None] - s_neg[None, :]
        return np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))

    K_a = _kernel(score_a[pos_idx], score_a[neg_idx])
    K_b = _kernel(score_b[pos_idx], score_b[neg_idx])
    V10_a, V01_a = K_a.mean(axis=1), K_a.mean(axis=0)
    V10_b, V01_b = K_b.mean(axis=1), K_b.mean(axis=0)
    auc_a = float(V10_a.mean())
    auc_b = float(V10_b.mean())
    if n1 < 2 or n0 < 2:
        return auc_a, auc_b, 0.0, 1.0
    V10_mat = np.column_stack([V10_a, V10_b])
    V01_mat = np.column_stack([V01_a, V01_b])
    S10 = np.cov(V10_mat.T) / n1
    S01 = np.cov(V01_mat.T) / n0
    var_d = (S10[0,0] + S10[1,1] - 2*S10[0,1] + S01[0,0] + S01[1,1] - 2*S01[0,1])
    if var_d <= 0:
        return auc_a, auc_b, 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var_d)
    p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return auc_a, auc_b, float(z), p

def mcnemar_test(pred_a, pred_b):
    """Continuity-corrected McNemar (Edwards 1948)."""
    pred_a = np.asarray(pred_a, int)
    pred_b = np.asarray(pred_b, int)
    b = int(((pred_a==1) & (pred_b==0)).sum())
    c = int(((pred_a==0) & (pred_b==1)).sum())
    n_disc = b + c
    agree_pp = int(((pred_a==1) & (pred_b==1)).sum())
    agree_nn = int(((pred_a==0) & (pred_b==0)).sum())
    if n_disc == 0:
        return b, c, n_disc, agree_pp, agree_nn, 0.0, 1.0
    chi2_stat = (abs(b - c) - 1.0)**2 / n_disc
    p = float(1.0 - chi2_dist.cdf(chi2_stat, df=1))
    return b, c, n_disc, agree_pp, agree_nn, float(chi2_stat), p

def paired_bootstrap(y_true, prob_a, prob_b, n_boot, seed):
    """Paired bootstrap: same row indices for both pipelines."""
    y_true = np.asarray(y_true, int)
    prob_a = np.asarray(prob_a, float)
    prob_b = np.asarray(prob_b, float)
    pred_a = (prob_a >= THRESHOLD).astype(int)
    pred_b = (prob_b >= THRESHOLD).astype(int)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    store = {k: [] for k in [
        "delta_auroc", "delta_sens", "delta_spec",
        "delta_fp", "delta_fn", "delta_accuracy", "delta_f1", "delta_brier"
    ]}
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        mA = metric_bundle(y_b, prob_a[idx], pred_a[idx])
        mB = metric_bundle(y_b, prob_b[idx], pred_b[idx])
        store["delta_auroc"].append(mB["AUROC"] - mA["AUROC"])
        store["delta_sens"].append(mB["Sensitivity"] - mA["Sensitivity"])
        store["delta_spec"].append(mB["Specificity"] - mA["Specificity"])
        store["delta_fp"].append(mB["FP"] - mA["FP"])
        store["delta_fn"].append(mB["FN"] - mA["FN"])
        store["delta_accuracy"].append(mB["Accuracy"] - mA["Accuracy"])
        store["delta_f1"].append(mB["F1_Score"] - mA["F1_Score"])
        store["delta_brier"].append(mB["Brier"] - mA["Brier"])
    return {k: np.asarray(v) for k, v in store.items()}

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
# 3. LOCK VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 1] Verifying upstream locks...")

if not os.path.exists(RQ2_FLAG):
    raise FileNotFoundError(f"Missing RQ2 flag: {RQ2_FLAG}")
if not os.path.exists(BD_CLEAN_PATH):
    raise FileNotFoundError(f"Missing bd_clean.csv: {BD_CLEAN_PATH}")

OOF_A_PATH = os.path.join(RQ2_PRED_DIR, "rq2_loso_s0_predictions.csv")
if not os.path.exists(OOF_A_PATH):
    raise FileNotFoundError(f"Missing Pipeline A OOF: {OOF_A_PATH}")

print("  ✓ RQ2 flag OK")
print(f"  ✓ Pipeline A OOF: {OOF_A_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. LOAD PIPELINE A (S0) — DO NOT RE-RUN
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 2] Loading Pipeline A (S0) from RQ2...")

oof_a = pd.read_csv(OOF_A_PATH)
assert len(oof_a) > 0, "Pipeline A OOF is empty"
assert oof_a["ID"].is_unique, "Duplicate IDs in Pipeline A"
assert "y_prob" in oof_a.columns and "y_pred" in oof_a.columns, \
    "Missing y_prob/y_pred in Pipeline A OOF"

print(f"  ✓ Pipeline A: {len(oof_a)} rows, columns: {list(oof_a.columns)}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. LOAD bd_clean & VERIFY S1 FEATURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 3] Loading bd_clean.csv...")

bd_clean = pd.read_csv(BD_CLEAN_PATH).reset_index(drop=True)

available_cols = list(bd_clean.columns)
print(f"  Available columns: {available_cols}")

missing_s1 = [c for c in S1_COLS if c not in available_cols]
if missing_s1:
    raise KeyError(f"Missing S1 features: {missing_s1}")

for c in S1_COLS:
    bd_clean[c] = bd_clean[c].astype(int)
bd_clean["label"] = bd_clean["label"].astype(int)

aq_score_col = "AQ_score_legacy" if "AQ_score_legacy" in available_cols else \
               ("AQ10_score" if "AQ10_score" in available_cols else None)
if aq_score_col is None:
    bd_clean["AQ_score_legacy"] = (bd_clean[S0_COLS] >= 2).sum(axis=1)
    aq_score_col = "AQ_score_legacy"
bd_clean[aq_score_col] = bd_clean[aq_score_col].astype(int)

aq_rule_col = "AQ_rule_pred" if "AQ_rule_pred" in available_cols else None
if aq_rule_col is None:
    bd_clean["AQ_rule_pred"] = (bd_clean[aq_score_col] >= 6).astype(int)
    aq_rule_col = "AQ_rule_pred"
bd_clean[aq_rule_col] = bd_clean[aq_rule_col].astype(int)

a_ids = set(oof_a["ID"].values)
c_ids = set(bd_clean["ID"].values)
if a_ids != c_ids:
    raise AssertionError(f"ID mismatch: A-only={len(a_ids-c_ids)}, clean-only={len(c_ids-a_ids)}")

SCHOOLS = sorted(bd_clean["school_id"].unique())
N_SCHOOLS = len(SCHOOLS)
N_TOTAL = len(bd_clean)

print(f"  ✓ bd_clean: {N_TOTAL} rows, {N_SCHOOLS} schools")
print(f"  ✓ S0: {len(S0_COLS)} features | S1: {len(S1_COLS)} features")

asd_schools  = bd_clean[bd_clean["label"]==1]["school_id"].unique()
norm_schools = bd_clean[bd_clean["label"]==0]["school_id"].unique()
print(f"  ASD schools: {len(asd_schools)} | Normal schools: {len(norm_schools)}")
if len(asd_schools) < 2 or len(norm_schools) < 2:
    raise ValueError("LOSO requires ≥2 schools per class")

# ══════════════════════════════════════════════════════════════════════════════
# 6. PIPELINE B — LOSO ON S1
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("[Step 4] Pipeline B: LOSO on S1 (14 features)")
print(f"         {N_SCHOOLS} folds — same sorted order as RQ2")
print("=" * 78)

oof_b_rows = []
t0 = time.time()

for fold_idx, sid in enumerate(SCHOOLS, 1):
    held_mask  = bd_clean["school_id"] == sid
    train_mask = ~held_mask
    held_df  = bd_clean[held_mask].copy()
    train_df = bd_clean[train_mask].copy()

    X_held  = held_df[S1_COLS].values
    X_train = train_df[S1_COLS].values
    y_train = train_df["label"].values
    y_held  = held_df["label"].values

    if len(np.unique(y_train)) < 2:
        raise RuntimeError(f"Fold {sid}: training mono-class — invalid LOSO")

    model = build_model()
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_held)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)

    for i in range(len(held_df)):
        oof_b_rows.append({
            "ID": held_df["ID"].iloc[i],
            "school_id": sid,
            "label": int(y_held[i]),
            "y_prob_s1": float(prob[i]),
            "y_pred_s1": int(pred[i]),
        })

    n_pos = int(y_held.sum())
    n_neg = len(y_held) - n_pos
    print(f"  Fold {fold_idx:2d}/{N_SCHOOLS}  {sid}  held={len(held_df):3d}  "
          f"train={len(train_df):3d}  pos={n_pos}  neg={n_neg}")

print(f"\n  Pipeline B complete in {time.time()-t0:.1f}s")

oof_b = pd.DataFrame(oof_b_rows)
assert len(oof_b) == N_TOTAL, f"OOF B size {len(oof_b)} != {N_TOTAL}"
assert oof_b["ID"].is_unique, "Duplicate IDs in Pipeline B"

OOF_B_PATH = os.path.join(DIRS["predictions"], "rq3_loso_s1_predictions.csv")
oof_b.to_csv(OOF_B_PATH, index=False)
print(f"  ✓ S1 OOF saved: {OOF_B_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. MERGE PIPELINES A & B
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 5] Merging Pipeline A and B on ID...")

oof_a_sorted = oof_a.sort_values("ID").reset_index(drop=True)
oof_b_sorted = oof_b.sort_values("ID").reset_index(drop=True)

oof_a_renamed = oof_a_sorted[["ID", "school_id", "label", "y_prob", "y_pred"]].rename(
    columns={"y_prob": "prob_s0", "y_pred": "pred_s0"}
)

paired = oof_a_renamed.merge(
    oof_b_sorted[["ID", "y_prob_s1", "y_pred_s1"]].rename(
        columns={"y_prob_s1": "prob_s1", "y_pred_s1": "pred_s1"}
    ),
    on="ID",
    how="inner"
)

assert len(paired) == N_TOTAL, f"Paired merge size {len(paired)} != {N_TOTAL}"
assert paired["ID"].is_unique

y_all   = paired["label"].values.astype(int)
prob_s0 = paired["prob_s0"].values.astype(float)
pred_s0 = paired["pred_s0"].values.astype(int)
prob_s1 = paired["prob_s1"].values.astype(float)
pred_s1 = paired["pred_s1"].values.astype(int)

print(f"  ✓ Paired table: {len(paired)} rows")

# ══════════════════════════════════════════════════════════════════════════════
# 8. POOLED METRICS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 6] Pooled global metrics...")

m_s0 = metric_bundle(y_all, prob_s0, pred_s0)
m_s1 = metric_bundle(y_all, prob_s1, pred_s1)

print(f"\n  {'Metric':<18} {'S0 (AQ-10)':>12} {'S1 (AQ-10+SQ)':>14} {'Δ':>10}")
print("  " + "-" * 58)
for k in ["AUROC", "Sensitivity", "Specificity", "Accuracy", "F1_Score", "Brier"]:
    d = m_s1[k] - m_s0[k]
    print(f"  {k:<18} {m_s0[k]:>12.4f} {m_s1[k]:>14.4f} {d:>+10.4f}")

delta_fp = int(m_s1["FP"] - m_s0["FP"])
delta_fn = int(m_s1["FN"] - m_s0["FN"])
print(f"\n  ΔFP = {delta_fp:+d}  |  ΔFN = {delta_fn:+d}")

# ══════════════════════════════════════════════════════════════════════════════
# 9. PRIMARY TEST: PAIRED DELONG
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("[Step 7] PRIMARY TEST: Paired DeLong (S0 vs S1)")
print("=" * 78)

auc_s0, auc_s1, z_delong, p_delong = delong_paired_test(y_all, prob_s0, prob_s1)
delta_auroc = auc_s1 - auc_s0
delong_sig = (not np.isnan(p_delong)) and (p_delong < ALPHA) and (delta_auroc > 0)

print(f"  AUROC S0     : {auc_s0:.4f}")
print(f"  AUROC S1     : {auc_s1:.4f}")
print(f"  ΔAUROC (S1-S0): {delta_auroc:+.4f}")
print(f"  z-statistic  : {z_delong:.4f}")
print(f"  p-value      : {p_delong:.6g}")
print(f"  Significant (α=0.05) and S1 better: {delong_sig}")

delong_result = {
    "test": "DeLong_paired_S0_vs_S1",
    "auroc_s0": round(float(auc_s0), 4),
    "auroc_s1": round(float(auc_s1), 4),
    "delta_auroc": round(float(delta_auroc), 4),
    "z_statistic": round(float(z_delong), 4) if not np.isnan(z_delong) else None,
    "p_value": round(float(p_delong), 6) if not np.isnan(p_delong) else None,
    "significant_and_s1_better": bool(delong_sig),
}

# ══════════════════════════════════════════════════════════════════════════════
# 10. SECONDARY TEST: MCNEMAR
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("[Step 8] SECONDARY TEST: McNemar (S0 binary vs S1 binary)")
print("=" * 78)

b_mc, c_mc, n_disc, agree_pp, agree_nn, chi2_mc, p_mc = mcnemar_test(pred_s0, pred_s1)
s1_mcnemar_better = c_mc > b_mc
mcnemar_sig = (p_mc < ALPHA) and s1_mcnemar_better

print(f"\n  Discordant pairs:")
print(f"    b = S0 flags, S1 misses : {b_mc}")
print(f"    c = S1 flags, S0 misses : {c_mc}")
print(f"  n_discordant = {n_disc}")
print(f"  chi² (continuity-corr) = {chi2_mc:.4f}")
print(f"  p-value = {p_mc:.6g}")
print(f"  S1 fixes more than breaks: {s1_mcnemar_better}")
print(f"  Significant (α=0.05) and S1 better: {mcnemar_sig}")

mcnemar_result = {
    "test": "McNemar_S0_binary_vs_S1_binary",
    "b_S0flags_S1misses": b_mc,
    "c_S1flags_S0misses": c_mc,
    "n_discordant": n_disc,
    "chi2_continuity_corrected": round(float(chi2_mc), 4),
    "p_value": round(float(p_mc), 6),
    "s1_fixes_more": bool(s1_mcnemar_better),
    "significant_and_s1_better": bool(mcnemar_sig),
}

# ══════════════════════════════════════════════════════════════════════════════
# 11. PAIRED BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[Step 9] Paired bootstrap (B={N_BOOTSTRAP})...")

boot = paired_bootstrap(y_all, prob_s0, prob_s1, N_BOOTSTRAP, SEED)

for b_idx in [500, 1000, 1500, 2000]:
    if len(boot["delta_auroc"]) >= b_idx:
        print(f"    bootstrap {b_idx}/{N_BOOTSTRAP}")

boot_summary = {}
for key, arr in boot.items():
    lo, hi = percentile_ci(arr)
    boot_summary[key] = {
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "ci_lower": round(float(lo), 4) if not np.isnan(lo) else None,
        "ci_upper": round(float(hi), 4) if not np.isnan(hi) else None,
        "formatted": f"{arr.mean():.4f} [{lo:.4f}, {hi:.4f}]" \
                     if not (np.isnan(lo) or np.isnan(hi)) else f"{arr.mean():.4f} [NA]",
    }

print("  ✓ Bootstrap complete")

# ══════════════════════════════════════════════════════════════════════════════
# 12. PRE-SPECIFIED F* LOCKING DECISION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("[Step 10] F* LOCKING DECISION (pre-specified rule)")
print("=" * 78)

statistical_improvement = delong_sig or mcnemar_sig
clinical_safety = (delta_fp <= MAX_FP_INFLATION) and (delta_fn < MAX_FN_INCREASE)

if statistical_improvement and clinical_safety:
    LOCKED_FSTAR = "S1"
    LOCKED_COLS = S1_COLS
    fstar_reason = (
        f"S1 showed statistically significant improvement over S0 "
        f"(DeLong p={p_delong:.4g}, McNemar p={p_mc:.4g}) "
        f"with acceptable clinical safety (ΔFP={delta_fp:+d} ≤ {MAX_FP_INFLATION}, "
        f"ΔFN={delta_fn:+d} < {MAX_FN_INCREASE}). "
        f"Locking F*=S1 (AQ-10 + SQ1-4)."
    )
elif statistical_improvement and not clinical_safety:
    LOCKED_FSTAR = "S0"
    LOCKED_COLS = S0_COLS
    fstar_reason = (
        f"S1 showed statistical improvement (DeLong p={p_delong:.4g}, "
        f"McNemar p={p_mc:.4g}) but violated clinical safety limits "
        f"(ΔFP={delta_fp:+d} or ΔFN={delta_fn:+d}). "
        f"For screening safety, lock F*=S0 (AQ-10 only)."
    )
else:
    LOCKED_FSTAR = "S0"
    LOCKED_COLS = S0_COLS
    fstar_reason = (
        f"S1 did not show statistically significant improvement over S0 "
        f"(DeLong p={p_delong:.4g}, McNemar p={p_mc:.4g}). "
        f"Locking F*=S0 (AQ-10 only) for parsimony and clinical feasibility."
    )

print(f"\n  ★ LOCKED F* = {LOCKED_FSTAR}")
print(f"    Features: {len(LOCKED_COLS)} columns")
print(f"    Reason : {fstar_reason}")

# ══════════════════════════════════════════════════════════════════════════════
# 13. TABLES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 11] Building tables...")

comparison_row = {
    "Comparison": "S1 (AQ-10+SQ1-4) vs S0 (AQ-10)",
    "N": N_TOTAL,
    "Schools": N_SCHOOLS,
    "AUROC_S0": round(float(m_s0["AUROC"]), 4),
    "AUROC_S1": round(float(m_s1["AUROC"]), 4),
    "Delta_AUROC_Point": round(float(delta_auroc), 4),
    "Delta_AUROC_Bootstrap": boot_summary["delta_auroc"]["formatted"],
    "DeLong_Z": round(float(z_delong), 4) if not np.isnan(z_delong) else "NaN",
    "DeLong_P": f"{p_delong:.6g}" if not np.isnan(p_delong) else "NaN",
    "DeLong_Significant_S1Better": delong_sig,
    "McNemar_Chi2": round(float(chi2_mc), 4),
    "McNemar_P": f"{p_mc:.6g}",
    "McNemar_Significant_S1Better": mcnemar_sig,
    "Sensitivity_S0": round(float(m_s0["Sensitivity"]), 4),
    "Sensitivity_S1": round(float(m_s1["Sensitivity"]), 4),
    "Specificity_S0": round(float(m_s0["Specificity"]), 4),
    "Specificity_S1": round(float(m_s1["Specificity"]), 4),
    "FP_S0": int(m_s0["FP"]),
    "FP_S1": int(m_s1["FP"]),
    "FN_S0": int(m_s0["FN"]),
    "FN_S1": int(m_s1["FN"]),
    "Delta_FP": delta_fp,
    "Delta_FN": delta_fn,
    "LOCKED_FSTAR": LOCKED_FSTAR,
    "FSTAR_Reason": fstar_reason,
}
comp_df = pd.DataFrame([comparison_row])
save_tbl(comp_df, "T_rq3_paired_comparison.csv")
save_pr(comp_df, "T_rq3_paper_paired_comparison.csv")

boot_rows = []
for key, label in [
    ("delta_auroc", "ΔAUROC"), ("delta_sens", "ΔSensitivity"),
    ("delta_spec", "ΔSpecificity"), ("delta_fp", "ΔFP"),
    ("delta_fn", "ΔFN"), ("delta_f1", "ΔF1"), ("delta_brier", "ΔBrier"),
]:
    s = boot_summary[key]
    boot_rows.append({
        "Metric": label, "Mean": s["mean"], "Median": s["median"],
        "CI_Lower": s["ci_lower"], "CI_Upper": s["ci_upper"],
        "Formatted": s["formatted"],
    })
boot_df = pd.DataFrame(boot_rows)
save_tbl(boot_df, "T_rq3_bootstrap_deltas.csv")
save_pr(boot_df, "T_rq3_paper_bootstrap_deltas.csv")

school_rows = []
for sid in SCHOOLS:
    mask = paired["school_id"] == sid
    sub = paired[mask]
    y_s = sub["label"].values
    p0 = sub["prob_s0"].values
    pr0 = sub["pred_s0"].values
    p1 = sub["prob_s1"].values
    pr1 = sub["pred_s1"].values

    n = len(y_s)
    n_pos = int(y_s.sum())
    n_neg = n - n_pos
    school_type = "ASD" if n_pos > 0 else "Normal"
    flag_rate_s0 = float(pr0.mean())
    flag_rate_s1 = float(pr1.mean())

    school_rows.append({
        "school_id": sid, "school_type": school_type, "n": n,
        "n_pos": n_pos, "n_neg": n_neg,
        "flag_rate_s0": round(flag_rate_s0, 4),
        "flag_rate_s1": round(flag_rate_s1, 4),
        "delta_flag_rate": round(flag_rate_s1 - flag_rate_s0, 4),
    })

school_df = pd.DataFrame(school_rows)
save_tbl(school_df, "T_rq3_per_school_descriptive.csv")
save_pr(school_df, "T_rq3_paper_per_school.csv")

print("  ✓ All tables saved")

# ══════════════════════════════════════════════════════════════════════════════
# 14. FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 12] Generating figures...")

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

C_S0 = "#ff7f0e"
C_S1 = "#9467bd"

# ── Figure 1: ROC Curves ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
fpr0, tpr0, _ = roc_curve(y_all, prob_s0)
fpr1, tpr1, _ = roc_curve(y_all, prob_s1)
ax.plot(fpr0, tpr0, "--", lw=2.0, color=C_S0,
        label=f"S0 — AQ-10 only      AUC = {m_s0['AUROC']:.4f}")
ax.plot(fpr1, tpr1, "-", lw=2.5, color=C_S1,
        label=f"S1 — AQ-10 + SQ1-4   AUC = {m_s1['AUROC']:.4f}")
ax.plot([0, 1], [0, 1], ":", color="gray", alpha=0.5)

ax.scatter(1-m_s0["Specificity"], m_s0["Sensitivity"],
           marker="D", s=90, color=C_S0, zorder=5, edgecolor="white",
           label=f"S0 @ 0.50  (Sens={m_s0['Sensitivity']:.3f})")
ax.scatter(1-m_s1["Specificity"], m_s1["Sensitivity"],
           marker="s", s=90, color=C_S1, zorder=5, edgecolor="white",
           label=f"S1 @ 0.50  (Sens={m_s1['Sensitivity']:.3f})")

pval_txt = f"DeLong p = {p_delong:.4g}" if not np.isnan(p_delong) else "DeLong p = N/A"
ax.set_title(f"RQ3 — Paired LOSO ROC: S0 vs S1\n"
             f"{pval_txt}  |  F* = {LOCKED_FSTAR}")
ax.set_xlabel("1 - Specificity (FPR)")
ax.set_ylabel("Sensitivity (TPR)")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, linestyle=":", alpha=0.4)
save_fig(fig, "F_rq3_1_roc_s0_vs_s1.png")
print("  ✓ F1: ROC comparison")

# ── Figure 2: Bootstrap ΔAUROC ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
arr = boot["delta_auroc"]
lo, hi = percentile_ci(arr)
ax.hist(arr, bins=60, color=C_S1, edgecolor="white", alpha=0.85, density=True)
ax.axvline(0, color="black", lw=1.5, ls="--", label="No difference")
ax.axvline(arr.mean(), color=C_S1, lw=2, label=f"Mean Δ = {arr.mean():.4f}")
if not np.isnan(lo):
    ax.axvline(lo, color="gray", lw=1.5, ls=":", label=f"95% CI [{lo:.4f}, {hi:.4f}]")
    ax.axvline(hi, color="gray", lw=1.5, ls=":")
ax.set_xlabel("ΔAUROC (S1 − S0)")
ax.set_ylabel("Density")
ax.set_title(f"RQ3 — Paired Bootstrap ΔAUROC (B={N_BOOTSTRAP})\n"
             f"F* = {LOCKED_FSTAR}")
ax.legend(fontsize=9)
ax.grid(True, linestyle=":", alpha=0.4)
save_fig(fig, "F_rq3_2_bootstrap_delta_auroc.png")
print("  ✓ F2: Bootstrap ΔAUROC")

# ── Figure 3: Per-School Flagging Rates ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(10, N_SCHOOLS*0.8), 5))
x = np.arange(N_SCHOOLS)
bar_w = 0.35
ax.bar(x - bar_w/2, school_df["flag_rate_s0"], bar_w, color="#cccccc",
       alpha=0.8, edgecolor="white", label="S0 flagging rate")
ax.bar(x + bar_w/2, school_df["flag_rate_s1"], bar_w, color=C_S1,
       alpha=0.85, edgecolor="white", label="S1 flagging rate")

ax.set_xticks(x)
ax.set_xticklabels(school_df["school_id"], rotation=45, ha="right", fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_ylabel("Flagging rate")
ax.set_title("RQ3 — Per-School Flagging Rates (Mono-Class Schools)\n"
             "ASD schools: higher = more sensitive  |  "
             "Normal schools: lower = more specific")
ax.legend(fontsize=9)
ax.grid(True, axis="y", linestyle=":", alpha=0.4)
save_fig(fig, "F_rq3_3_per_school_flagging_rates.png")
print("  ✓ F3: Per-school flagging rates")

# ── Figure 4: Confusion Matrices ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax_, prds, title in [
    (axes[0], pred_s0, f"S0 — AQ-10 only\nAUROC={m_s0['AUROC']:.4f}"),
    (axes[1], pred_s1, f"S1 — AQ-10 + SQ1-4\nAUROC={m_s1['AUROC']:.4f}"),
]:
    cm = confusion_matrix(y_all, prds, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "ASD"]).plot(
        ax=ax_, colorbar=False, cmap="Blues")
    ax_.set_title(title, fontsize=11)
fig.suptitle(f"RQ3 — Pooled OOF Confusion Matrices  (N={N_TOTAL}, F*={LOCKED_FSTAR})", y=1.04)
fig.tight_layout()
save_fig(fig, "F_rq3_4_confusion_matrices.png")
print("  ✓ F4: Confusion matrices")

# ── Figure 5: Calibration ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k:", alpha=0.6, label="Perfect calibration")
for probs, name, color, ls in [
    (prob_s0, f"S0 (AQ-10)       ECE={m_s0['ECE']:.4f}", C_S0, "--"),
    (prob_s1, f"S1 (AQ-10+SQ1-4) ECE={m_s1['ECE']:.4f}", C_S1, "-"),
]:
    cx, cy = reliability_curve(y_all, probs)
    if len(cx) > 0:
        ax.plot(cx, cy, ls+"o", color=color, lw=2, label=name)
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed fraction positive (ASD)")
ax.set_title(f"RQ3 — Calibration Comparison  (N={N_TOTAL})")
ax.legend(loc="upper left")
ax.grid(True, linestyle=":", alpha=0.4)
save_fig(fig, "F_rq3_5_calibration.png")
print("  ✓ F5: Calibration")

# ══════════════════════════════════════════════════════════════════════════════
# 15. MASTER SUMMARY JSON
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 13] Writing RQ3_Master_Summary.json...")

oof_fstar_path = OOF_B_PATH if LOCKED_FSTAR == "S1" else OOF_A_PATH

rq3_summary = {
    "module": "RQ3_Questionnaire_Augmentation_Paired_LOSO",
    "version": "FINAL-MONO-11.1-RQ3",
    "status": "COMPLETE",
    "timestamp": datetime.now().isoformat(),
    "locked_algorithm": "Support Vector Machine",
    "locked_hyperparameters": LOCKED_PARAMS,
    "locked_feature_set": LOCKED_FSTAR,
    "locked_feature_columns": LOCKED_COLS,
    "fstar_reason": fstar_reason,
    "loso_n_schools": N_SCHOOLS,
    "pooled_oof_n": N_TOTAL,
    "pooled_s0_metrics": {k: float(m_s0[k]) for k in METRIC_KEYS},
    "pooled_s1_metrics": {k: float(m_s1[k]) for k in METRIC_KEYS},
    "primary_test_delong": delong_result,
    "secondary_test_mcnemar": mcnemar_result,
    "clinical_safety": {
        "max_fp_inflation_allowed": MAX_FP_INFLATION,
        "max_fn_increase_allowed": MAX_FN_INCREASE,
        "actual_delta_fp": delta_fp,
        "actual_delta_fn": delta_fn,
        "clinical_safety_passed": clinical_safety,
    },
    "paired_bootstrap": {
        "n_boot": N_BOOTSTRAP,
        "deltas": boot_summary,
    },
    "predictions_file_s0": OOF_A_PATH,
    "predictions_file_s1": OOF_B_PATH,
    "predictions_file_fstar": oof_fstar_path,
    "limitation_note": (
        "Schools were mono-class by design. Per-school AUROC/sensitivity/"
        "specificity were undefined for held-out folds in LOSO. All primary "
        "comparisons used pooled out-of-fold predictions. Per-school tables "
        "report flagging rates as descriptive institutional characterizations."
    ),
}

for path in [
    os.path.join(DIRS["results"], "RQ3_Master_Summary.json"),
    os.path.join(DIRS["config"], "RQ3_Master_Summary.json"),
]:
    with open(path, "w") as f:
        json.dump(rq3_summary, f, indent=4, default=str)

print("  ✓ Master summary saved")

# ══════════════════════════════════════════════════════════════════════════════
# 16. DONE FLAG
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 14] Writing completion flag...")

flag_content = (
    f"RQ3 FINAL-MONO-11.1-RQ3 COMPLETE\n"
    f"timestamp={datetime.now().isoformat()}\n"
    f"locked_fstar={LOCKED_FSTAR}\n"
    f"locked_feature_columns={LOCKED_COLS}\n"
    f"auroc_s0={m_s0['AUROC']:.6f}\n"
    f"auroc_s1={m_s1['AUROC']:.6f}\n"
    f"delta_auroc={delta_auroc:+.6f}\n"
    f"delong_z={z_delong:.6f}\n"
    f"delong_p={p_delong:.6g}\n"
    f"mcnemar_chi2={chi2_mc:.6f}\n"
    f"mcnemar_p={p_mc:.6g}\n"
    f"delta_fp={delta_fp:+d}\n"
    f"delta_fn={delta_fn:+d}\n"
    f"fstar_reason={fstar_reason}\n"
    f"oof_fstar_path={oof_fstar_path}\n"
    f"next=RQ4_explainability\n"
)
for fp in [os.path.join(DIRS["logs"], "rq3_done.flag"),
           os.path.join(PROJECT, "logs", "rq3_done.flag")]:
    with open(fp, "w") as f:
        f.write(flag_content)
print(f"  ✓ Flag saved")

# ══════════════════════════════════════════════════════════════════════════════
# 17. FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("RQ3 SUMMARY — PAIRED LOSO: S0 vs S1")
print("=" * 78)

print(f"\nDESIGN")
print(f"  Schools           : {N_SCHOOLS} (mono-class)")
print(f"  Pooled OOF        : {N_TOTAL}")

print(f"\nMETRICS")
print(f"  {'Metric':<16} {'S0':>10} {'S1':>10} {'Δ':>10}")
print("  " + "-" * 50)
for k in ["AUROC", "Sensitivity", "Specificity", "Accuracy", "F1_Score", "Brier"]:
    print(f"  {k:<16} {m_s0[k]:>10.4f} {m_s1[k]:>10.4f} {m_s1[k]-m_s0[k]:>+10.4f}")

print(f"\nTESTS")
print(f"  DeLong (primary) : z={z_delong:.4f}, p={p_delong:.6g}, S1 better & sig: {delong_sig}")
print(f"  McNemar (sec)    : χ²={chi2_mc:.4f}, p={p_mc:.6g}, S1 better & sig: {mcnemar_sig}")
print(f"  Discordant       : b={b_mc} (S0 wins), c={c_mc} (S1 wins)")

print(f"\nCLINICAL SAFETY")
print(f"  ΔFP = {delta_fp:+d} (limit ≤{MAX_FP_INFLATION})")
print(f"  ΔFN = {delta_fn:+d} (limit <{MAX_FN_INCREASE})")
print(f"  Safety passed    : {clinical_safety}")

print(f"\n★ LOCKED F* = {LOCKED_FSTAR}")
print(f"  Reason: {fstar_reason}")

print(f"\nOUTPUTS")
print(f"  Tables    : {DIRS['tables']}")
print(f"  Figures   : {DIRS['figures']}")
print(f"  F* OOF    : {oof_fstar_path}")

print("\n" + "=" * 78)
print("STATUS: RQ3 COMPLETE")
print("=" * 78)

