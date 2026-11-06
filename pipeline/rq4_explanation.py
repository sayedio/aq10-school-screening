

import subprocess, sys, os, json, time, random, warnings
from datetime import datetime
from itertools import combinations

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

np.random.seed(SEED)

try:
    import shap
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "shap"],
                   capture_output=True)
    import shap

# from google.colab import drive  # Colab-only; locally, set the AQ10_BASE environment variable instead
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    roc_auc_score, recall_score, f1_score,
    accuracy_score, brier_score_loss, confusion_matrix,
)
from scipy.stats import spearmanr, pearsonr

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — PATHS & DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 78)
print("RQ4 — EXPLAINABILITY (XAI) & SUBGROUP PERFORMANCE ANALYSIS")
print("BD-AQ10(+SQ4) MULTI-CENTER SCHOOL SCREENING | FINAL-Q1-10.0-RQ4-XAI-POOLEDPI")
print("=" * 78)

# drive.mount("/content/drive", force_remount=True)

BASE    = os.environ.get("AQ10_BASE", "/content/drive/MyDrive/Autism")
PROJECT = os.path.join(BASE, "aq_10")          # FIXED: must match RQ3
PROC    = os.path.join(PROJECT, "data", "processed")

RQ3_FLAG     = os.path.join(PROJECT, "rq3_augmentation", "logs", "rq3_done.flag")
RQ3_FLAG_ALT = os.path.join(PROJECT, "logs", "rq3_done.flag")
RQ3_JSON     = os.path.join(PROJECT, "rq3_augmentation", "results", "RQ3_Master_Summary.json")
BD_CLEAN     = os.path.join(PROC, "bd_clean.csv")

RQ4 = os.path.join(PROJECT, "rq4")
DIRS = {
    "tables":      os.path.join(RQ4, "tables"),
    "paper_ready": os.path.join(RQ4, "tables", "paper_ready"),
    "figures":     os.path.join(RQ4, "figures"),
    "results":     os.path.join(RQ4, "results"),
    "logs":        os.path.join(RQ4, "logs"),
    "pt_tables":   os.path.join(PROJECT, "paper_results", "rq4", "tables"),
    "pt_figures":  os.path.join(PROJECT, "paper_results", "rq4", "figures"),
}
for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

print(f"\n  Project : {PROJECT}")
print(f"  Output  : {RQ4}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS, HELPERS
# ══════════════════════════════════════════════════════════════════════════════

THRESHOLD           = 0.50
N_SHAP_BG           = 50
N_PI_REPEATS_GLOBAL = 500
N_BOOT_SCHOOL       = 2000
N_BOOT_SUBGROUP     = 2000
SMALL_N             = 30
NOTEWORTHY_GAP      = 0.05
EPS                 = 1e-7

LOCKED_PARAMS = dict(C=1.0, kernel="rbf", gamma="scale",
                     probability=True, random_state=SEED)

AQ_LABELS = {
    # Display labels follow the published AQ-10 Child wording (Allison et al., 2012) and the
    # study codebook. Corrected after the locked run: labels only, no numbers change.
    "A_Q1":   "Q1 – Notices small sounds",
    "A_Q2":   "Q2 – Whole picture over details (R)",
    "A_Q3":   "Q3 – Follows several conversations (R)",
    "A_Q4":   "Q4 – Switches activities easily (R)",
    "A_Q5":   "Q5 – Cannot keep a conversation going",
    "A_Q6":   "Q6 – Good at social chit-chat (R)",
    "A_Q7":   "Q7 – Hard to infer characters' intentions",
    "A_Q8":   "Q8 – Enjoyed pretend play (R)",
    "A_Q9":   "Q9 – Reads feelings from faces (R)",
    "A_Q10":  "Q10 – Hard to make new friends",
    "A_SQ1":  "SQ1 – Distress when routines change",
    "A_SQ2":  "SQ2 – Repeats movements or play patterns",
    "A_SQ3":  "SQ3 – Unusual sensory sensitivity",
    "A_SQ4":  "SQ4 – Eye contact or response to name (R)",
}

def feat_label(col): return AQ_LABELS.get(col, col)

def to_native(obj):
    if isinstance(obj, np.ndarray):          return obj.tolist()
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    if isinstance(obj, dict):                return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):       return [to_native(v) for v in obj]
    return obj

def rho_interpretation(rho):
    a = abs(float(rho))
    if a < 0.20:   strength = "Slight"
    elif a < 0.40: strength = "Fair"
    elif a < 0.60: strength = "Moderate"
    elif a < 0.80: strength = "Substantial"
    else:          strength = "Almost Perfect"
    direction = "agreement" if float(rho) >= 0 else "disagreement (inverse ranking)"
    return f"{strength} {direction}"

def rho_strength_only(rho):
    a = abs(float(rho))
    if a < 0.20: return "Slight"
    if a < 0.40: return "Fair"
    if a < 0.60: return "Moderate"
    if a < 0.80: return "Substantial"
    return "Almost Perfect"

def safe_metrics(y_true, y_prob, y_pred):
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    y_pred = np.asarray(y_pred, int)
    n_pos, n_neg = int(y_true.sum()), int((y_true == 0).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return dict(
        AUROC=float(roc_auc_score(y_true, y_prob)) if n_pos > 0 and n_neg > 0 else np.nan,
        Sensitivity=float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
        Specificity=float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
        F1=float(f1_score(y_true, y_pred, zero_division=0)),
        Accuracy=float(accuracy_score(y_true, y_pred)),
        TP=tp, TN=tn, FP=fp, FN=fn, n=len(y_true), n_pos=n_pos, n_neg=n_neg,
    )

def pooled_log_loss(y_true, y_prob, eps=EPS):
    p = np.clip(np.asarray(y_prob, float), eps, 1 - eps)
    y = np.asarray(y_true, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

def percentile_ci(boot_vals, alpha=0.05):
    boot_vals = np.asarray(boot_vals, float)
    boot_vals = boot_vals[~np.isnan(boot_vals)]
    if len(boot_vals) < 50:
        return np.nan, np.nan
    return float(np.percentile(boot_vals, 100 * alpha / 2)), \
           float(np.percentile(boot_vals, 100 * (1 - alpha / 2)))

def subgroup_bootstrap_ci(y_true, y_prob, y_pred, n_boot=2000, seed=SEED):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    store = {"sens": [], "spec": [], "auroc": []}
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        m = safe_metrics(y_b, y_prob[idx], y_pred[idx])
        store["sens"].append(m["Sensitivity"])
        store["spec"].append(m["Specificity"])
        store["auroc"].append(m["AUROC"])
    out = {}
    for k, v in store.items():
        arr = np.asarray(v)
        arr = arr[~np.isnan(arr)]
        out[k] = percentile_ci(arr) if len(arr) >= 50 else (np.nan, np.nan)
    return out

def save_tbl(df, name):
    for d in [DIRS["tables"], DIRS["pt_tables"]]:
        df.to_csv(os.path.join(d, name), index=False)

def save_pr(df, name):
    for d in [DIRS["paper_ready"], DIRS["pt_tables"]]:
        df.to_csv(os.path.join(d, name), index=False)

def save_fig(fig, name):
    for d in [DIRS["figures"], DIRS["pt_figures"]]:
        fig.savefig(os.path.join(d, name), dpi=300, bbox_inches="tight")
    plt.close(fig)

def build_model():
    return Pipeline([("scaler", StandardScaler()), ("clf", SVC(**LOCKED_PARAMS))])


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — GATE CHECK & LOAD F*
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 1] Verifying upstream locks...")

if not (os.path.exists(RQ3_FLAG) or os.path.exists(RQ3_FLAG_ALT)):
    raise FileNotFoundError(f"Missing RQ3 flag.\nTried: {RQ3_FLAG}\n       {RQ3_FLAG_ALT}")
if not os.path.exists(RQ3_JSON):
    raise FileNotFoundError(f"Missing: {RQ3_JSON}")
if not os.path.exists(BD_CLEAN):
    raise FileNotFoundError(f"Missing: {BD_CLEAN}")

with open(RQ3_JSON) as f:
    rq3 = json.load(f)

FSTAR      = rq3["locked_feature_set"]
FSTAR_COLS = rq3["locked_feature_columns"]
FSTAR_N    = len(FSTAR_COLS)
OOF_PATH   = rq3["predictions_file_fstar"]

if not os.path.exists(OOF_PATH):
    raise FileNotFoundError(f"OOF file not found: {OOF_PATH}")

print(f"  RQ3 gate OK")
print(f"  F* = {FSTAR}  ({FSTAR_N} features): {FSTAR_COLS}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 2] Loading data...")

bd = pd.read_csv(BD_CLEAN).reset_index(drop=True)
for c in FSTAR_COLS:
    if c not in bd.columns:
        raise KeyError(f"Locked feature '{c}' missing from bd_clean.csv")
    bd[c] = bd[c].astype(int)
bd["label"] = bd["label"].astype(int)

# Normalize demographic columns to prevent duplicate subgroups (e.g. 'F' vs 'f')
if "Gender" in bd.columns:
    bd["Gender"] = bd["Gender"].astype(str).str.strip().str.lower()
if "family_hx" in bd.columns:
    bd["family_hx"] = bd["family_hx"].astype(str).str.strip().str.lower()
    bd["family_hx"] = bd["family_hx"].replace({
        "1": "yes", "0": "no", "true": "yes", "false": "no"
    })

AQ10_COLS = [f"A_Q{i}" for i in range(1, 11)]
if "AQ10_score" not in bd.columns:
    bd["AQ10_score"] = bd[AQ10_COLS].ge(2).sum(axis=1)

SCHOOLS   = sorted(bd["school_id"].unique())
N_SCHOOLS = len(SCHOOLS)
N_TOTAL   = len(bd)

oof = pd.read_csv(OOF_PATH)
assert len(oof) == N_TOTAL and oof["ID"].is_unique, "OOF ID mismatch"

oof_full = oof.merge(
    bd[["ID", "school_id", "label", "Age", "Gender", "family_hx", "AQ10_score"] + FSTAR_COLS],
    on="ID", how="left", suffixes=("", "_dup")
)
oof_full = oof_full[[c for c in oof_full.columns if not c.endswith("_dup")]]

y_all    = oof_full["label"].values.astype(int)
prob_all = oof_full["y_prob"].values.astype(float)
pred_all = (prob_all >= THRESHOLD).astype(int)
global_m = safe_metrics(y_all, prob_all, pred_all)
GLOBAL_SENS = global_m["Sensitivity"]
GLOBAL_SPEC = global_m["Specificity"]

feat_corr = bd[FSTAR_COLS].corr(method="pearson")
corr_vals = feat_corr.abs().values
max_corr  = float(corr_vals[np.triu_indices_from(corr_vals, k=1)].max())
mean_corr = float(corr_vals[np.triu_indices_from(corr_vals, k=1)].mean())
high_corr_pairs = []
for i, j in combinations(range(FSTAR_N), 2):
    r = float(feat_corr.iloc[i, j])
    if abs(r) >= 0.50:
        high_corr_pairs.append({
            "Feature_A": FSTAR_COLS[i], "Feature_B": FSTAR_COLS[j],
            "Pearson_r": round(r, 4), "Abs_r": round(abs(r), 4),
        })

print(f"  bd_clean: {N_TOTAL} rows | {N_SCHOOLS} schools")
print(f"  Global  : Sens={GLOBAL_SENS:.4f}  Spec={GLOBAL_SPEC:.4f}  AUROC={global_m['AUROC']:.4f}")
print(f"  Feature correlation: max|r|={max_corr:.3f}  mean|r|={mean_corr:.3f}")
if high_corr_pairs:
    print(f"  WARNING: {len(high_corr_pairs)} feature pairs with |r|>=0.50")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4A — LOSO LOOP: KernelSHAP + collect per-fold artifacts for pooled PI
# ══════════════════════════════════════════════════════════════════════════════

N_SHAP_SAMPLES = min(2 ** FSTAR_N, 1024) if FSTAR_N <= 10 else 1024

print("\n" + "=" * 78)
print("[Step 3] LOSO Loop: KernelSHAP  (+ storing fold models for pooled PI)")
print(f"         F*={FSTAR} | {FSTAR_N} features")
print(f"         SHAP: {N_SHAP_BG} background | {N_SHAP_SAMPLES} coalitions/instance")
print("=" * 78)

shap_matrix   = np.zeros((N_TOTAL, FSTAR_N))
feat_vals_raw = np.zeros((N_TOTAL, FSTAR_N))

fold_clf, fold_scaler    = [], []
fold_Xh_raw, fold_Xt_raw = [], []
fold_n, fold_isasd       = [], []
fold_yh, fold_base_prob  = [], []

ids_in_order       = []
school_id_in_order = []
additivity_errors  = []

school_types = []
t0 = time.time()
cursor = 0

for si, sid in enumerate(SCHOOLS):
    held_m  = bd["school_id"] == sid
    train_m = ~held_m
    held_df  = bd[held_m].copy()
    train_df = bd[train_m].copy()

    Xh_raw = held_df[FSTAR_COLS].values.astype(float)
    yh     = held_df["label"].values.astype(int)
    Xt_raw = train_df[FSTAR_COLS].values.astype(float)
    yt     = train_df["label"].values.astype(int)

    is_asd = (yh.sum() == len(yh))
    school_types.append("ASD" if is_asd else "Normal")
    n_held = len(yh)

    if len(np.unique(yt)) < 2:
        raise RuntimeError(f"Fold {sid}: training mono-class — invalid LOSO")

    mdl = build_model()
    mdl.fit(Xt_raw, yt)
    scaler = mdl.named_steps["scaler"]
    clf    = mdl.named_steps["clf"]

    Xh_sc = scaler.transform(Xh_raw)
    Xt_sc = scaler.transform(Xt_raw)

    rng_shap = np.random.RandomState(SEED + si * 7)
    bg_idx   = rng_shap.choice(len(Xt_sc), min(N_SHAP_BG, len(Xt_sc)), replace=False)
    background = Xt_sc[bg_idx]

    def predict_asd_prob(x):
        return clf.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_asd_prob, background)
    shap_vals = explainer.shap_values(Xh_sc, nsamples=N_SHAP_SAMPLES, silent=True)

    pred_check = predict_asd_prob(Xh_sc)
    expected_val = float(explainer.expected_value)
    shap_sum = shap_vals.sum(axis=1) + expected_val
    fold_add_err = float(np.max(np.abs(pred_check - shap_sum)))
    additivity_errors.append(fold_add_err)
    if fold_add_err > 0.05:
        print(f"    WARNING: Fold {sid} max additivity error = {fold_add_err:.4f}")

    shap_matrix[cursor:cursor+n_held]   = shap_vals
    feat_vals_raw[cursor:cursor+n_held] = Xh_raw
    ids_in_order.extend(held_df["ID"].tolist())
    school_id_in_order.extend([sid] * n_held)
    cursor += n_held

    base_prob = clf.predict_proba(Xh_sc)[:, 1]
    fold_clf.append(clf)
    fold_scaler.append(scaler)
    fold_Xh_raw.append(Xh_raw)
    fold_Xt_raw.append(Xt_raw)
    fold_n.append(n_held)
    fold_isasd.append(is_asd)
    fold_yh.append(yh)
    fold_base_prob.append(base_prob)

    top_shap_j = int(np.argmax(np.abs(shap_vals).mean(axis=0)))
    print(f"  Fold {si+1:2d}/{N_SCHOOLS}  {sid}  ({'ASD   ' if is_asd else 'Normal'})  "
          f"n={n_held:3d}  SHAP-top={FSTAR_COLS[top_shap_j]}  add_err={fold_add_err:.4f}")

print(f"\n  LOSO loop complete in {time.time()-t0:.1f}s")
print(f"  SHAP matrix: {shap_matrix.shape}")
print(f"  Max additivity error across folds: {max(additivity_errors):.6f}")

asd_idx    = [i for i, t in enumerate(school_types) if t == "ASD"]
normal_idx = [i for i, t in enumerate(school_types) if t == "Normal"]

y_pool         = np.concatenate(fold_yh)
base_prob_pool = np.concatenate(fold_base_prob)
assert len(y_pool) == N_TOTAL

base_pool_auroc   = roc_auc_score(y_pool, base_prob_pool)
base_pool_logloss = pooled_log_loss(y_pool, base_prob_pool)

print(f"\n  Sanity check — pooled baseline vs OOF-file global:")
print(f"    Pooled AUROC (this loop) = {base_pool_auroc:.4f}   "
      f"OOF-file AUROC = {global_m['AUROC']:.4f}")
if abs(base_pool_auroc - global_m['AUROC']) > 1e-4:
    print(f"    WARNING: AUROC mismatch >1e-4 — check determinism (Platt scaling)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4B — POOLED GLOBAL PERMUTATION IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("[Step 4] Pooled Global Permutation Importance")
print(f"         {N_PI_REPEATS_GLOBAL} repeats x {FSTAR_N} features x {N_SCHOOLS} folds")
print("         Metric: drop in pooled AUROC / rise in pooled log-loss")
print("         Permutation values resampled from each fold's TRAINING background")
print("=" * 78)

pi_repeats_auroc   = np.zeros((FSTAR_N, N_PI_REPEATS_GLOBAL))
pi_repeats_logloss = np.zeros((FSTAR_N, N_PI_REPEATS_GLOBAL))

t1 = time.time()
for j in range(FSTAR_N):
    for r in range(N_PI_REPEATS_GLOBAL):
        permuted_prob_pool = np.empty(N_TOTAL)
        cur = 0
        for fi in range(N_SCHOOLS):
            clf_f, scaler_f = fold_clf[fi], fold_scaler[fi]
            Xh_f, Xt_f, n_f = fold_Xh_raw[fi], fold_Xt_raw[fi], fold_n[fi]
            rng = np.random.RandomState(SEED + j*100_000 + r*1_000 + fi)
            sampled_vals = rng.choice(Xt_f[:, j], size=n_f, replace=True)
            Xp_raw = Xh_f.copy()
            Xp_raw[:, j] = sampled_vals
            Xp_sc = scaler_f.transform(Xp_raw)
            permuted_prob_pool[cur:cur+n_f] = clf_f.predict_proba(Xp_sc)[:, 1]
            cur += n_f

        auroc_perm   = roc_auc_score(y_pool, permuted_prob_pool)
        logloss_perm = pooled_log_loss(y_pool, permuted_prob_pool)
        pi_repeats_auroc[j, r]   = base_pool_auroc - auroc_perm
        pi_repeats_logloss[j, r] = logloss_perm - base_pool_logloss

    print(f"  Feature {j+1:2d}/{FSTAR_N}  {FSTAR_COLS[j]:<8}  "
          f"mean AUROC-drop={pi_repeats_auroc[j].mean():+.5f}  "
          f"mean LogLoss-rise={pi_repeats_logloss[j].mean():+.5f}")

print(f"\n  Pooled PI complete in {time.time()-t1:.1f}s")

global_pi_mean = pi_repeats_auroc.mean(axis=1)
pi_ci_lo       = np.percentile(pi_repeats_auroc, 2.5,  axis=1)
pi_ci_hi       = np.percentile(pi_repeats_auroc, 97.5, axis=1)
pi_sort_idx    = np.argsort(global_pi_mean)[::-1]

global_pi_logloss_mean = pi_repeats_logloss.mean(axis=1)
pi_logloss_ci_lo       = np.percentile(pi_repeats_logloss, 2.5,  axis=1)
pi_logloss_ci_hi       = np.percentile(pi_repeats_logloss, 97.5, axis=1)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — AGGREGATE SHAP RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 5] Aggregating SHAP results...")

mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
shap_sort_idx = np.argsort(mean_abs_shap)[::-1]

shap_school_means = np.zeros((N_SCHOOLS, FSTAR_N))
cursor2 = 0
school_n_list = []
for si, sid in enumerate(SCHOOLS):
    n_held = int((bd["school_id"] == sid).sum())
    shap_school_means[si] = np.abs(shap_matrix[cursor2:cursor2+n_held]).mean(axis=0)
    school_n_list.append(n_held)
    cursor2 += n_held

print(f"\n  SHAP Importance (Mean |SHAP|, descending):")
print(f"  {'Rank':<5} {'Feature':<10} {'Mean|SHAP|':>12} {'Label'}")
print("  " + "-" * 60)
for rank, j in enumerate(shap_sort_idx):
    print(f"  {rank+1:<5} {FSTAR_COLS[j]:<10} {mean_abs_shap[j]:>12.6f}  {feat_label(FSTAR_COLS[j])}")

print(f"\n  Pooled Global PI (Mean AUROC-drop, descending):")
print(f"  {'Rank':<5} {'Feature':<10} {'AUROC-drop':>12} {'95% CI':>22} {'Label'}")
print("  " + "-" * 78)
for rank, j in enumerate(pi_sort_idx):
    print(f"  {rank+1:<5} {FSTAR_COLS[j]:<10} {global_pi_mean[j]:>12.6f}  "
          f"[{pi_ci_lo[j]:.6f},{pi_ci_hi[j]:.6f}]  {feat_label(FSTAR_COLS[j])}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SCHOOL-LEVEL BOOTSTRAP CI (SHAP only)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[Step 6] School-level bootstrap CI for SHAP (B={N_BOOT_SCHOOL})...")

rng_boot = np.random.RandomState(SEED)
boot_shap = np.zeros((N_BOOT_SCHOOL, FSTAR_N))
for b in range(N_BOOT_SCHOOL):
    idx = rng_boot.choice(N_SCHOOLS, N_SCHOOLS, replace=True)
    boot_shap[b] = shap_school_means[idx].mean(axis=0)
shap_lo = np.percentile(boot_shap, 2.5,  axis=0)
shap_hi = np.percentile(boot_shap, 97.5, axis=0)

print(f"  Bootstrap complete")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SHAP vs PI CONSISTENCY CHECK
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 7] SHAP vs Pooled-PI consistency (Spearman rho)...")

rho_shap_pi, p_shap_pi = spearmanr(mean_abs_shap, global_pi_mean)
print(f"  Spearman rho (mean|SHAP| vs mean AUROC-drop PI) = {rho_shap_pi:.4f}  p={p_shap_pi:.4g}")
print(f"  -> {rho_interpretation(rho_shap_pi)}")

rho_shap_pi_ll, p_shap_pi_ll = spearmanr(mean_abs_shap, global_pi_logloss_mean)
print(f"  (Secondary, log-loss-rise PI) rho = {rho_shap_pi_ll:.4f}  p={p_shap_pi_ll:.4g}")

if rho_shap_pi > 0.7:
    print(f"  Both XAI methods agree — findings are robust")
elif rho_shap_pi < -0.7:
    print(f"  WARNING: rankings inverted — inspect pooled PI before reporting")
else:
    print(f"  Methods partially agree — both rankings reported")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SPEARMAN RANK STABILITY (SHAP-based, across schools)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 8] Spearman rank stability of SHAP rankings across schools...")

shap_ranks = np.zeros((N_SCHOOLS, FSTAR_N))
for s in range(N_SCHOOLS):
    order = np.argsort(shap_school_means[s])[::-1]
    for pos, j in enumerate(order):
        shap_ranks[s, j] = pos + 1

rho_vals, rho_rows = [], []
rho_mat = np.full((N_SCHOOLS, N_SCHOOLS), np.nan)
np.fill_diagonal(rho_mat, 1.0)

for i, j in combinations(range(N_SCHOOLS), 2):
    rho, p = spearmanr(shap_ranks[i], shap_ranks[j])
    rho_mat[i, j] = rho_mat[j, i] = rho
    rho_vals.append(float(rho))
    rho_rows.append({
        "School_A": SCHOOLS[i], "Type_A": school_types[i],
        "School_B": SCHOOLS[j], "Type_B": school_types[j],
        "Spearman_rho": round(float(rho), 4),
        "p_value": round(float(p), 6),
        "Pair_type": ("ASD-ASD" if school_types[i]==school_types[j]=="ASD" else
                      "Normal-Normal" if school_types[i]==school_types[j]=="Normal" else
                      "Cross-type"),
    })

rho_df  = pd.DataFrame(rho_rows)
mean_rho = float(np.mean(rho_vals))
min_rho  = float(np.min(rho_vals))
max_rho  = float(np.max(rho_vals))

print(f"  Mean pairwise Spearman rho = {mean_rho:.4f}  [{min_rho:.4f}, {max_rho:.4f}]")
print(f"  -> {rho_strength_only(mean_rho)} stability")

mean_shap_rank = shap_ranks.mean(axis=0)
rank_sd_shap   = shap_ranks.std(axis=0)
rank_cv_shap   = rank_sd_shap / (mean_shap_rank + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — FEATURE CLASSIFICATION (SHAP-based)
# ══════════════════════════════════════════════════════════════════════════════

n_top       = max(1, int(np.ceil(FSTAR_N * 0.40)))
top_shap_set = set(shap_sort_idx[:n_top])
cv_median   = np.median(rank_cv_shap)

classify_rows       = []
consistent_features = []
variable_features   = []

for j, col in enumerate(FSTAR_COLS):
    in_top = j in top_shap_set
    low_cv = rank_cv_shap[j] <= cv_median
    cls    = ("Consistent" if in_top and low_cv else
              "Moderate"   if in_top or  low_cv else "Variable")
    if cls == "Consistent": consistent_features.append(col)
    elif cls == "Variable":  variable_features.append(col)

    classify_rows.append({
        "Feature":               col,
        "Label":                 feat_label(col),
        "SHAP_Mean_Abs":         round(float(mean_abs_shap[j]), 6),
        "SHAP_Boot_95CI":        f"{mean_abs_shap[j]:.6f} [{shap_lo[j]:.6f}, {shap_hi[j]:.6f}]",
        "SHAP_Global_Rank":      int(np.where(shap_sort_idx==j)[0][0]) + 1,
        "PI_AUROC_Drop":         round(float(global_pi_mean[j]), 6),
        "PI_AUROC_Drop_95CI":    f"{global_pi_mean[j]:.6f} [{pi_ci_lo[j]:.6f}, {pi_ci_hi[j]:.6f}]",
        "PI_Global_Rank":        int(np.where(pi_sort_idx==j)[0][0]) + 1,
        "PI_LogLoss_Rise":       round(float(global_pi_logloss_mean[j]), 6),
        "SHAP_Mean_Rank":        round(float(mean_shap_rank[j]), 2),
        "SHAP_Rank_CV":          round(float(rank_cv_shap[j]), 3),
        "In_Top_SHAP":           in_top,
        "Low_Rank_CV":           low_cv,
        "Classification":        cls,
    })

classify_df = (pd.DataFrame(classify_rows)
               .sort_values("SHAP_Global_Rank")
               .reset_index(drop=True))

print(f"\n  Consistent features : {consistent_features}")
print(f"  Variable features   : {variable_features}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — MODEL vs RULE DISCRIMINABILITY + FEATURE CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 9] Model importance vs AQ-10 item discriminability...")

disc_vals, disc_rows = [], []
for j, col in enumerate(FSTAR_COLS):
    asd_rate  = (bd.loc[bd["label"]==1, col] >= 2).mean()
    norm_rate = (bd.loc[bd["label"]==0, col] >= 2).mean()
    disc      = float(asd_rate - norm_rate)
    disc_vals.append(disc)
    disc_rows.append({
        "Feature":          col,
        "Label":            feat_label(col),
        "P_pos_ASD":        round(float(asd_rate), 4),
        "P_pos_Normal":     round(float(norm_rate), 4),
        "Discriminability": round(disc, 4),
        "SHAP_Mean_Abs":    round(float(mean_abs_shap[j]), 6),
        "SHAP_Rank":        int(np.where(shap_sort_idx==j)[0][0]) + 1,
        "PI_AUROC_Drop":    round(float(global_pi_mean[j]), 6),
        "PI_Rank":          int(np.where(pi_sort_idx==j)[0][0]) + 1,
    })
disc_df = pd.DataFrame(disc_rows)

rho_shap_disc, p_shap_disc = spearmanr(mean_abs_shap, disc_vals)
rho_pi_disc,   p_pi_disc   = spearmanr(global_pi_mean, disc_vals)
print(f"  SHAP vs discriminability: rho={rho_shap_disc:.4f}  p={p_shap_disc:.4g}")
print(f"  PI   vs discriminability: rho={rho_pi_disc:.4f}  p={p_pi_disc:.4g}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — FAIRNESS / SUBGROUP AUDIT (with bootstrap CIs)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 10] Subgroup performance audit (descriptive, with bootstrap CIs)...")

def build_subgroups(df):
    groups = []
    if "Age" in df.columns:
        ages = df["Age"].dropna()
        cutoff = 8
        if ages.min() >= cutoff:
            cutoff = int(np.median(ages))
        groups += [(f"Age <= {cutoff-1}", "Age", df["Age"] < cutoff),
                   (f"Age >= {cutoff}",   "Age", df["Age"] >= cutoff)]
    if "Gender" in df.columns:
        for v in sorted(df["Gender"].dropna().unique()):
            groups.append((f"Gender = {v}", "Gender", df["Gender"] == v))
    if "family_hx" in df.columns:
        for v in sorted(df["family_hx"].dropna().unique()):
            lbl = "family_hx = Yes" if str(v).lower() in ["1","yes","true"] else "family_hx = No"
            groups.append((lbl, "family_hx", df["family_hx"] == v))
    asd_schools    = [SCHOOLS[i] for i in asd_idx]
    normal_schools = [SCHOOLS[i] for i in normal_idx]
    groups.append(("School type = ASD",    "school_type", df["school_id"].isin(asd_schools)))
    groups.append(("School type = Normal", "school_type", df["school_id"].isin(normal_schools)))
    return groups

subgroups     = build_subgroups(oof_full)
fairness_rows = []

print(f"\n  {'Subgroup':<28} {'n':>5} {'n_ASD':>6} {'Sens':>8} {'Spec':>8} {'DeltaSens':>10} Note")
print("  " + "-" * 90)

for sg_name, sg_col, sg_mask in subgroups:
    sub   = oof_full[sg_mask]
    n_sub = len(sub)
    if n_sub == 0:
        continue
    y_s    = sub["label"].values.astype(int)
    prob_s = sub["y_prob"].values.astype(float)
    pred_s = (prob_s >= THRESHOLD).astype(int)
    m      = safe_metrics(y_s, prob_s, pred_s)
    ds     = float(m["Sensitivity"] - GLOBAL_SENS) if not np.isnan(m["Sensitivity"]) else np.nan
    dp     = float(m["Specificity"] - GLOBAL_SPEC) if not np.isnan(m["Specificity"]) else np.nan
    small  = n_sub < SMALL_N
    nw     = (not small) and (not np.isnan(ds)) and abs(ds) >= NOTEWORTHY_GAP
    note   = "exploratory (n<30)" if small else (f"Delta={ds:+.3f}" if nw else "")

    # Bootstrap CIs
    boot_ci = subgroup_bootstrap_ci(y_s, prob_s, pred_s, N_BOOT_SUBGROUP, SEED)

    fairness_rows.append({
        "Subgroup": sg_name, "Subgroup_Column": sg_col,
        "n": int(n_sub), "n_ASD": int(y_s.sum()), "n_Normal": int((y_s==0).sum()),
        "AUROC": round(float(m["AUROC"]),4) if not np.isnan(m["AUROC"]) else "NaN",
        "Sensitivity": round(float(m["Sensitivity"]),4) if not np.isnan(m["Sensitivity"]) else "NaN",
        "Specificity": round(float(m["Specificity"]),4) if not np.isnan(m["Specificity"]) else "NaN",
        "Accuracy": round(float(m["Accuracy"]),4), "F1": round(float(m["F1"]),4),
        "TP":m["TP"],"FN":m["FN"],"TN":m["TN"],"FP":m["FP"],
        "Global_Sensitivity": round(GLOBAL_SENS,4), "Global_Specificity": round(GLOBAL_SPEC,4),
        "Delta_Sensitivity": round(float(ds),4) if not np.isnan(ds) else "NaN",
        "Delta_Specificity": round(float(dp),4) if not np.isnan(dp) else "NaN",
        "Sens_CI_Lower": round(boot_ci["sens"][0],4) if not np.isnan(boot_ci["sens"][0]) else "NaN",
        "Sens_CI_Upper": round(boot_ci["sens"][1],4) if not np.isnan(boot_ci["sens"][1]) else "NaN",
        "Spec_CI_Lower": round(boot_ci["spec"][0],4) if not np.isnan(boot_ci["spec"][0]) else "NaN",
        "Spec_CI_Upper": round(boot_ci["spec"][1],4) if not np.isnan(boot_ci["spec"][1]) else "NaN",
        "Is_Exploratory": small, "Noteworthy": nw, "Note": note,
        "Interpretation": "exploratory" if small else "descriptive",
    })
    s_d  = f"{m['Sensitivity']:.4f}" if not np.isnan(m["Sensitivity"]) else "  N/A "
    sp_d = f"{m['Specificity']:.4f}" if not np.isnan(m["Specificity"]) else "  N/A "
    ds_d = f"{ds:+.4f}" if not np.isnan(ds) else "  N/A "
    print(f"  {sg_name:<28} {n_sub:>5} {int(y_s.sum()):>6} {s_d:>8} {sp_d:>8} {ds_d:>10}  {note}")

fairness_df = pd.DataFrame(fairness_rows)
noteworthy_groups = fairness_df[fairness_df["Noteworthy"]==True]["Subgroup"].tolist()
print(f"\n  Noteworthy subgroups: {noteworthy_groups or 'None'}")
print(f"  NOTE: Descriptive only — no confirmatory claims. n<30 = exploratory.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — SAVE ALL TABLES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 11] Saving tables...")

# T1: SHAP values per instance
shap_val_df = pd.DataFrame(shap_matrix, columns=[f"SHAP_{c}" for c in FSTAR_COLS])
shap_val_df.insert(0, "ID", ids_in_order)
shap_val_df.insert(1, "label", y_pool)
shap_val_df.insert(2, "school_id", school_id_in_order)
for j, c in enumerate(FSTAR_COLS):
    shap_val_df[f"feat_{c}"] = feat_vals_raw[:, j]
save_tbl(shap_val_df, "T_rq4_shap_values_per_instance.csv")

# T2: SHAP + pooled PI global importance
save_tbl(classify_df, "T_rq4_shap_pi_importance_global.csv")

# T3: Per-school SHAP
school_shap_rows = []
for si, sid in enumerate(SCHOOLS):
    row = {"school_id": sid, "school_type": school_types[si], "n": school_n_list[si]}
    for j, col in enumerate(FSTAR_COLS):
        row[f"shap_{col}"] = round(float(shap_school_means[si,j]), 6)
    school_shap_rows.append(row)
school_shap_df = pd.DataFrame(school_shap_rows)
save_tbl(school_shap_df, "T_rq4_xai_by_school.csv")

# T4: Spearman pairwise rho
save_tbl(rho_df, "T_rq4_spearman_pairwise.csv")

# T5: Model vs rule discriminability
save_tbl(disc_df, "T_rq4_model_vs_rule_discriminability.csv")

# T6: Fairness
save_tbl(fairness_df, "T_rq4_fairness_subgroups.csv")

# T7: Feature correlation matrix (NEW)
feat_corr.index.name = "Feature"
save_tbl(feat_corr.reset_index(), "T_rq4_feature_correlation_matrix.csv")
if high_corr_pairs:
    save_tbl(pd.DataFrame(high_corr_pairs), "T_rq4_high_correlation_pairs.csv")

# Paper-ready tables
pr1 = classify_df[[
    "Feature","Label","SHAP_Global_Rank","SHAP_Boot_95CI",
    "PI_Global_Rank","PI_AUROC_Drop_95CI","Classification"
]].copy()
pr1.columns = [
    "Feature","Description","SHAP Rank","SHAP Mean|SHAP| [95% CI]",
    "PI Rank","PI Mean AUROC-drop [95% CI]","Consistency"
]
save_pr(pr1, "T_rq4_paper_feature_importance.csv")

pr2 = fairness_df[["Subgroup","n","n_ASD","Sensitivity","Specificity",
                    "Delta_Sensitivity","Delta_Specificity","Note"]].copy()
save_pr(pr2, "T_rq4_paper_fairness.csv")

rho_sum = []
rho_sum.append({"Pair_type":"Overall","n_pairs":len(rho_vals),
                "Mean_rho":round(mean_rho,4),"Range":f"[{min_rho:.4f},{max_rho:.4f}]",
                "Landis_Koch":rho_strength_only(mean_rho)})
for pt in ["ASD-ASD","Normal-Normal","Cross-type"]:
    sub = rho_df[rho_df["Pair_type"]==pt]["Spearman_rho"]
    if len(sub):
        rho_sum.append({"Pair_type":pt,"n_pairs":len(sub),
                        "Mean_rho":round(float(sub.mean()),4),
                        "Range":f"[{float(sub.min()):.4f},{float(sub.max()):.4f}]",
                        "Landis_Koch":rho_strength_only(float(sub.mean()))})
save_pr(pd.DataFrame(rho_sum), "T_rq4_paper_spearman_summary.csv")

shap_pi_comp = pd.DataFrame([{
    "Metric": "Spearman rho (SHAP vs pooled-PI ranking)",
    "Value": round(float(rho_shap_pi), 4),
    "p_value": round(float(p_shap_pi), 6),
    "Interpretation": rho_interpretation(rho_shap_pi),
    "Conclusion": ("Both XAI methods agree — findings are robust" if rho_shap_pi > 0.7 else
                   "Rankings inverted — do not report as agreement" if rho_shap_pi < -0.7 else
                   "Partial agreement — both rankings reported"),
}])
save_pr(shap_pi_comp, "T_rq4_paper_shap_vs_pi.csv")

print("  11 tables saved (7 full + 4 paper-ready)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — 10 PUBLICATION-QUALITY FIGURES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 12] Generating 10 figures...")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.labelsize": 11, "legend.fontsize": 9, "figure.dpi": 100,
    "axes.spines.top": False, "axes.spines.right": False,
})

C_CONSIST  = "#9467bd"
C_MODERATE = "#ff7f0e"
C_VARIABLE = "#cccccc"
C_ASD      = "#d62728"
C_NORMAL   = "#1f77b4"

def get_bar_color(col):
    row = classify_df[classify_df["Feature"]==col]
    if len(row)==0: return C_MODERATE
    cls = row["Classification"].values[0]
    return C_CONSIST if cls=="Consistent" else C_VARIABLE if cls=="Variable" else C_MODERATE

handles = [mpatches.Patch(color=C_CONSIST,  label="Consistent"),
           mpatches.Patch(color=C_MODERATE, label="Moderate"),
           mpatches.Patch(color=C_VARIABLE, label="Variable")]

# ── FIGURE 1: SHAP Beeswarm ───────────────────────────────────────────────────
print("  F1: SHAP Beeswarm...")
shap.summary_plot(
    shap_matrix, features=feat_vals_raw,
    feature_names=[feat_label(c) for c in FSTAR_COLS],
    plot_type="dot", max_display=FSTAR_N, show=False, color_bar=True, plot_size=None,
)
fig = plt.gcf()
fig.set_size_inches(10, 6)
for ax in fig.get_axes():
    if ax.get_xlabel():
        ax.set_title(
            f"RQ4 — SHAP Beeswarm Summary  (F*={FSTAR}, N={N_TOTAL} instances)\n"
            "Colour = feature value (red=high, blue=low)  |  x-axis = SHAP impact on P(ASD)",
            fontsize=11, pad=10)
        break
plt.tight_layout()
for d in [DIRS["figures"], DIRS["pt_figures"]]:
    fig.savefig(os.path.join(d, "F_rq4_1_shap_beeswarm.png"), dpi=300, bbox_inches="tight")
plt.close()

# ── FIGURE 2: SHAP Global Importance Bar ─────────────────────────────────────
print("  F2: SHAP importance bar...")
fig, ax = plt.subplots(figsize=(11, 5))
labels_s = [feat_label(FSTAR_COLS[j]) for j in shap_sort_idx]
vals_s   = mean_abs_shap[shap_sort_idx]
err_lo_s = vals_s - shap_lo[shap_sort_idx]
err_hi_s = shap_hi[shap_sort_idx] - vals_s
colors_s = [get_bar_color(FSTAR_COLS[j]) for j in shap_sort_idx]
x = np.arange(FSTAR_N)
ax.bar(x, vals_s, color=colors_s, alpha=0.85, edgecolor="white",
       yerr=[err_lo_s, err_hi_s], capsize=5, error_kw={"elinewidth":1.5, "ecolor":"#555555"})
ax.set_xticks(x); ax.set_xticklabels(labels_s, rotation=38, ha="right", fontsize=9.5)
ax.set_ylabel("Mean |SHAP value|  (impact on P(ASD))")
ax.set_title(f"RQ4 — Global Feature Importance: KernelSHAP  (F*={FSTAR})\n"
             f"Error bars = school-level bootstrap 95% CI (B={N_BOOT_SCHOOL})")
ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.4)
ax.grid(True, axis="y", linestyle=":", alpha=0.35)
ax.legend(handles=handles, loc="upper right", fontsize=9)
save_fig(fig, "F_rq4_2_shap_bar_global.png")

# ── FIGURE 3: Pooled PI Importance Bar ────────────────────────────────────────
print("  F3: Pooled PI bar...")
fig, ax = plt.subplots(figsize=(11, 5))
labels_p = [feat_label(FSTAR_COLS[j]) for j in pi_sort_idx]
vals_p   = global_pi_mean[pi_sort_idx]
err_lo_p = vals_p - pi_ci_lo[pi_sort_idx]
err_hi_p = pi_ci_hi[pi_sort_idx] - vals_p
colors_p = [get_bar_color(FSTAR_COLS[j]) for j in pi_sort_idx]
ax.bar(x, vals_p, color=colors_p, alpha=0.85, edgecolor="white",
       yerr=[err_lo_p, err_hi_p], capsize=5, error_kw={"elinewidth":1.5, "ecolor":"#555555"})
ax.set_xticks(x); ax.set_xticklabels(labels_p, rotation=38, ha="right", fontsize=9.5)
ax.set_ylabel("Mean Pooled Permutation Importance\n(drop in pooled AUROC when permuted)")
ax.set_title(f"RQ4 — Validation XAI: Pooled Global Permutation Importance  (F*={FSTAR})\n"
             f"Error bars = 95% CI over {N_PI_REPEATS_GLOBAL} repeats  |  "
             f"Spearman rho vs SHAP = {rho_shap_pi:.4f}")
ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.4)
ax.grid(True, axis="y", linestyle=":", alpha=0.35)
ax.legend(handles=handles, loc="upper right", fontsize=9)
save_fig(fig, "F_rq4_3_pi_bar_global.png")

# ── FIGURE 4: SHAP by School Type ────────────────────────────────────────────
print("  F4: SHAP by school type...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, idx_list, title, color in [
    (axes[0], asd_idx,    f"ASD Schools ({len(asd_idx)} folds)",    C_ASD),
    (axes[1], normal_idx, f"Normal Schools ({len(normal_idx)} folds)", C_NORMAL),
]:
    m_v  = shap_school_means[idx_list].mean(axis=0) if idx_list else np.zeros(FSTAR_N)
    ord_ = np.argsort(m_v)[::-1]
    x_   = np.arange(FSTAR_N)
    ax.bar(x_, m_v[ord_], color=color, alpha=0.80, edgecolor="white")
    ax.set_xticks(x_)
    ax.set_xticklabels([feat_label(FSTAR_COLS[j]) for j in ord_], rotation=38, ha="right", fontsize=9)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Mean |SHAP value|")
    ax.axhline(0, color="black", lw=0.8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    for xi, vi in enumerate(m_v[ord_]):
        if vi > 0:
            ax.text(xi, vi+1e-5, str(xi+1), ha="center", va="bottom", fontsize=8.5, color="black")
fig.suptitle(f"RQ4 — SHAP Importance by School Type  (F*={FSTAR})\nNumber = within-type rank",
             y=1.02, fontsize=12)
fig.tight_layout()
save_fig(fig, "F_rq4_4_shap_by_school_type.png")

# ── FIGURE 5: School x Feature SHAP Heatmap ──────────────────────────────────
print("  F5: SHAP heatmap...")
school_labels_fig = [f"{s}\n({t})" for s,t in zip(SCHOOLS, school_types)]
feat_labels_fig   = [feat_label(c) for c in FSTAR_COLS]
shap_heat_df = pd.DataFrame(shap_school_means, index=school_labels_fig, columns=feat_labels_fig)
fw = max(10, FSTAR_N*0.85); fh = max(6, N_SCHOOLS*0.55)
fig, ax = plt.subplots(figsize=(fw, fh))
sns.heatmap(shap_heat_df, ax=ax, cmap="YlOrRd", annot=True, fmt=".4f",
            linewidths=0.4, annot_kws={"fontsize":7.5}, cbar_kws={"label":"Mean |SHAP value|"})
ax.set_title(f"RQ4 — KernelSHAP: School x Feature  (F*={FSTAR})\n"
             "Each cell = mean |SHAP| for that school's held-out students")
ax.tick_params(axis="x", rotation=38); ax.tick_params(axis="y", rotation=0)
save_fig(fig, "F_rq4_5_shap_heatmap_school_x_feature.png")

# ── FIGURE 6: Spearman rho Heatmap ───────────────────────────────────────────
print("  F6: Spearman heatmap...")
fig, ax = plt.subplots(figsize=(max(8,N_SCHOOLS*0.7), max(7,N_SCHOOLS*0.65)))
rho_plot = rho_mat.copy()
np.fill_diagonal(rho_plot, np.nan)
annot_arr = np.array([[f"{rho_plot[i,j]:.2f}" if i!=j else "" for j in range(N_SCHOOLS)] for i in range(N_SCHOOLS)])
xlbl = [f"{s}\n({t[0]})" for s,t in zip(SCHOOLS, school_types)]
sns.heatmap(rho_plot, ax=ax, annot=annot_arr, fmt="", cmap="RdYlGn", vmin=-1, vmax=1,
            center=0, linewidths=0.5, xticklabels=xlbl, yticklabels=xlbl,
            annot_kws={"fontsize":8.5}, cbar_kws={"label":"Spearman rho"})
ax.set_title(f"RQ4 — Pairwise Spearman rho of SHAP Rankings\n"
             f"Mean rho = {mean_rho:.4f}  [{min_rho:.4f}, {max_rho:.4f}]  |  {rho_strength_only(mean_rho)} stability")
ax.tick_params(axis="x", rotation=40); ax.tick_params(axis="y", rotation=0)
save_fig(fig, "F_rq4_6_spearman_heatmap.png")

# ── FIGURE 7: SHAP vs Pooled-PI Scatter ─────────────────────────────────────
print("  F7: SHAP vs PI consistency...")
fig, ax = plt.subplots(figsize=(8, 6))
for j, col in enumerate(FSTAR_COLS):
    cls = classify_df.loc[classify_df["Feature"]==col,"Classification"].values[0]
    c_  = C_CONSIST if cls=="Consistent" else C_VARIABLE if cls=="Variable" else C_MODERATE
    ax.scatter(mean_abs_shap[j], global_pi_mean[j], s=130, color=c_,
               alpha=0.82, edgecolors="white", zorder=3)
    ax.annotate(col, (mean_abs_shap[j], global_pi_mean[j]),
                xytext=(4,2), textcoords="offset points", fontsize=8.5, color="#333333")
if FSTAR_N > 2:
    z = np.polyfit(mean_abs_shap, global_pi_mean, 1)
    xl = np.linspace(mean_abs_shap.min(), mean_abs_shap.max(), 100)
    ax.plot(xl, np.poly1d(z)(xl), "--", color="#777777", lw=1.5,
            label=f"Trend  (rho={rho_shap_pi:.3f}, p={p_shap_pi:.3g})")
ax.set_xlabel("Mean |SHAP value|  (KernelSHAP)")
ax.set_ylabel("Pooled Permutation Importance (AUROC-drop)")
ax.set_title(f"RQ4 — XAI Consistency: KernelSHAP vs Pooled Global PI\n"
             f"Spearman rho = {rho_shap_pi:.4f}  |  {rho_interpretation(rho_shap_pi)}")
ax.legend(handles=handles + [plt.Line2D([0],[0],color="#777777",lw=1.5,
          linestyle="--",label="Trend line")], fontsize=9)
ax.grid(True, linestyle=":", alpha=0.35)
save_fig(fig, "F_rq4_7_shap_vs_pi_consistency.png")

# ── FIGURE 8: Model vs Rule Discriminability ──────────────────────────────────
print("  F8: Model vs rule discriminability...")
fig, ax = plt.subplots(figsize=(8, 6))
for j, col in enumerate(FSTAR_COLS):
    c_ = get_bar_color(col)
    ax.scatter(disc_vals[j], mean_abs_shap[j], s=130, color=c_,
               alpha=0.82, edgecolors="white", zorder=3)
    ax.annotate(col, (disc_vals[j], mean_abs_shap[j]),
                xytext=(4,2), textcoords="offset points", fontsize=8.5, color="#333333")
if FSTAR_N > 2:
    z = np.polyfit(disc_vals, mean_abs_shap, 1)
    xl = np.linspace(min(disc_vals), max(disc_vals), 100)
    ax.plot(xl, np.poly1d(z)(xl), "--", color="#777777", lw=1.5,
            label=f"Trend  (rho={rho_shap_disc:.3f}, p={p_shap_disc:.3g})")
ax.set_xlabel("Item Discriminability  P(item>=2|ASD) - P(item>=2|Normal)")
ax.set_ylabel("Mean |SHAP value|  (model importance)")
ax.set_title(f"RQ4 — Model Importance vs AQ-10 Item Discriminability\n"
             f"Spearman rho = {rho_shap_disc:.4f}  (p={p_shap_disc:.4g})")
ax.legend(handles=handles + [plt.Line2D([0],[0],color="#777777",lw=1.5,
          linestyle="--",label="Trend line")], fontsize=9)
ax.grid(True, linestyle=":", alpha=0.35)
save_fig(fig, "F_rq4_8_model_vs_rule_discriminability.png")

# ── FIGURE 9: Fairness Subgroup Chart ────────────────────────────────────────
print("  F9: Fairness subgroups...")
if len(fairness_df) > 0:
    n_sg   = len(fairness_df)
    sg_names = fairness_df["Subgroup"].values
    sg_sens  = pd.to_numeric(fairness_df["Sensitivity"], errors="coerce").values
    sg_spec  = pd.to_numeric(fairness_df["Specificity"], errors="coerce").values
    sg_small = fairness_df["Is_Exploratory"].values
    sg_nw    = fairness_df["Noteworthy"].values
    x_sg     = np.arange(n_sg)

    fig, axes = plt.subplots(2, 1, figsize=(max(11, n_sg*1.15), 10))
    for ax, vals, global_v, ylabel, col_ok in [
        (axes[0], sg_sens, GLOBAL_SENS, "Sensitivity",  "#2ca02c"),
        (axes[1], sg_spec, GLOBAL_SPEC, "Specificity",  "#1f77b4"),
    ]:
        bar_c = []
        for i in range(n_sg):
            if np.isnan(vals[i]): bar_c.append("#eeeeee")
            elif sg_small[i]:     bar_c.append("#aaaaaa")
            elif sg_nw[i]:        bar_c.append(C_ASD)
            else:                 bar_c.append(col_ok)
        bv = np.where(np.isnan(vals), 0, vals)
        ax.bar(x_sg, bv, color=bar_c, alpha=0.85, edgecolor="white")
        ax.axhline(global_v, color="black", lw=2, linestyle="--", label=f"Global = {global_v:.4f}")
        ax.axhline(global_v-NOTEWORTHY_GAP, color=C_ASD, lw=1.5, linestyle=":",
                   label=f"-{NOTEWORTHY_GAP*100:.0f}pp threshold")
        ax.fill_between([-0.5,n_sg-0.5], global_v-NOTEWORTHY_GAP, global_v, alpha=0.06, color=col_ok)
        ax.set_xticks(x_sg); ax.set_xticklabels(sg_names, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.10); ax.set_ylabel(ylabel)
        ax.set_title(f"RQ4 — {ylabel} by Subgroup  (F*={FSTAR})")
        for i, v in enumerate(vals):
            if np.isnan(v):
                ax.text(i, 0.02, "N/A", ha="center", va="bottom", fontsize=7.5, color="#999999")
        h = [mpatches.Patch(color=col_ok,    label="Within range"),
             mpatches.Patch(color=C_ASD,     label=f"Noteworthy |Delta|>={NOTEWORTHY_GAP*100:.0f}pp"),
             mpatches.Patch(color="#aaaaaa", label="Exploratory n<30"),
             mpatches.Patch(color="#eeeeee", label="Undefined (mono-class)")]
        ax.legend(handles=h, loc="lower right", fontsize=8.5)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35)
    fig.suptitle("RQ4 — Subgroup Performance Audit  (Descriptive only)\n"
                 "No confirmatory claims — subgroups n<30 labelled exploratory", y=1.01, fontsize=12)
    fig.tight_layout()
    save_fig(fig, "F_rq4_9_fairness_subgroups.png")

# ── FIGURE 10: Feature Correlation Heatmap (NEW) ─────────────────────────────
print("  F10: Feature correlation heatmap...")
fig, ax = plt.subplots(figsize=(max(8, FSTAR_N*0.8), max(7, FSTAR_N*0.7)))
mask = np.triu(np.ones_like(feat_corr, dtype=bool), k=1)
sns.heatmap(feat_corr, mask=mask, annot=True, fmt=".2f", cmap="Blues",
            vmin=0, vmax=1, square=True, linewidths=0.5,
            xticklabels=[feat_label(c) for c in FSTAR_COLS],
            yticklabels=[feat_label(c) for c in FSTAR_COLS],
            annot_kws={"fontsize":9}, ax=ax,
            cbar_kws={"shrink": 0.75, "label": "Pearson r"})
ax.set_title(f"RQ4 — Feature Intercorrelation Matrix  (F*={FSTAR})\n"
             f"max|r|={max_corr:.3f}  mean|r|={mean_corr:.3f}  |  "
             f"{len(high_corr_pairs)} pairs with |r|>=0.50")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
plt.setp(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
save_fig(fig, "F_rq4_10_feature_correlation_heatmap.png")

print(f"\n  All 10 figures saved")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — MASTER SUMMARY JSON
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 13] Writing RQ4_Master_Summary.json...")

rq4_summary = {
    "module":    "RQ4_XAI_Explainability_Subgroup_Analysis",
    "version":   "FINAL-Q1-10.0-RQ4-XAI-POOLEDPI",
    "status":    "COMPLETE",
    "timestamp": datetime.now().isoformat(),
    "research_question": (
        "Which questionnaire items consistently influence model predictions "
        "across schools, and does model performance vary across demographic groups?"
    ),
    "design_note": (
        "Descriptive/exploratory. No hypotheses. LOSO folds are class-homogeneous "
        "by school design; Permutation Importance is computed as a POOLED, "
        "mixed-class metric (drop in pooled AUROC / rise in pooled log-loss) "
        "rather than per-fold, to avoid degeneracy on single-class subsets. "
        "KernelSHAP uses mixed-class training backgrounds per fold."
    ),
    "reproducibility": {
        "seed": SEED,
        "pythonhashseed": str(SEED),
        "shap_version": shap.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": "see environment",
        "caveat": (
            "SVC(probability=True) uses Platt scaling with 5-fold internal CV. "
            "With random_state fixed, results should be deterministic on the same "
            "platform. Minor cross-platform variation in the C++ SMO solver is possible."
        ),
    },
    "xai_methods": {
        "primary":    "KernelSHAP (Lundberg & Lee, 2017)",
        "validation": "Pooled Global Permutation Importance (Breiman 2001; Fisher/Rudin/Dominici 2019)",
        "settings": {
            "shap_background_n":    N_SHAP_BG,
            "shap_nsamples":        N_SHAP_SAMPLES,
            "shap_additivity_max_error": round(float(max(additivity_errors)), 6),
            "pi_repeats":           N_PI_REPEATS_GLOBAL,
            "pi_metric_primary":    "drop in pooled AUROC (mixed-class, 843 instances)",
            "pi_metric_secondary":  "rise in pooled log-loss (cross-entropy)",
            "pi_permutation_source": "resampled from each fold's own training background",
        },
    },
    "limitations": [
        "KernelSHAP assumes feature independence when sampling coalitions; AQ items are correlated (see correlation matrix).",
        "Permutation Importance on correlated features can split importance across redundant items; rankings remain valid but exact marginal attributions are uncertain.",
        "SVM probability outputs (Platt scaling) are approximations; SHAP explains these approximations, not the true decision function.",
        "Schools are mono-class by design; per-school AUROC/Sens/Spec are undefined for held-out folds.",
        "Subgroup analysis is descriptive/exploratory; no Bonferroni correction applied (inappropriate without pre-registered hypotheses).",
        "All findings are specific to the BD-AQ10(+SQ4) instrument and the recruited school population; generalisation requires external validation."
    ],
    "locked_feature_set":     FSTAR,
    "locked_feature_columns": FSTAR_COLS,
    "loso_n_schools":         N_SCHOOLS,
    "pooled_oof_n":           N_TOTAL,
    "pooled_baseline_auroc":  round(float(base_pool_auroc), 4),
    "pooled_baseline_logloss": round(float(base_pool_logloss), 4),
    "feature_correlation": {
        "max_abs_pearson_r": round(max_corr, 4),
        "mean_abs_pearson_r": round(mean_corr, 4),
        "n_pairs_ge_0.50": len(high_corr_pairs),
        "high_pairs": high_corr_pairs,
    },
    "shap_importance": {
        col: {
            "mean_abs_shap":  round(float(mean_abs_shap[j]), 6),
            "boot_95ci_lo":   round(float(shap_lo[j]), 6),
            "boot_95ci_hi":   round(float(shap_hi[j]), 6),
            "global_rank":    int(np.where(shap_sort_idx==j)[0][0]) + 1,
            "classification": classify_df.loc[classify_df["Feature"]==col, "Classification"].values[0],
        }
        for j, col in enumerate(FSTAR_COLS)
    },
    "pi_importance": {
        col: {
            "mean_auroc_drop":   round(float(global_pi_mean[j]), 6),
            "ci95_lo":           round(float(pi_ci_lo[j]), 6),
            "ci95_hi":           round(float(pi_ci_hi[j]), 6),
            "mean_logloss_rise": round(float(global_pi_logloss_mean[j]), 6),
            "global_rank":       int(np.where(pi_sort_idx==j)[0][0]) + 1,
        }
        for j, col in enumerate(FSTAR_COLS)
    },
    "top_feature_shap":       FSTAR_COLS[shap_sort_idx[0]],
    "top_feature_shap_label": feat_label(FSTAR_COLS[shap_sort_idx[0]]),
    "top_feature_pi":         FSTAR_COLS[pi_sort_idx[0]],
    "top_feature_pi_label":   feat_label(FSTAR_COLS[pi_sort_idx[0]]),
    "consistent_features":    consistent_features,
    "variable_features":      variable_features,
    "xai_consistency": {
        "spearman_rho_shap_vs_pi_auroc":    round(float(rho_shap_pi), 4),
        "p_value_auroc":                    round(float(p_shap_pi), 6),
        "spearman_rho_shap_vs_pi_logloss":  round(float(rho_shap_pi_ll), 4),
        "p_value_logloss":                  round(float(p_shap_pi_ll), 6),
        "agreement_label":                  rho_interpretation(rho_shap_pi),
        "conclusion": ("Both XAI methods agree — findings are robust" if rho_shap_pi > 0.7 else
                       "Rankings inverted — not reportable as agreement" if rho_shap_pi < -0.7 else
                       "Partial agreement — both reported"),
    },
    "spearman_stability": {
        "mean_rho": round(float(mean_rho), 4),
        "range":    [round(float(min_rho),4), round(float(max_rho),4)],
        "landis_koch": rho_strength_only(mean_rho),
        "n_pairs":  len(rho_vals),
    },
    "model_vs_rule": {
        "shap_vs_discriminability_rho": round(float(rho_shap_disc), 4),
        "shap_vs_discriminability_p":   round(float(p_shap_disc), 6),
        "pi_vs_discriminability_rho":   round(float(rho_pi_disc), 4),
        "pi_vs_discriminability_p":     round(float(p_pi_disc), 6),
    },
    "fairness": {
        "method":             "Descriptive subgroup analysis with bootstrap 95% CIs",
        "note":               "No Bonferroni correction. n<30 labelled exploratory.",
        "global_sensitivity": round(float(GLOBAL_SENS), 4),
        "global_specificity": round(float(GLOBAL_SPEC), 4),
        "n_subgroups":        len(fairness_df),
        "noteworthy_subgroups": noteworthy_groups,
    },
    "figures": [f"F_rq4_{i+1}*.png" for i in range(10)],
    "tables":  [
        "T_rq4_shap_values_per_instance.csv",
        "T_rq4_shap_pi_importance_global.csv",
        "T_rq4_xai_by_school.csv",
        "T_rq4_spearman_pairwise.csv",
        "T_rq4_model_vs_rule_discriminability.csv",
        "T_rq4_fairness_subgroups.csv",
        "T_rq4_feature_correlation_matrix.csv",
        "T_rq4_high_correlation_pairs.csv",
        "T_rq4_paper_feature_importance.csv",
        "T_rq4_paper_fairness.csv",
        "T_rq4_paper_spearman_summary.csv",
        "T_rq4_paper_shap_vs_pi.csv",
    ],
}

sp = os.path.join(DIRS["results"], "RQ4_Master_Summary.json")
with open(sp, "w") as f:
    json.dump(to_native(rq4_summary), f, indent=4)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — DONE FLAG
# ══════════════════════════════════════════════════════════════════════════════

flag_content = (
    f"RQ4 FINAL-Q1-10.0-RQ4-XAI-POOLEDPI COMPLETE\n"
    f"timestamp={datetime.now().isoformat()}\n"
    f"locked_fstar={FSTAR}\n"
    f"xai_primary=KernelSHAP\n"
    f"xai_validation=PooledGlobalPermutationImportance(AUROC-drop)\n"
    f"top_feature_shap={FSTAR_COLS[shap_sort_idx[0]]}\n"
    f"top_feature_pi={FSTAR_COLS[pi_sort_idx[0]]}\n"
    f"consistent_features={consistent_features}\n"
    f"shap_pi_rho={rho_shap_pi:.4f}\n"
    f"mean_spearman_rho={mean_rho:.4f}\n"
    f"noteworthy_subgroups={noteworthy_groups}\n"
    f"figures=10\ntables=12\nnext=phase_paper_compile\n"
)
for fp in [os.path.join(DIRS["logs"],"rq4_done.flag"),
           os.path.join(PROJECT,"logs","rq4_done.flag")]:
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    with open(fp,"w") as f: f.write(flag_content)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("RQ4 SUMMARY — XAI EXPLAINABILITY & SUBGROUP PERFORMANCE")
print("=" * 78)

print("\n  XAI METHODS USED")
print(f"  PRIMARY   : KernelSHAP (Lundberg & Lee 2017)")
print(f"              Background={N_SHAP_BG} | Coalitions={N_SHAP_SAMPLES}/instance")
print(f"  VALIDATION: Pooled Global Permutation Importance | Repeats={N_PI_REPEATS_GLOBAL}")
print(f"              Metric: pooled AUROC-drop (mixed-class, N={N_TOTAL})")
print(f"  AGREEMENT : Spearman rho(SHAP,PI) = {rho_shap_pi:.4f}")

print(f"\n  SHAP RANKING (top -> bottom)")
print("  " + "-" * 70)
for rank, j in enumerate(shap_sort_idx):
    cls = classify_df.loc[classify_df["Feature"]==FSTAR_COLS[j],"Classification"].values[0]
    ci  = f"[{shap_lo[j]:.5f}, {shap_hi[j]:.5f}]"
    print(f"  {rank+1:>2}. {FSTAR_COLS[j]:<8}  {feat_label(FSTAR_COLS[j]):<35}  "
          f"|SHAP|={mean_abs_shap[j]:.5f} {ci}  [{cls}]")

print(f"\n  POOLED PI RANKING (top -> bottom)")
print("  " + "-" * 70)
for rank, j in enumerate(pi_sort_idx):
    ci = f"[{pi_ci_lo[j]:.5f}, {pi_ci_hi[j]:.5f}]"
    print(f"  {rank+1:>2}. {FSTAR_COLS[j]:<8}  {feat_label(FSTAR_COLS[j]):<35}  "
          f"AUROC-drop={global_pi_mean[j]:.5f} {ci}")

print(f"\n  RANK STABILITY (SHAP, across schools)")
print(f"  Mean Spearman rho = {mean_rho:.4f}  [{min_rho:.4f},{max_rho:.4f}]  -> {rho_strength_only(mean_rho)}")

print(f"\n  FEATURE CORRELATION")
print(f"  max|r| = {max_corr:.3f}  mean|r| = {mean_corr:.3f}  pairs>=0.50 = {len(high_corr_pairs)}")

print(f"\n  SUBGROUP AUDIT  (Descriptive)")
print(f"  Noteworthy: {noteworthy_groups or 'None'}")

print(f"\n  OUTPUTS")
print(f"  Figures (10) : {DIRS['figures']}")
print(f"  Tables  (12) : {DIRS['tables']}")
print(f"  Paper-ready  : {DIRS['paper_ready']}")
print(f"  Summary JSON : {sp}")

print("\n" + "=" * 78)
print("STATUS: RQ4 FINAL-Q1-10.0-RQ4-XAI-POOLEDPI COMPLETE")
print("NEXT  : phase_paper_compile.py -> numbers.md")
print("=" * 78)

