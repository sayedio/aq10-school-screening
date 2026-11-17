"""RQ1 - Five learners, one locked model and one test against the AQ-10 rule.

Reads the files written by pipeline/prepare_from_zenodo.py. Selects the model by five-fold
cross-validation on the development set (86 settings), refits it and tests it once on the
169 test children against the AQ-10 rule (score of 6 or more). Writes tables, predictions and
the locked SVM settings to <work>/aq_10/rq1_baseline/, and three figures to <work>/aq_10/figures/:
model selection, the test set (ROC curves and the SVM against the rule), and calibration with
net benefit.
"""
import os, json, time, random, itertools, warnings
from datetime import datetime

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
np.random.seed(SEED)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score,
    brier_score_loss, confusion_matrix,
    cohen_kappa_score,
)
from scipy.stats import chi2 as chi2_dist


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — PATHS & DIRECTORIES
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 78)
print("RQ1  |  BD-AQ10 SCHOOL SCREENING  |  FINAL-7.0-RQ1")
print("PRIMARY: McNemar (M* binary vs AQ-10 ≥6)  |  Fixed t=0.50  |  80/20 split")
print("NO AUROC comparison with binary rule — methodologically invalid")
print("=" * 78)

BASE    = os.environ.get("AQ10_BASE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work"))
PROJECT = os.path.join(BASE, "aq_10")

SPLITS_DIR    = os.path.join(PROJECT, "data/splits")
PROCESSED_DIR = os.path.join(PROJECT, "data/processed")
CONFIG_DIR    = os.path.join(PROJECT, "config")
LOGS_DIR      = os.path.join(PROJECT, "logs")
PHASE0_FLAG   = os.path.join(LOGS_DIR, "phase0_done.flag")

RQ1_DIR = os.path.join(PROJECT, "rq1_baseline")
DIRS = {
    "root":          RQ1_DIR,
    "tables":        os.path.join(RQ1_DIR, "tables"),
    "paper_ready":   os.path.join(RQ1_DIR, "tables", "paper_ready"),
    "results":       os.path.join(RQ1_DIR, "results"),
    "models":        os.path.join(RQ1_DIR, "models"),
    "predictions":   os.path.join(RQ1_DIR, "predictions"),
    "logs":          os.path.join(RQ1_DIR, "logs"),
    "config":        os.path.join(RQ1_DIR, "config"),
    "paper_tables":  os.path.join(PROJECT, "paper_results", "rq1", "tables"),
}
for p in DIRS.values():
    os.makedirs(p, exist_ok=True)

print(f"  Project : {PROJECT}")
print(f"  Output  : {RQ1_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS & HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [f"A_Q{i}" for i in range(1, 11)]
AQ_SCORE_COL = "AQ_score_legacy"
THRESHOLD    = 0.50        # FIXED from Phase 0 — NEVER post-hoc modified
N_BOOTSTRAP  = 2000

# Binary metrics (valid for BOTH M* and AQ-10 rule)
BINARY_KEYS = [
    "Sensitivity", "Specificity", "Precision_PPV", "NPV",
    "F1_Score", "Accuracy", "Kappa",
]
# All metrics (used for M* only; prob-based metrics NOT attributed to the rule)
ALL_METRIC_KEYS = [
    "AUROC", "AUPRC", "Accuracy", "Precision_PPV", "NPV",
    "Sensitivity", "Specificity", "F1_Score", "Kappa", "Brier", "ECE",
]

ALGO_ORDER = [
    "Logistic Regression",
    "Support Vector Machine",
    "Random Forest",
    "K-Nearest Neighbors",
    "Naive Bayes",
]
MODEL_FILE = {
    "Logistic Regression":    "logistic_regression.joblib",
    "Support Vector Machine": "svm.joblib",
    "Random Forest":          "random_forest.joblib",
    "K-Nearest Neighbors":    "knn.joblib",
    "Naive Bayes":            "naive_bayes.joblib",
}


def short_name(algo):
    return {"Logistic Regression":"LR","Support Vector Machine":"SVM",
            "Random Forest":"RF","K-Nearest Neighbors":"KNN",
            "Naive Bayes":"NB","AQ-10 Clinical Rule":"Rule"}.get(algo, algo)


def json_safe(obj):
    if isinstance(obj, dict):   return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [json_safe(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)): return None
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, set):         return list(obj)
    return obj


# ── Expected Calibration Error ─────────────────────────────────────────────
def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    bins   = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    if n == 0: return np.nan
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & ((y_prob <= hi) if i == n_bins-1 else (y_prob < hi))
        if mask.sum() == 0: continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)


# ── Core metric bundle ─────────────────────────────────────────────────────
def metric_bundle(y_true, y_prob, y_pred):
    """
    y_prob is used for AUROC/AUPRC/Brier/ECE.
    y_pred is used for all binary metrics.
    NOTE: When called for the AQ-10 rule, y_prob = train-calibrated probability
          (for internal consistency only); AUROC/Brier/ECE are NOT reported
          as rule metrics in any output table.
    """
    y_true = np.asarray(y_true, int)
    y_prob = np.asarray(y_prob, float)
    y_pred = np.asarray(y_pred, int)
    if len(np.unique(y_true)) < 2:
        return {k: np.nan for k in ALL_METRIC_KEYS} | {"TP":0,"TN":0,"FP":0,"FN":0}
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "AUROC":         float(roc_auc_score(y_true, y_prob)),
        "AUPRC":         float(average_precision_score(y_true, y_prob)),
        "Accuracy":      float(accuracy_score(y_true, y_pred)),
        "Precision_PPV": float(precision_score(y_true, y_pred, zero_division=0)),
        "NPV":           float(tn/(tn+fn)) if (tn+fn)>0 else 0.0,
        "Sensitivity":   float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity":   float(tn/(tn+fp)) if (tn+fp)>0 else 0.0,
        "F1_Score":      float(f1_score(y_true, y_pred, zero_division=0)),
        "Kappa":         float(cohen_kappa_score(y_true, y_pred)),
        "Brier":         float(brier_score_loss(y_true, y_prob)),
        "ECE":           float(expected_calibration_error(y_true, y_prob)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# ── McNemar's test (continuity-corrected, Edwards 1948) ───────────────────
def mcnemar_test(pred_a, pred_b):
    """
    Tests whether two binary classifiers produce systematically different
    decisions on the SAME participants (paired test on discordant cells).
    H0: P(b) = P(c)  [symmetric disagreements → no directional asymmetry]
    chi2 = (|b−c| − 1)² / (b+c)   with Edwards continuity correction
    Interpretation: Significantly different DECISION PATTERNS, NOT superiority.
    """
    pred_a = np.asarray(pred_a, int)
    pred_b = np.asarray(pred_b, int)
    b = int(((pred_a == 1) & (pred_b == 0)).sum())   # M* flags, Rule misses
    c = int(((pred_a == 0) & (pred_b == 1)).sum())   # Rule flags, M* misses
    agree_pp = int(((pred_a == 1) & (pred_b == 1)).sum())
    agree_nn = int(((pred_a == 0) & (pred_b == 0)).sum())
    n_disc = b + c
    if n_disc == 0:
        return b, c, n_disc, agree_pp, agree_nn, 0.0, 1.0
    chi2_stat = (abs(b - c) - 1.0)**2 / n_disc
    p = float(1.0 - chi2_dist.cdf(chi2_stat, df=1))
    return b, c, n_disc, agree_pp, agree_nn, float(chi2_stat), p


# ── Holm-Bonferroni correction ─────────────────────────────────────────────
def holm_bonferroni(pvals):
    pvals   = np.asarray(pvals, float)
    m       = len(pvals)
    order   = np.argsort(pvals)
    adj     = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        a = max(min((m - rank) * pvals[idx], 1.0), running)
        adj[idx] = a; running = a
    return adj


# ── DeLong paired AUROC (for secondary 5-model comparison only) ───────────
def delong_paired(y_true, score_a, score_b):
    y_true  = np.asarray(y_true,  int)
    score_a = np.asarray(score_a, float)
    score_b = np.asarray(score_b, float)
    pos = np.where(y_true == 1)[0]; neg = np.where(y_true == 0)[0]
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0: return np.nan, np.nan, 0.0, 1.0
    def _K(sp, sn):
        d = sp[:,None] - sn[None,:]
        return np.where(d>0,1., np.where(d==0,.5,0.))
    Ka = _K(score_a[pos], score_a[neg]); Kb = _K(score_b[pos], score_b[neg])
    auc_a = float(Ka.mean()); auc_b = float(Kb.mean())
    if n1 < 2 or n0 < 2: return auc_a, auc_b, 0.0, 1.0
    V10 = np.column_stack([Ka.mean(1), Kb.mean(1)])
    V01 = np.column_stack([Ka.mean(0), Kb.mean(0)])
    S10 = np.cov(V10.T)/n1; S01 = np.cov(V01.T)/n0
    var_d = S10[0,0]+S10[1,1]-2*S10[0,1] + S01[0,0]+S01[1,1]-2*S01[0,1]
    if var_d <= 0: return auc_a, auc_b, 0.0, 1.0
    from scipy.stats import norm
    z = (auc_a - auc_b) / np.sqrt(var_d)
    p = float(2.0*(1.0 - norm.cdf(abs(z))))
    return auc_a, auc_b, float(z), p


# ── Stratified percentile bootstrap ───────────────────────────────────────
def stratified_boot_idx(y, n_boot, seed):
    """Stratified bootstrap preserving original class counts."""
    y   = np.asarray(y, int)
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    return [np.concatenate([rng.choice(pos,len(pos),replace=True),
                            rng.choice(neg,len(neg),replace=True)])
            for _ in range(n_boot)]


def percentile_ci(vals, alpha=0.05):
    """Simple percentile 95% CI. No BCa."""
    v = np.asarray(vals, float); v = v[~np.isnan(v)]
    if len(v) < 50: return np.nan, np.nan
    return float(np.percentile(v, 100*alpha/2)), float(np.percentile(v, 100*(1-alpha/2)))


def fmt_ci(pt, lo, hi, decimals=4):
    if np.isnan(lo) or np.isnan(hi):
        return f"{pt:.{decimals}f} [NA]"
    return f"{pt:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


# ── Model builder ──────────────────────────────────────────────────────────
def build_model(algo, params):
    p = dict(params)
    if algo == "Naive Bayes":
        mt = p.pop("model_type")
        return (CategoricalNB(alpha=p["alpha"], min_categories=4)
                if mt == "categorical" else GaussianNB(var_smoothing=p["var_smoothing"]))
    if algo == "Logistic Regression":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(C=p["C"], penalty=p.get("penalty","l2"),
                             solver=p.get("solver","lbfgs"), max_iter=2000, random_state=SEED))])
    if algo == "Support Vector Machine":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(C=p["C"], kernel=p["kernel"], gamma=p["gamma"],
                             probability=True, random_state=SEED))])
    if algo == "Random Forest":
        return RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"], max_features="sqrt",
            random_state=SEED, n_jobs=-1)
    if algo == "K-Nearest Neighbors":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier(n_neighbors=p["n_neighbors"],
                             weights=p["weights"], metric=p["metric"], n_jobs=-1))])
    raise ValueError(f"Unknown algorithm: {algo}")


# ── AQ-10 helpers ──────────────────────────────────────────────────────────
def compute_aq10(X_arr):
    """Returns (raw integer score 0-10, binary rule pred score>=6)."""
    score = (np.asarray(X_arr) >= 2).sum(axis=1).astype(int)
    return score, (score >= 6).astype(int)


def fit_aq10_calibrator(X_train_arr, y_train_arr):
    """Logistic calibration: AQ-10 integer score → P(ASD), fitted on TRAIN only."""
    scores = (np.asarray(X_train_arr) >= 2).sum(axis=1).reshape(-1,1).astype(float)
    lr = LogisticRegression(random_state=SEED, max_iter=2000, C=1.0)
    lr.fit(scores, np.asarray(y_train_arr, int))
    return lr


def apply_aq10_calibrator(lr, X_arr):
    scores = (np.asarray(X_arr) >= 2).sum(axis=1).reshape(-1,1).astype(float)
    return lr.predict_proba(scores)[:, 1]


# ── DCA helpers ────────────────────────────────────────────────────────────
def net_benefit(y_true, y_prob, pt):
    if pt <= 0 or pt >= 1: return np.nan
    n = len(y_true); pred = (np.asarray(y_prob) >= pt).astype(int)
    tp = ((pred==1) & (np.asarray(y_true)==1)).sum()
    fp = ((pred==1) & (np.asarray(y_true)==0)).sum()
    return float(tp/n - (fp/n)*(pt/(1-pt)))


def net_benefit_all(y_true, pt):
    if pt <= 0 or pt >= 1: return np.nan
    prev = np.asarray(y_true).mean()
    return float(prev - (1-prev)*(pt/(1-pt)))


# ── Save helpers ───────────────────────────────────────────────────────────
def save_tbl(df, name):
    for d in [DIRS["tables"], DIRS["paper_tables"]]:
        df.to_csv(os.path.join(d, name), index=False)


def save_pr(df, name):
    for d in [DIRS["paper_ready"], DIRS["paper_tables"]]:
        df.to_csv(os.path.join(d, name), index=False)




# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PHASE 0 VERIFICATION + LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 1]  Phase 0 lock verification ...")

if not os.path.exists(PHASE0_FLAG):
    raise FileNotFoundError(f"Phase-0 flag missing: {PHASE0_FLAG}\nRun Phase 0 first.")
print(f"  OK  phase0_done.flag")

with open(os.path.join(CONFIG_DIR, "master_config.json")) as f:
    master_config = json.load(f)
with open(os.path.join(CONFIG_DIR, "cv_fold_indices.json")) as f:
    fold_indices = json.load(f)

N_OUTER_FOLDS = master_config["N_OUTER_FOLDS"]
print(f"  OK  master_config   SEED={master_config['SEED']}  N_FOLDS={N_OUTER_FOLDS}")

print("\n[Step 2]  Loading locked splits ...")

TRAIN_PATH = os.path.join(SPLITS_DIR, "train.csv")
TEST_PATH  = os.path.join(SPLITS_DIR, "test.csv")

for p in [TRAIN_PATH, TEST_PATH]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required split: {p}")

df_train = pd.read_csv(TRAIN_PATH)
df_test  = pd.read_csv(TEST_PATH)

required_cols = FEATURE_COLS + ["label", AQ_SCORE_COL, "AQ_rule_pred", "ID"]
for name, df_ in [("train", df_train), ("test", df_test)]:
    missing = [c for c in required_cols if c not in df_.columns]
    if missing:
        raise ValueError(f"{name}.csv missing Phase-0 columns: {missing}")

# Support both n_train (v5.0) and n_dev (v4.x) in master_config
exp_train = master_config.get("n_train", master_config.get("n_dev"))
exp_test  = master_config["n_test"]
if exp_train is not None and len(df_train) != exp_train:
    raise ValueError(f"train.csv: {len(df_train)} rows (expected {exp_train})")
if len(df_test) != exp_test:
    raise ValueError(f"test.csv: {len(df_test)} rows (expected {exp_test})")

X_train = df_train[FEATURE_COLS].astype(int)
y_train = df_train["label"].astype(int)
X_test  = df_test[FEATURE_COLS].astype(int)
y_test  = df_test["label"].astype(int)

N_TRAIN = len(df_train); N_TEST = len(df_test)

print(f"  Train : N={N_TRAIN}  ASD={int((y_train==1).sum())}  Non-ASD={int((y_train==0).sum())}")
print(f"  Test  : N={N_TEST}   ASD={int((y_test==1).sum())}   Non-ASD={int((y_test==0).sum())}")
print(f"  ASD prevalence (test): {(y_test==1).mean()*100:.1f}%")

# Participant ID leakage audit
train_ids = set(df_train["ID"].astype(str))
test_ids  = set(df_test["ID"].astype(str))
overlap   = len(train_ids & test_ids)
if overlap > 0:
    raise RuntimeError(f"Participant ID leakage: {overlap} IDs in both train and test — HALT.")
print(f"  Participant ID leakage: {overlap}  [PASS]")

# Verify AQ-10 rule encoding vs Phase 0
aq10_score_train, rule_pred_train = compute_aq10(X_train.values)
aq10_score_test,  rule_pred_test  = compute_aq10(X_test.values)

assert np.array_equal(rule_pred_train, df_train["AQ_rule_pred"].astype(int).values), \
    "AQ_rule_pred mismatch in train — check Phase 0."
assert np.array_equal(rule_pred_test,  df_test["AQ_rule_pred"].astype(int).values), \
    "AQ_rule_pred mismatch in test — check Phase 0."
print("  AQ-10 rule (score ≥6) verified against Phase 0  [PASS]")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — AQ-10 CALIBRATION (TRAIN SET ONLY)
# ──────────────────────────────────────────────────────────────────────────────
# Fitted here (before any modelling) so aq10_test_prob is available for
# consistent metric_bundle computation from the start.
# Used ONLY for DCA and calibration figures — secondary analyses.
# NOT used as "the AQ-10 rule's probability output."
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 3]  Fitting AQ-10 probability calibrator on TRAIN set only ...")
print("  (logistic regression: AQ-10 integer score → P(ASD))")
print("  Used ONLY for DCA and calibration figures (secondary).")

aq10_calibrator  = fit_aq10_calibrator(X_train.values, y_train.values)
aq10_train_prob  = apply_aq10_calibrator(aq10_calibrator, X_train.values)
aq10_test_prob   = apply_aq10_calibrator(aq10_calibrator, X_test.values)

score_vals  = np.arange(0, 11)
score_probs = aq10_calibrator.predict_proba(
    score_vals.reshape(-1,1).astype(float))[:, 1]
print(f"\n  {'Score':>6} {'P(ASD)':>10}")
for sv, sp in zip(score_vals, score_probs):
    marker = "  ← clinical threshold" if sv == 6 else ""
    print(f"  {sv:>6d} {sp:>10.4f}{marker}")

joblib.dump(aq10_calibrator, os.path.join(DIRS["models"], "aq10_calibrator.joblib"))

# AQ-10 RULE point estimates
# Binary metrics use rule_pred_test; prob-based use aq10_test_prob (internal only)
rule_point = metric_bundle(y_test.values, aq10_test_prob, rule_pred_test)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HYPERPARAMETER SEARCH SPACES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 4]  Hyperparameter search spaces ...")

search_spaces = {
    "Logistic Regression": [
        {"C": c, "penalty": "l2", "solver": "lbfgs"}
        for c in [0.01, 0.1, 1.0, 10.0, 100.0]
    ],
    "Support Vector Machine": [
        {"C": c, "kernel": k, "gamma": g}
        for c, k, g in itertools.product([0.1,1.0,10.0], ["rbf","linear"], ["scale","auto"])
    ],
    "Random Forest": [
        {"n_estimators": ne, "max_depth": md, "min_samples_leaf": msl}
        for ne, md, msl in itertools.product([300,500,800], [None,4,8,12], [1,2,5])
    ],
    "K-Nearest Neighbors": [
        {"n_neighbors": k, "weights": w, "metric": m}
        for k, w, m in itertools.product([3,5,7,9,11,15],
                                          ["uniform","distance"],
                                          ["euclidean","manhattan"])
    ],
    "Naive Bayes": (
        [{"model_type":"categorical","alpha":a} for a in [0.1,0.5,1.0,2.0]] +
        [{"model_type":"gaussian","var_smoothing":v} for v in [1e-9,1e-8,1e-7,1e-6,1e-5]]
    ),
}

total_configs = sum(len(g) for g in search_spaces.values())
print(f"  {'Algorithm':<28} {'Configs':>7} {'x Folds':>8} {'= Fits':>7}")
print("  " + "-" * 55)
for algo, grid in search_spaces.items():
    print(f"  {algo:<28} {len(grid):>7d} {N_OUTER_FOLDS:>8d} {len(grid)*N_OUTER_FOLDS:>7d}")
print(f"  {'TOTAL':<28} {total_configs:>7d} {N_OUTER_FOLDS:>8d} {total_configs*N_OUTER_FOLDS:>7d}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — 5-FOLD CV HYPERPARAMETER SEARCH  (sealed indices)
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[Step 5]  5-Fold CV HP search (identical folds across all algorithms) ...")

search_logs = []
t_start = time.time()

for algo_name, param_grid in search_spaces.items():
    print(f"\n  >>> {algo_name}  ({len(param_grid)} configs)")
    best_cv = -1.0

    for cfg_i, params in enumerate(param_grid, 1):
        t0 = time.time()
        fold_vals = {k: [] for k in ALL_METRIC_KEYS}

        for fold_k in range(1, N_OUTER_FOLDS + 1):
            fk   = fold_indices[f"fold_{fold_k}"]
            tr_i = fk["train_indices"]; vl_i = fk["val_indices"]
            Xtr  = X_train.iloc[tr_i].values; ytr = y_train.iloc[tr_i].values
            Xvl  = X_train.iloc[vl_i].values; yvl = y_train.iloc[vl_i].values

            model = build_model(algo_name, params)
            model.fit(Xtr, ytr)
            vp = model.predict_proba(Xvl)[:, 1]
            vr = (vp >= THRESHOLD).astype(int)
            vm = metric_bundle(yvl, vp, vr)
            for k in ALL_METRIC_KEYS:
                fold_vals[k].append(vm[k])

        row = {"Algorithm": algo_name, "Config_ID": cfg_i,
               "Hyperparameters": json.dumps(params, default=str),
               "Fit_Time_Sec": round(time.time()-t0, 4)}
        for k in ALL_METRIC_KEYS:
            row[f"Mean_CV_{k}"] = float(np.nanmean(fold_vals[k]))
            row[f"Std_CV_{k}"]  = float(np.nanstd(fold_vals[k]))
        search_logs.append(row)

        if row["Mean_CV_AUROC"] > best_cv:
            best_cv = row["Mean_CV_AUROC"]
        if cfg_i % 10 == 0 or cfg_i == len(param_grid):
            print(f"    [{cfg_i:3d}/{len(param_grid)}]  best CV AUROC={best_cv:.4f}")

search_df = pd.DataFrame(search_logs)
search_df.to_csv(os.path.join(DIRS["results"],"hyperparameter_search_log.csv"), index=False)
save_tbl(search_df, "T_rq1_hyperparameter_search_log.csv")
print(f"\n  CV search done in {(time.time()-t_start)/60:.1f} min")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — SELECT CHAMPION M*
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 6]  Selecting champion model M* ...")

ranked_df = search_df.sort_values(
    ["Mean_CV_AUROC","Mean_CV_Sensitivity","Mean_CV_ECE"],
    ascending=[False, False, True]).reset_index(drop=True)
save_tbl(ranked_df, "T_rq1_ranked_all_configs.csv")

best_row  = ranked_df.iloc[0]
best_algo = best_row["Algorithm"]
best_params = json.loads(best_row["Hyperparameters"])

print(f"\n  *** CHAMPION M* = {best_algo} ***")
print(f"      Params     : {best_params}")
print(f"      CV AUROC   : {best_row['Mean_CV_AUROC']:.4f} ± {best_row['Std_CV_AUROC']:.4f}")

best_per_algo = {}
best_val_rows = []
print(f"\n  {'Algorithm':<30} {'CV-AUROC':>10} {'Std':>8}  Winner?")
print("  " + "-" * 58)
for algo in ALGO_ORDER:
    sub = ranked_df[ranked_df["Algorithm"]==algo].iloc[0]
    best_per_algo[algo] = json.loads(sub["Hyperparameters"])
    best_val_rows.append(sub)
    mark = "  ← M*" if algo == best_algo else ""
    print(f"  {algo:<30} {sub['Mean_CV_AUROC']:>10.4f} {sub['Std_CV_AUROC']:>8.4f}{mark}")

save_tbl(pd.DataFrame(best_val_rows), "T_rq1_best_config_per_algo_cv.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — RETRAIN ALL MODELS ON FULL TRAIN → ONE-SHOT TEST EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[Step 7]  Retrain on full train (N={N_TRAIN}) → one-shot test evaluation ...")

keep_cols = [c for c in ["ID","label","school_id","Age","Gender",
                          AQ_SCORE_COL,"AQ_rule_pred"] if c in df_test.columns]
test_export = df_test[keep_cols].copy()
test_export["AQ10_raw_score"] = aq10_score_test
test_export["pred_Rule"]      = rule_pred_test

test_probs  = {}   # continuous probability
test_preds  = {}   # binary at THRESHOLD
point_est   = {"AQ-10 Clinical Rule": rule_point}

refit_models = {}
for algo in ALGO_ORDER:
    model = build_model(algo, best_per_algo[algo])
    model.fit(X_train.values, y_train.values)
    refit_models[algo] = model

    prob = model.predict_proba(X_test.values)[:, 1]
    pred = (prob >= THRESHOLD).astype(int)
    m    = metric_bundle(y_test.values, prob, pred)
    sk   = short_name(algo)

    test_export[f"prob_{sk}"] = prob
    test_export[f"pred_{sk}"] = pred
    test_probs[algo] = prob
    test_preds[algo] = pred
    point_est[algo]  = m

    joblib.dump(model, os.path.join(DIRS["models"], MODEL_FILE[algo]))

# Champion model exports
joblib.dump(refit_models[best_algo],
            os.path.join(DIRS["models"], "best_model_train.joblib"))
bsk = short_name(best_algo)
test_export["prob_Mstar"] = test_export[f"prob_{bsk}"]
test_export["pred_Mstar"] = test_export[f"pred_{bsk}"]
test_export["Mstar_Name"] = best_algo
test_export.to_csv(os.path.join(DIRS["predictions"],"test_predictions.csv"), index=False)
save_tbl(test_export, "T_rq1_test_predictions.csv")

print(f"\n  {'Model':<30} {'Sens':>7} {'Spec':>7} {'PPV':>7} {'NPV':>7} "
      f"{'F1':>7} {'Acc':>7} {'AUROC':>7}")
print("  " + "-" * 82)
for algo in ALGO_ORDER + ["AQ-10 Clinical Rule"]:
    m  = point_est[algo]
    mk = "  [M*]" if algo == best_algo else ""
    auroc_str = f"{m['AUROC']:.4f}" if algo != "AQ-10 Clinical Rule" else "(n/a)"
    print(f"  {algo+mk:<30} {m['Sensitivity']:>7.4f} {m['Specificity']:>7.4f} "
          f"{m['Precision_PPV']:>7.4f} {m['NPV']:>7.4f} "
          f"{m['F1_Score']:>7.4f} {m['Accuracy']:>7.4f} {auroc_str:>7}")
print("  NOTE: AUROC shown (n/a) for AQ-10 rule — rule is binary, not probabilistic.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — STRATIFIED PERCENTILE BOOTSTRAP CIs
# ══════════════════════════════════════════════════════════════════════════════

print(f"\n[Step 8]  Bootstrap CIs (B={N_BOOTSTRAP}, stratified by class) ...")
print("  M*   : binary metrics + AUROC/Brier/ECE (secondary)")
print("  Rule : binary metrics ONLY (rule is binary, no prob-based metrics reported)")

boot_idx = stratified_boot_idx(y_test.values, N_BOOTSTRAP, SEED)

boot_mstar = {k: [] for k in ALL_METRIC_KEYS}
boot_rule  = {k: [] for k in BINARY_KEYS}

for b_i, idx in enumerate(boot_idx):
    y_b = y_test.values[idx]
    if len(np.unique(y_b)) < 2: continue

    # M* — all metrics
    mb_m = metric_bundle(y_b,
                         test_probs[best_algo][idx],
                         test_preds[best_algo][idx])
    for k in ALL_METRIC_KEYS:
        boot_mstar[k].append(mb_m[k])

    # AQ-10 rule — binary metrics only (prob-based NOT attributed to rule)
    mb_r = metric_bundle(y_b,
                         aq10_test_prob[idx],
                         rule_pred_test[idx])
    for k in BINARY_KEYS:
        boot_rule[k].append(mb_r[k])

    if (b_i + 1) % 500 == 0:
        print(f"    bootstrap {b_i+1}/{N_BOOTSTRAP}")

# Build CI rows
ci_rows = []
for label_, boot_d, keys_, pt_src in [
    (best_algo,            boot_mstar, ALL_METRIC_KEYS, point_est[best_algo]),
    ("AQ-10 Clinical Rule", boot_rule,  BINARY_KEYS,    rule_point),
]:
    row = {"Model": label_, "Is_Mstar": label_ == best_algo}
    for k in keys_:
        lo, hi = percentile_ci(boot_d[k])
        pt     = float(pt_src[k])
        row[f"{k}_Point"]     = round(pt, 4)
        row[f"{k}_95CI_Lo"]   = round(lo,4) if not np.isnan(lo) else None
        row[f"{k}_95CI_Hi"]   = round(hi,4) if not np.isnan(hi) else None
        row[f"{k}_Formatted"] = fmt_ci(pt, lo, hi)
    for k in ["TP","TN","FP","FN"]:
        row[k] = pt_src[k]
    ci_rows.append(row)

ci_df = pd.DataFrame(ci_rows)
save_tbl(ci_df, "T_rq1_bootstrap_CI_all.csv")

# Paper-ready screening table (binary metrics, NO AUROC column for rule)
print("\n  Binary screening performance with 95% CI:")
print(f"  {'Metric':<16} {'M*':<32} {'AQ-10 Rule (≥6)':<32}")
print("  " + "-" * 82)
mstar_row = ci_df[ci_df["Is_Mstar"]].iloc[0]
rule_row  = ci_df[~ci_df["Is_Mstar"]].iloc[0]
for k in BINARY_KEYS:
    mf = mstar_row.get(f"{k}_Formatted", "NA")
    rf = rule_row.get(f"{k}_Formatted", "NA")
    print(f"  {k:<16} {mf:<32} {rf:<32}")

# Paper-ready table (binary only)
binary_paper_cols = ["Model","Sensitivity_Formatted","Specificity_Formatted",
                     "Precision_PPV_Formatted","NPV_Formatted","F1_Score_Formatted",
                     "Accuracy_Formatted","Kappa_Formatted","TP","FP","FN","TN"]
paper_binary = []
for row_ in ci_rows:
    paper_binary.append({
        "Model":       row_["Model"],
        "Threshold":   f"t={THRESHOLD}" if row_["Is_Mstar"] else "score ≥6",
        "Sensitivity": row_.get("Sensitivity_Formatted","NA"),
        "Specificity": row_.get("Specificity_Formatted","NA"),
        "PPV":         row_.get("Precision_PPV_Formatted","NA"),
        "NPV":         row_.get("NPV_Formatted","NA"),
        "F1":          row_.get("F1_Score_Formatted","NA"),
        "Accuracy":    row_.get("Accuracy_Formatted","NA"),
        "Kappa":       row_.get("Kappa_Formatted","NA"),
        "TP":          row_.get("TP","NA"), "FP": row_.get("FP","NA"),
        "FN":          row_.get("FN","NA"), "TN": row_.get("TN","NA"),
    })
paper_binary_df = pd.DataFrame(paper_binary)
save_pr(paper_binary_df, "T_rq1_paper_Table1_screening_performance.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — PRIMARY ANALYSIS: McNEMAR'S TEST
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("[Step 9]  PRIMARY ANALYSIS: McNemar's Test (continuity-corrected)")
print(f"  M* binary (t={THRESHOLD})  vs  AQ-10 ≥6 rule")
print("  H0: Symmetric discordance (P(b) = P(c))")
print("  Interpretation: PAIRED DECISION DIFFERENCES — not superiority")
print("=" * 78)

b_mc, c_mc, n_disc, agree_pp, agree_nn, chi2_mc, p_mc = mcnemar_test(
    test_preds[best_algo], rule_pred_test)
sig_mcnemar = p_mc < 0.05

total_test = len(y_test)
print(f"\n  Prediction agreement table  (N={total_test}):")
print(f"  ┌──────────────────────────────────────────────┐")
print(f"  │                   AQ-10 Neg    AQ-10 Pos     │")
print(f"  │  M* Pos (flag)    {b_mc:>7d}      {agree_pp:>7d}        │")
print(f"  │  M* Neg (miss)    {agree_nn:>7d}      {c_mc:>7d}        │")
print(f"  └──────────────────────────────────────────────┘")
print(f"  b (M*+/Rule−) : {b_mc}   c (M*−/Rule+) : {c_mc}")
print(f"  Discordant pairs total : {n_disc}")
print(f"  χ² (continuity-corrected) : {chi2_mc:.4f}")
print(f"  p-value                   : {p_mc:.6g}")
print(f"  Significant (α=0.05)      : {'YES — different decision patterns' if sig_mcnemar else 'NO — no significant difference'}")

if n_disc > 0:
    print(f"\n  Among {n_disc} discordant pairs:")
    print(f"    M* flags, Rule misses : {b_mc} ({100*b_mc/n_disc:.1f}%)")
    print(f"    Rule flags, M* misses : {c_mc} ({100*c_mc/n_disc:.1f}%)")

mcnemar_result = {
    "test":                     "McNemar continuity-corrected (Edwards 1948)",
    "comparison":               f"{best_algo} binary (t={THRESHOLD}) vs AQ-10 >=6 rule",
    "threshold_Mstar":          THRESHOLD,
    "threshold_Rule":           "score >= 6",
    "b_Mstar1_Rule0":           b_mc,
    "c_Mstar0_Rule1":           c_mc,
    "n_discordant":             n_disc,
    "n_concordant_both_pos":    agree_pp,
    "n_concordant_both_neg":    agree_nn,
    "chi2_continuity_corrected":round(chi2_mc, 4),
    "p_value":                  round(p_mc, 6),
    "alpha":                    0.05,
    "significant":              bool(sig_mcnemar),
    "correct_interpretation":   (
        "Significantly different paired decision patterns "
        if sig_mcnemar else
        "No significant difference in paired decision patterns"),
    "incorrect_interpretation_to_avoid": (
        "DO NOT say McNemar proves M* has higher accuracy. "
        "It tests symmetry of disagreements, not accuracy."),
}
mcnemar_df = pd.DataFrame([mcnemar_result])
save_tbl(mcnemar_df, "T_rq1_primary_mcnemar.csv")
save_pr(mcnemar_df,  "T_rq1_paper_primary_mcnemar.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — M* SECONDARY CHARACTERISATION (AUROC, Brier, ECE)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 10]  M* secondary characterisation (AUROC, Calibration) ...")
print("  NOTE: These metrics characterise M* as a MODEL, not for rule comparison.")
print("  AQ-10 INSTRUMENT AUROC shown separately, labeled 'context only'.")

mstar_auroc     = float(point_est[best_algo]["AUROC"])
mstar_brier     = float(point_est[best_algo]["Brier"])
mstar_ece       = float(point_est[best_algo]["ECE"])
aq10_inst_auroc = float(roc_auc_score(y_test.values, aq10_score_test))

auroc_lo, auroc_hi = percentile_ci(boot_mstar["AUROC"])
brier_lo, brier_hi = percentile_ci(boot_mstar["Brier"])
ece_lo,   ece_hi   = percentile_ci(boot_mstar["ECE"])

print(f"\n  M* AUROC         : {fmt_ci(mstar_auroc, auroc_lo, auroc_hi)}")
print(f"  M* Brier Score   : {fmt_ci(mstar_brier, brier_lo, brier_hi)}")
print(f"  M* ECE           : {fmt_ci(mstar_ece,   ece_lo,   ece_hi)}")
print(f"\n  AQ-10 instrument AUROC : {aq10_inst_auroc:.4f}  [context only — NOT rule AUROC]")
print("  (The instrument's potential discrimination differs from the rule's fixed ≥6 operating point)")

secondary_m = {
    "model":           best_algo,
    "AUROC_point":     round(mstar_auroc, 4),
    "AUROC_95CI_Lo":   round(auroc_lo, 4) if not np.isnan(auroc_lo) else None,
    "AUROC_95CI_Hi":   round(auroc_hi, 4) if not np.isnan(auroc_hi) else None,
    "AUROC_Formatted": fmt_ci(mstar_auroc, auroc_lo, auroc_hi),
    "Brier_Formatted": fmt_ci(mstar_brier, brier_lo, brier_hi),
    "ECE_Formatted":   fmt_ci(mstar_ece,   ece_lo,   ece_hi),
    "AQ10_instrument_AUROC_context_only": round(aq10_inst_auroc, 4),
    "note": (
        "AUROC is a secondary, descriptive metric characterising M* as a model. "
        "The AQ-10 instrument AUROC is for context only and does NOT represent "
        "the AQ-10 clinical rule (score>=6) which is a binary classifier."),
}
secondary_df = pd.DataFrame([secondary_m])
save_tbl(secondary_df, "T_rq1_secondary_mstar_auroc_calibration.csv")
save_pr(secondary_df,  "T_rq1_paper_secondary_discrimination.csv")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — AQ-10 THRESHOLD ANALYSIS  (0–10)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 11]  AQ-10 threshold analysis (scores 0–10) ...")
print("  Separates instrument discrimination from the ≥6 rule's operating point.")

thresh_rows = []
for t in range(0, 11):
    pred_t = (aq10_score_test >= t).astype(int)
    n_pos  = int(pred_t.sum()); n_neg = int((1-pred_t).sum())
    if n_pos == 0 or n_neg == 0:
        thresh_rows.append({
            "AQ10_threshold": t,
            "N_flagged": n_pos,
            "Sensitivity": np.nan, "Specificity": np.nan,
            "F1": np.nan, "Accuracy": np.nan, "PPV": np.nan, "NPV": np.nan,
            "Note": "All positive" if n_pos == N_TEST else "All negative",
        })
        continue
    tn, fp, fn, tp = confusion_matrix(y_test.values, pred_t, labels=[0,1]).ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0
    ppv  = tp/(tp+fp) if (tp+fp)>0 else 0
    npv  = tn/(tn+fn) if (tn+fn)>0 else 0
    f1   = f1_score(y_test.values, pred_t, zero_division=0)
    acc  = accuracy_score(y_test.values, pred_t)
    thresh_rows.append({
        "AQ10_threshold": t, "N_flagged": n_pos,
        "Sensitivity": round(sens,4), "Specificity": round(spec,4),
        "F1": round(f1,4), "Accuracy": round(acc,4),
        "PPV": round(ppv,4), "NPV": round(npv,4),
        "Note": "Clinical rule" if t == 6 else "",
    })

thresh_df = pd.DataFrame(thresh_rows)
save_tbl(thresh_df, "T_rq1_aq10_threshold_analysis.csv")

print(f"\n  {'Threshold':>10} {'Sensitivity':>12} {'Specificity':>12} "
      f"{'F1':>8} {'Accuracy':>10}  Note")
print("  " + "-" * 65)
for _, r in thresh_df.iterrows():
    note = "  ← clinical rule" if r["AQ10_threshold"] == 6 else ""
    sens = f"{r['Sensitivity']:.4f}" if not pd.isna(r['Sensitivity']) else "  NA"
    spec = f"{r['Specificity']:.4f}" if not pd.isna(r['Specificity']) else "  NA"
    f1_  = f"{r['F1']:.4f}"          if not pd.isna(r['F1'])          else "  NA"
    acc_ = f"{r['Accuracy']:.4f}"    if not pd.isna(r['Accuracy'])    else "  NA"
    print(f"  {int(r['AQ10_threshold']):>10d} {sens:>12} {spec:>12} "
          f"{f1_:>8} {acc_:>10}{note}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — DECISION CURVE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 12]  Decision Curve Analysis (clinical window 0.10–0.50) ...")

pt_grid  = np.round(np.arange(0.01, 1.00, 0.01), 2)
dca_rows = []
for pt in pt_grid:
    nb_m   = net_benefit(y_test.values, test_probs[best_algo], pt)
    nb_aq  = net_benefit(y_test.values, aq10_test_prob, pt)
    nb_all = net_benefit_all(y_test.values, pt)
    dca_rows.append({"Threshold": float(pt),
                     "NB_Mstar": nb_m, "NB_AQ10_calibrated": nb_aq,
                     "NB_TreatAll": nb_all, "NB_TreatNone": 0.0,
                     "Delta_Mstar_vs_AQ10": (nb_m-nb_aq
                                              if not (np.isnan(nb_m) or np.isnan(nb_aq))
                                              else np.nan)})
dca_df = pd.DataFrame(dca_rows)
save_tbl(dca_df, "T_rq1_dca_results.csv")

win = dca_df[(dca_df["Threshold"]>=0.10) & (dca_df["Threshold"]<=0.50)]
dca_summary = {"clinical_window":"0.10-0.50",
               "mean_NB_Mstar":   round(win["NB_Mstar"].mean(),4),
               "mean_NB_AQ10":    round(win["NB_AQ10_calibrated"].mean(),4),
               "mean_NB_TreatAll":round(win["NB_TreatAll"].mean(),4),
               "pct_Mstar_wins":  round(100*(win["Delta_Mstar_vs_AQ10"]>0).mean(),1),
               "note": "Pct_Mstar_wins = % of grid points, not an inferential statistic"}
save_pr(pd.DataFrame([dca_summary]), "T_rq1_paper_dca_summary.csv")
print(f"  Clinical window 0.10-0.50:")
for k, v in dca_summary.items():
    if k != "note": print(f"    {k:<25} {v}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — SECONDARY: PAIRWISE DELONG (5 ML MODELS, EXPLORATORY)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 13]  SECONDARY/EXPLORATORY: Pairwise DeLong (5 ML models) ...")
print("  Holm-Bonferroni corrected | Model ranking ONLY")
print("  NOTE: AQ-10 rule is NOT included — rule comparison is the PRIMARY (McNemar).")

sec_rows = []; sec_pvals = []
pairs = list(itertools.combinations(range(len(ALGO_ORDER)), 2))
for i, j in pairs:
    ma, mb_ = ALGO_ORDER[i], ALGO_ORDER[j]
    a_a, a_b, z_, p_ = delong_paired(y_test.values, test_probs[ma], test_probs[mb_])
    sec_rows.append({"Model_A":ma, "Model_B":mb_,
                     "AUROC_A":round(a_a,4), "AUROC_B":round(a_b,4),
                     "Delta":round(a_a-a_b,4), "z":round(z_,4), "p":round(p_,6)})
    sec_pvals.append(p_)

adj = holm_bonferroni(sec_pvals)
for row_, pa in zip(sec_rows, adj):
    row_["p_holm"] = round(float(pa),6)
    row_["sig_holm"] = bool(pa < 0.05)

sec_df = pd.DataFrame(sec_rows).sort_values("Delta",ascending=False).reset_index(drop=True)
save_tbl(sec_df, "T_rq1_secondary_delong_5models.csv")

n_sig = int(sec_df["sig_holm"].sum())
print(f"  {len(sec_df)} pairs | {n_sig} significant after Holm")
print(f"\n  {'A':<26} {'B':<26} {'Δ':>7} {'z':>7} {'p':>10} {'p_holm':>10} Sig")
print("  " + "-" * 95)
for _, r in sec_df.iterrows():
    am = " [M*]" if r["Model_A"]==best_algo else ""
    bm = " [M*]" if r["Model_B"]==best_algo else ""
    print(f"  {r['Model_A']+am:<26} {r['Model_B']+bm:<26} "
          f"{r['Delta']:>+7.4f} {r['z']:>7.4f} {r['p']:>10.6g} {r['p_holm']:>10.6g} "
          f"{'*' if r['sig_holm'] else 'ns'}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — SAVE M* CONFIGURATION (locked for RQ2)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 14]  Saving locked M* configuration for RQ2 ...")

best_config = {
    "selected_algorithm":  best_algo,
    "short_name":          short_name(best_algo),
    "hyperparameters":     best_params,
    "feature_set":         "S0",
    "feature_columns":     FEATURE_COLS,
    "decision_threshold":  THRESHOLD,
    "threshold_note":      "Fixed at 0.50, pre-specified Phase 0, NEVER post-hoc modified",
    "selection_rule":      "Mean_CV_AUROC desc → Mean_CV_Sensitivity desc → Mean_CV_ECE asc",
    "cv_performance": {
        "Mean_CV_AUROC":       float(best_row["Mean_CV_AUROC"]),
        "Std_CV_AUROC":        float(best_row["Std_CV_AUROC"]),
        "Mean_CV_Sensitivity": float(best_row["Mean_CV_Sensitivity"]),
        "N_Folds":             N_OUTER_FOLDS,
    },
    "test_binary_metrics": {k: float(point_est[best_algo][k]) for k in BINARY_KEYS},
    "test_discrimination": {
        "AUROC": round(mstar_auroc, 4),
        "AUROC_95CI": [round(auroc_lo,4), round(auroc_hi,4)],
        "note":  "Secondary characterisation of M* — NOT compared to AQ-10 rule AUROC",
    },
    "test_calibration": {
        "Brier": round(mstar_brier, 4), "ECE": round(mstar_ece, 4),
    },
    "primary_test_mcnemar": json_safe(mcnemar_result),
    "dca_summary": json_safe(dca_summary),
    "model_file":       "best_model_train.joblib",
    "calibrator_file":  "aq10_calibrator.joblib",
    "rq2_note": (
        f"M*={best_algo}. Threshold={THRESHOLD}. "
        "For LOSO: lock algorithm+hyperparameters+threshold. "
        "Retrain FRESH model inside each LOSO fold — do NOT reuse joblib model."),
    "created_at": datetime.now().isoformat(),
}

for path in [os.path.join(DIRS["models"],  "best_model_config.json"),
             os.path.join(DIRS["config"],  "best_model_config.json"),
             os.path.join(DIRS["results"], "best_model_config.json")]:
    with open(path, "w") as f:
        json.dump(json_safe(best_config), f, indent=4, default=str)

with open(os.path.join(DIRS["models"], "best_model_name.txt"), "w") as f:
    f.write(best_algo + "\n")
with open(os.path.join(DIRS["models"], "all_best_hyperparams.json"), "w") as f:
    json.dump(json_safe(best_per_algo), f, indent=4, default=str)

print(f"  Locked M* = {best_algo}  |  Params = {best_params}  |  t = {THRESHOLD}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15: FIGURES
# ══════════════════════════════════════════════════════════════════════════════

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Arial is often missing on Linux; no warning flood
FIG_STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
             "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
             "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False}
FIG_DIR = os.path.join(PROJECT, "figures")
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


print("\n[Step 15]  Figures ...")
SHORT = {"Logistic Regression": "Logistic regression", "Support Vector Machine": "SVM", "Random Forest": "Random forest",
         "K-Nearest Neighbors": "k-nearest neighbors", "Naive Bayes": "Naive Bayes"}
LCOL = {"Logistic Regression": "#999999", "Support Vector Machine": "#0072B2", "Random Forest": "#009E73",
        "K-Nearest Neighbors": "#CC79A7", "Naive Bayes": "#E69F00"}
C_MODEL, C_RULE, C_TOT = "#0072B2", "#D55E00", "#7F7F7F"
sel = SHORT[best_algo]

# Model selection: mean CV AUROC of every setting, by learner
with plt.rc_context(FIG_STYLE):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.2, top=0.93)
    jit = np.random.default_rng(0)                    # jitter only; no effect on any number
    for i, algo in enumerate(ALGO_ORDER):
        v = search_df.loc[search_df["Algorithm"] == algo, "Mean_CV_AUROC"].values
        ax.scatter(i + jit.uniform(-0.2, 0.2, len(v)), v, s=6, color="#9DB9D5", lw=0, zorder=2)
        best_v = v.max()
        ax.plot([i - 0.3, i + 0.3], [best_v, best_v], color=C_MODEL if algo == best_algo else "#444444", lw=1.4, zorder=3)
        ax.text(i, best_v, f" {best_v:.3f}", ha="center", va="bottom", fontsize=6,
                color=C_MODEL if algo == best_algo else "#444444")
    ax.set_xticks(range(len(ALGO_ORDER)))
    ax.set_xticklabels([f"{SHORT[a].replace(' ', chr(10), 1)}\n({int((search_df['Algorithm'] == a).sum())})" for a in ALGO_ORDER],
                       fontsize=6.3)
    for tl, algo in zip(ax.get_xticklabels(), ALGO_ORDER):
        if algo == best_algo:
            tl.set_color(C_MODEL); tl.set_fontweight("bold")
    ax.set_ylabel("Mean AUROC in five-fold CV")
    ax.set_title("Points: settings (number in brackets). Bars: best setting. Blue: selected.", fontsize=6.2, loc="left")
    save_figure(fig, "rq1_model_selection_cv_auroc.png")

# Test set: ROC curves of the five learners, and the selected model against the rule
with plt.rc_context(FIG_STYLE):
    fig = plt.figure(figsize=(7.16, 2.7))
    ax_a = fig.add_axes([0.065, 0.16, 0.3, 0.76])
    ax_b = fig.add_axes([0.5, 0.16, 0.43, 0.7])
    for algo in ALGO_ORDER:
        fpr_, tpr_, _ = roc_curve(y_test.values, test_probs[algo])
        ax_a.plot(fpr_, tpr_, color=LCOL[algo], lw=1.5 if algo == best_algo else 0.9, zorder=3 if algo == best_algo else 2,
                  label=f"{SHORT[algo]} ({roc_auc_score(y_test.values, test_probs[algo]):.3f})")
    fpr_, tpr_, _ = roc_curve(y_test.values, aq10_score_test)
    ax_a.plot(fpr_, tpr_, color=C_TOT, ls=(0, (4, 2)), lw=1.0,
              label=f"AQ-10 total score ({roc_auc_score(y_test.values, aq10_score_test):.3f})")
    rp_ = point_est["AQ-10 Clinical Rule"]
    ax_a.plot(1 - rp_["Specificity"], rp_["Sensitivity"], "s", color=C_RULE, ms=4, zorder=4, label="AQ-10 rule (score \u2265 6)")
    ax_a.plot([0, 1], [0, 1], ":", color="#BBBBBB", lw=0.6)
    ax_a.set_xlabel("False-positive rate"); ax_a.set_ylabel("Sensitivity")
    ax_a.legend(title="Test set (AUROC)", loc="lower right", frameon=False, fontsize=5.8, title_fontsize=6)
    mets = [("Sensitivity", "Sensitivity"), ("Specificity", "Specificity"), ("PPV", "Precision_PPV"), ("NPV", "NPV"),
            ("Accuracy", "Accuracy"), ("F1 score", "F1_Score")]
    lows = []
    for i, (lab, key) in enumerate(mets):
        yy = len(mets) - 1 - i
        for row_, col, mk, off, name in ((mstar_row, C_MODEL, "o", 0.13, f"{sel}, threshold {THRESHOLD:.2f}"),
                                         (rule_row, C_RULE, "s", -0.13, "AQ-10 rule (score \u2265 6)")):
            pt = float(row_[f"{key}_Point"])
            lo = float(row_[f"{key}_95CI_Lo"]) if pd.notna(row_[f"{key}_95CI_Lo"]) else pt
            hi = float(row_[f"{key}_95CI_Hi"]) if pd.notna(row_[f"{key}_95CI_Hi"]) else pt
            lows.append(lo)
            ax_b.errorbar(pt, yy + off, xerr=[[max(pt - lo, 0)], [max(hi - pt, 0)]], fmt=mk, color=col, ms=3.5,
                          elinewidth=0.9, capsize=1.5, label=name if i == 0 else None)
            ax_b.text(1.012, yy + off, f"{pt:.3f}", transform=ax_b.get_yaxis_transform(), va="center", fontsize=6, color=col)
    ax_b.set_yticks(range(len(mets))); ax_b.set_yticklabels([m[0] for m in mets][::-1])
    ax_b.set_xlim(max(0.0, min(lows) - 0.05), 1.0); ax_b.set_ylim(-0.6, len(mets) - 0.4)
    ax_b.set_xlabel(f"Test set, {len(y_test)} children (95% bootstrap interval, 2,000 stratified resamples)")
    ax_b.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, frameon=False, fontsize=6.3)
    for ax_, letter, dx, yy_ in ((ax_a, "(a)", -0.2, 1.02), (ax_b, "(b)", -0.17, 1.1)):
        ax_.text(dx, yy_, letter, transform=ax_.transAxes, fontsize=8, fontweight="bold", va="bottom")
    save_figure(fig, "rq1_test_set_roc_and_svm_vs_rule.png")

# Test set: calibration of the selected model, and net benefit
with plt.rc_context(FIG_STYLE):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.16, 2.6))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.17, top=0.9, wspace=0.28)
    mp, ob, nb_ = reliability_bins(y_test.values, test_probs[best_algo])
    ax_a.plot([0, 1], [0, 1], ":", color="#999999", lw=0.7)
    ax_a.plot(mp, ob, color=C_MODEL, lw=0.8)
    ax_a.scatter(mp, ob, s=6 + 40 * nb_ / nb_.max(), color=C_MODEL, zorder=3)
    ax_a.text(0.04, 0.96, f"{sel}\nBrier {mstar_brier:.3f}\nECE {mstar_ece:.3f}", transform=ax_a.transAxes, va="top", fontsize=6.5)
    ax_a.set_xlabel("Predicted probability (mean of ten equal bins; point size = children)")
    ax_a.set_ylabel("Observed share of ASD-school children")
    ax_a.set_xlim(-0.02, 1.02); ax_a.set_ylim(-0.02, 1.02)
    d_ = dca_df[dca_df["Threshold"] <= 0.8]
    ax_b.axvspan(0.10, 0.50, color="#EEEEEE", lw=0, label="Thresholds 0.10 to 0.50")
    ax_b.plot(d_["Threshold"], d_["NB_Mstar"], color=C_MODEL, lw=1.2, label=sel)
    ax_b.plot(d_["Threshold"], d_["NB_AQ10_calibrated"], color=C_RULE, lw=1.0, ls=(0, (4, 2)),
              label="AQ-10 total score (logistic calibration)")
    ax_b.plot(d_["Threshold"], d_["NB_TreatAll"], color="#777777", lw=0.8, ls=":", label="Refer all")
    ax_b.axhline(0, color="#222222", lw=0.6, label="Refer none")
    top_ = float(np.nanmax(d_[["NB_Mstar", "NB_AQ10_calibrated", "NB_TreatAll"]].values))
    ax_b.set_ylim(-0.05, top_ + 0.05); ax_b.set_xlim(0, 0.8)
    ax_b.set_xlabel("Threshold probability"); ax_b.set_ylabel("Net benefit")
    ax_b.legend(loc="lower left", bbox_to_anchor=(0.0, 0.06), frameon=False, fontsize=6)
    for ax_, letter in ((ax_a, "(a)"), (ax_b, "(b)")):
        ax_.text(-0.17, 1.02, letter, transform=ax_.transAxes, fontsize=8, fontweight="bold", va="bottom")
    save_figure(fig, "rq1_test_set_calibration_and_net_benefit.png")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — MASTER SUMMARY JSON + DONE FLAG
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 16]  Master summary + done flag ...")

master_summary = {
    "module":  "RQ1_Local_Performance_vs_AQ10_Rule",
    "version": "FINAL-7.0-RQ1",
    "status":  "COMPLETE",
    "timestamp": datetime.now().isoformat(),

    "methodology": {
        "split":          "80/20 school-stratified (Phase 0 FINAL-5.0)",
        "hp_search":      f"{N_OUTER_FOLDS}-fold StratifiedKFold CV on train (sealed)",
        "threshold":      THRESHOLD,
        "threshold_note": "Fixed Phase 0, NEVER post-hoc",
        "bootstrap":      f"Stratified percentile (no BCa), B={N_BOOTSTRAP}, 95% CI",
        "seed":           SEED,
        "primary_test":   "McNemar continuity-corrected (M* binary vs AQ-10 >=6)",
        "why_no_auroc_primary": (
            "AQ-10 clinical rule is binary (score>=6). It has no continuous "
            "probability output. AUROC comparison between M* (prob) and rule "
            "(binary) conflates instrument potential with rule operating point. "
            "Methodologically invalid — removed from primary analysis."),
    },

    "champion_Mstar": {
        "algorithm":     best_algo,
        "short_name":    short_name(best_algo),
        "hyperparameters": best_params,
        "cv_auroc_mean": float(best_row["Mean_CV_AUROC"]),
        "cv_auroc_std":  float(best_row["Std_CV_AUROC"]),
    },

    "PRIMARY_McNemar": json_safe(mcnemar_result),

    "binary_performance_Mstar":      json_safe({k: float(point_est[best_algo][k]) for k in BINARY_KEYS}),
    "binary_performance_AQ10_rule":  json_safe({k: float(rule_point[k]) for k in BINARY_KEYS}),

    "secondary_Mstar_discrimination": json_safe(secondary_m),
    "secondary_calibration": {
        "Brier": round(mstar_brier,4), "ECE": round(mstar_ece,4),
    },
    "secondary_dca":        json_safe(dca_summary),
    "secondary_delong_5models": {
        "n_comparisons":      len(sec_df),
        "n_significant_holm": n_sig,
        "correction":         "Holm-Bonferroni",
        "results": json_safe(sec_df.to_dict(orient="records")),
    },

    "figures": FIGURES_SAVED,

    "tables": [
        "T_rq1_hyperparameter_search_log.csv",
        "T_rq1_ranked_all_configs.csv",
        "T_rq1_best_config_per_algo_cv.csv",
        "T_rq1_test_predictions.csv",
        "T_rq1_bootstrap_CI_all.csv",
        "T_rq1_primary_mcnemar.csv",
        "T_rq1_secondary_mstar_auroc_calibration.csv",
        "T_rq1_aq10_threshold_analysis.csv",
        "T_rq1_dca_results.csv",
        "T_rq1_secondary_delong_5models.csv",
    ],

    "leakage_audit": {"participant_id_overlap": overlap, "status": "PASS"},
    "next": f"RQ2 (LOSO) | M*={best_algo} | t={THRESHOLD} | Lock alg+params+threshold",
}

with open(os.path.join(DIRS["results"], "RQ1_Master_Summary.json"), "w") as f:
    json.dump(json_safe(master_summary), f, indent=4, default=str)

flag_txt = (
    f"RQ1 COMPLETE — FINAL-7.0-RQ1\n"
    f"timestamp={datetime.now().isoformat()}\n"
    f"winner_algorithm={best_algo}\n"
    f"winner_short={short_name(best_algo)}\n"
    f"winner_params={json.dumps(best_params)}\n"
    f"mean_cv_auroc={float(best_row['Mean_CV_AUROC']):.6f}\n"
    f"primary_test=McNemar_continuity_corrected\n"
    f"mcnemar_b={b_mc}  mcnemar_c={c_mc}  n_discordant={n_disc}\n"
    f"mcnemar_chi2={chi2_mc:.6f}\n"
    f"mcnemar_p={p_mc:.6g}\n"
    f"mcnemar_significant={sig_mcnemar}\n"
    f"mstar_auroc_secondary={mstar_auroc:.6f}\n"
    f"mstar_brier={mstar_brier:.6f}\n"
    f"mstar_ece={mstar_ece:.6f}\n"
    f"aq10_inst_auroc_context_only={aq10_inst_auroc:.6f}\n"
    f"threshold_fixed={THRESHOLD}\n"
    f"bootstrap=stratified_percentile_B{N_BOOTSTRAP}\n"
    f"no_delong_vs_rule=TRUE\n"
    f"no_youden=TRUE\n"
    f"no_bca=TRUE\n"
    f"feature_set=S0\n"
    f"train_n={N_TRAIN}  test_n={N_TEST}\n"
)
for fp in [os.path.join(DIRS["logs"], "rq1_done.flag"),
           os.path.join(LOGS_DIR,     "rq1_done.flag")]:
    with open(fp, "w") as f:
        f.write(flag_txt)

print("  Master summary and done flag saved.")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL CONSOLE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 78)
print("RQ1  FINAL SUMMARY  —  FINAL-7.0-RQ1")
print("=" * 78)
print(f"\n  CHAMPION  M* = {best_algo}")
print(f"    Params    : {best_params}")
print(f"    CV AUROC  : {best_row['Mean_CV_AUROC']:.4f} ± {best_row['Std_CV_AUROC']:.4f}")
print(f"    Threshold : {THRESHOLD}  (FIXED, pre-specified Phase 0)\n")

print(f"  {'Metric':<14} {'M* [95% CI]':<36} {'AQ-10 Rule ≥6 [95% CI]'}")
print("  " + "-" * 80)
for k in BINARY_KEYS:
    mf = mstar_row.get(f"{k}_Formatted","NA")
    rf = rule_row.get(f"{k}_Formatted","NA")
    print(f"  {k:<14} {mf:<36} {rf}")

print(f"""
  ┌──────────────────────────────────────────────────────────────┐
  │  PRIMARY: McNemar (M* binary vs AQ-10 ≥6 rule)              │
  │  b (M*+/Rule−) = {b_mc}    c (M*−/Rule+) = {c_mc}                    │
  │  Discordant pairs = {n_disc}                                       │
  │  χ² (continuity-corrected) = {chi2_mc:.4f}                    │
  │  p = {p_mc:.6g}                                           │
  │  Significant (α=0.05): {'YES — different decision patterns' if sig_mcnemar else 'NO'}          │
  ├──────────────────────────────────────────────────────────────┤
  │  SECONDARY: M* discrimination (not compared to rule)         │
  │  M* AUROC = {fmt_ci(mstar_auroc,auroc_lo,auroc_hi)}          │
  │  AQ-10 instrument AUROC = {aq10_inst_auroc:.4f} [CONTEXT ONLY]     │
  ├──────────────────────────────────────────────────────────────┤
  │  SECONDARY: M* calibration                                   │
  │  Brier = {fmt_ci(mstar_brier,brier_lo,brier_hi)}             │
  │  ECE   = {fmt_ci(mstar_ece,ece_lo,ece_hi)}                   │
  └──────────────────────────────────────────────────────────────┘
""")

print(f"  OUTPUTS:")
print(f"    Tables      : {DIRS['tables']}")
print(f"    Figures     : {FIG_DIR}")
print(f"    Paper-ready : {DIRS['paper_ready']}")
print(f"    Models      : {DIRS['models']}")
print("\n" + "=" * 78)
print(f"STATUS : RQ1 FINAL-7.0-RQ1  COMPLETE")
print(f"NEXT   : RQ2 (LOSO)  |  M* = {best_algo}  |  t = {THRESHOLD}")
print("=" * 78)

