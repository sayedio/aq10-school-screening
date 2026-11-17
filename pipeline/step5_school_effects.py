"""Step 5 - School effects: does the model learn school-specific response patterns?

Run after rq4_explanation.py (it reads the RQ1 to RQ3 outputs and needs the RQ4 done flag):

    python pipeline/step5_school_effects.py

Added after the locked RQ1-RQ4 run, to examine where leave-one-school-out validation (RQ2)
lost specificity. It uses the locked SVM settings and the locked predictions and proposes no
new model. Output: <work>/aq_10/step5_school_effects/ (tables/ and report_step5.md),
logs/step5_done.flag, and two figures in <work>/aq_10/figures/: item 1 and school effects, and the
influence of each training school. Only school codes S01-S12 are written; no school names and no
child IDs. Takes about 5-10 minutes.

A. The locked leave-one-school-out decisions reproduced in this environment
B. Do schools leave a fingerprint in the answers? School profiles, answer-by-school association,
   between-school variance (ICC), rater response style, school identifiability, drift check
C. Seen versus unseen schools: random child-level CV, local adaptation, leave-one-pair-out
D. Mechanism of the S06 false positives: item 1 counterfactual, item response curves, training
   composition, training-school influence, each item removed, S06 profile, who the SVM adds
"""
import os
import sys
import time
import warnings

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 - PATHS, DATA AND HELPERS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("STEP 5 - SCHOOL EFFECTS")
print("=" * 78)
BASE = os.environ.get("AQ10_BASE", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work"))
PROJECT = os.path.join(BASE, "aq_10")
OUT = os.path.join(PROJECT, "step5_school_effects")
TAB, FIG = os.path.join(OUT, "tables"), os.path.join(PROJECT, "figures")
RQ4_FLAG = os.path.join(PROJECT, "logs", "rq4_done.flag")
if not os.path.exists(RQ4_FLAG):
    sys.exit(f"Run rq1_baseline.py to rq4_explanation.py first (missing {RQ4_FLAG}).")
for d in (OUT, TAB, FIG):
    os.makedirs(d, exist_ok=True)
T_START = time.time()

bd = pd.read_csv(os.path.join(PROJECT, "data", "processed", "bd_clean.csv"))
bd = bd.drop(columns=["school_name"], errors="ignore")
assert len(bd) == 843 and bd.ID.is_unique
AQ = [f"A_Q{i}" for i in range(1, 11)]        # aligned values 0-3 (the model's ten features)
RAW = [f"Q{i}" for i in range(1, 11)]         # raw codes 1-4 (1 = definitely agree, 4 = definitely disagree)
bd["code"] = "S" + bd.school_id.str[-2:]      # SCH_006 -> S06
bd["sex"] = bd.Gender.astype(str).str.strip().str.lower()
bd.loc[bd.ID == "a362", "sex"] = "f"          # blank in the original entry; the paper form shows female
bd["total"] = bd.AQ_score_legacy.astype(int)  # AQ-10 score 0-10
bd["rule"] = (bd.total >= 6).astype(int)
bd["ordsum"] = bd[AQ].sum(axis=1)             # unweighted sum of aligned values 0-30
bd["ERS"] = bd[RAW].isin([1, 4]).mean(axis=1) # extreme response style: share of "definitely" answers
assert (bd.rule == bd.AQ_rule_pred).all()

lock = pd.read_csv(os.path.join(PROJECT, "rq2_loso", "predictions", "rq2_loso_s0_predictions.csv"))
lock = lock[["ID", "y_prob", "y_pred"]].rename(columns={"y_prob": "p_lock", "y_pred": "f_lock"})
bd = bd.merge(lock, on="ID", how="left")      # a left merge keeps the bd_clean row order
assert len(bd) == 843 and bd.p_lock.notna().all()
bd["svm_logit"] = np.log(np.clip(bd.p_lock, 1e-6, 1 - 1e-6) / (1 - np.clip(bd.p_lock, 1e-6, 1 - 1e-6)))

test = pd.read_csv(os.path.join(PROJECT, "rq1_baseline", "predictions", "test_predictions.csv"))
assert len(test) == 169

y = bd.label.values.astype(int)
XA = bd[AQ].values.astype(float)
SCHOOLS = sorted(bd.school_id.unique())
SP = sorted(bd.loc[bd.label == 1, "school_id"].unique())
MN = sorted(bd.loc[bd.label == 0, "school_id"].unique())
CODE = dict(zip(bd.school_id, bd.code))
SID = {c: s for s, c in CODE.items()}
IDX = {s: np.where(bd.school_id.values == s)[0] for s in SCHOOLS}
ERSV = bd.ERS.values
LABELS = {**{a: a.replace("A_", "") for a in AQ}, "Age": "Age", "total": "AQ-10 total", "ordsum": "Ordinal sum",
          "ERS": "Extreme responding", "svm_logit": "SVM output (logit)", "f_lock": "SVM flag", "rule": "Rule flag"}
S06 = (bd.school_id == "SCH_006").values
OTHER_MN = (y == 0) & ~S06
ALL = np.ones(843, bool)
LOCKED = dict(C=1.0, kernel="rbf", gamma="scale")
Q1_ANSWER = {3: "definitely agree", 2: "slightly agree", 1: "slightly disagree", 0: "definitely disagree"}


def svm(C=1.0, kernel="rbf", gamma="scale", probability=True):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(C=C, kernel=kernel, gamma=gamma, probability=probability, random_state=SEED))])


def fit_predict(train_mask, test_mask, cols=AQ, params=LOCKED, return_model=False):
    m = svm(**params).fit(bd.loc[train_mask, cols].values, y[train_mask])
    p = m.predict_proba(bd.loc[test_mask, cols].values)[:, 1]
    return (p, m) if return_model else p


def loso(cols=AQ, params=LOCKED, return_models=False):
    """Leave-one-school-out exactly as in RQ2: schools in sorted order, rows in bd_clean order."""
    prob, models = np.full(843, np.nan), {}
    for s in SCHOOLS:
        held = bd.school_id.values == s
        p, m = fit_predict(~held, held, cols, params, return_model=True)
        prob[held] = p
        models[s] = m
    return (prob, models) if return_models else prob


def rates(yt, yp):
    yt, yp = np.asarray(yt).astype(int), np.asarray(yp).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum()); fn = int(((yp == 0) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum()); tn = int(((yp == 0) & (yt == 0)).sum())
    return {"TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "Sens": tp / (tp + fn) if tp + fn else np.nan,
            "Spec": tn / (tn + fp) if tn + fp else np.nan}


def auc(yt, s):
    yt = np.asarray(yt)
    return roc_auc_score(yt, s) if 0 < yt.sum() < len(yt) else np.nan


def cluster_boot(stat, idx=IDX, sp=SP, mn=MN, B=2000, seed=SEED):
    """95% percentile interval; ASD and mainstream schools resampled separately, with replacement."""
    rng_b = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        ss = list(rng_b.choice(sp, len(sp))) + list(rng_b.choice(mn, len(mn)))
        vals.append(stat(np.concatenate([idx[s] for s in ss])))
    return np.nanpercentile(np.asarray(vals, float), [2.5, 97.5])


def icc1(v, g):
    """ICC(1) from one-way ANOVA; g = integer school codes 0..k-1."""
    v = np.asarray(v, float); g = np.asarray(g)
    k, N = g.max() + 1, len(v)
    n = np.bincount(g, minlength=k).astype(float)
    m = np.bincount(g, v, minlength=k) / n
    msb = (n * (m - v.mean()) ** 2).sum() / (k - 1)
    msw = ((v - m[g]) ** 2).sum() / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    den = msb + (n0 - 1) * msw
    return (msb - msw) / den if den > 0 else np.nan


def holm(p):
    p = np.asarray(p, float); adj = np.empty(len(p)); run = 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj


def f3(x, d=3):
    return "n/a" if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) else f"{x:.{d}f}"


def ci(lo_hi, d=3):
    return f"[{f3(lo_hi[0], d)}, {f3(lo_hi[1], d)}]"


def pfmt(p):
    return "n/a" if p is None or np.isnan(p) else (f"{p:.2g}" if p < 0.001 else f"{p:.3f}")


def save(df, name):
    df.to_csv(os.path.join(TAB, name), index=False)
    print("  [saved]", name)


REPORT = ["# Step 5 - School effects", "",
          f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}; seed {SEED}. "
          "Added after the locked RQ1-RQ4 run; uses the locked SVM settings and predictions.", ""]


def write_report(section_lines):
    """Append a section, print it and rewrite report_step5.md (so finished sections survive a crash)."""
    REPORT.extend(section_lines + [""])
    with open(os.path.join(OUT, "report_step5.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT) + "\n")
    print("\n".join(section_lines))


print(f"Project: {PROJECT}")
print(f"{len(bd)} children | {len(SP)} ASD and {len(MN)} mainstream schools | test set {len(test)}")

# ══════════════════════════════════════════════════════════════════════════════
# A. REPRODUCE THE LOCKED LEAVE-ONE-SCHOOL-OUT RUN
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
p_ref, M_ref = loso(return_models=True)
f_ref = (p_ref >= 0.5).astype(int)
agree = int((f_ref == bd.f_lock.values).sum())
changed = sorted(bd.code[f_ref != bd.f_lock.values].unique())
write_report(["## A. Reproducing the locked leave-one-school-out run", "",
              f"- The refit reproduced {agree} of 843 locked decisions "
              f"(largest probability difference {np.abs(p_ref - bd.p_lock.values).max():.4f}; "
              f"schools with a changed decision: {', '.join(changed) if changed else 'none'}).",
              "- Pooled numbers use the locked predictions; analyses that refit the model are compared with this refit."])

# ══════════════════════════════════════════════════════════════════════════════
# B. DO SCHOOLS LEAVE A FINGERPRINT IN THE ANSWERS?
# ══════════════════════════════════════════════════════════════════════════════
rng = np.random.default_rng(SEED)
R = ["## B. Do schools leave a fingerprint in the answers?", "",
     "### B1. School profiles (T_s5_B1_school_profile.csv, T_s5_B1_item_endorsement_by_school.csv)", "",
     "| School | Group | n | Age | Boys % | AQ total | Rule + % | SVM flagged % | ERS | Q1 definitely disagree % |",
     "|---|---|---|---|---|---|---|---|---|---|"]
prof = []
for s in SP + MN:
    g = bd[bd.school_id == s]
    q1 = g.Q1.value_counts(normalize=True)
    prof.append({"school": CODE[s], "group": "ASD" if s in SP else "mainstream", "n": len(g),
                 "age_mean": g.Age.mean(), "boys_pct": 100 * (g.sex == "m").mean(),
                 "AQ_total_mean": g.total.mean(), "AQ_total_sd": g.total.std(),
                 "rule_positive_pct": 100 * g.rule.mean(), "svm_flagged_pct": 100 * g.f_lock.mean(),
                 "ERS_mean": g.ERS.mean(),
                 "Q1_def_agree_pct": 100 * q1.get(1, 0), "Q1_sl_agree_pct": 100 * q1.get(2, 0),
                 "Q1_sl_disagree_pct": 100 * q1.get(3, 0), "Q1_def_disagree_pct": 100 * q1.get(4, 0)})
prof = pd.DataFrame(prof)
save(prof.round(3), "T_s5_B1_school_profile.csv")
R += [f"| {r.school} | {r.group} | {r.n} | {r.age_mean:.1f} | {r.boys_pct:.0f} | {r.AQ_total_mean:.2f} | "
      f"{r.rule_positive_pct:.1f} | {r.svm_flagged_pct:.1f} | {r.ERS_mean:.2f} | {r.Q1_def_disagree_pct:.1f} |"
      for r in prof.itertuples()]

order = [CODE[s] for s in SP + MN]
endorse = (bd[AQ] >= 2).groupby(bd.code).mean().loc[order]
endorse.columns = RAW
save((100 * endorse).round(1).reset_index().rename(columns={"code": "school"}), "T_s5_B1_item_endorsement_by_school.csv")


def cramers_v(a, b):
    a = np.asarray(a, int); b = np.asarray(b, int)
    na, nb = a.max() + 1, b.max() + 1
    tab = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(float)
    tab = tab[tab.sum(1) > 0][:, tab.sum(0) > 0]
    if min(tab.shape) < 2:
        return np.nan
    e = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    return float(np.sqrt(((tab - e) ** 2 / e).sum() / (tab.sum() * (min(tab.shape) - 1))))


def perm_v(sub, item, n_perm=2000):
    a = pd.factorize(sub.school_id)[0]; b = sub[item].values.astype(int) - 1
    obs = cramers_v(a, b)
    null = np.array([cramers_v(rng.permutation(a), b) for _ in range(n_perm)])
    return obs, (1 + (null >= obs - 1e-12).sum()) / (n_perm + 1)


SETS = {"ASD": bd[bd.label == 1], "mainstream": bd[bd.label == 0], "mainstream_without_S06": bd[(bd.label == 0) & ~S06]}
rows = []
for it in RAW:
    row = {"item": it, "V_answer_by_group": cramers_v(bd.label.values, bd[it].values.astype(int) - 1)}
    for nm, sub in SETS.items():
        row[f"V_{nm}"], row[f"p_{nm}"] = perm_v(sub, it)
    rows.append(row)
B2 = pd.DataFrame(rows)
for nm in SETS:
    B2[f"p_holm_{nm}"] = holm(B2[f"p_{nm}"].values)
save(B2.round(4), "T_s5_B2_item_by_school.csv")
R += ["", "### B2. Does each answer depend on the school within each group? (T_s5_B2_item_by_school.csv)", "",
      "Cramer's V for answer (4 levels) by school within a group; p from 2,000 shuffles of school within the group, "
      "Holm-adjusted over the ten items. V for answer by group (ASD vs mainstream) is shown for comparison.", "",
      "| Item | V by group | V school (ASD) | p Holm | V school (mainstream) | p Holm | V school (mainstream without S06) | p Holm |",
      "|---|---|---|---|---|---|---|---|"]
R += [f"| {r.item} | {r.V_answer_by_group:.2f} | {r.V_ASD:.2f} | {pfmt(r.p_holm_ASD)} | {r.V_mainstream:.2f} | "
      f"{pfmt(r.p_holm_mainstream)} | {r.V_mainstream_without_S06:.2f} | {pfmt(r.p_holm_mainstream_without_S06)} |"
      for r in B2.itertuples()]


def perm_icc(sub, var, n_perm=2000):
    g = pd.factorize(sub.school_id)[0]; v = sub[var].values.astype(float)
    obs = icc1(v, g)
    if np.isnan(obs):
        return obs, np.nan
    null = np.array([icc1(v, rng.permutation(g)) for _ in range(n_perm)])
    return obs, (1 + (null >= obs - 1e-12).sum()) / (n_perm + 1)


VARS = AQ + ["total", "ordsum", "ERS", "svm_logit", "f_lock", "rule"]
rows = []
for v in VARS:
    row = {"variable": LABELS[v]}
    for nm, sub in SETS.items():
        row[f"ICC_{nm}"], row[f"p_{nm}"] = perm_icc(sub, v)
    rows.append(row)
B3 = pd.DataFrame(rows)
save(B3.round(4), "T_s5_B3_icc_by_school.csv")
R += ["", "### B3. Share of variance between schools, within each group (ICC(1); T_s5_B3_icc_by_school.csv)", "",
      "| Variable | ICC ASD | p | ICC mainstream | p | ICC mainstream without S06 | p |", "|---|---|---|---|---|---|---|"]
R += [f"| {r.variable} | {f3(r.ICC_ASD)} | {pfmt(r.p_ASD)} | {f3(r.ICC_mainstream)} | {pfmt(r.p_mainstream)} | "
      f"{f3(r.ICC_mainstream_without_S06)} | {pfmt(r.p_mainstream_without_S06)} |" for r in B3.itertuples()]
fp_mask = (y == 0) & (bd.f_lock.values == 1); tn_mask = (y == 0) & (bd.f_lock.values == 0)
miss = (y == 1) & (bd.rule.values == 0)
svm_only = miss & (bd.f_lock.values == 1); neither = miss & (bd.f_lock.values == 0)
E = ERSV
auc_e = auc(y, E); ci_e = cluster_boot(lambda ii: auc(y[ii], E[ii]))
R += ["", "### B4. Rater response style: share of 'definitely' answers (ERS)", "",
      f"- ERS mean: ASD {E[y == 1].mean():.3f}, mainstream {E[y == 0].mean():.3f} "
      f"(Mann-Whitney p = {pfmt(stats.mannwhitneyu(E[y == 1], E[y == 0]).pvalue)}).",
      f"- ERS alone as a score separates the groups with AUROC {auc_e:.3f} {ci(ci_e)} (school-clustered interval).",
      f"- Mainstream children: false positives {E[fp_mask].mean():.3f} (n = {fp_mask.sum()}) vs true negatives {E[tn_mask].mean():.3f} "
      f"(p = {pfmt(stats.mannwhitneyu(E[fp_mask], E[tn_mask]).pvalue)}); S06 {E[S06].mean():.3f} vs other mainstream {E[OTHER_MN].mean():.3f} "
      f"(p = {pfmt(stats.mannwhitneyu(E[S06], E[OTHER_MN]).pvalue)}).",
      f"- ASD-school children scoring below 6: found by the SVM only {E[svm_only].mean():.3f} (n = {svm_only.sum()}) vs missed by both "
      f"{E[neither].mean():.3f} (n = {neither.sum()}; p = {pfmt(stats.mannwhitneyu(E[svm_only], E[neither]).pvalue)})."]


def school_cv(X, lab, repeats, seed):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    pred = np.zeros((repeats, len(lab)), int)
    for i, (tr, te) in enumerate(rskf.split(X, lab)):
        pred[i // 5, te] = svm(probability=False).fit(X[tr], lab[tr]).predict(X[te])
    K = lab.max() + 1
    bal = float(np.mean([balanced_accuracy_score(lab, pr) for pr in pred]))
    rec = np.mean([confusion_matrix(lab, pr, labels=list(range(K))).diagonal() / np.bincount(lab, minlength=K) for pr in pred], axis=0)
    return bal, rec


rows = []
for gname, gmask in (("ASD", y == 1), ("mainstream", y == 0)):
    sub = bd[gmask]
    lab, uniq = pd.factorize(sub.school_id)
    for fname, cols in (("10 items", AQ), ("9 items without Q1", AQ[1:])):
        X = sub[cols].values.astype(float)
        bal, rec = school_cv(X, lab, 5, SEED)
        if fname == "10 items":
            null = np.array([school_cv(X, rng.permutation(lab), 1, SEED + k)[0] for k in range(200)])
            p = (1 + (null >= bal).sum()) / 201
        else:
            p = np.nan
        row = {"group": gname, "features": fname, "balanced_accuracy": bal, "chance": 1 / len(uniq), "perm_p": p}
        row.update({f"recall_{CODE[s]}": r for s, r in zip(uniq, rec)})
        rows.append(row)
B5 = pd.DataFrame(rows)
save(B5.round(4), "T_s5_B5_school_identifiability.csv")
R += ["", "### B5. Can the ten answers identify the school within each group? (T_s5_B5_school_identifiability.csv)", "",
      "Same RBF SVM settings, multiclass, 5-fold CV repeated 5 times; p from 200 shuffles of school labels.", ""]
for r in B5.itertuples():
    recs = ", ".join(f"{c.replace('recall_', '')} {getattr(r, c):.2f}" for c in B5.columns
                     if c.startswith("recall_") and not pd.isna(getattr(r, c)))
    R += [f"- {r.group}, {r.features}: balanced accuracy {r.balanced_accuracy:.3f} (chance {r.chance:.3f}; p = {pfmt(r.perm_p)}). "
          f"Recall by school: {recs}."]

rows = []
for s in SP + MN:
    sub = bd[bd.label.values == (1 if s in SP else 0)]
    is_s = (sub.school_id == s).values
    for it in RAW:
        v = sub[it].values.astype(int) - 1
        a_c, b_c = np.bincount(v[is_s], minlength=4), np.bincount(v[~is_s], minlength=4)
        keep = (a_c + b_c) > 0
        tvd = 0.5 * np.abs(a_c / a_c.sum() - b_c / b_c.sum()).sum()
        p = stats.chi2_contingency(np.vstack([a_c[keep], b_c[keep]]))[1] if keep.sum() > 1 else 1.0
        rows.append({"school": CODE[s], "group": "ASD" if s in SP else "mainstream", "item": it, "TVD": tvd, "chi2_p": p})
B6 = pd.DataFrame(rows)
B6["p_bonferroni"] = np.minimum(1, B6.chi2_p * len(B6))
save(B6.round(5), "T_s5_B6_item_drift_by_school.csv")
top = B6.sort_values("TVD", ascending=False).head(10)
R += ["", "### B6. Answer-distribution check per school and item (school vs the other schools of its group; T_s5_B6_item_drift_by_school.csv)", "",
      f"- {int((B6.p_bonferroni < 0.05).sum())} of {len(B6)} school-item pairs differ after Bonferroni correction. Largest total variation distances:"]
R += [f"  - {r.school} ({r.group}) {r.item}: TVD {r.TVD:.2f}, Bonferroni p = {pfmt(r.p_bonferroni)}" for r in top.itertuples()]
write_report(R + ["", f"(A and B took {time.time() - t0:.0f} s)"])

# ══════════════════════════════════════════════════════════════════════════════
# C. SEEN VERSUS UNSEEN SCHOOLS
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
rng = np.random.default_rng(SEED + 1)
REP = 10
P = np.full((REP, 843), np.nan)
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=REP, random_state=SEED)
for i, (tr, te) in enumerate(rskf.split(XA, y)):
    P[i // 5, te] = svm().fit(XA[tr], y[tr]).predict_proba(XA[te])[:, 1]
cvs = pd.DataFrame([{**rates(y, P[r] >= .5), "AUROC": auc(y, P[r])} for r in range(REP)])
lo = {**rates(y, bd.f_lock.values), "AUROC": auc(y, bd.p_lock.values)}
ho = {**rates(test.label.values, test.pred_Mstar.values), "AUROC": auc(test.label.values, test.prob_Mstar.values)}
rows = []
for s in SP + MN:
    ii = IDX[s]; tt = (test.school_id == s).values
    rows.append({"school": CODE[s], "group": "ASD" if s in SP else "mainstream", "n": len(ii),
                 "flagged_pct_LOSO_unseen": 100 * bd.f_lock.values[ii].mean(),
                 "flagged_pct_randomCV_seen": 100 * (P[:, ii] >= .5).mean(),
                 "holdout_n": int(tt.sum()), "holdout_flagged": int(test.pred_Mstar.values[tt].sum())})
C1 = pd.DataFrame(rows)
save(C1.round(2), "T_s5_C1_seen_vs_unseen_by_school.csv")
save(cvs.round(4), "T_s5_C1_random_cv_repeats.csv")
rng_txt = lambda k: f"{cvs[k].mean():.3f} [{cvs[k].min():.3f} to {cvs[k].max():.3f}]"
R = ["## C. Seen versus unseen schools", "",
     "### C1. Random child-level CV (schools seen) vs leave-one-school-out (school unseen)", "",
     "| Validation | AUROC | Sensitivity | Specificity |", "|---|---|---|---|",
     f"| Leave-one-school-out (locked) | {lo['AUROC']:.3f} | {lo['Sens']:.3f} | {lo['Spec']:.3f} |",
     f"| Random 5-fold CV, 10 repeats: mean [range] | {rng_txt('AUROC')} | {rng_txt('Sens')} | {rng_txt('Spec')} |",
     f"| Hold-out test set (schools seen in development) | {ho['AUROC']:.3f} | {ho['Sens']:.3f} | {ho['Spec']:.3f} |", "",
     "| School | Group | n | Flagged %, unseen (LOSO) | Flagged %, seen (random CV) | Hold-out flagged / n |", "|---|---|---|---|---|---|"]
R += [f"| {r.school} | {r.group} | {r.n} | {r.flagged_pct_LOSO_unseen:.1f} | {r.flagged_pct_randomCV_seen:.1f} | {r.holdout_flagged} / {r.holdout_n} |"
      for r in C1.itertuples()]

KS, REPS = (5, 10, 20), 10
rows = []
for s in SCHOOLS:
    ids = IDX[s]
    for k in KS:
        for r in range(REPS):
            add = rng.choice(ids, k, replace=False); rest = np.setdiff1d(ids, add)
            train = ~(bd.school_id.values == s); train[add] = True
            test_m = np.zeros(843, bool); test_m[rest] = True
            p = fit_predict(train, test_m)
            rows.append({"school": CODE[s], "group": "ASD" if s in SP else "mainstream", "k": k, "rep": r,
                         "n_rest": len(rest), "n_flagged": int((p >= .5).sum())})
C2 = pd.DataFrame(rows)
C2["flagged_pct"] = 100 * C2.n_flagged / C2.n_rest
save(C2, "T_s5_C2_local_adaptation_runs.csv")
by_school = C2.groupby(["school", "group", "k"]).flagged_pct.mean().unstack("k").reset_index()
by_school.insert(2, "k0_LOSO", [100 * bd.f_lock.values[IDX[SID[c]]].mean() for c in by_school.school])
save(by_school.round(2), "T_s5_C2_local_adaptation_by_school.csv")
pool = C2.groupby(["k", "rep", "group"])[["n_flagged", "n_rest"]].sum().reset_index()
pool["rate"] = pool.n_flagged / pool.n_rest
summ = pool.groupby(["k", "group"]).rate.agg(["mean", "min", "max"]).reset_index()
R += ["", "### C2. Local adaptation: k children of the held-out school added to training (10 random draws each)", "",
      "| k | Sensitivity, mean [range] | Specificity, mean [range] | S06 flagged %, mean |", "|---|---|---|---|",
      f"| 0 (LOSO) | {lo['Sens']:.3f} | {lo['Spec']:.3f} | {100 * bd.f_lock.values[S06].mean():.1f} |"]
for k in KS:
    se = summ[(summ.k == k) & (summ.group == "ASD")].iloc[0]; mn_ = summ[(summ.k == k) & (summ.group == "mainstream")].iloc[0]
    s06k = C2[(C2.school == "S06") & (C2.k == k)].flagged_pct.mean()
    R += [f"| {k} | {se['mean']:.3f} [{se['min']:.3f} to {se['max']:.3f}] | {1 - mn_['mean']:.3f} [{1 - mn_['max']:.3f} to {1 - mn_['min']:.3f}] | {s06k:.1f} |"]

rows = []
for a in SP:
    for b in MN:
        held = bd.school_id.isin([a, b]).values
        p = fit_predict(~held, held)
        yy, tt = y[held], bd.total.values[held]
        rs, rr = rates(yy, p >= .5), rates(yy, tt >= 6)
        rows.append({"ASD": CODE[a], "mainstream": CODE[b], "n": int(held.sum()),
                     "AUROC_svm": auc(yy, p), "AUROC_total": auc(yy, tt),
                     "Sens_svm": rs["Sens"], "Spec_svm": rs["Spec"], "Sens_rule": rr["Sens"], "Spec_rule": rr["Spec"]})
C3 = pd.DataFrame(rows)
save(C3.round(4), "T_s5_C3_leave_pair_out.csv")
q = lambda v: f"{np.median(v):.3f} [IQR {np.percentile(v, 25):.3f} to {np.percentile(v, 75):.3f}; range {np.min(v):.3f} to {np.max(v):.3f}]"
w06 = C3.mainstream == "S06"
R += ["", "### C3. Leave one ASD and one mainstream school out together (35 test folds with both groups)", "",
      f"- AUROC SVM: {q(C3.AUROC_svm)}; AUROC total score: {q(C3.AUROC_total)}; SVM higher in {(C3.AUROC_svm > C3.AUROC_total).sum()} of 35.",
      f"- Sensitivity SVM: {q(C3.Sens_svm)}; rule: {q(C3.Sens_rule)}; SVM higher in {(C3.Sens_svm > C3.Sens_rule).sum()} of 35.",
      f"- Specificity SVM: {q(C3.Spec_svm)}; rule: {q(C3.Spec_rule)}; SVM lower in {(C3.Spec_svm < C3.Spec_rule).sum()} of 35.",
      f"- Folds containing S06: SVM specificity {q(C3.Spec_svm[w06])}; other folds {q(C3.Spec_svm[~w06])}."]
write_report(R + ["", f"(C took {time.time() - t0:.0f} s)"])

# ══════════════════════════════════════════════════════════════════════════════
# D. MECHANISM OF THE S06 FALSE POSITIVES
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
f_ref = p_ref >= .5
flag06 = S06 & f_ref
R = ["## D. Mechanism of the S06 false positives", "",
     f"- In this refit the model flagged {int(flag06.sum())} of {int(S06.sum())} children in S06 (locked run {int(bd.f_lock.values[S06].sum())}).", ""]


def predict_with(j, value):
    """Probability for every child from its own LOSO model, with aligned item j set to value."""
    out = np.empty(843)
    for s in SCHOOLS:
        X = XA[IDX[s]].copy(); X[:, j] = value
        out[IDX[s]] = M_ref[s].predict_proba(X)[:, 1]
    return out


rows = [{"item1_answer": "as given", "flagged_of_the_S06_flagged": int(flag06.sum()), "S06_flagged": int(f_ref[S06].sum()),
         "other_mainstream_flagged_pct": 100 * f_ref[OTHER_MN].mean(), "ASD_flagged_pct": 100 * f_ref[y == 1].mean()}]
for v in (3, 2, 1, 0):
    fv = predict_with(0, v) >= .5
    rows.append({"item1_answer": Q1_ANSWER[v], "flagged_of_the_S06_flagged": int((fv & flag06).sum()), "S06_flagged": int(fv[S06].sum()),
                 "other_mainstream_flagged_pct": 100 * fv[OTHER_MN].mean(), "ASD_flagged_pct": 100 * fv[y == 1].mean()})
D1 = pd.DataFrame(rows)
save(D1.round(2), "T_s5_D1_item1_counterfactual.csv")
R += ["### D1. Counterfactual: every child's answer to item 1 replaced, all other answers kept", "",
      f"| Item 1 answer | Still flagged, of the {int(flag06.sum())} S06 children | S06 flagged, of 63 | Other mainstream flagged % | ASD flagged % |",
      "|---|---|---|---|---|"]
R += [f"| {r.item1_answer} | {r.flagged_of_the_S06_flagged} | {r.S06_flagged} | {r.other_mainstream_flagged_pct:.1f} | {r.ASD_flagged_pct:.1f} |"
      for r in D1.itertuples()]

rows, ice = [], []
for j, a in enumerate(AQ):
    probs = np.vstack([predict_with(j, v) for v in range(4)])
    for gname, gmask in (("ASD", y == 1), ("mainstream", y == 0), ("all", ALL)):
        curve = probs[:, gmask].mean(1)
        rows.append({"item": RAW[j], "group": gname, **{f"mean_p_aligned_{v}": curve[v] for v in range(4)},
                     "monotone_increasing": bool(np.all(np.diff(curve) >= -0.005))})
    dec = (np.diff(probs, axis=0) < -0.01).any(axis=0)
    ice.append({"item": RAW[j], "pct_children_with_a_decrease_all": 100 * dec.mean(),
                "pct_ASD": 100 * dec[y == 1].mean(), "pct_mainstream": 100 * dec[y == 0].mean()})
D2 = pd.DataFrame(rows); D2i = pd.DataFrame(ice)
save(D2.round(4), "T_s5_D2_partial_dependence.csv"); save(D2i.round(2), "T_s5_D2_nonmonotone_children.csv")
nonmono = D2[(D2.group == "all") & ~D2.monotone_increasing].item.tolist()
R += ["", "### D2. Does a higher trait value on each item always raise the prediction? (T_s5_D2_*.csv)", "",
      f"- Items whose average effect is not monotone (all children): {', '.join(nonmono) if nonmono else 'none'}.",
      "| Item | Mean p at 0 / 1 / 2 / 3 (all children) | Children with any decrease % |", "|---|---|---|"]
for it in RAW:
    r = D2[(D2.item == it) & (D2.group == "all")].iloc[0]; ri = D2i[D2i.item == it].iloc[0]
    R += [f"| {it} | {r.mean_p_aligned_0:.3f} / {r.mean_p_aligned_1:.3f} / {r.mean_p_aligned_2:.3f} / {r.mean_p_aligned_3:.3f} | "
          f"{ri.pct_children_with_a_decrease_all:.1f} |"]

dd = XA[:, 0] == 0
rows = []
for h in SCHOOLS:
    tr = bd.school_id.values != h
    n_sp, n_mn = int((tr & dd & (y == 1)).sum()), int((tr & dd & (y == 0)).sum())
    rows.append({"held_out": CODE[h], "train_Q1_def_disagree_ASD": n_sp, "train_Q1_def_disagree_mainstream": n_mn,
                 "P_ASD_given_Q1_def_disagree": n_sp / (n_sp + n_mn) if n_sp + n_mn else np.nan,
                 "held_out_Q1_def_disagree": int((~tr & dd).sum())})
D3 = pd.DataFrame(rows)
save(D3.round(4), "T_s5_D3_training_composition.csv")
r06 = D3[D3.held_out == "S06"].iloc[0]
by_sp = {CODE[s]: int((dd & (bd.school_id.values == s)).sum()) for s in SP}
R += ["", "### D3. What the training data taught about item 1 (T_s5_D3_training_composition.csv)", "",
      f"- When S06 was held out, 'definitely disagree' to item 1 appeared for {r06.train_Q1_def_disagree_ASD} ASD-school and "
      f"{r06.train_Q1_def_disagree_mainstream} mainstream children in training, so P(ASD | that answer) = {r06.P_ASD_given_Q1_def_disagree:.3f}.",
      f"- 'Definitely disagree' to item 1 by ASD school: {by_sp}."]

base = {s: f_ref[IDX[s]].mean() for s in SCHOOLS}
codes = [CODE[s] for s in SCHOOLS]
infl = pd.DataFrame(np.nan, index=codes, columns=codes)      # change in flag rate, percentage points
infl_n = pd.DataFrame(np.nan, index=codes, columns=codes)    # children flagged
for h in SCHOOLS:
    held = bd.school_id.values == h
    for e in SCHOOLS:
        if e != h:
            p = fit_predict(~held & (bd.school_id.values != e), held)
            infl.loc[CODE[h], CODE[e]] = 100 * ((p >= .5).mean() - base[h])
            infl_n.loc[CODE[h], CODE[e]] = int((p >= .5).sum())
infl.index.name = infl_n.index.name = "held_out_school"
save(infl.round(2).reset_index(), "T_s5_D4_training_school_influence_pp.csv")
save(infl_n.reset_index(), "T_s5_D4_training_school_influence_counts.csv")
top3 = sorted(SP, key=lambda s: -dd[IDX[s]].mean())[:3]      # ASD schools with the highest share of 'definitely disagree'
p_wo = fit_predict(~S06 & ~bd.school_id.isin(top3).values, S06)
R += ["", "### D4. Which training school taught the pattern? (one training school removed at a time; T_s5_D4_*.csv)", "",
      "Entries are the change in the held-out school's flag rate, in percentage points, when one training school is removed.", "",
      f"- S06 (baseline {int(f_ref[S06].sum())} of 63 flagged): " +
      ", ".join(f"without {c} {int(v)}" for c, v in infl_n.loc["S06"].dropna().items()) + ".",
      f"- Without the three ASD schools with the most 'definitely disagree' answers to item 1 ({', '.join(CODE[s] for s in top3)}): "
      f"{int((p_wo >= .5).sum())} of 63 flagged in S06.",
      "- Largest changes anywhere (held-out school, removed school, change in pp):"]
stack = infl.stack().sort_values(key=np.abs, ascending=False).head(8)
R += [f"  - {h}, without {e}: {v:+.1f}" for (h, e), v in stack.items()]
base_auc = auc(y, p_ref)
rows = [{"item_dropped": "none", "AUROC": base_auc, "dAUROC": 0.0, "dAUROC_lo": 0.0, "dAUROC_hi": 0.0,
         **{k: v for k, v in rates(y, f_ref).items() if k in ("Sens", "Spec")},
         "flagged_S06": int(f_ref[S06].sum()), "flagged_S07": int(f_ref[bd.school_id.values == "SCH_007"].sum()),
         "FP_other_mainstream": int(f_ref[OTHER_MN].sum())}]
for j, a in enumerate(AQ):
    p = loso([c for c in AQ if c != a]); f = p >= .5
    lohi = cluster_boot(lambda ii, p=p: auc(y[ii], p[ii]) - auc(y[ii], p_ref[ii]))
    rows.append({"item_dropped": RAW[j], "AUROC": auc(y, p), "dAUROC": auc(y, p) - base_auc, "dAUROC_lo": lohi[0], "dAUROC_hi": lohi[1],
                 **{k: v for k, v in rates(y, f).items() if k in ("Sens", "Spec")},
                 "flagged_S06": int(f[S06].sum()), "flagged_S07": int(f[bd.school_id.values == "SCH_007"].sum()),
                 "FP_other_mainstream": int(f[OTHER_MN].sum())})
D5 = pd.DataFrame(rows)
save(D5.round(4), "T_s5_D5_drop_one_item.csv")
R += ["", "### D5. Each item removed in turn (LOSO refit; T_s5_D5_drop_one_item.csv)", "",
      "| Item removed | AUROC | Change [school-clustered 95% interval] | Sens | Spec | S06 flagged | S07 flagged | FP other mainstream |",
      "|---|---|---|---|---|---|---|---|"]
R += [f"| {r.item_dropped} | {r.AUROC:.3f} | {r.dAUROC:+.3f} [{r.dAUROC_lo:+.3f}, {r.dAUROC_hi:+.3f}] | {r.Sens:.3f} | {r.Spec:.3f} | "
      f"{r.flagged_S06} | {r.flagged_S07} | {r.FP_other_mainstream} |" for r in D5.itertuples()]

oth = bd[OTHER_MN]; s6 = bd[S06]
rows = []
for var in ["Age", "total", "ordsum", "ERS"] + AQ:
    U, p = stats.mannwhitneyu(s6[var], oth[var], alternative="two-sided")
    rows.append({"variable": LABELS.get(var, var), "S06_mean": s6[var].mean(), "other_mainstream_mean": oth[var].mean(),
                 "rank_biserial": 2 * U / (len(s6) * len(oth)) - 1, "p": p})
D6 = pd.DataFrame(rows); D6["p_holm"] = holm(D6.p.values)
boys = stats.fisher_exact([[int((s6.sex == "m").sum()), int((s6.sex != "m").sum())], [int((oth.sex == "m").sum()), int((oth.sex != "m").sum())]])[1]
save(D6.round(4), "T_s5_D6_S06_vs_other_mainstream.csv")
R += ["", "### D6. S06 compared with the other mainstream schools (T_s5_D6_*.csv)", "",
      f"- Boys: S06 {100 * (s6.sex == 'm').mean():.0f}% vs {100 * (oth.sex == 'm').mean():.0f}% (Fisher p = {pfmt(boys)}).",
      "| Variable | S06 mean | Other mainstream mean | Rank-biserial r | p (Holm) |", "|---|---|---|---|---|"]
R += [f"| {r.variable} | {r.S06_mean:.3f} | {r.other_mainstream_mean:.3f} | {r.rank_biserial:+.2f} | {pfmt(r.p_holm)} |" for r in D6.itertuples()]

fl, ru = bd.f_lock.values == 1, bd.rule.values == 1
cats = {"both flag": fl & ru, "SVM only": fl & ~ru, "rule only": ~fl & ru, "neither": ~fl & ~ru}
rows = []
for gname, gmask in (("ASD", y == 1), ("mainstream", y == 0)):
    for cname, cm in cats.items():
        m = gmask & cm
        rows.append({"group": gname, "category": cname, "n": int(m.sum()), "ERS_mean": ERSV[m].mean() if m.any() else np.nan,
                     **{f"total_{t}": int((m & (bd.total.values == t)).sum()) for t in range(11)}})
D7 = pd.DataFrame(rows)
save(D7.round(3), "T_s5_D7_who_the_svm_adds.csv")
R += ["", "### D7. AQ-10 totals of children by decision (T_s5_D7_who_the_svm_adds.csv)", ""]
for r in D7[D7.n > 0].itertuples():
    dist = ", ".join(f"{t}: {getattr(r, f'total_{t}')}" for t in range(11) if getattr(r, f"total_{t}") > 0)
    R += [f"- {r.group}, {r.category} (n = {r.n}): totals {dist}"]
write_report(R + ["", f"(D took {time.time() - t0:.0f} s; step 5 total {time.time() - T_START:.0f} s)"])

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════

import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Arial is often missing on Linux; no warning flood
FIG_STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
             "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
             "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.spines.top": False, "axes.spines.right": False}
FIG_DIR = FIG
os.makedirs(FIG_DIR, exist_ok=True)
FIGURES_SAVED = []


def save_figure(fig, name):
    """Save a figure at 600 dpi in <work>/aq_10/figures/ and close it."""
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=600)
    plt.close(fig)
    FIGURES_SAVED.append(name)
    print(f"  [figure] {path}")


print("\nFigures...")
sh5 = pd.read_csv(os.path.join(PROJECT, "rq4", "tables", "T_rq4_shap_values_per_instance.csv"), usecols=["ID", "SHAP_A_Q1"])
d5 = bd[["ID", "label", "code", "Q1"]].merge(sh5, on="ID")
assert len(d5) == 843
ANS = [(4, "Definitely disagree", "#D55E00"), (3, "Slightly disagree", "#F3C6A5"),
       (2, "Slightly agree", "#BFD7EA"), (1, "Definitely agree", "#4A8FC7")]   # raw codes of item 1
with plt.rc_context(FIG_STYLE):
    fig = plt.figure(figsize=(7.16, 2.85))
    ax_a = fig.add_axes([0.165, 0.30, 0.24, 0.64])
    ax_b = fig.add_axes([0.475, 0.30, 0.205, 0.64])
    ax_c = fig.add_axes([0.775, 0.30, 0.215, 0.64])
    # (a) answers to item 1 in each ASD school, in S06 and in the other mainstream schools
    groups5 = [(f"{CODE[s]} (ASD, {len(IDX[s])})", (d5.code == CODE[s]).values) for s in SP]
    groups5 += [(f"S06 (mainstream, {int(S06.sum())})", (d5.code == "S06").values),
                (f"Other mainstream ({int(OTHER_MN.sum())})", ((d5.label == 0) & (d5.code != "S06")).values)]
    for i, (name, m) in enumerate(groups5):
        yy = len(groups5) - 1 - i
        left = 0.0
        for code_, _, col in ANS:
            wv = 100 * (d5.Q1.values[m] == code_).mean()
            ax_a.barh(yy, wv, left=left, height=0.72, color=col, edgecolor="white", linewidth=0.5)
            if wv >= 14:
                ax_a.text(left + wv / 2, yy, f"{wv:.0f}%", ha="center", va="center", fontsize=6,
                          color="white" if code_ in (4, 1) else "#222222")
            left += wv
    ax_a.set_yticks(range(len(groups5))[::-1]); ax_a.set_yticklabels([g[0] for g in groups5], fontsize=6.4)
    ax_a.axhline(1.5, color="#444444", lw=0.6, ls=(0, (2, 2)))
    ax_a.set_xlim(0, 100); ax_a.set_xticks([0, 25, 50, 75, 100]); ax_a.set_xlabel("Answers to item 1 (%)")
    ax_a.spines["left"].set_visible(False)
    ax_a.tick_params(axis="y", length=0)
    ax_a.legend(handles=[Patch(color=col) for _, _, col in ANS], labels=[l for _, l, _ in ANS],
                loc="upper center", bbox_to_anchor=(0.30, -0.20), ncol=2, frameon=False, fontsize=6.2,
                handlelength=1.0, columnspacing=1.0)
    # (b) SHAP value of item 1 for every child, by answer, with the means
    jit = np.random.default_rng(1)                               # jitter only; no effect on any number
    pos5 = {4: 0, 3: 1, 2: 2, 1: 3}
    for lab, col, nm in ((1, "#9B2C5E", "ASD school"), (0, "#4C9BD5", "mainstream")):
        m = (d5.label == lab).values
        xj = d5.Q1.values[m].astype(int)
        ax_b.scatter(np.vectorize(pos5.get)(xj) + jit.uniform(-0.22, 0.22, m.sum()), d5.SHAP_A_Q1.values[m],
                     s=2.2, color=col, alpha=0.55, linewidths=0, label=nm)
    for code_, xp in pos5.items():
        sel = d5.Q1.values == code_
        if sel.any():
            mu = d5.SHAP_A_Q1.values[sel].mean()
            ax_b.plot([xp - 0.30, xp + 0.30], [mu, mu], color="#111111", lw=1.6, solid_capstyle="butt")
    ax_b.plot([], [], color="#111111", lw=1.6, label="mean")
    ax_b.axhline(0, color="#888888", lw=0.6, ls=(0, (2, 2)))
    ax_b.set_xticks(range(4)); ax_b.set_xticklabels(["Def.\ndis.", "Sl.\ndis.", "Sl.\nagree", "Def.\nagree"])
    ax_b.set_xlabel("Answer to item 1"); ax_b.set_ylabel("SHAP value of item 1")
    ax_b.legend(loc="upper right", frameon=False, fontsize=6.2, markerscale=3, handlelength=1.2)
    # (c) children flagged in each mainstream school: seen, unseen, unseen + 10 local children
    mn5 = [CODE[s] for s in MN]
    c1_ = C1.set_index("school"); k10_ = by_school.set_index("school")
    series5 = [("seen in training", c1_.loc[mn5, "flagged_pct_randomCV_seen"].values, "#B8B8B8"),
               ("unseen", c1_.loc[mn5, "flagged_pct_LOSO_unseen"].values, "#D55E00"),
               ("unseen + 10 local children", k10_.loc[mn5, 10].values, "#4A8FC7")]
    hgt = 0.26
    for k, (nm, vals, col) in enumerate(series5):
        ax_c.barh(np.arange(len(mn5))[::-1] + (1 - k) * hgt, vals, height=hgt, color=col, label=nm, linewidth=0)
    ax_c.set_yticks(np.arange(len(mn5))[::-1]); ax_c.set_yticklabels(mn5, fontsize=6.4)
    xmax5 = max(5.0, 1.12 * max(float(np.nanmax(v)) for _, v, _ in series5))
    ax_c.set_xlabel("Children flagged (%)"); ax_c.set_xlim(0, xmax5)
    ax_c.tick_params(axis="y", length=0)
    ax_c.legend(loc="upper center", bbox_to_anchor=(0.40, -0.20), frameon=False, fontsize=6.2, handlelength=1.0)
    for ax_, letter, xoff in ((ax_a, "(a)", -0.62), (ax_b, "(b)", -0.30), (ax_c, "(c)", -0.22)):
        ax_.text(xoff, 1.04, letter, transform=ax_.transAxes, fontsize=8, fontweight="bold", va="bottom")
    save_figure(fig, "step5_item1_and_school_effects.png")

# Influence of each training school on the share flagged in each left-out school
with plt.rc_context(FIG_STYLE):
    fig = plt.figure(figsize=(4.6, 3.9))
    ax = fig.add_axes([0.12, 0.14, 0.7, 0.78])
    cax = fig.add_axes([0.85, 0.2, 0.025, 0.66])
    vals_ = infl.values.astype(float)
    lim_ = max(1.0, float(np.nanmax(np.abs(vals_))))
    im_ = ax.imshow(vals_, cmap="RdBu_r", vmin=-lim_, vmax=lim_, aspect="auto")
    for i in range(vals_.shape[0]):
        for j in range(vals_.shape[1]):
            if not np.isnan(vals_[i, j]):
                ax.text(j, i, f"{vals_[i, j]:+.0f}", ha="center", va="center", fontsize=5,
                        color="white" if abs(vals_[i, j]) > 0.6 * lim_ else "#222222")
    ax.set_xticks(range(len(infl.columns))); ax.set_xticklabels(infl.columns, rotation=90)
    ax.set_yticks(range(len(infl.index))); ax.set_yticklabels(infl.index)
    ax.set_xlabel("Training school removed"); ax.set_ylabel("Left-out school")
    ax.set_title("Change in children flagged in the left-out school (percentage points)", fontsize=6.5, loc="left")
    fig.colorbar(im_, cax=cax)
    save_figure(fig, "step5_training_school_influence.png")


with open(os.path.join(PROJECT, "logs", "step5_done.flag"), "w") as fh:
    fh.write(f"step5_school_effects complete\nscikit-learn={sklearn.__version__}\nrefit_agreement={agree}/843\n")
print("\n" + "=" * 78)
print(f"STEP 5 COMPLETE  |  report: {os.path.join(OUT, 'report_step5.md')}")
print("=" * 78)
