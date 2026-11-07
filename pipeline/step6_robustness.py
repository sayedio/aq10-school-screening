"""Step 6 - Robustness: school-level statistics, nested tuning and the AQ-10 cut-off.

Run after rq4_explanation.py (it reads the RQ1 to RQ3 outputs and needs the RQ4 done flag):

    python pipeline/step6_robustness.py

Added after the locked RQ1-RQ4 run, to check whether the RQ1 to RQ3 conclusions hold when the
school, not the child, is treated as the unit, when the SVM settings are re-tuned inside every
fold, and under both readings of the AQ-10 referral cut-off. Output:
<AQ10_BASE>/aq_10/step6_robustness/ (tables/, report_step6.md) and logs/step6_done.flag.
Takes about 10-20 minutes; section E is the slow part.

A. Paired decisions with the school as the unit (clustered McNemar, exact school-level sign-flip test)
B. Hold-out set with school-clustered intervals
C. Fourteen versus ten items with school-clustered intervals
D. AQ-10 cut-offs 4 to 8 (6 or more, as published, and more than 6, as on the scoring sheet)
E. Nested leave-one-school-out: the SVM grid re-tuned on the training schools of every fold
"""
import itertools
import os
import sys
import time
import warnings

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 - PATHS, DATA AND HELPERS
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 78)
print("STEP 6 - ROBUSTNESS")
print("=" * 78)
BASE = os.environ.get("AQ10_BASE", "/content/drive/MyDrive/Autism")
PROJECT = os.path.join(BASE, "aq_10")
OUT = os.path.join(PROJECT, "step6_robustness")
TAB = os.path.join(OUT, "tables")
RQ4_FLAG = os.path.join(PROJECT, "logs", "rq4_done.flag")
if not os.path.exists(RQ4_FLAG):
    sys.exit(f"Run rq1_baseline.py to rq4_explanation.py first (missing {RQ4_FLAG}).")
for d in (OUT, TAB):
    os.makedirs(d, exist_ok=True)
T_START = time.time()

bd = pd.read_csv(os.path.join(PROJECT, "data", "processed", "bd_clean.csv"))
bd = bd.drop(columns=["school_name"], errors="ignore")
assert len(bd) == 843 and bd.ID.is_unique
AQ = [f"A_Q{i}" for i in range(1, 11)]
bd["code"] = "S" + bd.school_id.str[-2:]
bd["total"] = bd.AQ_score_legacy.astype(int)
lock = pd.read_csv(os.path.join(PROJECT, "rq2_loso", "predictions", "rq2_loso_s0_predictions.csv"))
lock = lock[["ID", "y_prob", "y_pred"]].rename(columns={"y_prob": "p_lock", "y_pred": "f_lock"})
s14 = pd.read_csv(os.path.join(PROJECT, "rq3_augmentation", "predictions", "rq3_loso_s1_predictions.csv"))
s14 = s14[["ID", "y_prob_s1", "y_pred_s1"]].rename(columns={"y_prob_s1": "p14", "y_pred_s1": "f14"})
bd = bd.merge(lock, on="ID", how="left").merge(s14, on="ID", how="left")   # a left merge keeps the bd_clean row order
assert len(bd) == 843 and bd[["p_lock", "p14"]].notna().all().all()

test = pd.read_csv(os.path.join(PROJECT, "rq1_baseline", "predictions", "test_predictions.csv"))
test["code"] = "S" + test.school_id.str[-2:]
test["total"] = test.AQ_score_legacy.astype(int)
assert len(test) == 169

y = bd.label.values.astype(int)
SCHOOLS = sorted(bd.school_id.unique())
SP = sorted(bd.loc[bd.label == 1, "school_id"].unique())
MN = sorted(bd.loc[bd.label == 0, "school_id"].unique())
CODE = dict(zip(bd.school_id, bd.code))
IDX = {s: np.where(bd.school_id.values == s)[0] for s in SCHOOLS}
S06 = (bd.school_id == "SCH_006").values
LOCKED = dict(C=1.0, kernel="rbf", gamma="scale")


def svm(C=1.0, kernel="rbf", gamma="scale"):
    return Pipeline([("scaler", StandardScaler()),
                     ("clf", SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=SEED))])


def fit_predict(train_mask, test_mask, cols=AQ, params=LOCKED):
    m = svm(**params).fit(bd.loc[train_mask, cols].values, y[train_mask])
    return m.predict_proba(bd.loc[test_mask, cols].values)[:, 1]


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
    """95% percentile interval; special and mainstream schools resampled separately, with replacement."""
    rng_b = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        ss = list(rng_b.choice(sp, len(sp))) + list(rng_b.choice(mn, len(mn)))
        vals.append(stat(np.concatenate([idx[s] for s in ss])))
    return np.nanpercentile(np.asarray(vals, float), [2.5, 97.5])


def wilson(x, n, z=1.959964):
    if n == 0:
        return (np.nan, np.nan)
    ph = x / n; den = 1 + z * z / n; c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def f3(x, d=3):
    return "n/a" if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)) else f"{x:.{d}f}"


def ci(lo_hi, d=3):
    return f"[{f3(lo_hi[0], d)}, {f3(lo_hi[1], d)}]"


def pfmt(p):
    return "n/a" if p is None or np.isnan(p) else (f"{p:.2g}" if p < 0.001 else f"{p:.3f}")


def save(df, name):
    df.to_csv(os.path.join(TAB, name), index=False)
    print("  [saved]", name)


REPORT = ["# Step 6 - Robustness", "",
          f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}; seed {SEED}. "
          "Added after the locked RQ1-RQ4 run; uses the locked predictions.", ""]


def write_report(section_lines):
    """Append a section, print it and rewrite report_step6.md (so finished sections survive a crash)."""
    REPORT.extend(section_lines + [""])
    with open(os.path.join(OUT, "report_step6.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT) + "\n")
    print("\n".join(section_lines))


# ══════════════════════════════════════════════════════════════════════════════
# A. PAIRED DECISIONS WITH THE SCHOOL AS THE UNIT
# ══════════════════════════════════════════════════════════════════════════════
def clustered_mcnemar(fa, fb, groups):
    """Durkalski et al. (2003) clustered McNemar test plus an exact school-level sign-flip test.
    b = children flagged by A only, c = flagged by B only."""
    t = pd.DataFrame({"g": groups, "b": (fa == 1) & (fb == 0), "c": (fa == 0) & (fb == 1)}).groupby("g")[["b", "c"]].sum()
    d = (t.b - t.c).values.astype(float)
    den = (d ** 2).sum()
    chi = d.sum() ** 2 / den if den > 0 else 0.0
    K = len(d)
    signs = np.array(list(itertools.product((1, -1), repeat=K))) if K <= 16 else np.random.default_rng(SEED).choice((1, -1), size=(200000, K))
    p_exact = float((np.abs(signs @ d) >= abs(d.sum()) - 1e-9).mean())
    b, c = int(t.b.sum()), int(t.c.sum())
    chi_child = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return {"schools": K, "b": b, "c": c, "schools_b_gt_c": int((d > 0).sum()), "schools_b_lt_c": int((d < 0).sum()),
            "p_child_level": float(stats.chi2.sf(chi_child, 1)) if b + c else 1.0,
            "p_clustered_Durkalski": float(stats.chi2.sf(chi, 1)) if den > 0 else 1.0, "p_school_signflip_exact": p_exact}


tot = bd.total.values; fl = bd.f_lock.values.astype(int); grp = bd.code.values
tt_y, tt_f, tt_tot, tt_g = test.label.values, test.pred_Mstar.values.astype(int), test.total.values, test.code.values
COMP = [("Unseen schools: SVM vs rule (S>=6)", fl, (tot >= 6).astype(int), y, grp),
        ("Unseen schools: SVM vs S>=5", fl, (tot >= 5).astype(int), y, grp),
        ("Unseen schools: SVM vs S>=4", fl, (tot >= 4).astype(int), y, grp),
        ("Unseen schools: 14 items vs 10 items", bd.f14.values.astype(int), fl, y, grp),
        ("Hold-out: SVM vs rule (S>=6)", tt_f, (tt_tot >= 6).astype(int), tt_y, tt_g)]
rows = []
for name, fa, fb, yy, gg in COMP:
    for sub, m in (("all children", np.ones(len(yy), bool)), ("special (sensitivity)", yy == 1), ("mainstream (specificity)", yy == 0)):
        rows.append({"comparison": name, "children": sub, **clustered_mcnemar(fa[m], fb[m], gg[m])})
TA = pd.DataFrame(rows)
save(TA.round(6), "T_s6_A_clustered_mcnemar.csv")
R = ["## A. Paired decisions with the school as the unit (T_s6_A_clustered_mcnemar.csv)", "",
     "b = flagged by the first method only, c = by the second only. The exact sign-flip test uses school-level differences "
     "(with 5 special schools its smallest possible p is 0.0625).", "",
     "| Comparison | Children | b vs c | Schools b>c / b<c | p child-level | p clustered (Durkalski) | p exact sign-flip |", "|---|---|---|---|---|---|---|"]
R += [f"| {r.comparison} | {r.children} | {r.b} vs {r.c} | {r.schools_b_gt_c} / {r.schools_b_lt_c} | {pfmt(r.p_child_level)} | "
      f"{pfmt(r.p_clustered_Durkalski)} | {pfmt(r.p_school_signflip_exact)} |" for r in TA.itertuples()]
write_report(R)

# ══════════════════════════════════════════════════════════════════════════════
# B. HOLD-OUT SET WITH SCHOOL-CLUSTERED INTERVALS (its children come from the same 12 schools)
# ══════════════════════════════════════════════════════════════════════════════
TIDX = {s: np.where(test.school_id.values == s)[0] for s in sorted(test.school_id.unique())}
TSP = sorted(test.loc[test.label == 1, "school_id"].unique()); TMN = sorted(test.loc[test.label == 0, "school_id"].unique())
tt_p = test.prob_Mstar.values
stat_ho = {"Sensitivity, SVM": lambda ii: rates(tt_y[ii], tt_f[ii])["Sens"],
           "Specificity, SVM": lambda ii: rates(tt_y[ii], tt_f[ii])["Spec"],
           "Sensitivity, rule": lambda ii: rates(tt_y[ii], tt_tot[ii] >= 6)["Sens"],
           "Specificity, rule": lambda ii: rates(tt_y[ii], tt_tot[ii] >= 6)["Spec"],
           "Sensitivity difference": lambda ii: rates(tt_y[ii], tt_f[ii])["Sens"] - rates(tt_y[ii], tt_tot[ii] >= 6)["Sens"],
           "AUROC, SVM": lambda ii: auc(tt_y[ii], tt_p[ii]), "AUROC, total score": lambda ii: auc(tt_y[ii], tt_tot[ii])}
rows = []
for name, fn in stat_ho.items():
    lohi = cluster_boot(fn, idx=TIDX, sp=TSP, mn=TMN)
    rows.append({"metric": name, "estimate": fn(np.arange(len(test))), "school_clustered_lo": lohi[0], "school_clustered_hi": lohi[1]})
TB = pd.DataFrame(rows)
save(TB.round(4), "T_s6_B_holdout_school_clustered_ci.csv")
R = [f"## B. Hold-out set with school-clustered intervals ({len(TSP)} special and {len(TMN)} mainstream schools; T_s6_B_*.csv)", ""]
R += [f"- {r.metric}: {r.estimate:.3f} [{r.school_clustered_lo:.3f}, {r.school_clustered_hi:.3f}]" for r in TB.itertuples()]
write_report(R)

# ══════════════════════════════════════════════════════════════════════════════
# C. FOURTEEN VERSUS TEN ITEMS WITH SCHOOL-CLUSTERED INTERVALS
# ══════════════════════════════════════════════════════════════════════════════
p14, f14 = bd.p14.values, bd.f14.values.astype(int)
cfun = {"Sensitivity difference (14 - 10)": lambda ii: rates(y[ii], f14[ii])["Sens"] - rates(y[ii], fl[ii])["Sens"],
        "Specificity difference (14 - 10)": lambda ii: rates(y[ii], f14[ii])["Spec"] - rates(y[ii], fl[ii])["Spec"],
        "AUROC difference (14 - 10)": lambda ii: auc(y[ii], p14[ii]) - auc(y[ii], bd.p_lock.values[ii])}
rows = []
for name, fn in cfun.items():
    lohi = cluster_boot(fn)
    rows.append({"metric": name, "estimate": fn(np.arange(843)), "lo": lohi[0], "hi": lohi[1]})
TC = pd.DataFrame(rows)
save(TC.round(4), "T_s6_C_14_vs_10_items_clustered.csv")
R = ["## C. Fourteen vs ten items, school-clustered intervals", ""]
R += [f"- {r.metric}: {r.estimate:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" for r in TC.itertuples()]
write_report(R)

# ══════════════════════════════════════════════════════════════════════════════
# D. AQ-10 CUT-OFFS 4 TO 8
# ══════════════════════════════════════════════════════════════════════════════
rows = []
for t in range(4, 9):
    rh, ra = rates(tt_y, tt_tot >= t), rates(y, tot >= t)
    lo_s = cluster_boot(lambda ii, t=t: rates(y[ii], tot[ii] >= t)["Sens"])
    lo_p = cluster_boot(lambda ii, t=t: rates(y[ii], tot[ii] >= t)["Spec"])
    rows.append({"cutoff": f"S >= {t}", "note": {6: "Allison et al. 2012 cut-off (6 or more)", 7: "scoring-sheet wording (more than 6)"}.get(t, ""),
                 "holdout_sens": rh["Sens"], "holdout_sens_wilson": ci(wilson(rh["TP"], rh["TP"] + rh["FN"])),
                 "holdout_spec": rh["Spec"], "holdout_spec_wilson": ci(wilson(rh["TN"], rh["TN"] + rh["FP"])),
                 "all843_sens": ra["Sens"], "all843_sens_school_ci": ci(lo_s), "all843_spec": ra["Spec"], "all843_spec_school_ci": ci(lo_p),
                 "all843_flagged": ra["TP"] + ra["FP"]})
TD = pd.DataFrame(rows)
save(TD.round(4), "T_s6_D_aq10_cutoffs.csv")
R = ["## D. AQ-10 cut-offs (the rule needs no training, so 'all 843' is the same as unseen schools)", "",
     "| Cut-off | Note | Hold-out sens | Hold-out spec | All 843 sens [school CI] | All 843 spec [school CI] |", "|---|---|---|---|---|---|"]
R += [f"| {r.cutoff} | {r.note} | {r.holdout_sens:.3f} {r.holdout_sens_wilson} | {r.holdout_spec:.3f} {r.holdout_spec_wilson} | "
      f"{r.all843_sens:.3f} {r.all843_sens_school_ci} | {r.all843_spec:.3f} {r.all843_spec_school_ci} |" for r in TD.itertuples()]
write_report(R)

# ══════════════════════════════════════════════════════════════════════════════
# E. NESTED LEAVE-ONE-SCHOOL-OUT: SVM SETTINGS RE-TUNED INSIDE EVERY OUTER FOLD
# ══════════════════════════════════════════════════════════════════════════════
t0 = time.time()
GRID = [dict(C=c, kernel=k, gamma=g) for c, k, g in itertools.product([0.1, 1.0, 10.0], ["rbf", "linear"], ["scale", "auto"])]
p_nest = np.full(843, np.nan); chosen = []
for oi, h in enumerate(SCHOOLS, 1):
    outer_train = bd.school_id.values != h
    best_key, best_prm, best_auc = None, None, None
    for prm in GRID:
        pin = np.full(843, np.nan)
        for s in SCHOOLS:
            if s != h:
                held = bd.school_id.values == s
                pin[held] = fit_predict(outer_train & ~held, held, params=prm)
        a_ = auc(y[outer_train], pin[outer_train])
        sens_ = ((pin[outer_train] >= .5) & (y[outer_train] == 1)).sum() / (y[outer_train] == 1).sum()
        key = (round(a_, 8), round(sens_, 8))
        if best_key is None or key > best_key:
            best_key, best_prm, best_auc = key, prm, a_
    held = bd.school_id.values == h
    p_nest[held] = fit_predict(outer_train, held, params=best_prm)
    chosen.append({"held_out": CODE[h], **best_prm, "inner_pooled_AUROC": best_auc})
    print(f"  nested fold {oi}/12 {CODE[h]}: chose {best_prm} (inner AUROC {best_auc:.4f}) [{time.time() - t0:.0f} s]")
TE = pd.DataFrame(chosen)
save(TE.round(5), "T_s6_E_nested_chosen_settings.csv")
f_nest = (p_nest >= .5).astype(int)
rn, rl = rates(y, f_nest), rates(y, fl)
d_auc = cluster_boot(lambda ii: auc(y[ii], p_nest[ii]) - auc(y[ii], bd.p_lock.values[ii]))
nest_ci = {k: cluster_boot(fn) for k, fn in {"AUROC": lambda ii: auc(y[ii], p_nest[ii]),
                                                "Sens": lambda ii: rates(y[ii], f_nest[ii])["Sens"],
                                                "Spec": lambda ii: rates(y[ii], f_nest[ii])["Spec"]}.items()}
save(pd.DataFrame({"school": [CODE[s] for s in SCHOOLS], "n": [len(IDX[s]) for s in SCHOOLS],
                   "flagged_locked": [int(fl[IDX[s]].sum()) for s in SCHOOLS],
                   "flagged_nested": [int(f_nest[IDX[s]].sum()) for s in SCHOOLS]}), "T_s6_E_nested_flags_by_school.csv")
R = ["## E. Nested leave-one-school-out: the SVM grid (12 settings) re-tuned on the 11 training schools of every fold", "",
     "Inner selection: pooled inner leave-one-school-out AUROC (tie-break: sensitivity).", "",
     "- Settings chosen: " + "; ".join(f"{r.held_out} C={r.C}, {r.kernel}, gamma={r.gamma}" for r in TE.itertuples()) + ".",
     f"- Nested: AUROC {auc(y, p_nest):.3f} {ci(nest_ci['AUROC'])}, sensitivity {rn['Sens']:.3f} {ci(nest_ci['Sens'])}, "
     f"specificity {rn['Spec']:.3f} {ci(nest_ci['Spec'])}.",
     f"- Locked: AUROC {auc(y, bd.p_lock.values):.3f}, sensitivity {rl['Sens']:.3f}, specificity {rl['Spec']:.3f}. "
     f"AUROC difference (nested - locked) {auc(y, p_nest) - auc(y, bd.p_lock.values):+.3f} {ci(d_auc)}.",
     f"- Same decision for {int((f_nest == fl).sum())} of 843 children; S06 flagged nested {int(f_nest[S06].sum())} vs locked {int(fl[S06].sum())}."]
write_report(R + ["", f"(E took {time.time() - t0:.0f} s; step 6 total {time.time() - T_START:.0f} s)"])

with open(os.path.join(PROJECT, "logs", "step6_done.flag"), "w") as fh:
    fh.write(f"step6_robustness complete\nscikit-learn={sklearn.__version__}\n")
print("\n" + "=" * 78)
print(f"STEP 6 COMPLETE  |  report: {os.path.join(OUT, 'report_step6.md')}")
print("=" * 78)
