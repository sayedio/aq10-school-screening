"""Step 7 - Summary of all steps, one CSV file per topic.

Run after step6_robustness.py:

    python pipeline/step7_summary.py

Reads the outputs of rq1_baseline.py to step6_robustness.py, prints each result and writes it to
<work>/aq_10/step7_summary/ as a CSV file. Results of the locked run (model selection, the test
set, ten versus fourteen items, item importance) are read from the tables of RQ1 to RQ4. The rest
is computed here and is post hoc: school-clustered bootstrap intervals, DeLong tests, lower
cut-offs, children flagged by school and by score, refits with logistic regression and without
item 1, the false positives in school S06, subgroups with Wilson intervals and the positive
predictive value at screening prevalences. Only school codes are written; no child IDs.
"""
import json
import os
import re
import sys
import warnings

warnings.filterwarnings("ignore")                     # as in the other steps; library notices only

import numpy as np
import pandas as pd
import sklearn
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.environ.get("AQ10_BASE", os.path.join(REPO, "work")), "aq_10")
OUT = os.path.join(ROOT, "step7_summary")
AQ = [f"A_Q{i}" for i in range(1, 11)]
SEED = 42
ITEM = {"A_Q1": "Q1 Notices small sounds", "A_Q2": "Q2 Whole picture over details (R)",
        "A_Q3": "Q3 Follows several conversations (R)", "A_Q4": "Q4 Switches activities easily (R)",
        "A_Q5": "Q5 Cannot keep a conversation going", "A_Q6": "Q6 Good at social chit-chat (R)",
        "A_Q7": "Q7 Hard to infer characters' intentions", "A_Q8": "Q8 Enjoyed pretend play (R)",
        "A_Q9": "Q9 Reads feelings from faces (R)", "A_Q10": "Q10 Hard to make new friends"}


# ---------------------------------------------------------------- statistics
def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong(y, p1, p2):
    """Paired DeLong test; returns (auc1, auc2, z, p)."""
    y = np.asarray(y)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    preds = np.vstack([p1, p2]).astype(float)
    tx = np.array([_midrank(r[pos]) for r in preds])
    ty = np.array([_midrank(r[neg]) for r in preds])
    tz = np.array([_midrank(np.concatenate([r[pos], r[neg]])) for r in preds])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    lvec = np.array([1, -1])
    z = (aucs[0] - aucs[1]) / np.sqrt(lvec @ s @ lvec)
    return aucs[0], aucs[1], z, 2 * stats.norm.sf(abs(z))


def mcnemar(a, b):
    """Continuity-corrected McNemar; b = a only, c = b only."""
    bb = int(((a == 1) & (b == 0)).sum())
    cc = int(((a == 0) & (b == 1)).sum())
    if bb + cc == 0:
        return bb, cc, 0.0, 1.0
    chi = (abs(bb - cc) - 1) ** 2 / (bb + cc)
    return bb, cc, chi, stats.chi2.sf(chi, 1)


def wilson(k, n, z=1.959963984540054):
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return ph, c - h, c + h


def cal_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    return LogisticRegression(C=1e10, max_iter=5000).fit(lo, y).coef_[0][0]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        mk = (p >= edges[i]) & (p <= edges[i + 1]) if i == 0 else (p > edges[i]) & (p <= edges[i + 1])
        if mk.sum():
            e += mk.mean() * abs(y[mk].mean() - p[mk].mean())
    return e


def svm_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("c", SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=SEED))])


def lr_pipe():
    return Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(C=0.1, penalty="l2", max_iter=2000, random_state=SEED))])


def loso(d, y, feats, make):
    prob = np.zeros(len(d))
    for sc in sorted(d.school_id.unique()):
        te = d.school_id.values == sc
        m = make().fit(d.loc[~te, feats].values, y[~te])
        prob[te] = m.predict_proba(d.loc[te, feats].values)[:, 1]
    return prob


# ---------------------------------------------------------------- output
def code(s):
    """SCH_001 -> S01."""
    return "S" + s[-2:]


def numbers(text):
    """All decimal numbers in a string such as '0.9506 [0.9012, 0.9877]'."""
    return [float(v) for v in re.findall(r"-?\d+\.\d+(?:e-?\d+)?", str(text))]


def locked(*parts):
    path = os.path.join(ROOT, *parts)
    if not os.path.exists(path):
        sys.exit(f"Missing {'/'.join(parts)}. Run the earlier steps first.")
    return pd.read_csv(path)


def emit(name, df, note=""):
    """Print a result and save it as <name>.csv."""
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    show = df.copy()                                  # the CSV keeps full precision; the console shows 6 digits
    for c in show.columns:
        if show[c].dtype == object:
            show[c] = show[c].map(lambda v: f"{v:.6g}" if isinstance(v, (float, np.floating)) and not np.isnan(v)
                                  else ("" if v is None or (isinstance(v, float) and np.isnan(v)) else v))
    print("\n" + "-" * 78 + f"\n{name}.csv" + (f"\n{note}" if note else "") + "\n" + "-" * 78)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_colwidth", 70):
        print(show.to_string(index=False, float_format=lambda v: f"{v:.6g}"))


def kv(rows):
    """Rows of (quantity, value, interval low, interval high, note)."""
    return pd.DataFrame(rows, columns=["quantity", "value", "ci_low", "ci_high", "note"])


# ---------------------------------------------------------------- main
def main():
    sys.stdout.reconfigure(encoding="utf-8")
    flag = os.path.join(ROOT, "logs", "step6_done.flag")
    if not os.path.exists(flag):
        sys.exit(f"Run step6_robustness.py first (missing {flag}).")
    os.makedirs(OUT, exist_ok=True)
    print("=" * 78 + "\nSTEP 7 - SUMMARY OF ALL STEPS\n" + "=" * 78)
    print(f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, pandas {pd.__version__}; seed {SEED}")

    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "bd_clean.csv"))
    # Child a362: gender was left blank at data entry; the paper form shows female (the Zenodo file holds F).
    df.loc[(df.ID == "a362") & df.Gender.isna(), "Gender"] = "F"
    df["sex"] = df.Gender.astype(str).str.strip().str.lower()

    # ---- cohort
    sp, mn = df[df.label == 1], df[df.label == 0]
    sa = pd.read_csv(os.path.join(REPO, "config", "split_assignment.csv"))
    part = df[["ID", "label"]].merge(sa, left_on="ID", right_on="child_id")
    npart = lambda lab, p: int(((part.label == lab) & (part.partition == p)).sum())
    rows = [("children", len(sp), len(mn)),
            ("age_mean", sp.Age.mean(), mn.Age.mean()), ("age_sd", sp.Age.std(), mn.Age.std()),
            ("age_min", sp.Age.min(), mn.Age.min()), ("age_max", sp.Age.max(), mn.Age.max()),
            ("aged_4_to_7", int((sp.Age <= 7).sum()), int((mn.Age <= 7).sum())),
            ("aged_8_to_11", int((sp.Age >= 8).sum()), int((mn.Age >= 8).sum())),
            ("boys", int((sp.sex == "m").sum()), int((mn.sex == "m").sum())),
            ("girls", int((sp.sex == "f").sum()), int((mn.sex == "f").sum())),
            ("aq10_total_mean", sp.AQ_score_legacy.mean(), mn.AQ_score_legacy.mean()),
            ("aq10_total_sd", sp.AQ_score_legacy.std(), mn.AQ_score_legacy.std()),
            ("aq10_total_median", sp.AQ_score_legacy.median(), mn.AQ_score_legacy.median()),
            ("aq10_rule_positive_score_6_or_more", int(sp.AQ_rule_pred.sum()), int(mn.AQ_rule_pred.sum())),
            ("development_set", npart(1, "development"), npart(0, "development")),
            ("test_set", npart(1, "test"), npart(0, "test")),
            ("schools", sp.school_id.nunique(), mn.school_id.nunique())]
    emit("cohort", pd.DataFrame(rows, columns=["measure", "ASD_schools", "mainstream_schools"]))

    # ---- model selection (locked run). The SVM grid holds two kernels and naive Bayes two forms; each
    # is shown on its own row. Where the locked best setting of a learner falls in a row it is used as
    # is, otherwise the best setting of that row.
    cv = locked("rq1_baseline", "tables", "T_rq1_best_config_per_algo_cv.csv").set_index("Algorithm")
    log = locked("rq1_baseline", "tables", "T_rq1_hyperparameter_search_log.csv")

    def family(algo, hp):
        h = json.loads(hp)
        if algo == "Support Vector Machine":
            return f"{algo}, {h['kernel']} kernel"
        return f"{algo}, {h['model_type']}" if algo == "Naive Bayes" else algo

    log["family"] = [family(a, h) for a, h in zip(log.Algorithm, log.Hyperparameters)]
    rows = []
    for fam, g in log.groupby("family"):
        lk = cv.loc[g.Algorithm.iloc[0]]
        best = lk if family(g.Algorithm.iloc[0], lk.Hyperparameters) == fam else g.loc[g.Mean_CV_AUROC.idxmax()]
        ties = int((abs(g.Mean_CV_AUROC - best.Mean_CV_AUROC) < 1e-9).sum())
        rows.append({"learner": fam, "settings": len(g), "best_setting": best.Hyperparameters,
                     "other_settings_tied_at_best": ties - 1,
                     **{f"cv_{k.lower()}_mean": best[f"Mean_CV_{k}"] for k in ("AUROC", "Sensitivity", "Specificity", "Brier", "ECE")},
                     "cv_auroc_sd": best.Std_CV_AUROC})
    emit("model_selection_cv", pd.DataFrame(rows).sort_values("cv_auroc_mean", ascending=False),
         f"{len(log)} settings, each fitted in the same five folds of the development set.")

    # ---- test set, each learner at its best setting
    ta = pd.read_csv(os.path.join(ROOT, "rq1_baseline", "predictions", "test_predictions.csv"))
    rows = []
    for name, col in [("SVM, RBF kernel", "SVM"), ("Random forest", "RF"), ("Naive Bayes", "NB"),
                      ("k-nearest neighbors", "KNN"), ("Logistic regression", "LR"), ("AQ-10 rule (score >= 6)", None)]:
        yy = ta.label.values
        pr = ta[f"pred_{col}"].values if col else ta.AQ_rule_pred.values
        sc = ta[f"prob_{col}"].values if col else ta.AQ_score_legacy.values
        tp, fn = int(((pr == 1) & (yy == 1)).sum()), int(((pr == 0) & (yy == 1)).sum())
        fp, tn = int(((pr == 1) & (yy == 0)).sum()), int(((pr == 0) & (yy == 0)).sum())
        rows.append({"method": name, "auroc": roc_auc_score(yy, sc), "auroc_of": "probability" if col else "AQ-10 total score",
                     "accuracy": (tp + tn) / len(yy), "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
                     "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan,
                     "f1": 2 * tp / (2 * tp + fp + fn), "TP": tp, "FP": fp, "FN": fn, "TN": tn})
    emit("test_set_all_learners", pd.DataFrame(rows), f"{len(ta)} test children; threshold 0.50 for every learner.")

    # ---- test set, selected model against the rule (locked bootstrap intervals)
    ci_t = locked("rq1_baseline", "tables", "T_rq1_bootstrap_CI_all.csv")
    svm_r, rul_r = ci_t[ci_t.Is_Mstar.astype(str) == "True"].iloc[0], ci_t[ci_t.Is_Mstar.astype(str) != "True"].iloc[0]
    rows = []
    for k in ("Sensitivity", "Specificity", "Precision_PPV", "NPV", "F1_Score", "Accuracy", "Kappa", "AUROC", "AUPRC", "Brier", "ECE"):
        get = lambda r, s: r.get(f"{k}_{s}", np.nan)
        rows.append({"metric": k, "svm": get(svm_r, "Point"), "svm_ci_low": get(svm_r, "95CI_Lo"), "svm_ci_high": get(svm_r, "95CI_Hi"),
                     "rule": get(rul_r, "Point"), "rule_ci_low": get(rul_r, "95CI_Lo"), "rule_ci_high": get(rul_r, "95CI_Hi")})
    emit("test_set_svm_vs_rule", pd.DataFrame(rows),
         "95% intervals: stratified bootstrap, 2,000 resamples. The rule is binary, so it has no probability-based metrics.")
    cal = locked("rq1_baseline", "tables", "T_rq1_secondary_mstar_auroc_calibration.csv").iloc[0]
    t = ta
    yt, pt = t.label.values, t.prob_SVM.values
    yht, rt = (pt >= 0.5).astype(int), t.AQ_rule_pred.values
    m_all, m_sp, m_mn = mcnemar(yht, rt), mcnemar(yht[yt == 1], rt[yt == 1]), mcnemar(yht[yt == 0], rt[yt == 0])
    dl_t = delong(yt, pt, t.AQ_score_legacy.values)
    emit("test_set_tests", kv([
        ("McNemar, all children: SVM only / rule only", f"{m_all[0]} / {m_all[1]}", None, None, f"chi2 {m_all[2]:.4f}, p {m_all[3]:.3g}"),
        ("McNemar, ASD-school children: SVM only / rule only", f"{m_sp[0]} / {m_sp[1]}", None, None, f"p {m_sp[3]:.3g}"),
        ("McNemar, mainstream children: SVM only / rule only", f"{m_mn[0]} / {m_mn[1]}", None, None, f"p {m_mn[3]:.3g}"),
        ("AUROC, SVM", dl_t[0], None, None, ""),
        ("AUROC, AQ-10 total score", dl_t[1], None, None, f"locked run {cal.AQ10_instrument_AUROC_context_only}"),
        ("DeLong p, SVM vs AQ-10 total score", dl_t[3], None, None, f"z {dl_t[2]:.4f}"),
        ("Calibration slope, SVM", cal_slope(yt, pt), None, None, "")]))

    # ---- leave one school out
    o = pd.read_csv(os.path.join(ROOT, "rq2_loso", "predictions", "rq2_loso_s0_predictions.csv"))
    d = df.set_index("ID").loc[o.ID].reset_index()
    y, p, yh = o.label.values, o.y_prob.values, o.y_pred.values
    tot = o.AQ_score_legacy.values.astype(float)
    osum = d[AQ].sum(axis=1).values.astype(float)
    cut = {6: (tot >= 6).astype(int), 5: (tot >= 5).astype(int), 4: (tot >= 4).astype(int)}

    rng = np.random.default_rng(SEED)
    sp_s, mn_s = o[o.label == 1].school_id.unique(), o[o.label == 0].school_id.unique()
    idx = {s: np.where(o.school_id.values == s)[0] for s in o.school_id.unique()}
    boot = []
    for _ in range(2000):
        ss = list(rng.choice(sp_s, len(sp_s))) + list(rng.choice(mn_s, len(mn_s)))
        ii = np.concatenate([idx[s] for s in ss])
        yy = y[ii]
        se = lambda pr: pr[ii][yy == 1].mean()
        spf = lambda pr: 1 - pr[ii][yy == 0].mean()
        r6 = cut[6]
        boot.append(dict(
            auc=roc_auc_score(yy, p[ii]), auc_t=roc_auc_score(yy, tot[ii]),
            se=se(yh), sp=spf(yh), ppv=(yh[ii] & yy).sum() / yh[ii].sum(), acc=(yh[ii] == yy).mean(),
            se6=se(r6), sp6=spf(r6), ppv6=(r6[ii] & yy).sum() / r6[ii].sum(), acc6=(r6[ii] == yy).mean(),
            d_auc_t=roc_auc_score(yy, p[ii]) - roc_auc_score(yy, tot[ii]),
            d_auc_o=roc_auc_score(yy, p[ii]) - roc_auc_score(yy, osum[ii]),
            d_se6=se(yh) - se(r6), d_sp6=spf(yh) - spf(r6), d_se5=se(yh) - se(cut[5])))
    q = pd.DataFrame(boot).quantile([0.025, 0.975])
    Q = lambda k: (q[k].iloc[0], q[k].iloc[1])

    rows = []
    for name, pr, key in [("SVM, threshold 0.50", yh, ""), ("AQ-10 score >= 6 (rule)", cut[6], "6"),
                          ("AQ-10 score >= 5", cut[5], None), ("AQ-10 score >= 4", cut[4], None)]:
        tp, fn = int(pr[y == 1].sum()), int((1 - pr[y == 1]).sum())
        fp, tn = int(pr[y == 0].sum()), int((1 - pr[y == 0]).sum())
        r = {"method": name, "TP": tp, "FN": fn, "FP": fp, "TN": tn, "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp),
             "ppv": tp / (tp + fp), "npv": tn / (tn + fn), "accuracy": (tp + tn) / len(y), "f1": 2 * tp / (2 * tp + fp + fn)}
        for m_, b_ in (("sensitivity", "se"), ("specificity", "sp"), ("ppv", "ppv"), ("accuracy", "acc")):
            lo, hi = Q(b_ + key) if key is not None else (np.nan, np.nan)
            r[f"{m_}_ci_low"], r[f"{m_}_ci_high"] = lo, hi
        rows.append(r)
    emit("loso_svm_and_cutoffs", pd.DataFrame(rows),
         f"{len(y)} children, each scored with its school left out. Intervals: school-clustered bootstrap "
         "(2,000 resamples of schools within each group).")

    dl_t, dl_o, dl_ot = delong(y, p, tot), delong(y, p, osum), delong(y, osum, tot)
    se_full = lambda pr: pr[y == 1].mean()
    sp_full = lambda pr: 1 - pr[y == 0].mean()
    rows = [("AUROC, SVM", dl_t[0], *Q("auc"), "school-clustered interval"),
            ("AUROC, AQ-10 total score", dl_t[1], *Q("auc_t"), "school-clustered interval"),
            ("AUROC, unweighted ordinal sum", dl_o[1], None, None, ""),
            ("AUROC difference, SVM minus total score", dl_t[0] - dl_t[1], *Q("d_auc_t"), f"DeLong p {dl_t[3]:.3g}"),
            ("AUROC difference, SVM minus ordinal sum", dl_o[0] - dl_o[1], *Q("d_auc_o"), f"DeLong p {dl_o[3]:.3g}"),
            ("AUROC difference, ordinal sum minus total score", dl_ot[0] - dl_ot[1], None, None, f"DeLong p {dl_ot[3]:.3g}"),
            ("Sensitivity difference, SVM minus rule", se_full(yh) - se_full(cut[6]), *Q("d_se6"), ""),
            ("Specificity difference, SVM minus rule", sp_full(yh) - sp_full(cut[6]), *Q("d_sp6"), ""),
            ("Sensitivity difference, SVM minus score >= 5", se_full(yh) - se_full(cut[5]), *Q("d_se5"), "")]
    for k, lab in [(6, "rule"), (5, "score >= 5")]:
        a, b = mcnemar(yh[y == 1], cut[k][y == 1]), mcnemar(yh[y == 0], cut[k][y == 0])
        rows.append((f"ASD-school children, SVM only / {lab} only", f"{a[0]} / {a[1]}", None, None, f"child-level McNemar p {a[3]:.3g}"))
        rows.append((f"Mainstream children, SVM only / {lab} only", f"{b[0]} / {b[1]}", None, None, f"child-level McNemar p {b[3]:.3g}"))
    rows += [("Brier score, SVM", brier_score_loss(y, p), None, None, ""), ("ECE, SVM (ten bins)", ece(y, p), None, None, ""),
             ("Calibration slope, SVM", cal_slope(y, p), None, None, "")]
    youden = []
    for sc in sorted(d.school_id.unique()):
        tr = d.school_id.values != sc
        j = [((tot[tr] >= c)[y[tr] == 1].mean() - (tot[tr] >= c)[y[tr] == 0].mean(), c) for c in range(11)]
        youden.append(max(j)[1])
    rows.append(("Youden-optimal AQ-10 cut-off on the training schools of each fold", ", ".join(map(str, youden)), None, None, ""))
    emit("loso_tests", kv(rows), "School-level versions of the paired tests are in step6_robustness (T_s6_A_clustered_mcnemar.csv).")

    fl = o.assign(svm=yh, rule=cut[6]).groupby("school_id").agg(
        lab=("label", "first"), n=("label", "size"), svm=("svm", "sum"), rule=("rule", "sum"))
    fl = fl.sort_values(["lab", "school_id"], ascending=[False, True])
    emit("loso_flags_by_school", pd.DataFrame(
        [{"school": code(s), "group": "ASD" if r.lab == 1 else "mainstream", "children": r.n, "flagged_svm": r.svm,
          "flagged_rule": r.rule, "share_flagged_svm": r.svm / r.n, "share_flagged_rule": r.rule / r.n} for s, r in fl.iterrows()]),
        "For ASD schools the share flagged is the sensitivity; for mainstream schools it is the false-positive rate.")
    rows = []
    for s_ in range(11):
        row = {"aq10_total": s_}
        for g, gname in ((1, "ASD"), (0, "mainstream")):
            m = (y == g) & (tot == s_)
            row[f"{gname}_children"], row[f"{gname}_flagged_svm"] = int(m.sum()), int(yh[m].sum())
        rows.append(row)
    emit("loso_flags_by_score", pd.DataFrame(rows))

    # ---- post hoc refits
    # Fit with the rows in bd_clean order, exactly as rq2_loso.py does (the SVM's internal Platt
    # cross-validation depends on row order), then put the probabilities back in the order of o.
    in_o = lambda pr: pd.Series(pr, index=df.ID.values).loc[o.ID].values
    p10 = in_o(loso(df, df.label.values, AQ, svm_pipe))
    p9 = in_o(loso(df, df.label.values, [f for f in AQ if f != "A_Q1"], svm_pipe))
    y9 = (p9 >= 0.5).astype(int)
    plr = in_o(loso(df, df.label.values, AQ, lr_pipe))
    ylr = (plr >= 0.5).astype(int)
    dd = []
    for _ in range(2000):
        ss = list(rng.choice(sp_s, 5)) + list(rng.choice(mn_s, 7))
        ii = np.concatenate([idx[s] for s in ss])
        dd.append(roc_auc_score(y[ii], p9[ii]) - roc_auc_score(y[ii], p[ii]))
    s06 = d.school_id.values == "SCH_006"
    dd_lo, dd_hi = np.percentile(dd, [2.5, 97.5])
    emit("posthoc_refits", kv([
        ("Locked leave-one-school-out decisions reproduced by a refit", f"{int(((p10 >= .5) == (p >= .5)).sum())} of {len(p)}",
         None, None, f"scikit-learn {sklearn.__version__}"),
        ("Logistic regression (L2, C = 0.1), AUROC", roc_auc_score(y, plr), None, None, f"DeLong p vs AQ-10 total {delong(y, plr, tot)[3]:.3g}"),
        ("ASD-school children found by the SVM only / by logistic regression only",
         f"{int(((yh == 1) & (ylr == 0) & (y == 1)).sum())} / {int(((yh == 0) & (ylr == 1) & (y == 1)).sum())}", None, None, ""),
        ("SVM without item 1, sensitivity", y9[y == 1].mean(), None, None, ""),
        ("SVM without item 1, specificity", 1 - y9[y == 0].mean(), None, None, ""),
        ("SVM without item 1, AUROC", roc_auc_score(y, p9), None, None, ""),
        ("AUROC difference, without item 1 minus ten items", roc_auc_score(y, p9) - roc_auc_score(y, p), dd_lo, dd_hi,
         "school-clustered interval"),
        ("Flagged in S06, without item 1 / ten items", f"{y9[s06].sum()} / {yh[s06].sum()}", None, None, ""),
        ("False positives in the other mainstream schools, without item 1 / ten items",
         f"{y9[(y == 0) & ~s06].sum()} / {yh[(y == 0) & ~s06].sum()}", None, None, "")]),
        "Post hoc refits under leave-one-school-out validation.")

    # ---- ten versus fourteen items (locked run)
    pc = locked("rq3_augmentation", "tables", "paper_ready", "T_rq3_paper_paired_comparison.csv").iloc[0]
    bd3 = locked("rq3_augmentation", "tables", "T_rq3_bootstrap_deltas.csv")
    rows = [("AUROC, 10 items", pc.AUROC_S0, None, None, ""), ("AUROC, 14 items", pc.AUROC_S1, None, None, ""),
            ("Sensitivity, 10 items", pc.Sensitivity_S0, None, None, ""), ("Sensitivity, 14 items", pc.Sensitivity_S1, None, None, ""),
            ("Specificity, 10 items", pc.Specificity_S0, None, None, ""), ("Specificity, 14 items", pc.Specificity_S1, None, None, ""),
            ("Paired DeLong p", float(pc.DeLong_P), None, None, ""), ("Paired McNemar p", float(pc.McNemar_P), None, None, ""),
            ("Adopted item set", pc.LOCKED_FSTAR, None, None, "pre-specified adoption rule")]
    for r in bd3.itertuples():
        rows.append((f"Difference, 14 minus 10 items: {r.Metric}", r.Mean, r.CI_Lower, r.CI_Upper, "paired bootstrap over children (mean)"))
    emit("items_10_vs_14", kv(rows))

    # ---- item importance (locked run)
    gi = locked("rq4", "tables", "T_rq4_shap_pi_importance_global.csv")
    rows = []
    for r in gi.itertuples():
        sh, pi = numbers(r.SHAP_Boot_95CI), numbers(r.PI_AUROC_Drop_95CI)
        rows.append({"item": ITEM.get(r.Feature, r.Feature), "mean_abs_shap": sh[0], "shap_ci_low": sh[1], "shap_ci_high": sh[2],
                     "shap_rank": r.SHAP_Global_Rank, "permutation_auroc_drop": pi[0], "permutation_p2_5": pi[1],
                     "permutation_p97_5": pi[2], "permutation_rank": r.PI_Global_Rank, "permutation_logloss_rise": r.PI_LogLoss_Rise,
                     "rank_mean_across_schools": r.SHAP_Mean_Rank, "rank_cv_across_schools": r.SHAP_Rank_CV})
    emit("item_importance", pd.DataFrame(rows),
         "Mean |SHAP|: 95% interval from a bootstrap over schools. Permutation importance: 2.5th to 97.5th percentiles of 500 repetitions.")
    rho_pi = locked("rq4", "tables", "paper_ready", "T_rq4_paper_shap_vs_pi.csv").Value.iloc[0]
    pw = locked("rq4", "tables", "T_rq4_spearman_pairwise.csv")
    keep = (pw.School_A != "SCH_006") & (pw.School_B != "SCH_006")
    sh = pd.read_csv(os.path.join(ROOT, "rq4", "tables", "T_rq4_shap_values_per_instance.csv"))
    d2 = d.assign(yh=yh).merge(sh[["ID"] + [f"SHAP_{a}" for a in AQ]], on="ID")
    end_diff = [(d2[a][d2.label == 1] >= 2).mean() - (d2[a][d2.label == 0] >= 2).mean() for a in AQ]
    rho_end = stats.spearmanr([d2[f"SHAP_{a}"].abs().mean() for a in AQ], end_diff).statistic
    e1 = lambda m: (d2.A_Q1[m] >= 2).mean()
    emit("item_importance_agreement", kv([
        ("Spearman rho, SHAP ranking vs permutation ranking", rho_pi, None, None, ""),
        ("Spearman rho, SHAP ranking vs group difference in endorsement", rho_end, None, None, ""),
        ("Rank stability across left-out schools, mean pairwise rho", pw.Spearman_rho.mean(), pw.Spearman_rho.min(),
         pw.Spearman_rho.max(), "low/high = range"),
        ("Rank stability without S06, mean pairwise rho", pw.Spearman_rho[keep].mean(), pw.Spearman_rho[keep].min(),
         pw.Spearman_rho[keep].max(), "low/high = range"),
        ("Item 1 endorsed, ASD-school children", e1(d2.label == 1), None, None, ""),
        ("Item 1 endorsed, mainstream children", e1(d2.label == 0), None, None, "")]))

    # ---- false positives in school S06
    f6 = (d2.school_id == "SCH_006") & (d2.yh == 1)
    other = (d2.label == 0) & (d2.school_id != "SCH_006")
    shap6 = d2.loc[f6, [f"SHAP_{a}" for a in AQ]].mean()
    emit("s06_false_positives", kv([
        ("S06 children flagged by the SVM", f"{int(f6.sum())} of {int((d2.school_id == 'SCH_006').sum())}", None, None, ""),
        ("S06 children flagged by the rule", int(((d2.school_id == "SCH_006") & (d2.AQ_score_legacy >= 6)).sum()), None, None, ""),
        ("AQ-10 totals of the S06 children flagged by the SVM", f"{d2.AQ_score_legacy[f6].min()} to {d2.AQ_score_legacy[f6].max()}",
         None, None, ""),
        ("Answered 'definitely disagree' to item 1: S06 children flagged", f"{int((d2.Q1[f6] == 4).sum())} of {int(f6.sum())}", None, None, ""),
        ("Answered 'definitely disagree' to item 1: other mainstream children", f"{int((d2.Q1[other] == 4).sum())} of {int(other.sum())}",
         None, None, ""),
        ("Answered 'definitely disagree' to item 1: ASD-school children",
         f"{int((d2.Q1[d2.label == 1] == 4).sum())} of {int((d2.label == 1).sum())}", None, None, ""),
        ("Mean SHAP value of item 1, S06 children flagged", shap6["SHAP_A_Q1"], None, None, ""),
        ("Largest mean SHAP value of any other item, S06 children flagged", shap6.drop("SHAP_A_Q1").max(), None, None, ""),
        ("False positives outside S06: SVM / rule", f"{int(d2.yh[other].sum())} / {int((d2.AQ_score_legacy[other] >= 6).sum())}",
         None, None, f"of {int(other.sum())} children")]))

    # ---- subgroups and screening prevalence
    d2["rule"] = (d2.AQ_score_legacy >= 6).astype(int)
    d2["sex"] = d2.Gender.astype(str).str.strip().str.lower()
    rows = []
    for name, mk in [("Aged 4 to 7", d2.Age <= 7), ("Aged 8 to 11", d2.Age >= 8), ("Boys", d2.sex == "m"), ("Girls", d2.sex == "f")]:
        s_, m_ = mk & (d2.label == 1), mk & (d2.label == 0)
        a, b = wilson(d2.yh[s_].sum(), s_.sum()), wilson((1 - d2.yh[m_]).sum(), m_.sum())
        rows.append({"subgroup": name, "ASD_children": int(s_.sum()), "sensitivity_svm": a[0], "sens_ci_low": a[1], "sens_ci_high": a[2],
                     "sensitivity_rule": d2.rule[s_].mean(), "mainstream_children": int(m_.sum()), "specificity_svm": b[0],
                     "spec_ci_low": b[1], "spec_ci_high": b[2], "specificity_rule": 1 - d2.rule[m_].mean()})
    dx = d2.doctor_dx.astype(str).str.strip().str.lower()
    for v, lab in [("yes", "Prior diagnosis reported"), ("no", "Prior diagnosis not reported")]:
        s_ = (d2.label == 1) & (dx == v)
        a = wilson(d2.yh[s_].sum(), s_.sum())
        rows.append({"subgroup": lab, "ASD_children": int(s_.sum()), "sensitivity_svm": a[0], "sens_ci_low": a[1], "sens_ci_high": a[2],
                     "sensitivity_rule": d2.rule[s_].mean()})
    emit("subgroups", pd.DataFrame(rows), "Leave-one-school-out decisions; 95% Wilson intervals.")
    young = (d2.label == 0) & (d2.Age <= 7)
    rest = young & (d2.school_id != "SCH_006")
    emit("subgroups_younger_mainstream", kv([
        ("Younger mainstream children flagged by the SVM", int(d2.yh[young].sum()), None, None,
         f"{int(d2.yh[young & (d2.school_id == 'SCH_006')].sum())} of them in S06"),
        ("Specificity at 4 to 7 years outside S06, SVM", 1 - d2.yh[rest].mean(), None, None, ""),
        ("Specificity at 4 to 7 years outside S06, rule", 1 - d2.rule[rest].mean(), None, None, "")]))
    se, sp_ = d2.yh[d2.label == 1].mean(), 1 - d2.yh[d2.label == 0].mean()
    se_r, sp_r = d2.rule[d2.label == 1].mean(), 1 - d2.rule[d2.label == 0].mean()
    rows = []
    for pi in (0.01, 0.03, 0.05):
        ref, ref_r = se * pi + (1 - sp_) * (1 - pi), se_r * pi + (1 - sp_r) * (1 - pi)
        rows.append({"prevalence": pi, "ppv_svm": se * pi / ref, "ppv_rule": se_r * pi / ref_r,
                     "referred_per_1000_svm": 1000 * ref, "referred_per_1000_rule": 1000 * ref_r,
                     "found_per_1000_svm": 1000 * pi * se, "found_per_1000_rule": 1000 * pi * se_r})
    emit("screening_prevalence", pd.DataFrame(rows),
         "From the leave-one-school-out sensitivity and specificity; school placement as the reference.")

    with open(os.path.join(ROOT, "logs", "step7_done.flag"), "w", encoding="utf-8") as fh:
        fh.write(f"step7_summary complete\nscikit-learn={sklearn.__version__}\n")
    print("\n" + "=" * 78 + f"\nSTEP 7 COMPLETE  |  CSV files: {OUT}\n" + "=" * 78)


if __name__ == "__main__":
    main()
