"""Write results/paper_numbers.md: the numbers a reviewer needs to check the paper.

Run after rq1_baseline.py to rq4_explanation.py, with AQ10_BASE pointing at the
folder that holds the private aq_10 project folder:

    python analysis/paper_numbers.py > my_paper_numbers.md

Then compare my_paper_numbers.md with results/paper_numbers.md.

Locked-run results (model selection, Table II, ten versus fourteen items, item
importance) are read from the tables that rq1_baseline.py to rq4_explanation.py
write. Everything else is computed here: the cohort, school-clustered bootstrap
intervals, DeLong tests, lower cut-offs, per-school flags, the error analysis in
school S06, the nine-item refit, subgroups and prevalence-adjusted PPV.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = os.path.join(os.environ.get("AQ10_BASE", os.path.join(os.getcwd(), "work")), "aq_10")
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


# ---------------------------------------------------------------- formatting
def f3(x):
    return f"{x:.3f}"


def pval(p):
    if p < 0.001:
        m, e = f"{p:.1e}".split("e")
        return f"{m} × 10<sup>{int(e)}</sup>"
    return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"


def ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def r3(text):
    """Round every number in a locked-run string such as '0.9506 [0.9012, 0.9877]' to 3 decimals."""
    return re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.3f}", str(text))


def r4(text):
    """As r3, with 4 decimals, for small values such as item importance."""
    return re.sub(r"-?\d+\.\d+", lambda m: f"{float(m.group()):.4f}", str(text))


def code(s):
    """SCH_001 -> S01, as in the paper."""
    return "S" + s[-2:]


def mdtable(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def locked(*parts):
    path = os.path.join(ROOT, *parts)
    if not os.path.exists(path):
        sys.exit(f"Missing {'/'.join(parts)}. Run the earlier steps first.")
    return pd.read_csv(path)


# ---------------------------------------------------------------- main
def main():
    md = []
    add = md.append

    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "bd_clean.csv"))
    # Child a362: gender was left blank at data entry; the paper form shows female (the Zenodo file holds F).
    df.loc[(df.ID == "a362") & df.Gender.isna(), "Gender"] = "F"
    df["sex"] = df.Gender.astype(str).str.strip().str.lower()

    add("# Numbers reported in the paper\n")
    add("Generated by `analysis/paper_numbers.py` from the locked run. Brackets are 95% intervals. "
        "School codes S01 to S12 are those used in the paper; special schools are S01, S04, S08, S09 and S10.\n")

    # ---- Table I
    sp, mn = df[df.label == 1], df[df.label == 0]
    sa = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "split_assignment.csv"))
    part = df[["ID", "label"]].merge(sa, left_on="ID", right_on="child_id")
    dv = lambda lab: f"{((part.label == lab) & (part.partition == 'development')).sum()} / {((part.label == lab) & (part.partition == 'test')).sum()}"
    add("## Cohort (Table I)\n")
    add(mdtable(["", "Special schools", "Mainstream schools"], [
        ["Children", len(sp), len(mn)],
        ["Age, mean (SD)", f"{sp.Age.mean():.2f} ({sp.Age.std():.2f})", f"{mn.Age.mean():.2f} ({mn.Age.std():.2f})"],
        ["Aged 4 to 7 years", (sp.Age <= 7).sum(), (mn.Age <= 7).sum()],
        ["Boys / girls", f"{(sp.sex == 'm').sum()} / {(sp.sex == 'f').sum()}", f"{(mn.sex == 'm').sum()} / {(mn.sex == 'f').sum()}"],
        ["AQ-10 total, mean (SD)", f"{sp.AQ_score_legacy.mean():.2f} ({sp.AQ_score_legacy.std():.2f})",
         f"{mn.AQ_score_legacy.mean():.2f} ({mn.AQ_score_legacy.std():.2f})"],
        ["AQ-10 rule positive (S ≥ 6)", sp.AQ_rule_pred.sum(), mn.AQ_rule_pred.sum()],
        ["Development / test set", dv(1), dv(0)]]))
    add("")
    ex_path = os.path.join(ROOT, "phase0_qc", "exclusion_log.csv")
    if os.path.exists(ex_path):
        ex = pd.read_csv(ex_path).set_index("Stage").Count_Change
        add(f"Forms received {ex.filter(like='Total Raw').iloc[0]}; excluded for age outside 4 to 11 years "
            f"{-ex.filter(like='Age').iloc[0]}; no other exclusions.\n")

    # ---- Model selection and Table II (locked run)
    cv = locked("rq1_baseline", "tables", "T_rq1_best_config_per_algo_cv.csv")
    add("## Model selection (Section IV-B)\n")
    rows = []
    for _, r in cv.sort_values("Mean_CV_AUROC", ascending=False).iterrows():
        hp = ", ".join(f"{k} = {v}" for k, v in json.loads(r.Hyperparameters).items())
        rows.append([r.Algorithm, hp, f"{r.Mean_CV_AUROC:.3f} ({r.Std_CV_AUROC:.3f})"])
    add(mdtable(["Learner", "Best setting", "Mean CV AUROC (SD)"], rows))
    add("")

    perf = locked("rq1_baseline", "tables", "paper_ready", "T_rq1_paper_Table1_screening_performance.csv")
    cal = locked("rq1_baseline", "tables", "T_rq1_secondary_mstar_auroc_calibration.csv").iloc[0]
    svm, rul = perf.iloc[0], perf.iloc[1]
    t = pd.read_csv(os.path.join(ROOT, "rq1_baseline", "predictions", "test_predictions.csv"))
    yt, pt = t.label.values, t.prob_SVM.values
    yht, rt = (pt >= 0.5).astype(int), t.AQ_rule_pred.values
    m_all, m_sp, m_mn = mcnemar(yht, rt), mcnemar(yht[yt == 1], rt[yt == 1]), mcnemar(yht[yt == 0], rt[yt == 0])
    dl_t = delong(yt, pt, t.AQ_score_legacy.values)
    add("## Hold-out test, 169 children (Table II)\n")
    add(mdtable(["Metric", "SVM", "AQ-10 rule (S ≥ 6)"], [
        ["TP / FP / FN / TN", f"{svm.TP} / {svm.FP} / {svm.FN} / {svm.TN}", f"{rul.TP} / {rul.FP} / {rul.FN} / {rul.TN}"],
        *[[m, r3(svm[c]), r3(rul[c])] for m, c in [("Sensitivity", "Sensitivity"), ("Specificity", "Specificity"),
                                                  ("PPV", "PPV"), ("NPV", "NPV"), ("Accuracy", "Accuracy")]],
        ["AUROC", r3(cal.AUROC_Formatted), f"{cal.AQ10_instrument_AUROC_context_only:.3f} (total score)"],
        ["Brier score / ECE", f"{r3(cal.Brier_Formatted)} / {r3(cal.ECE_Formatted)}", "n/a"]]))
    add("")
    add(f"- McNemar, all children (primary): SVM only {m_all[0]}, rule only {m_all[1]}, p = {pval(m_all[3])}")
    add(f"- McNemar, special-school children: {m_sp[0]} vs {m_sp[1]}, p = {pval(m_sp[3])}; "
        f"mainstream children: {m_mn[0]} vs {m_mn[1]}, p = {pval(m_mn[3])}")
    add(f"- DeLong, SVM vs AQ-10 total score: {f3(dl_t[0])} vs {f3(dl_t[1])}, p = {pval(dl_t[3])}")
    add(f"- Calibration slope {cal_slope(yt, pt):.2f}\n")

    # ---- Table III
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
    B = lambda k: ci(q[k].iloc[0], q[k].iloc[1])

    def col(pr, k=None):
        tp, fn = pr[y == 1].sum(), (1 - pr[y == 1]).sum()
        fp, tn = pr[y == 0].sum(), (1 - pr[y == 0]).sum()
        vals = [tp / (tp + fn), tn / (tn + fp), tp / (tp + fp), (tp + tn) / len(y)]
        txt = [f3(v) for v in vals]
        if k is not None:
            txt = [f"{t} {B(b)}" for t, b in zip(txt, [f"se{k}", f"sp{k}", f"ppv{k}", f"acc{k}"])]
        return [f"{tp} / {fn}", f"{fp} / {tn}"] + txt

    cs, c6, c5, c4 = col(yh, ""), col(cut[6], 6), col(cut[5]), col(cut[4])
    add("## Unseen schools, leave-one-school-out, 843 children (Table III)\n")
    add("Intervals are school-clustered bootstrap intervals (2,000 resamples of schools within each group).\n")
    names = ["TP / FN (special)", "FP / TN (mainstream)", "Sensitivity", "Specificity", "PPV", "Accuracy"]
    add(mdtable(["Metric", "SVM", "S ≥ 6 (rule)", "S ≥ 5", "S ≥ 4"],
                [[n, a, b, c, e] for n, a, b, c, e in zip(names, cs, c6, c5, c4)]))
    add("")
    dl_t, dl_o, dl_ot = delong(y, p, tot), delong(y, p, osum), delong(y, osum, tot)
    add(f"- AUROC: SVM {f3(dl_t[0])} {B('auc')}, AQ-10 total {f3(dl_t[1])} {B('auc_t')}, ordinal sum {f3(dl_o[1])}")
    add(f"- SVM vs total: DeLong p = {pval(dl_t[3])}, school-clustered difference {B('d_auc_t')}; "
        f"SVM vs ordinal sum: p = {pval(dl_o[3])}, difference {B('d_auc_o')}; ordinal sum vs total: p = {pval(dl_ot[3])}")
    for k, lab in [(6, "rule"), (5, "S ≥ 5")]:
        a, b = mcnemar(yh[y == 1], cut[k][y == 1])[:2], mcnemar(yh[y == 0], cut[k][y == 0])[:2]
        add(f"- Discordant children, SVM only vs {lab} only: special {a[0]} vs {a[1]}, mainstream {b[0]} vs {b[1]}")
    add(f"- Sensitivity difference, SVM minus rule {B('d_se6')}; specificity difference {B('d_sp6')}; "
        f"sensitivity difference, SVM minus S ≥ 5 {B('d_se5')}")
    add(f"- Calibration: Brier {brier_score_loss(y, p):.3f}, ECE {ece(y, p):.3f}, slope {cal_slope(y, p):.2f}")
    youden = []
    for sc in sorted(d.school_id.unique()):
        tr = d.school_id.values != sc
        j = [((tot[tr] >= c)[y[tr] == 1].mean() - (tot[tr] >= c)[y[tr] == 0].mean(), c) for c in range(11)]
        youden.append(max(j)[1])
    yc = sorted(set(youden))
    add(f"- Youden-optimal cut-off on the training schools: {yc[0]} in all 12 folds\n" if len(yc) == 1
        else f"- Youden-optimal cut-offs on the training schools: {youden}\n")

    add("### Children flagged in each held-out school (Fig. 2(a))\n")
    fl = o.assign(svm=yh, rule=cut[6]).groupby("school_id").agg(
        lab=("label", "first"), n=("label", "size"), svm=("svm", "sum"), rule=("rule", "sum"))
    fl = fl.sort_values(["lab", "school_id"], ascending=[False, True])
    add(mdtable(["School", "Group", "Children", "Flagged by SVM", "Flagged by rule"],
                [[code(s), "special" if r.lab == 1 else "mainstream", r.n, r.svm, r.rule] for s, r in fl.iterrows()]))
    add("")

    # ---- Post hoc refits
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
    add("### Post hoc refits (Section IV-C and IV-G)\n")
    add(f"- A refit with scikit-learn {__import__('sklearn').__version__} reproduced "
        f"{int(((p10 >= .5) == (p >= .5)).sum())} of {len(p)} locked decisions")
    add(f"- Logistic regression (L2, C = 0.1): AUROC {roc_auc_score(y, plr):.3f}, vs AQ-10 total p = {pval(delong(y, plr, tot)[3])}; "
        f"special-school children found by the SVM only vs logistic regression only: "
        f"{int(((yh == 1) & (ylr == 0) & (y == 1)).sum())} vs {int(((yh == 0) & (ylr == 1) & (y == 1)).sum())}")
    add(f"- SVM without item 1: sensitivity {y9[y == 1].mean():.3f}, specificity {1 - y9[y == 0].mean():.3f}, "
        f"AUROC {roc_auc_score(y, p9):.3f} (difference from ten items {ci(*np.percentile(dd, [2.5, 97.5]))}); "
        f"flagged in S06 {y9[s06].sum()} (ten items {yh[s06].sum()}); "
        f"false positives in the other mainstream schools {y9[(y == 0) & ~s06].sum()} (ten items {yh[(y == 0) & ~s06].sum()})\n")

    # ---- Ten vs fourteen items (locked run)
    pc = locked("rq3_augmentation", "tables", "paper_ready", "T_rq3_paper_paired_comparison.csv").iloc[0]
    bd = locked("rq3_augmentation", "tables", "paper_ready", "T_rq3_paper_bootstrap_deltas.csv").set_index("Metric")
    add("## Ten versus fourteen items (Section IV-D)\n")
    add(mdtable(["", "10 items", "14 items"], [
        ["AUROC", f"{pc.AUROC_S0:.4f}", f"{pc.AUROC_S1:.4f}"],
        ["Sensitivity", f3(pc.Sensitivity_S0), f3(pc.Sensitivity_S1)],
        ["Specificity", f3(pc.Specificity_S0), f3(pc.Specificity_S1)]]))
    add("")
    add(f"- Paired DeLong p = {pc.DeLong_P:.2f}; paired McNemar p = {pc.McNemar_P:.2f}; "
        f"sensitivity difference {r3(bd.loc['ΔSensitivity', 'Formatted'])} (paired bootstrap over children)\n")

    # ---- Item importance (locked run)
    fi = locked("rq4", "tables", "paper_ready", "T_rq4_paper_feature_importance.csv")
    rho_pi = locked("rq4", "tables", "paper_ready", "T_rq4_paper_shap_vs_pi.csv").Value.iloc[0]
    pw = locked("rq4", "tables", "T_rq4_spearman_pairwise.csv")
    keep = (pw.School_A != "SCH_006") & (pw.School_B != "SCH_006")
    add("## What the model relies on (Section IV-E, Fig. 3)\n")
    add(mdtable(["Item", "Mean abs. SHAP", "Permutation importance (AUROC drop)"],
                [[ITEM[r.Feature], r4(r["SHAP Mean|SHAP| [95% CI]"]), r4(r["PI Mean AUROC-drop [95% CI]"])]
                 for _, r in fi.iterrows()]))
    add("")
    sh = pd.read_csv(os.path.join(ROOT, "rq4", "tables", "T_rq4_shap_values_per_instance.csv"))
    d2 = d.assign(yh=yh).merge(sh[["ID"] + [f"SHAP_{a}" for a in AQ]], on="ID")
    end_diff = [(d2[a][d2.label == 1] >= 2).mean() - (d2[a][d2.label == 0] >= 2).mean() for a in AQ]
    rho_end = stats.spearmanr([d2[f"SHAP_{a}"].abs().mean() for a in AQ], end_diff).statistic
    add(f"- Spearman ρ, SHAP vs permutation ranking {rho_pi:.2f}; SHAP ranking vs group difference in endorsement {rho_end:.2f}")
    add(f"- Rank stability across held-out schools: mean pairwise ρ {pw.Spearman_rho.mean():.2f} "
        f"(range {pw.Spearman_rho.min():.2f} to {pw.Spearman_rho.max():.2f}); without S06 {pw.Spearman_rho[keep].mean():.2f} "
        f"(range {pw.Spearman_rho[keep].min():.2f} to {pw.Spearman_rho[keep].max():.2f})")
    e1 = lambda m: (d2.A_Q1[m] >= 2).mean()
    add(f"- Item 1 endorsed: special schools {e1(d2.label == 1):.3f}, mainstream schools {e1(d2.label == 0):.3f}\n")

    # ---- S06 (Fig. 4)
    f6 = (d2.school_id == "SCH_006") & (d2.yh == 1)
    other = (d2.label == 0) & (d2.school_id != "SCH_006")
    shap6 = d2.loc[f6, [f"SHAP_{a}" for a in AQ]].mean()
    add("## False positives in school S06 (Section IV-G, Fig. 4)\n")
    add(f"- Flagged by the SVM {int(f6.sum())} of {int((d2.school_id == 'SCH_006').sum())}; by the rule "
        f"{int(((d2.school_id == 'SCH_006') & (d2.AQ_score_legacy >= 6)).sum())}; AQ-10 totals of those flagged "
        f"{d2.AQ_score_legacy[f6].min()} to {d2.AQ_score_legacy[f6].max()}")
    add(f"- Answered definitely disagree to item 1: {int((d2.Q1[f6] == 4).sum())} of the {int(f6.sum())} flagged; "
        f"{int((d2.Q1[other] == 4).sum())} of {int(other.sum())} children in the other mainstream schools; "
        f"{int((d2.Q1[d2.label == 1] == 4).sum())} of {int((d2.label == 1).sum())} special-school children")
    add(f"- Mean SHAP value among the {int(f6.sum())} flagged: item 1 {shap6['SHAP_A_Q1']:+.2f}; "
        f"largest other item {shap6.drop('SHAP_A_Q1').max():+.2f}")
    add(f"- False positives outside S06: SVM {int(d2.yh[other].sum())}, rule {int((d2.AQ_score_legacy[other] >= 6).sum())}, "
        f"of {int(other.sum())}\n")

    # ---- Subgroups and prevalence
    d2["rule"] = (d2.AQ_score_legacy >= 6).astype(int)
    d2["sex"] = d2.Gender.astype(str).str.strip().str.lower()
    add("## Subgroups and screening prevalence (Section IV-F)\n")
    rows = []
    for name, mk in [("Aged 4 to 7", d2.Age <= 7), ("Aged 8 to 11", d2.Age >= 8), ("Boys", d2.sex == "m"), ("Girls", d2.sex == "f")]:
        s_, m_ = mk & (d2.label == 1), mk & (d2.label == 0)
        a, b = wilson(d2.yh[s_].sum(), s_.sum()), wilson((1 - d2.yh[m_]).sum(), m_.sum())
        rows.append([name, s_.sum(), f"{f3(a[0])} {ci(a[1], a[2])}", m_.sum(), f"{f3(b[0])} {ci(b[1], b[2])}"])
    add(mdtable(["Subgroup", "Special", "Sensitivity", "Mainstream", "Specificity"], rows))
    add("")
    young = (d2.label == 0) & (d2.Age <= 7)
    rest = young & (d2.school_id != "SCH_006")
    add(f"- Younger mainstream children flagged {int(d2.yh[young].sum())}, of whom {int(d2.yh[young & (d2.school_id == 'SCH_006')].sum())} "
        f"in S06; specificity at 5 to 7 years outside S06 {1 - d2.yh[rest].mean():.3f} (rule {1 - d2.rule[rest].mean():.3f})")
    dx = d2.doctor_dx.astype(str).str.strip().str.lower()
    for v, lab in [("yes", "reported"), ("no", "not reported")]:
        s_ = (d2.label == 1) & (dx == v)
        a = wilson(d2.yh[s_].sum(), s_.sum())
        add(f"- Prior diagnosis {lab} (n = {s_.sum()}): sensitivity {f3(a[0])} {ci(a[1], a[2])}, rule {d2.rule[s_].mean():.2f}")
    add("")
    se, sp_ = d2.yh[d2.label == 1].mean(), 1 - d2.yh[d2.label == 0].mean()
    se_r, sp_r = d2.rule[d2.label == 1].mean(), 1 - d2.rule[d2.label == 0].mean()
    rows = []
    for pi in (0.01, 0.03, 0.05):
        ref, ref_r = se * pi + (1 - sp_) * (1 - pi), se_r * pi + (1 - sp_r) * (1 - pi)
        rows.append([f"{pi:.0%}", f"{100 * se * pi / ref:.1f}%", f"{100 * se_r * pi / ref_r:.1f}%",
                     f"{1000 * ref:.1f} / {1000 * ref_r:.1f}", f"{1000 * pi * se:.1f} / {1000 * pi * se_r:.1f}"])
    add(mdtable(["Prevalence", "PPV, SVM", "PPV, rule", "Referred per 1,000 (SVM / rule)", "Cases found per 1,000 (SVM / rule)"], rows))

    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
