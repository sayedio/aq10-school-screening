"""Fig. 2 of the paper: AQ-10 total score by group, and children flagged by the SVM in unseen schools at each score.

Run after rq2_loso.py, with AQ10_BASE pointing at the folder that holds the private aq_10 project folder:

    python analysis/make_figure_scores.py [output.png]   (default fig2_scores.png, 600 dpi)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = os.path.join(os.environ.get("AQ10_BASE", os.path.join(os.getcwd(), "work")), "aq_10")
MIN_N = 5                                    # scores held by fewer children of a group are not plotted in (b)
ASD_C, MN_C, RULE_C, INK = "#CC79A7", "#56B4E9", "#D55E00", "#333333"

o = pd.read_csv(os.path.join(ROOT, "rq2_loso", "predictions", "rq2_loso_s0_predictions.csv"))
score, lab, svm = o.AQ_score_legacy.values.astype(int), o.label.values, o.y_pred.values
S = np.arange(11)

plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 2.0))
fig.subplots_adjust(left=0.105, right=0.985, bottom=0.185, top=0.8, wspace=0.42)

# (a) distribution of the total score in each group
for k, (g, c, name) in enumerate(((1, ASD_C, "ASD schools"), (0, MN_C, "Mainstream"))):
    pct = [100 * np.mean(score[lab == g] == s) for s in S]
    a.bar(S + (k - 0.5) * 0.4, pct, 0.4, color=c, label=name, lw=0)
a.axvline(5.5, color=INK, ls="--", lw=0.7)
a.text(5.65, 38, "cut-off", fontsize=6, color=INK, va="top")
a.set_xlabel("AQ-10 total score"); a.set_ylabel("Children in group (%)")
a.set_xticks(S[::2]); a.set_xlim(-0.6, 10.6); a.set_ylim(0, 40)

# (b) flagged by the SVM when the school was unseen, at each total score
for g, c, m, name in ((1, ASD_C, "o", "SVM, ASD schools"), (0, MN_C, "s", "SVM, mainstream")):
    n = np.array([np.sum((lab == g) & (score == s)) for s in S])
    f = np.array([svm[(lab == g) & (score == s)].sum() for s in S])
    keep = n >= MIN_N
    b.plot(S[keep], 100 * f[keep] / n[keep], marker=m, ms=3, color=c, lw=1.1, label=name)
b.step(S, 100 * (S >= 6), where="mid", color=RULE_C, ls="--", lw=0.9, label="AQ-10 rule")
b.set_xlabel("AQ-10 total score"); b.set_ylabel("Flagged by the SVM (%)")
b.set_xticks(S[::2]); b.set_xlim(-0.4, 10.4); b.set_ylim(-3, 103)

for ax, t in ((a, "(a)"), (b, "(b)")):
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.3, 1.06, t, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")

fig.legend(handles=[Patch(color=ASD_C, label="ASD schools"), Patch(color=MN_C, label="Mainstream schools"),
                    Line2D([], [], color=RULE_C, ls="--", lw=0.9, label="AQ-10 rule")],
           loc="upper center", ncol=3, frameon=False, fontsize=6.5, handlelength=1.4, columnspacing=1.4, bbox_to_anchor=(0.53, 1.0))
fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "fig2_scores.png", dpi=600)
print("ok")
