"""Rebuild the locked Phase-0 outputs from the Zenodo file.

The original cleaner started from school spreadsheets, which contain school
names and are not shared. Approved users of the BD-AQ10 dataset
(doi: 10.5281/zenodo.23988059) receive full_dataset.csv instead, with every form
received (863 children). This script keeps the children aged
4 to 11 (843) and turns them into the exact inputs used by the analysis steps:

    <AQ10_BASE>/aq_10/data/processed/bd_clean.csv
    <AQ10_BASE>/aq_10/data/splits/train.csv      (development set, 674 children)
    <AQ10_BASE>/aq_10/data/splits/test.csv       (hold-out set, 169 children)
    <AQ10_BASE>/aq_10/config/master_config.json
    <AQ10_BASE>/aq_10/config/cv_fold_indices.json
    <AQ10_BASE>/aq_10/logs/phase0_done.flag
    <AQ10_BASE>/aq_10/figures/study_design.png

Row order, the development/test split and the five cross-validation folds come
from config/split_assignment.csv, so the downstream scripts see the same rows in
the same order as in the locked run. School names are replaced by school codes.

AQ10_BASE is the working folder. If it is not set, every script uses the folder
work/ inside this repository (ignored by git). Nothing is written anywhere else.

Usage:
    python pipeline/prepare_from_zenodo.py path/to/full_dataset.csv
    python pipeline/prepare_from_zenodo.py        # asks for the path
    %run pipeline/prepare_from_zenodo.py          # in a Colab notebook: shows an upload button
"""
import json
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
AGE_RANGE = (4, 11)
FORWARD = ["Q1", "Q5", "Q7", "Q10", "SQ1", "SQ2", "SQ3"]
REVERSE = ["Q2", "Q3", "Q4", "Q6", "Q8", "Q9", "SQ4"]
AQ_ITEMS = [f"Q{i}" for i in range(1, 11)]
SQ_ITEMS = [f"SQ{i}" for i in range(1, 5)]
COLUMNS = (["ID", "source_form", "school_id", "school_name", "school_type", "consent", "Age", "Gender",
            "family_hx", "doctor_dx"] + AQ_ITEMS + SQ_ITEMS + ["label"]
           + [f"A_{c}" for c in FORWARD] + [f"A_{c}" for c in REVERSE]
           + ["AQ_score_legacy", "AQ_rule_pred"])


def build(full_dataset_csv, base):
    z = pd.read_csv(full_dataset_csv)
    need = {"child_id", "school_code", "group", "age", "gender", "family_hx", "doctor_dx"} | set(AQ_ITEMS + SQ_ITEMS)
    missing = need - set(z.columns)
    if missing:
        sys.exit(f"full_dataset.csv is missing columns: {sorted(missing)}")
    if z.child_id.duplicated().any():
        sys.exit("full_dataset.csv has duplicate child_id values")
    bad = ~z[AQ_ITEMS + SQ_ITEMS].isin([1, 2, 3, 4]).all(axis=1)
    if bad.any():
        sys.exit(f"{int(bad.sum())} rows have item codes outside 1-4")

    group = z.group.astype(str).str.strip().str.lower()
    unknown = sorted(set(group) - {"asd", "special", "mainstream"})
    if unknown:
        sys.exit(f"Unexpected values in the group column: {unknown} (expected ASD or mainstream)")
    asd = group.isin(["asd", "special"])                  # ASD school; the file first uploaded said "special"

    in_range = z.age.between(*AGE_RANGE)
    received = (int(asd.sum()), int((~asd).sum()))        # ASD schools, mainstream schools
    excluded = (int((asd & ~in_range).sum()), int((~asd & ~in_range).sum()))
    z, asd = z[in_range], asd[in_range]

    d = pd.DataFrame({
        "ID": z.child_id,
        "source_form": asd.map({True: "ASD.xlsx", False: "Normal.xlsx"}),
        "school_id": "SCH_" + z.school_code.str[1:].str.zfill(3),
        "school_name": z.school_code,                      # names withheld
        "school_type": asd.map({True: "asd", False: "normal"}),
        "consent": "yes",
        "Age": z.age.astype(int),
        "Gender": z.gender,
        "family_hx": z.family_hx,
        "doctor_dx": z.doctor_dx,
    })
    for c in AQ_ITEMS + SQ_ITEMS:
        d[c] = z[c].astype(int)
    d["label"] = asd.astype(int)
    for c in FORWARD:                                      # aligned 0 (no trait) to 3 (strongest trait)
        d[f"A_{c}"] = 4 - d[c]
    for c in REVERSE:
        d[f"A_{c}"] = d[c] - 1
    d["AQ_score_legacy"] = (d[[f"A_{c}" for c in AQ_ITEMS]] >= 2).sum(axis=1)
    d["AQ_rule_pred"] = (d.AQ_score_legacy >= 6).astype(int)
    d = d[COLUMNS]

    sa = pd.read_csv(os.path.join(HERE, "config", "split_assignment.csv"))
    if set(sa.child_id) != set(d.ID):
        sys.exit("child_id values in full_dataset.csv do not match config/split_assignment.csv")
    d = d.set_index("ID")
    bd = d.loc[sa.sort_values("bd_clean_row").child_id].reset_index()
    dev = d.loc[sa[sa.partition == "development"].sort_values("split_row").child_id].reset_index()
    test = d.loc[sa[sa.partition == "test"].sort_values("split_row").child_id].reset_index()

    proj = os.path.join(base, "aq_10")
    for sub in ("data/processed", "data/splits", "config", "logs"):
        os.makedirs(os.path.join(proj, sub), exist_ok=True)
    bd.to_csv(os.path.join(proj, "data/processed/bd_clean.csv"), index=False)
    dev.to_csv(os.path.join(proj, "data/splits/train.csv"), index=False)
    test.to_csv(os.path.join(proj, "data/splits/test.csv"), index=False)
    for f in ("master_config.json", "cv_fold_indices.json"):
        shutil.copy(os.path.join(HERE, "config", f), os.path.join(proj, "config", f))
    with open(os.path.join(proj, "logs", "phase0_done.flag"), "w") as fh:
        fh.write("phase0 rebuilt from Zenodo full_dataset.csv by prepare_from_zenodo.py\n"
                 f"n={len(bd)} dev={len(dev)} test={len(test)}\n")

    folds = json.load(open(os.path.join(HERE, "config", "cv_fold_indices.json")))
    val = sorted(sum((folds[f"fold_{k}"]["val_indices"] for k in range(1, 6)), []))
    assert val == list(range(len(dev))), "fold indices do not cover the development set"
    print(f"Wrote {proj}")
    print(f"  forms {sum(received)}, left out for age outside {AGE_RANGE[0]} to {AGE_RANGE[1]}: {sum(excluded)} "
          f"(ASD school {excluded[0]}, mainstream {excluded[1]})")
    print(f"  children {len(bd)} (ASD school {int(bd.label.sum())}, mainstream {int((1 - bd.label).sum())})")
    print(f"  development {len(dev)}, test {len(test)}, AQ-10 rule positive {int(bd.AQ_rule_pred.sum())}")
    design = os.path.join(proj, "figures", "study_design.png")
    draw_study_design(bd, dev, test, design, received, excluded)
    print(f"  [figure] {design}")
    return design


def draw_study_design(bd, dev, test, out_path, received, excluded):
    """Study design, with the cohort counts taken from the data."""
    import logging

    import matplotlib
    from matplotlib.figure import Figure                  # no pyplot: safe inside a notebook kernel
    from matplotlib.patches import FancyBboxPatch, Rectangle

    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    cnt = lambda d: (int(d.label.sum()), int((1 - d.label).sum()))
    a_all, a_dev, a_test = cnt(bd), cnt(dev), cnt(test)
    sizes = bd.school_id.value_counts()
    n_sch = len(sizes)
    n_asd_sch = bd.loc[bd.label == 1, "school_id"].nunique()
    words = {12: "Twelve"}
    style = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"], "font.size": 7}
    with matplotlib.rc_context(style):
        W, H = 7.16, 3.55
        fig = Figure(figsize=(W, H))
        ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
        INK = "#333333"; BLUE = "#2b5c8a"; BLUEF = "#eaf0f7"; GREEN = "#2e6e4e"; GREENF = "#eaf4ee"

        def box(x, y, w, h, title, lines, fc="white", ec=INK, lw=0.8, ts=7.3, ls=6.8, dashed=False):
            ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", fc=fc, ec=ec, lw=lw,
                                        ls="--" if dashed else "-"))
            n = len(lines) + (1 if title else 0); step = 0.125; top = y + h / 2 + (n - 1) * step / 2
            k = 0
            if title:
                ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=ts, fontweight="bold", color=INK); k = 1
            for i, l in enumerate(lines):
                ax.text(x + w / 2, top - (i + k) * step, l, ha="center", va="center", fontsize=ls, color=INK)

        def arrow(x0, y0, x1, y1, c=INK):
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=7))

        def line(xs, ys, c=INK):
            ax.plot(xs, ys, color=c, lw=0.8, solid_capstyle="butt")

        y1, h1 = 2.93, 0.50; bw = 1.27; gap = (W - 0.10 - 5 * bw) / 4; xs = [0.05 + i * (bw + gap) for i in range(5)]
        r1 = [(f"{words.get(n_sch, n_sch)} schools", ["Bangladesh", f"{n_asd_sch} ASD, {n_sch - n_asd_sch} mainstream"]),
              ("Forms received", [f"n = {sum(received)}", f"{received[0]} / {received[1]}"]),
              ("Analysis cohort", [f"n = {len(bd)}, aged {bd.Age.min()} to {bd.Age.max()}", f"{a_all[0]} / {a_all[1]}"]),
              ("Predictors", ["AQ-10 Child, Q1 to Q10", "aligned 0 to 3"]),
              ("Label (proxy) and rule", ["ASD-school placement", "AQ-10 score ≥ 6"])]
        for x, (t, l) in zip(xs, r1):
            box(x, y1, bw, h1, t, l)
        for i in range(4):
            arrow(xs[i] + bw, y1 + h1 / 2, xs[i + 1], y1 + h1 / 2)
        ax.text(xs[0] + 0.02, y1 - 0.09, "Counts are ASD / mainstream", fontsize=6.2, style="italic", color="#666666", va="center")
        ax.text(xs[3] + bw / 2, y1 - 0.09, "SQ1 to SQ4 added for the 14-item set", fontsize=6.2, style="italic",
                color="#666666", va="center", ha="center")
        ex_x = xs[1] + bw + gap / 2; line([ex_x, ex_x], [y1 + h1 / 2, y1 - 0.17])
        box(ex_x - 0.62, y1 - 0.47, 1.24, 0.30, None,
            [f"Excluded: age outside {bd.Age.min()} to {bd.Age.max()}",
             f"n = {sum(excluded)} ({excluded[0]} / {excluded[1]})"],
            fc="#f6f6f6", ec="#888888", dashed=True, ls=6.4)
        arrow(ex_x, y1 - 0.17, ex_x, y1 - 0.17)
        cx = xs[2] + bw / 2; yb = 2.28; line([cx, cx], [y1, yb])
        pA = (0.05, 0.98, 3.50, 1.22); pB = (3.61, 0.98, 3.50, 1.22)
        line([pA[0] + pA[2] / 2, pB[0] + pB[2] / 2], [yb, yb])
        arrow(pA[0] + pA[2] / 2, yb, pA[0] + pA[2] / 2, pA[1] + pA[3]); arrow(pB[0] + pB[2] / 2, yb, pB[0] + pB[2] / 2, pB[1] + pB[3])
        ax.add_patch(FancyBboxPatch(pA[:2], pA[2], pA[3], boxstyle="round,pad=0,rounding_size=0.05", fc=BLUEF, ec="#8fa9c6", lw=0.9))
        ax.text(pA[0] + 0.10, pA[1] + pA[3] - 0.12, "A. Internal test-set validation", fontsize=7.6, fontweight="bold", color=BLUE, va="center")
        bx, by, bwid, bh = pA[0] + 0.12, pA[1] + 0.77, pA[2] - 0.24, 0.17
        ax.add_patch(Rectangle((bx, by), bwid * 0.8, bh, fc="#c5d5e8", ec="#8fa9c6", lw=0.6))
        ax.add_patch(Rectangle((bx + bwid * 0.8, by), bwid * 0.2, bh, fc=BLUE, ec="#8fa9c6", lw=0.6))
        ax.text(bx + bwid * 0.4, by + bh / 2, f"Development set, n = {len(dev)} ({a_dev[0]} / {a_dev[1]})",
                ha="center", va="center", fontsize=6.6, color=INK)
        ax.text(bx + bwid * 0.9, by + bh / 2, f"Test, n = {len(test)}", ha="center", va="center", fontsize=6.6, color="white")
        ax.text(pA[0] + pA[2] / 2, by - 0.09, "Stratified 80/20 split and five folds fixed with a seed before training",
                ha="center", va="center", fontsize=6.1, style="italic", color="#555555")
        sw = 1.02; sg = (pA[2] - 0.24 - 3 * sw) / 2; sy, sh = pA[1] + 0.08, 0.48
        A3 = [("Model selection", ["5 learners, 86 settings", "5-fold CV by AUROC"]),
              ("Locked model", ["RBF SVM, C = 1", "fixed threshold 0.50"]),
              ("Single test", [f"{a_test[0]} / {a_test[1]} children", "evaluated once"])]
        ax_ = [pA[0] + 0.12 + i * (sw + sg) for i in range(3)]
        for x, (t, l) in zip(ax_, A3):
            box(x, sy, sw, sh, t, l, ls=6.3, ts=7)
        for i in range(2):
            arrow(ax_[i] + sw, sy + sh / 2, ax_[i + 1], sy + sh / 2)
        ax.add_patch(FancyBboxPatch(pB[:2], pB[2], pB[3], boxstyle="round,pad=0,rounding_size=0.05", fc=GREENF, ec="#93bba3", lw=0.9))
        ax.text(pB[0] + 0.10, pB[1] + pB[3] - 0.12, "B. Leave-one-school-out validation", fontsize=7.6, fontweight="bold", color=GREEN, va="center")
        ax.text(pB[0] + 0.10, pB[1] + pB[3] - 0.29, "Hyper-parameters locked from arm A", fontsize=6.1, style="italic", color="#555555", va="center")
        g0x, g0y = pB[0] + 0.30, pB[1] + 0.07
        cs = 0.612 / n_sch
        for i in range(n_sch):
            for j in range(n_sch):
                ax.add_patch(Rectangle((g0x + j * cs, g0y + (n_sch - 1 - i) * cs), cs * 0.92, cs * 0.92,
                                       fc=GREEN if i == j else "#d3e6da", ec="none"))
        ax.text(g0x + n_sch * cs / 2, g0y + n_sch * cs + 0.06, f"Schools S01 to S{n_sch:02d}", ha="center", va="center", fontsize=6.1, color="#555555")
        ax.text(g0x - 0.09, g0y + n_sch * cs / 2, f"Folds 1 to {n_sch}", rotation=90, ha="center", va="center", fontsize=6.1, color="#555555")
        lx = g0x + n_sch * cs + 0.12
        ax.add_patch(Rectangle((lx, g0y + 0.36), 0.08, 0.08, fc=GREEN, ec="none")); ax.text(lx + 0.12, g0y + 0.40, "left-out school", fontsize=6.3, va="center")
        ax.add_patch(Rectangle((lx, g0y + 0.18), 0.08, 0.08, fc="#d3e6da", ec="none")); ax.text(lx + 0.12, g0y + 0.22, "training schools", fontsize=6.3, va="center")
        rb_x = pB[0] + pB[2] - 1.30
        box(rb_x, pB[1] + 0.46, 1.20, 0.44, "Refit per fold",
            [f"on {n_sch - 1} schools", f"{len(bd) - sizes.max()} to {len(bd) - sizes.min()} children"], ls=6.3, ts=7)
        box(rb_x, pB[1] + 0.05, 1.20, 0.34, "Pooled predictions", [f"out-of-fold, n = {len(bd)}"], ls=6.3, ts=7)
        arrow(rb_x + 0.60, pB[1] + 0.46, rb_x + 0.60, pB[1] + 0.39)
        y3, h3 = 0.04, 0.52; aw = (W - 0.10 - 4 * 0.07) / 5; axs = [0.05 + i * (aw + 0.07) for i in range(5)]
        A = [("Model versus AQ-10 rule", ["McNemar, child and school", "DeLong test, bootstrap CIs"]),
             ("Lower cut-offs (post hoc)", ["score ≥ 5, matched specificity", "score ≥ 4 by Youden index"]),
             ("Supplementary items", ["10 vs 14 items, same children", "pre-specified adoption rule"]),
             ("Explanation and audit", ["Kernel SHAP, permutation", "importance, subgroups"]),
             ("School effects (post hoc)", ["seen vs unseen schools,", "local children, nested tuning"])]
        for x, (t, l) in zip(axs, A):
            box(x, y3, aw, h3, t, l, ls=6.1, ts=6.8)
        yA = 0.80
        line([pA[0] + pA[2] / 2, pA[0] + pA[2] / 2], [pA[1], yA + 0.07], BLUE)
        line([axs[0] + aw * 0.35, pA[0] + pA[2] / 2], [yA + 0.07, yA + 0.07], BLUE)
        arrow(axs[0] + aw * 0.35, yA + 0.07, axs[0] + aw * 0.35, y3 + h3, BLUE)
        bcx = pB[0] + pB[2] / 2
        line([axs[0] + aw * 0.65, axs[4] + aw / 2], [yA, yA], GREEN); line([bcx, bcx], [pB[1], yA], GREEN)
        for k, x in enumerate(axs):
            xx = x + aw * 0.65 if k == 0 else x + aw / 2
            arrow(xx, yA, xx, y3 + h3, GREEN)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=600)


def ask_for_dataset(base):
    """Get full_dataset.csv from the user: an upload button in a Colab notebook, otherwise a typed path."""
    try:
        from IPython import get_ipython
        in_colab = get_ipython() is not None and "google.colab" in sys.modules
    except ImportError:
        in_colab = False
    if in_colab:
        from google.colab import files
        print("Choose full_dataset.csv from your computer.")
        uploaded = files.upload()
        if len(uploaded) != 1:
            sys.exit("Upload exactly one file: full_dataset.csv")
        name, content = next(iter(uploaded.items()))
        if os.path.isfile(name):                     # files.upload() also saves a copy in the current folder
            os.remove(name)
        dest = os.path.join(base, "input", "full_dataset.csv")   # private working folder, never the repository
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(content)
        print(f"Saved the upload as {dest}")
        return dest
    try:
        return input("Path to full_dataset.csv (or the folder that holds it): ")
    except EOFError:
        sys.exit(__doc__)


def resolve(path):
    path = os.path.expanduser(path.strip().strip('"').strip("'"))
    if os.path.isdir(path):
        path = os.path.join(path, "full_dataset.csv")
    if not os.path.isfile(path):
        sys.exit(f"File not found: {path}")
    return path


if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit(__doc__)
    base = os.environ.get("AQ10_BASE", os.path.join(HERE, "work"))
    source = sys.argv[1] if len(sys.argv) == 2 else ask_for_dataset(base)
    design = build(resolve(source), base)
    try:                                             # in a notebook, show the figure below the cell
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import Image, display
            display(Image(filename=design, width=900))
    except ImportError:
        pass
