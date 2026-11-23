"""Run the analysis steps in order, keep a log of each, and collect all results in one zip file.

From the repository folder, after preparing the data with pipeline/prepare_from_zenodo.py:

    python run_pipeline.py all                      # every step, in order, then collect
    python run_pipeline.py rq1                      # one step
    python run_pipeline.py all --data path/to/full_dataset.csv   # prepare the data first, then everything

In a notebook (Google Colab), the same steps run with their figures shown below the cell:

    import run_pipeline as rp
    rp.run("rq1")

Steps: rq1, rq2, rq3, rq4, step5, step6, step7, collect. Each step checks that the step before it
finished. Everything is written to the working folder: AQ10_BASE if set, otherwise work/ inside
this repository (ignored by git). Nothing is written anywhere else. `collect` writes
<work>/aq10_results.zip with the figures, the summary CSV files, the step reports, the summary
tables of every step (no child-level rows), the log of every step and the software versions.
"""
import argparse
import csv
import datetime
import glob
import importlib.metadata as md
import os
import platform
import queue
import shutil
import subprocess
import sys
import threading
import time

REPO = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.environ.get("AQ10_BASE", os.path.join(REPO, "work")))
PROJECT = os.path.join(BASE, "aq_10")
LOGS = os.path.join(PROJECT, "logs")
FIGURES = os.path.join(PROJECT, "figures")
RESULTS = os.path.join(BASE, "aq10_results")

STEPS = {
    "rq1": ("pipeline/rq1_baseline.py", "RQ1: five learners, model selection, one test against the AQ-10 rule"),
    "rq2": ("pipeline/rq2_loso.py", "RQ2: leave-one-school-out validation"),
    "rq3": ("pipeline/rq3_augmentation.py", "RQ3: ten versus fourteen items"),
    "rq4": ("pipeline/rq4_explanation.py", "RQ4: Kernel SHAP, permutation importance, subgroups"),
    "step5": ("pipeline/step5_school_effects.py", "Step 5 (post hoc): school effects"),
    "step6": ("pipeline/step6_robustness.py", "Step 6 (post hoc): school-level tests, nested tuning, AQ-10 cut-offs"),
    "step7": ("pipeline/step7_summary.py", "Step 7: summary of all steps as CSV files"),
}
ORDER = list(STEPS) + ["collect"]
TIMING = {"prepare": (1, 10), "rq1": (6, 45), "rq2": (1, 15), "rq3": (1, 15), "rq4": (30, 150),
          "step5": (10, 60), "step6": (20, 120), "step7": (3, 30)}
QUIET_MINUTES = 2          # print a "still running" line after this long without output
TABLE_DIRS = {"rq1": "rq1_baseline/tables", "rq2": "rq2_loso/tables", "rq3": "rq3_augmentation/tables",
              "rq4": "rq4/tables", "step5": "step5_school_effects/tables", "step6": "step6_robustness/tables"}
CHILD_LEVEL_COLUMNS = {"ID", "child_id"}
TIMED_OUT = -999
_NO_LINE = object()


class StepError(RuntimeError):
    """A step failed or ran past its time limit."""


def _delong_pvalue(y_true, first, second):
    """Return the paired DeLong p-value used for the locked comparisons."""
    import numpy as np
    from scipy import stats

    y_true = np.asarray(y_true)
    pos, neg = np.where(y_true == 1)[0], np.where(y_true == 0)[0]
    m, n = len(pos), len(neg)
    scores = np.vstack([first, second]).astype(float)

    def midranks(values):
        order = np.argsort(values)
        ranks = np.empty(len(values), float)
        i = 0
        while i < len(values):
            j = i + 1
            while j < len(values) and values[order[j]] == values[order[i]]:
                j += 1
            ranks[order[i:j]] = (i + j + 1) / 2
            i = j
        return ranks

    tx = np.array([midranks(row[pos]) for row in scores])
    ty = np.array([midranks(row[neg]) for row in scores])
    tz = np.array([midranks(np.concatenate([row[pos], row[neg]])) for row in scores])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    contrast = np.array([1, -1])
    variance = contrast @ covariance @ contrast
    if variance == 0:
        return 1.0
    z = (aucs[0] - aucs[1]) / np.sqrt(variance)
    return float(2 * stats.norm.sf(abs(z)))


def print_locked_findings():
    """Print the additional locked-run findings not shown by the step scripts."""
    import numpy as np
    import pandas as pd

    root = os.path.join(PROJECT)
    data = pd.read_csv(os.path.join(root, "data", "processed", "bd_clean.csv"))
    items = [f"A_Q{i}" for i in range(1, 11)]
    alpha_items = data[items].astype(float)
    alpha = len(items) / (len(items) - 1) * (
        1 - alpha_items.var(ddof=1).sum() / alpha_items.sum(axis=1).var(ddof=1)
    )
    patterns = data.groupby("label")[items].apply(lambda frame: set(map(tuple, frame.to_numpy())))
    shared = len(patterns.loc[1] & patterns.loc[0])
    total_patterns = len(patterns.loc[1] | patterns.loc[0])
    corr = alpha_items.corr()
    pairs = [(abs(corr.iloc[i, j]), corr.index[i], corr.index[j], corr.iloc[i, j])
             for i in range(len(items)) for j in range(i)]
    max_corr, first_item, second_item, signed_corr = max(pairs)
    item1_negative = bool((corr.loc["A_Q1", [c for c in items if c != "A_Q1"]] < 0).all())

    shap = pd.read_csv(os.path.join(root, "rq4", "tables", "T_rq4_shap_values_per_instance.csv"))
    item1_shap = shap.groupby("feat_A_Q1")["SHAP_A_Q1"].mean()
    test = pd.read_csv(os.path.join(root, "rq1_baseline", "predictions", "test_predictions.csv"))
    logistic_p = _delong_pvalue(test.label, test.prob_LR, test.AQ_score_legacy)
    loso = pd.read_csv(os.path.join(root, "rq2_loso", "predictions", "rq2_loso_s0_predictions.csv"))
    loso["rule"] = (loso.AQ_score_legacy >= 6).astype(int)
    s07 = loso[(loso.school_id == "SCH_007") & (loso.label == 0) & (loso.y_pred == 1)]
    s07_shap = shap[shap.ID.isin(s07.ID)]
    s07_means = s07_shap[[f"SHAP_{c}" for c in items]].mean().sort_values(ascending=False)
    school_shap = pd.read_csv(os.path.join(root, "rq4", "tables", "T_rq4_xai_by_school.csv"))
    s06 = school_shap.loc[school_shap.school_id == "SCH_006",
                          [f"shap_{c}" for c in items]].iloc[0]
    s06_rank = int(s06.rank(ascending=False, method="min")[f"shap_A_Q1"])

    print("\n" + "=" * 78 + "\nLOCKED RUN FINDINGS\n" + "=" * 78)
    print(f"Cronbach's alpha (AQ-10): {alpha:.2f}")
    print(f"Distinct response patterns shared by both groups: {shared} of {total_patterns}")
    print(f"Item 1 correlation with every other item is negative: {'yes' if item1_negative else 'no'}")
    print(f"Maximum absolute item correlation: {max_corr:.2f} ({first_item.replace('A_', '')} and "
          f"{second_item.replace('A_', '')}; signed r={signed_corr:.2f})")
    print(f"Mean SHAP for item 1, definitely disagree: {item1_shap.get(0, np.nan):+.3f}")
    print(f"Mean SHAP for item 1, definitely agree: {item1_shap.get(3, np.nan):+.3f}")
    print(f"Test-set logistic regression versus AQ-10 total score: DeLong p={logistic_p:.2f}")
    print(f"S07 false positives: {len(s07)}; also scoring 6 or more: {int((s07.AQ_score_legacy >= 6).sum())}; "
          f"largest mean SHAP: {s07_means.index[0].replace('SHAP_A_', 'Q')} {s07_means.iloc[0]:+.2f}")
    print(f"S06 item 1 SHAP rank: {s06_rank} (out of {len(items)})")
    print("Specificity check: all ten reported pairs involving S06 or S07 were below the AQ-10 rule.")


def child_env():
    return dict(os.environ, AQ10_BASE=BASE, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")


def run_script(args, log_path, name, limit_min):
    """Run a script, show its output live and keep a copy in log_path. Returns the exit code.

    The script is stopped if it runs longer than limit_min minutes (returns TIMED_OUT), and it is
    always stopped when this runner stops (for example when a Colab cell is interrupted), so no
    step keeps running in the background.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([sys.executable, "-u"] + args, cwd=REPO, env=child_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        lines = queue.Queue()

        def pump():
            for ln in proc.stdout:
                lines.put(ln)
            lines.put(None)                                   # the script closed its output

        threading.Thread(target=pump, daemon=True).start()
        t0 = last_out = last_note = time.time()
        try:
            while True:
                try:
                    line = lines.get(timeout=5)
                except queue.Empty:
                    line = _NO_LINE
                now = time.time()
                if line is None:
                    break
                if line is not _NO_LINE:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                    last_out = last_note = now
                elif now - last_note >= QUIET_MINUTES * 60:
                    note = (f"... {name} is still running: {(now - t0) / 60:.0f} min so far, "
                            f"no new output for {(now - last_out) / 60:.0f} min (normal for the long calculations)\n")
                    print(note, end="", flush=True)
                    log.write(note)
                    last_note = now
                if now - t0 > limit_min * 60:
                    proc.kill()
                    proc.wait()
                    return TIMED_OUT
            proc.wait()
        finally:
            if proc.poll() is None:                           # runner interrupted: stop the script too
                proc.kill()
                proc.wait()
    return proc.returncode


def run_step(name, max_minutes=None):
    script, what = STEPS[name]
    if not os.path.exists(os.path.join(LOGS, "phase0_done.flag")):
        raise StepError(f"The data are not prepared in {BASE}. Run pipeline/prepare_from_zenodo.py first.")
    usual, limit = TIMING[name]
    limit = max_minutes or limit
    print("\n" + "=" * 78 + f"\n{what}\n"
          f"Usually about {usual} min on a Colab CPU runtime; stopped automatically after {limit} min.\n"
          + "=" * 78, flush=True)
    t0 = time.time()
    log_path = os.path.join(LOGS, f"run_{name}.log")
    code = run_script([script], log_path, name, limit)
    if code == TIMED_OUT:
        raise StepError(f"{name} STOPPED after {limit} min, the time limit for this step. The results of the earlier "
                        f"steps are kept. If the machine is just slow, run it again with a higher limit "
                        f"(--max-minutes {2 * limit}). The log is {log_path}.")
    if code != 0:
        raise StepError(f"{name} FAILED (exit code {code}). The log is {log_path}.")
    print(f"{name} finished in {(time.time() - t0) / 60:.1f} min", flush=True)


def has_child_rows(path):
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        header = next(csv.reader(fh), [])
    return bool(CHILD_LEVEL_COLUMNS & {h.strip() for h in header})


def collect():
    print("\n" + "=" * 78 + "\nCollect all results in one zip file\n" + "=" * 78, flush=True)
    if os.path.isdir(RESULTS):
        shutil.rmtree(RESULTS)                           # this folder only ever holds output of collect()
    os.makedirs(RESULTS)
    copied, skipped = [], []

    def copy(src, rel):
        dst = os.path.join(RESULTS, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied.append(rel)

    for f in sorted(glob.glob(os.path.join(FIGURES, "*.png"))):
        copy(f, os.path.join("figures", os.path.basename(f)))
    for f in sorted(glob.glob(os.path.join(PROJECT, "step7_summary", "*.csv"))):
        copy(f, os.path.join("summary", os.path.basename(f)))
    for rel in ("step5_school_effects/report_step5.md", "step6_robustness/report_step6.md"):
        if os.path.exists(os.path.join(PROJECT, rel)):
            copy(os.path.join(PROJECT, rel), os.path.join("reports", os.path.basename(rel)))
    for f in sorted(glob.glob(os.path.join(PROJECT, "*", "results", "*.json"))):
        copy(f, os.path.join("step_summaries", os.path.basename(f)))
    for step, rel in TABLE_DIRS.items():
        for f in sorted(glob.glob(os.path.join(PROJECT, rel, "**", "*.csv"), recursive=True)):
            if has_child_rows(f):
                skipped.append(os.path.relpath(f, PROJECT).replace(os.sep, "/"))
            else:
                copy(f, os.path.join("tables", step, os.path.relpath(f, os.path.join(PROJECT, rel))))
    for f in sorted(glob.glob(os.path.join(LOGS, "run_*.log"))):
        copy(f, os.path.join("logs", os.path.basename(f)))

    info = [f"date: {datetime.datetime.now().isoformat(timespec='seconds')}",
            f"python: {platform.python_version()} ({platform.platform()})"]
    for pkg in ("scikit-learn", "numpy", "pandas", "scipy", "shap", "matplotlib"):
        try:
            info.append(f"{pkg}: {md.version(pkg)}")
        except md.PackageNotFoundError:
            info.append(f"{pkg}: not installed")
    try:
        head = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        info.append(f"code: commit {head or 'unknown'}{' with local changes' if dirty else ''}")
    except OSError:
        info.append("code: git not available")
    if skipped:
        info.append("left out (one row per child): " + ", ".join(skipped))
    with open(os.path.join(RESULTS, "versions.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(info) + "\n")

    zip_path = shutil.make_archive(RESULTS, "zip", RESULTS)
    n_fig = sum(1 for c in copied if c.startswith("figures"))
    print(f"Figures: {n_fig}; files collected: {len(copied) + 1}; child-level tables left out: {len(skipped)}")
    print(f"ZIP FILE: {zip_path}")


def _show_new_figures(since):
    """In a notebook, show the figures written after `since` below the cell."""
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        return
    if get_ipython() is None:
        return
    for f in sorted(glob.glob(os.path.join(FIGURES, "*.png"))):
        if os.path.getmtime(f) >= since - 1:
            print(f"\n{os.path.basename(f)}")
            display(Image(filename=f, width=900))


def run(*steps, max_minutes=None):
    """For a notebook: run one or more steps (or "all") and show their figures below the cell."""
    steps = ORDER if steps == ("all",) else steps
    bad = [s for s in steps if s not in ORDER]
    if bad:
        raise ValueError(f"unknown step(s): {', '.join(bad)}; choose from {', '.join(ORDER)}")
    for s in steps:
        t0 = time.time()
        if s == "collect":
            collect()
        else:
            run_step(s, max_minutes)
            if s == "step7":
                print_locked_findings()
            _show_new_figures(t0)


def main():
    for s in (sys.stdout, sys.stderr):                 # the scripts print symbols such as ≥; Windows consoles may not be UTF-8
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("steps", nargs="+", help="all, or one or more of: " + ", ".join(ORDER))
    ap.add_argument("--data", help="path to full_dataset.csv; prepares the data before the first step")
    ap.add_argument("--max-minutes", type=float, help="time limit per step, in minutes (default: about five times "
                                                      "the usual time of each step)")
    a = ap.parse_args()
    steps = ORDER if a.steps == ["all"] else a.steps
    bad = [s for s in steps if s not in ORDER]
    if bad:
        ap.error(f"unknown step(s): {', '.join(bad)}")
    if a.max_minutes is not None and a.max_minutes <= 0:
        ap.error("--max-minutes must be positive")
    print(f"Working folder: {BASE}")
    t0 = time.time()
    try:
        if a.data:
            limit = a.max_minutes or TIMING["prepare"][1]
            code = run_script(["pipeline/prepare_from_zenodo.py", a.data], os.path.join(LOGS, "run_prepare.log"), "prepare", limit)
            if code != 0:
                raise StepError("prepare " + (f"STOPPED after {limit} min." if code == TIMED_OUT else f"FAILED (exit code {code})."))
        for s in steps:
            if s == "collect":
                collect()
            else:
                run_step(s, a.max_minutes)
                if s == "step7":
                    print_locked_findings()
    except StepError as e:
        sys.exit(f"\n{e}")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
