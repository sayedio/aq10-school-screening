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
# Minutes: (usual time on a free Colab CPU runtime, limit after which the step is stopped).
# No step can loop forever, but a stuck process would otherwise wait without end; the limit is
# about five times the usual time. --max-minutes (or max_minutes=) changes it.
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


def child_env():
    # UTF-8 mode: the scripts write symbols such as Δ and ≥, and Windows does not default to UTF-8
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
            collect() if s == "collect" else run_step(s, a.max_minutes)
    except StepError as e:
        sys.exit(f"\n{e}")
    print(f"\nDone in {(time.time() - t0) / 60:.1f} min.")


if __name__ == "__main__":
    main()
