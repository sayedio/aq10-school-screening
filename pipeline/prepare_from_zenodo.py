"""Rebuild the locked Phase-0 outputs from the Zenodo file.

phase0_qc.py starts from the original school spreadsheets, which contain school
names and are not shared. Approved users of the BD-AQ10 dataset
(doi: 10.5281/zenodo.23988060) receive full_dataset.csv instead. This script turns
that file into the exact inputs that rq1_baseline.py to rq4_explanation.py read:

    <AQ10_BASE>/aq_10_v2/data/processed/bd_clean.csv
    <AQ10_BASE>/aq_10_v2/data/splits/train.csv      (development set, 674 children)
    <AQ10_BASE>/aq_10_v2/data/splits/test.csv       (hold-out set, 169 children)
    <AQ10_BASE>/aq_10_v2/config/master_config.json
    <AQ10_BASE>/aq_10_v2/config/cv_fold_indices.json
    <AQ10_BASE>/aq_10_v2/logs/phase0_done.flag

Row order, the development/test split and the five cross-validation folds come
from config/split_assignment.csv, so the downstream scripts see the same rows in
the same order as in the locked run. School names are replaced by school codes.

Usage:
    python pipeline/prepare_from_zenodo.py path/to/full_dataset.csv
with AQ10_BASE set to an empty working folder (default: ./work).
"""
import json
import os
import shutil
import sys

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
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

    special = z.group.str.strip().str.lower().eq("special")
    d = pd.DataFrame({
        "ID": z.child_id,
        "source_form": special.map({True: "ASD.xlsx", False: "Normal.xlsx"}),
        "school_id": "SCH_" + z.school_code.str[1:].str.zfill(3),
        "school_name": z.school_code,                      # names withheld
        "school_type": special.map({True: "asd", False: "normal"}),
        "consent": "yes",
        "Age": z.age.astype(int),
        "Gender": z.gender,
        "family_hx": z.family_hx,
        "doctor_dx": z.doctor_dx,
    })
    for c in AQ_ITEMS + SQ_ITEMS:
        d[c] = z[c].astype(int)
    d["label"] = special.astype(int)
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

    proj = os.path.join(base, "aq_10_v2")
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
    print(f"  children {len(bd)} (special {int(bd.label.sum())}, mainstream {int((1 - bd.label).sum())})")
    print(f"  development {len(dev)}, test {len(test)}, AQ-10 rule positive {int(bd.AQ_rule_pred.sum())}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    build(sys.argv[1], os.environ.get("AQ10_BASE", os.path.join(os.getcwd(), "work")))
