# Converted from v2_Thesis_phase_0.ipynb with outputs removed.
# This script contains no data files. Set AQ10_BASE to the folder that holds your private copy.
# Do not commit CSV, XLSX, PDF, ZIP or joblib files.

# ==============================================================================
# BD-AQ10(+SQ4) MULTI-CENTER SCHOOL SCREENING
# PHASE 0: DATA LOCK, CONSORT ATTRITION, SCHOOL-STRATIFIED 80/20 SPLIT,
#           & PRE-SEALED 5-FOLD CV INDICES
# VERSION: FINAL-5.0
#
# ARCHITECTURAL PRINCIPLES:
#   [P1] Participant-level independence ONLY.
#        Identical AQ-10 response patterns across DIFFERENT children are natural
#        clinical overlap (expected in a screener with 10 binary-scored items),
#        NOT data leakage.  Only participant IDs must not overlap.
#   [P2] School-stratified 80/20 split — every school that can contributes to
#        both dev and test.  Fallback: global stratified 80/20.
#   [P3] Pre-sealed 5-fold StratifiedKFold CV indices — sealed ONCE, identical
#        for all algorithms (LR, SVM, RF, KNN, NB).  Never re-drawn.
#   [P4] No modelling decisions — no scaling, SMOTE, feature selection, or
#        threshold tuning in Phase 0.
#
# OUTPUTS:
#   bd_clean.csv          — full validated cohort (all columns)
#   dev.csv               — 80% development set (HP search + final retrain)
#   test.csv              — 20% locked holdout (touched EXACTLY ONCE)
#   cv_fold_indices.json  — 5-fold CV train/val index lists over dev.csv
#   master_config.json    — all parameters, maps, counts, architecture notes
#   hashes.json           — MD5 cryptographic fingerprints
#   phase0_done.flag      — completion sentinel for downstream modules
#   exclusion_log.csv     — CONSORT attrition counts + percentages
#   school_split_log.csv  — per-school dev/test counts
#   T0_cv_fold_summary.csv
#   T0_cohort_stats.csv
#   T0_per_school_counts.csv
#
# LEAKAGE GUARD:
#   Participant ID uniqueness enforced (a001 / n001 format).
#   AQ-10 pattern identity across DIFFERENT participants is NOT guarded.
# ==============================================================================

import os
import sys

# ── Step 0A: Environmental locks ───────────────────────────────────────────────
SEED = 42
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"]          = str(SEED)

import json
import hashlib
import random
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
# from google.colab import drive  # Colab-only; locally, set the AQ10_BASE environment variable instead

random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore")

print("=" * 72)
print("PHASE 0  |  BD-AQ10 SCHOOL SCREENING  |  FINAL-5.0")
print("School-stratified 80/20 split  |  Participant ID guard  |  5-fold CV")
print("=" * 72)
print(f"  SEED={SEED}  PYTHONHASHSEED={SEED}  CUBLAS_WORKSPACE_CONFIG=:4096:8")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0B — DRIVE MOUNT & DIRECTORY TREE
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 0B]  Mounting drive & initializing directory tree ...")
# drive.mount("/content/drive", force_remount=True)

BASE_DIR    = os.environ.get("AQ10_BASE", "/content/drive/MyDrive/Autism")
DATA_DIR    = os.path.join(BASE_DIR,    "Data")
PROJECT_DIR = os.path.join(BASE_DIR,    "aq_10_v2")

# Verify raw data files exist before allocating anything
for fname in ["ASD.xlsx", "Normal.xlsx"]:
    fp = os.path.join(DATA_DIR, fname)
    if not os.path.exists(fp):
        raise FileNotFoundError(
            f"CRITICAL: {fp} not found.\n"
            f"Ensure the 'Autism/Data/' folder exists in My Drive.")

SUBDIRS = {
    "raw":         os.path.join(PROJECT_DIR, "data/raw"),
    "processed":   os.path.join(PROJECT_DIR, "data/processed"),
    "splits":      os.path.join(PROJECT_DIR, "data/splits"),
    "config":      os.path.join(PROJECT_DIR, "config"),
    "phase0_qc":   os.path.join(PROJECT_DIR, "phase0_qc"),
    "tables":      os.path.join(PROJECT_DIR, "tables"),
    "logs":        os.path.join(PROJECT_DIR, "logs"),
    "rq1":         os.path.join(PROJECT_DIR, "rq1_baseline"),
    "rq2":         os.path.join(PROJECT_DIR, "rq2_loso"),
    "rq3":         os.path.join(PROJECT_DIR, "rq3_sq4_augment"),
    "rq4":         os.path.join(PROJECT_DIR, "rq4_explain"),
    "paper_fig":   os.path.join(PROJECT_DIR, "paper_results/figures"),
    "paper_tab":   os.path.join(PROJECT_DIR, "paper_results/tables"),
    "predictions": os.path.join(PROJECT_DIR, "predictions"),
    "figures":     os.path.join(PROJECT_DIR, "figures"),
}
for d in SUBDIRS.values():
    os.makedirs(d, exist_ok=True)

print(f"  Project  : {PROJECT_DIR}")
print(f"  Raw data : {DATA_DIR}")
print("  Directory tree initialized.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — INGESTION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 1]  Ingesting raw spreadsheets ...")

# Spurious ID column names to drop
ID_ALIASES = {"id", "sl", "sl no", "sl.", "serial", "student_id", "no",
              "sr", "sr.", "s.no", "sno", "s no", "roll", "roll no"}

def _drop_id_cols(df: pd.DataFrame) -> pd.DataFrame:
    to_drop = [c for c in df.columns
               if str(c).strip().lower() in ID_ALIASES]
    return df.drop(columns=to_drop)

df_asd  = _drop_id_cols(pd.read_excel(os.path.join(DATA_DIR, "ASD.xlsx")))
df_norm = _drop_id_cols(pd.read_excel(os.path.join(DATA_DIR, "Normal.xlsx")))

# Synthetic master IDs  (participant-level, globally unique)
df_asd["ID"]  = [f"a{str(i+1).zfill(3)}" for i in range(len(df_asd))]
df_norm["ID"] = [f"n{str(i+1).zfill(3)}" for i in range(len(df_norm))]

df_asd["label"]        = 1
df_norm["label"]       = 0
df_asd["source_form"]  = "ASD.xlsx"
df_norm["source_form"] = "Normal.xlsx"

combined_df     = pd.concat([df_asd, df_norm], ignore_index=True)
RAW_TOTAL_COUNT = len(combined_df)

print(f"  ASD.xlsx     : {len(df_asd)} rows")
print(f"  Normal.xlsx  : {len(df_norm)} rows")
print(f"  Combined raw : {RAW_TOTAL_COUNT}  (ASD={len(df_asd)}, Normal={len(df_norm)})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — SCHOOL NAME HARDENING
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 2]  Hardening school strings ...")

INVALID_SCHOOL_STRINGS = {"nan", "none", "null", "", "unknown", "n/a", "na",
                           "not available", "-", "–"}

combined_df["school_name"] = (
    combined_df["school_name"].astype(str).str.strip().str.title()
)
combined_df["school_name"] = combined_df["school_name"].apply(
    lambda x: np.nan if x.lower() in INVALID_SCHOOL_STRINGS else x
)

valid_schools = sorted(combined_df["school_name"].dropna().unique())
school_id_map = {s: f"SCH_{i+1:03d}" for i, s in enumerate(valid_schools)}
combined_df["school_id"] = combined_df["school_name"].map(school_id_map)

print(f"  Unique school names found : {len(valid_schools)}")
for s, sid in school_id_map.items():
    n_ = (combined_df["school_name"] == s).sum()
    print(f"    {sid}  {s}  (n={n_})")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SCHEMA STANDARDIZATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 3]  Schema standardization ...")

def _build_col_map(df: pd.DataFrame) -> dict:
    col_map = {}
    for col in df.columns:
        cc  = str(col).strip()
        ccl = cc.lower()
        if   ccl == "age":                                   col_map[col] = "Age"
        elif ccl == "gender":                                col_map[col] = "Gender"
        elif ccl == "school_name":                           col_map[col] = "school_name"
        elif ccl == "school_type":                           col_map[col] = "school_type"
        elif ccl == "consent":                               col_map[col] = "consent"
        elif ccl == "family_hx":                             col_map[col] = "family_hx"
        elif ccl == "doctor_dx":                             col_map[col] = "doctor_dx"
        elif ccl == "label":                                 col_map[col] = "label"
        elif ccl == "id":                                    col_map[col] = "ID"
        elif ccl.startswith("q")  and ccl[1:].isdigit():    col_map[col] = cc.upper()
        elif ccl.startswith("sq") and ccl[2:].isdigit():    col_map[col] = cc.upper()
    return col_map

combined_df.rename(columns=_build_col_map(combined_df), inplace=True)
combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep="last")]

MANDATORY_COLS = [
    "ID", "school_name", "school_type", "consent",
    "Age", "Gender", "family_hx", "doctor_dx",
    "Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10",
    "SQ1", "SQ2", "SQ3", "SQ4", "label",
]
missing_mandatory = []
for col in MANDATORY_COLS:
    if col not in combined_df.columns:
        print(f"  WARNING: Mandatory column '{col}' missing — initializing with NaN.")
        combined_df[col] = np.nan
        missing_mandatory.append(col)

if not missing_mandatory:
    print("  Schema verified — all mandatory columns present.")
else:
    print(f"  Initialized {len(missing_mandatory)} missing columns with NaN.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — CONSORT ATTRITION FUNNEL  (7 stages)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 4]  CONSORT 7-stage attrition funnel ...")

# Restrict to known columns only
SAVE_COLS = [
    "ID", "source_form", "school_id", "school_name", "school_type",
    "consent", "Age", "Gender", "family_hx", "doctor_dx",
    "Q1",  "Q2",  "Q3",  "Q4",  "Q5",  "Q6",  "Q7",  "Q8",  "Q9",  "Q10",
    "SQ1", "SQ2", "SQ3", "SQ4", "label",
]
combined_df = combined_df[[c for c in SAVE_COLS if c in combined_df.columns]].copy()

# Coerce numeric columns
Q_COLS  = [f"Q{i}"  for i in range(1, 11)]
SQ_COLS = [f"SQ{i}" for i in range(1, 5)]
ALL_Q   = Q_COLS + SQ_COLS

for col in ALL_Q:
    if col in combined_df.columns:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

combined_df["Age"] = pd.to_numeric(
    combined_df["Age"].astype(str).str.extract(r"(\d+\.?\d*)")[0],
    errors="coerce"
)

# Stage 1: Invalid consent
VALID_CONSENT = {"yes", "y", "1", "true", "approved", "ok"}
consent_ok   = (combined_df["consent"].astype(str)
                .str.strip().str.lower()
                .str.replace(r"\.0$", "", regex=True)
                .isin(VALID_CONSENT))
n_no_consent = int((~consent_ok).sum())
combined_df  = combined_df[consent_ok].copy()

# Stage 2: Missing school
school_ok   = combined_df["school_name"].notna() & combined_df["school_id"].notna()
n_no_school = int((~school_ok).sum())
combined_df = combined_df[school_ok].copy()

# Stage 3: Missing Q1-Q10
q_ok    = combined_df[Q_COLS].notna().all(axis=1)
n_no_q  = int((~q_ok).sum())
combined_df = combined_df[q_ok].copy()

# Stage 4: Missing SQ1-SQ4
sq_ok    = combined_df[SQ_COLS].notna().all(axis=1)
n_no_sq  = int((~sq_ok).sum())
combined_df = combined_df[sq_ok].copy()

# Stage 5: Out-of-domain Likert (valid: 1, 2, 3, 4)
likert_ok    = combined_df[ALL_Q].isin([1, 2, 3, 4]).all(axis=1)
n_bad_likert = int((~likert_ok).sum())
combined_df  = combined_df[likert_ok].copy()

# Stage 6: Age out of bounds (4–11 inclusive)
age_ok   = (combined_df["Age"] >= 4.0) & (combined_df["Age"] <= 11.0)
n_bad_age= int((~age_ok).sum())
combined_df = combined_df[age_ok].copy().reset_index(drop=True)

# Stage 7: Final cohort
N_FINAL      = len(combined_df)
N_ASD_FINAL  = int((combined_df["label"] == 1).sum())
N_NORM_FINAL = int((combined_df["label"] == 0).sum())
N_SCHOOLS    = combined_df["school_id"].nunique()

print("=" * 72)
print("  CONSORT ATTRITION SUMMARY")
print("=" * 72)
print(f"  Total Raw                    : {RAW_TOTAL_COUNT:>6,d}")
print(f"  (-) Invalid Consent          : {n_no_consent:>6,d}")
print(f"  (-) Missing School           : {n_no_school:>6,d}")
print(f"  (-) Missing Q1-Q10           : {n_no_q:>6,d}")
print(f"  (-) Missing SQ1-SQ4          : {n_no_sq:>6,d}")
print(f"  (-) Out-of-Domain Likert     : {n_bad_likert:>6,d}")
print(f"  (-) Age Out of Bounds (4-11) : {n_bad_age:>6,d}")
print(f"  FINAL COHORT                 : {N_FINAL:>6,d}  "
      f"(ASD={N_ASD_FINAL}, Normal={N_NORM_FINAL})")
print(f"  School Centers               : {N_SCHOOLS:>6,d}")
print("=" * 72)

exclusion_df = pd.DataFrame({
    "Stage": [
        "0. Total Raw",
        "1. Invalid Consent",
        "2. Missing School",
        "3. Missing Q1-Q10",
        "4. Missing SQ1-SQ4",
        "5. Out-of-Domain Likert",
        "6. Age Out of Bounds (4-11)",
        "7. Final Cohort",
    ],
    "Count_Change": [
        RAW_TOTAL_COUNT,
        -n_no_consent, -n_no_school, -n_no_q,
        -n_no_sq, -n_bad_likert, -n_bad_age,
        N_FINAL,
    ],
    "Pct_of_Raw": [
        "100.0%",
        f"{n_no_consent/RAW_TOTAL_COUNT*100:.1f}%",
        f"{n_no_school/RAW_TOTAL_COUNT*100:.1f}%",
        f"{n_no_q/RAW_TOTAL_COUNT*100:.1f}%",
        f"{n_no_sq/RAW_TOTAL_COUNT*100:.1f}%",
        f"{n_bad_likert/RAW_TOTAL_COUNT*100:.1f}%",
        f"{n_bad_age/RAW_TOTAL_COUNT*100:.1f}%",
        f"{N_FINAL/RAW_TOTAL_COUNT*100:.1f}%",
    ],
})
exclusion_log_path = os.path.join(SUBDIRS["phase0_qc"], "exclusion_log.csv")
exclusion_df.to_csv(exclusion_log_path, index=False)
print(f"\n  Exclusion log: {exclusion_log_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — ORDINAL ENCODING  (0–3 scale)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 5]  Aligned Ordinal Intensity Encoding (0–3 scale) ...")

# High scores = more autistic trait expression (forward-scored)
FORWARD_ITEMS = ["Q1", "Q5", "Q7", "Q10", "SQ1", "SQ2", "SQ3"]
# High scores = fewer autistic trait expression (reverse-scored)
REVERSE_ITEMS = ["Q2", "Q3", "Q4",  "Q6",  "Q8",  "Q9", "SQ4"]

# After mapping: 3 = strongest autistic trait expression, 0 = weakest
FORWARD_MAP = {1: 3, 2: 2, 3: 1, 4: 0}
REVERSE_MAP = {1: 0, 2: 1, 3: 2, 4: 3}

for raw_col in FORWARD_ITEMS:
    enc_col = f"A_{raw_col}"
    if raw_col in combined_df.columns:
        combined_df[enc_col] = combined_df[raw_col].map(FORWARD_MAP)
    else:
        print(f"  WARNING: {raw_col} not found — {enc_col} will be NaN.")
        combined_df[enc_col] = np.nan

for raw_col in REVERSE_ITEMS:
    enc_col = f"A_{raw_col}"
    if raw_col in combined_df.columns:
        combined_df[enc_col] = combined_df[raw_col].map(REVERSE_MAP)
    else:
        print(f"  WARNING: {raw_col} not found — {enc_col} will be NaN.")
        combined_df[enc_col] = np.nan

AQ_ALIGNED_COLS = [f"A_Q{i}"  for i in range(1, 11)]
SQ_ALIGNED_COLS = [f"A_SQ{i}" for i in range(1, 5)]

# AQ_score_legacy: count of AQ-10 items with encoded value >= 2
# (This mirrors the original AQ-10 scoring where score >= 2 on an item counts)
combined_df["AQ_score_legacy"] = (
    combined_df[AQ_ALIGNED_COLS] >= 2
).sum(axis=1).astype(int)

# AQ_rule_pred: clinical positive at score >= 6
combined_df["AQ_rule_pred"] = (
    combined_df["AQ_score_legacy"] >= 6
).astype(int)

# Verify encoding completeness
encoding_na = combined_df[AQ_ALIGNED_COLS + SQ_ALIGNED_COLS].isna().sum().sum()
if encoding_na > 0:
    raise ValueError(
        f"ENCODING ERROR: {encoding_na} NaN values in encoded features. "
        f"All Q/SQ values must be integer 1-4 after the CONSORT funnel.")

print(f"  Encoded AQ features : {', '.join(AQ_ALIGNED_COLS)}")
print(f"  Encoded SQ features : {', '.join(SQ_ALIGNED_COLS)}")
print(f"  AQ_score_legacy range: "
      f"{combined_df['AQ_score_legacy'].min()}–{combined_df['AQ_score_legacy'].max()}")
print(f"  AQ+ (rule=1)        : "
      f"{combined_df['AQ_rule_pred'].sum()} / {N_FINAL} "
      f"({combined_df['AQ_rule_pred'].mean()*100:.1f}%)")
print(f"  Encoding NaN check  : {encoding_na}  [PASS]")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SCHOOL-STRATIFIED 80/20 SPLIT
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 6]  School-stratified 80/20 split ...")
print("  PRINCIPLE: Each school independently split; pools merged.")
print("  GUARD    : Per-school fallback to 'all dev' on any split failure.")
print("  FALLBACK : Global stratified 80/20 if NO school can contribute to test.")

TEST_RATIO = 0.20

# Reproducible global shuffle before splitting
combined_df = combined_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

dev_parts        = []
test_parts       = []
school_split_log = []

for school_id, grp in combined_df.groupby("school_id"):
    grp       = grp.copy().reset_index(drop=True)
    n_school  = len(grp)
    n_classes = grp["label"].nunique()

    # Guard: too few samples or single class
    if n_school < 2 or n_classes < 2:
        dev_parts.append(grp)
        school_split_log.append({
            "school_id": school_id,
            "Total":     n_school,
            "Dev_N":     n_school, "Test_N": 0,
            "Dev_ASD":   int((grp["label"] == 1).sum()),
            "Dev_Norm":  int((grp["label"] == 0).sum()),
            "Test_ASD":  0, "Test_Norm": 0,
            "Note":      "Too small or single class -> all to dev",
        })
        continue

    # Attempt stratified split; sklearn raises ValueError for very small schools
    try:
        s_dev, s_test = train_test_split(
            grp,
            test_size=TEST_RATIO,
            random_state=SEED,
            stratify=grp["label"],
        )
        dev_parts.append(s_dev)
        test_parts.append(s_test)
        school_split_log.append({
            "school_id": school_id,
            "Total":     n_school,
            "Dev_N":     len(s_dev),
            "Test_N":    len(s_test),
            "Dev_ASD":   int((s_dev["label"]  == 1).sum()),
            "Dev_Norm":  int((s_dev["label"]  == 0).sum()),
            "Test_ASD":  int((s_test["label"] == 1).sum()),
            "Test_Norm": int((s_test["label"] == 0).sum()),
            "Note":      "OK",
        })
    except Exception as exc:
        dev_parts.append(grp)
        school_split_log.append({
            "school_id": school_id,
            "Total":     n_school,
            "Dev_N":     n_school, "Test_N": 0,
            "Dev_ASD":   int((grp["label"] == 1).sum()),
            "Dev_Norm":  int((grp["label"] == 0).sum()),
            "Test_ASD":  0, "Test_Norm": 0,
            "Note":      f"Split failed ({exc}) -> all to dev",
        })

# Safe pooling — crash-proof concat
if not dev_parts:
    raise RuntimeError(
        "CRITICAL: dev_parts is empty — no records survived the split.")

dev_df = pd.concat(dev_parts, ignore_index=True).sample(
    frac=1.0, random_state=SEED).reset_index(drop=True)

if not test_parts:
    print("\n  [WARN] No school contributed to test via per-school split.")
    print("  [WARN] Falling back to GLOBAL stratified 80/20 split.")
    dev_df, test_df = train_test_split(
        combined_df, test_size=TEST_RATIO,
        random_state=SEED, stratify=combined_df["label"])
    dev_df  = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    fallback_used = True
else:
    test_df = pd.concat(test_parts, ignore_index=True).sample(
        frac=1.0, random_state=SEED).reset_index(drop=True)
    fallback_used = False

# Save school split log
school_split_df   = pd.DataFrame(school_split_log)
school_split_path = os.path.join(SUBDIRS["phase0_qc"], "school_split_log.csv")
school_split_df.to_csv(school_split_path, index=False)

n_schools_test = int((school_split_df["Test_N"] > 0).sum())

print(f"\n  {'School':<12} {'Total':>6} {'Dev':>6} {'Test':>6}  "
      f"{'Dev(A/N)':>10}  {'Test(A/N)':>10}  Note")
print("  " + "-" * 75)
for _, row in school_split_df.iterrows():
    dv = f"{int(row['Dev_ASD'])}/{int(row['Dev_Norm'])}"
    tv = f"{int(row['Test_ASD'])}/{int(row['Test_Norm'])}"
    print(f"  {str(row['school_id']):<12} {int(row['Total']):>6} "
          f"{int(row['Dev_N']):>6} {int(row['Test_N']):>6}  "
          f"{dv:>10}  {tv:>10}  {row['Note']}")

N_DEV   = len(dev_df);  N_TEST  = len(test_df)
dev_asd  = int((dev_df["label"]  == 1).sum())
dev_norm = int((dev_df["label"]  == 0).sum())
test_asd = int((test_df["label"] == 1).sum())
test_norm= int((test_df["label"] == 0).sum())

print(f"\n  Schools contributing to test : {n_schools_test} / {N_SCHOOLS}")
print(f"  Fallback global split used   : {fallback_used}")
print(f"  Dev   : N={N_DEV}   [ASD={dev_asd},  Normal={dev_norm}]  "
      f"({N_DEV/(N_DEV+N_TEST)*100:.1f}%)")
print(f"  Test  : N={N_TEST}  [ASD={test_asd}, Normal={test_norm}] "
      f"({N_TEST/(N_DEV+N_TEST)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — PARTICIPANT-LEVEL LEAKAGE CHECK  (ID uniqueness ONLY)
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 7]  Participant-level ID leakage check ...")
print("  RULE: No participant ID (a001, n001...) may appear in both dev and test.")
print("  NOTE: Identical A_Q1..A_Q10 patterns across DIFFERENT children are")
print("        natural clinical overlap — NOT leakage.  Not guarded here.")

dev_ids    = set(dev_df["ID"].astype(str))
test_ids   = set(test_df["ID"].astype(str))
id_overlap = len(dev_ids & test_ids)

if id_overlap > 0:
    raise RuntimeError(
        f"CRITICAL: {id_overlap} participant IDs appear in BOTH dev and test.\n"
        f"Overlap IDs: {sorted(dev_ids & test_ids)[:10]}\n"
        f"This is a split error — HALT.")

total_accounted = len(dev_ids) + len(test_ids)
count_ok        = total_accounted == N_FINAL

print(f"  Participant ID overlap : {id_overlap}              [PASS]")
print(f"  Total IDs accounted    : {total_accounted} == {N_FINAL}  "
      f"[{'PASS' if count_ok else 'FAIL'}]")

if not count_ok:
    raise RuntimeError(
        f"ID count mismatch: {total_accounted} != {N_FINAL}. "
        f"Some participants lost in the split — HALT.")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — PRE-SEAL 5-FOLD STRATIFIEDKFOLD CV INDICES
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 8]  Pre-sealing 5-fold StratifiedKFold CV indices over dev ...")
print("  All 5 algorithms (LR, SVM, RF, KNN, NB) will use IDENTICAL folds.")
print("  Sealed once here — never re-drawn in downstream modules.")

N_OUTER_FOLDS = 5
skf = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)

fold_indices      = {}
fold_summary_rows = []

for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(dev_df, dev_df["label"]), start=1):

    fold_key = f"fold_{fold_idx}"
    fold_indices[fold_key] = {
        "train_indices": train_idx.tolist(),
        "val_indices":   val_idx.tolist(),
    }

    y_tr = dev_df.loc[train_idx, "label"]
    y_vl = dev_df.loc[val_idx,   "label"]

    fold_summary_rows.append({
        "Fold":          fold_idx,
        "Train_N":       len(train_idx),
        "Train_ASD":     int((y_tr == 1).sum()),
        "Train_Normal":  int((y_tr == 0).sum()),
        "Train_ASD_Pct": f"{(y_tr==1).mean()*100:.1f}%",
        "Val_N":         len(val_idx),
        "Val_ASD":       int((y_vl == 1).sum()),
        "Val_Normal":    int((y_vl == 0).sum()),
        "Val_ASD_Pct":   f"{(y_vl==1).mean()*100:.1f}%",
    })

    print(f"  Fold {fold_idx}: "
          f"Train={len(train_idx):4d}  (ASD={int((y_tr==1).sum()):3d}, "
          f"Norm={int((y_tr==0).sum()):3d}) | "
          f"Val={len(val_idx):4d}  (ASD={int((y_vl==1).sum()):3d}, "
          f"Norm={int((y_vl==0).sum()):3d})")

cv_indices_path = os.path.join(SUBDIRS["config"], "cv_fold_indices.json")
with open(cv_indices_path, "w") as f:
    json.dump(fold_indices, f, indent=2)

fold_summary_df   = pd.DataFrame(fold_summary_rows)
fold_summary_path = os.path.join(SUBDIRS["tables"], "T0_cv_fold_summary.csv")
fold_summary_df.to_csv(fold_summary_path, index=False)

print(f"\n  CV indices sealed : {cv_indices_path}")
print(f"  Fold summary      : {fold_summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — SAVE DATASETS & CRYPTOGRAPHIC LOCKS
# ══════════════════════════════════════════════════════════════════════════════

print("\n[Step 9]  Writing physical datasets ...")

clean_path = os.path.join(SUBDIRS["processed"], "bd_clean.csv")
train_path = os.path.join(SUBDIRS["splits"],    "train.csv")
test_path  = os.path.join(SUBDIRS["splits"],    "test.csv")

combined_df.to_csv(clean_path, index=False)
dev_df.to_csv(train_path, index=False)
test_df.to_csv(test_path,  index=False)

print(f"  bd_clean.csv : {len(combined_df)} rows  {len(combined_df.columns)} cols")
print(f"  train.csv    : {len(dev_df)} rows")
print(f"  test.csv     : {len(test_df)} rows")

# MD5 hashes
def md5_file(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

hashes = {
    "bd_clean.csv":         md5_file(clean_path),
    "train.csv":            md5_file(train_path),
    "test.csv":             md5_file(test_path),
    "cv_fold_indices.json": md5_file(cv_indices_path),
}
hash_path = os.path.join(PROJECT_DIR, "data/hashes.json")
with open(hash_path, "w") as f:
    json.dump(hashes, f, indent=4)

print("\n  MD5 cryptographic fingerprints:")
for fname, md5 in hashes.items():
    print(f"  {fname:<28}  {md5}")

# Summary tables
t0_stats = pd.DataFrame({
    "Metric": [
        "Raw Ingested Records",
        "(-) Invalid Consent",     "(-) Missing School",
        "(-) Missing Q1-Q10",      "(-) Missing SQ1-SQ4",
        "(-) Out-of-Domain Likert","(-) Age Out of Bounds (4-11)",
        "Final Validated Cohort",  "  ASD Cases (label=1)",
        "  Mainstream Controls (label=0)",
        "Development Set",  "  Dev ASD",   "  Dev Normal",
        "Locked Test Set",  "  Test ASD",  "  Test Normal",
        "Active School Centers", "Schools Contributing to Test",
        "CV Outer Folds",
    ],
    "Count": [
        RAW_TOTAL_COUNT,
        n_no_consent, n_no_school, n_no_q, n_no_sq,
        n_bad_likert, n_bad_age,
        N_FINAL, N_ASD_FINAL, N_NORM_FINAL,
        N_DEV,  dev_asd,  dev_norm,
        N_TEST, test_asd, test_norm,
        N_SCHOOLS, n_schools_test, N_OUTER_FOLDS,
    ],
})
t0_stats_path = os.path.join(SUBDIRS["tables"], "T0_cohort_stats.csv")
t0_stats.to_csv(t0_stats_path, index=False)

# Per-school dev/test breakdown
school_dev  = dev_df.groupby(["school_id", "school_name", "school_type"]).agg(
    Dev_Count=("label", "size"), Dev_ASD=("label", "sum")).reset_index()
school_dev["Dev_Normal"] = school_dev["Dev_Count"] - school_dev["Dev_ASD"]

school_test = test_df.groupby(["school_id", "school_name", "school_type"]).agg(
    Test_Count=("label", "size"), Test_ASD=("label", "sum")).reset_index()
school_test["Test_Normal"] = school_test["Test_Count"] - school_test["Test_ASD"]

t0_school = school_dev.merge(
    school_test, on=["school_id", "school_name", "school_type"], how="outer"
).fillna(0)
for c in ["Dev_Count","Dev_ASD","Dev_Normal","Test_Count","Test_ASD","Test_Normal"]:
    t0_school[c] = t0_school[c].astype(int)
t0_school["Total"] = t0_school["Dev_Count"] + t0_school["Test_Count"]
t0_school = t0_school.sort_values("school_id").reset_index(drop=True)

t0_school_path = os.path.join(SUBDIRS["tables"], "T0_per_school_counts.csv")
t0_school.to_csv(t0_school_path, index=False)

# Master configuration
master_config = {
    "VERSION":         "FINAL-5.0",
    "SEED":            SEED,
    "N_OUTER_FOLDS":   N_OUTER_FOLDS,
    "TEST_RATIO":      TEST_RATIO,
    "THRESHOLD_FIXED": 0.50,

    "final_clean_count":        N_FINAL,
    "n_train":                  N_DEV,
    "n_test":                   N_TEST,
    "train_class_counts":       {"ASD": dev_asd,   "Normal": dev_norm},
    "test_class_counts":        {"ASD": test_asd,  "Normal": test_norm},
    "n_schools":                N_SCHOOLS,
    "n_schools_in_test":        n_schools_test,
    "fallback_global_split":    fallback_used,

    "split_strategy": (
        f"School-stratified {int((1-TEST_RATIO)*100)}/{int(TEST_RATIO*100)} "
        f"(FINAL-5.0): per-school stratified split, pooled. "
        f"Schools contributing to test: {n_schools_test}/{N_SCHOOLS}. "
        f"Fallback global split used: {fallback_used}."),

    "leakage_guard": {
        "type":       "participant_id_uniqueness_only",
        "id_overlap": 0,
        "status":     "PASS",
        "note": (
            "Identical A_Q1..A_Q10 patterns across DIFFERENT children are "
            "natural clinical overlap in a 10-item screener — NOT leakage. "
            "Only participant ID uniqueness is enforced."),
    },

    "forward_items": FORWARD_ITEMS,
    "reverse_items": REVERSE_ITEMS,
    "forward_map":   {str(k): v for k, v in FORWARD_MAP.items()},
    "reverse_map":   {str(k): v for k, v in REVERSE_MAP.items()},
    "aq_aligned_cols": AQ_ALIGNED_COLS,
    "sq_aligned_cols": SQ_ALIGNED_COLS,
    "q_cols_raw":      Q_COLS,
    "sq_cols_raw":     SQ_COLS,
    "all_q_cols_raw":  ALL_Q,

    "paths": {
        "project_dir":     PROJECT_DIR,
        "bd_clean":        clean_path,
        "train":           train_path,
        "test":            test_path,
        "cv_fold_indices": cv_indices_path,
        "hashes":          hash_path,
    },

    "consort": {
        "raw_total":        RAW_TOTAL_COUNT,
        "excl_no_consent":  n_no_consent,
        "excl_no_school":   n_no_school,
        "excl_missing_q":   n_no_q,
        "excl_missing_sq":  n_no_sq,
        "excl_bad_likert":  n_bad_likert,
        "excl_bad_age":     n_bad_age,
        "final_cohort":     N_FINAL,
        "final_asd":        N_ASD_FINAL,
        "final_normal":     N_NORM_FINAL,
    },

    "algorithms": ["LR", "SVM", "RF", "KNN", "NB"],

    "architecture_notes": {
        "split_strategy": (
            "School-stratified 80/20 (FINAL-5.0). Each school independently "
            "split; schools unable to contribute send all to dev. "
            "Fallback to global stratified 80/20 if no school can contribute."),
        "leakage_prevention": (
            "Participant ID uniqueness enforced. "
            "Identical AQ-10 patterns across different children are NOT leakage."),
        "cv_strategy": (
            "5-fold StratifiedKFold on train set, sealed once in Phase 0. "
            "Identical fold assignments for all 5 algorithms."),
        "threshold_protocol": (
            "Decision threshold fixed at 0.50, pre-specified in Phase 0. "
            "No Youden's J or post-hoc threshold selection."),
        "final_model_protocol": (
            "M* (champion) retrained on FULL train set with best HPs from CV. "
            "Evaluated exactly once on locked test set."),
        "bootstrap_protocol": (
            "Stratified bootstrap preserving original class counts, "
            "2000 iterations, 95% percentile CI. No BCa correction."),
        "no_modeling_in_phase0": (
            "No scaling, SMOTE, feature selection, or threshold tuning. "
            "All transforms applied only inside CV fold training sets."),
        "rq2_note": (
            "For LOSO (RQ2): lock the algorithm + hyperparameters + threshold "
            "from RQ1. Retrain a FRESH model inside each LOSO fold — do NOT "
            "reuse the joblib model trained on all dev data."),
    },

    "md5_hashes": hashes,
}

config_path = os.path.join(SUBDIRS["config"], "master_config.json")
with open(config_path, "w") as f:
    json.dump(master_config, f, indent=4)

# Completion flag
flag_path = os.path.join(SUBDIRS["logs"], "phase0_done.flag")
with open(flag_path, "w") as f:
    f.write(
        f"PHASE 0 FINAL-5.0 SUCCESSFULLY COMPLETED\n"
        f"timestamp={pd.Timestamp.now().isoformat()}\n"
        f"version=FINAL-5.0\n"
        f"seed={SEED}\n"
        f"split=school_stratified_80_20\n"
        f"fallback_global_split={fallback_used}\n"
        f"leakage_guard=participant_id_only\n"
        f"n_final={N_FINAL}\n"
        f"n_train={N_DEV}  (ASD={dev_asd}, Normal={dev_norm})\n"
        f"n_test={N_TEST}  (ASD={test_asd}, Normal={test_norm})\n"
        f"n_schools={N_SCHOOLS}  schools_in_test={n_schools_test}\n"
        f"n_cv_folds={N_OUTER_FOLDS}\n"
        f"threshold_fixed=0.50\n"
        f"md5_bd_clean={hashes['bd_clean.csv']}\n"
        f"md5_train={hashes['train.csv']}\n"
        f"md5_test={hashes['test.csv']}\n"
        f"md5_cv_indices={hashes['cv_fold_indices.json']}\n"
    )


# ══════════════════════════════════════════════════════════════════════════════
# DELIVERABLES CHECKLIST
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("  PHASE 0 FINAL-5.0 — DELIVERABLES CHECKLIST")
print("=" * 72)

CHECKLIST = [
    ("bd_clean.csv",              clean_path),
    ("train.csv",                 train_path),
    ("test.csv",                  test_path),
    ("cv_fold_indices.json",      cv_indices_path),
    ("master_config.json",        config_path),
    ("hashes.json",               hash_path),
    ("phase0_done.flag",          flag_path),
    ("exclusion_log.csv",         exclusion_log_path),
    ("school_split_log.csv",      school_split_path),
    ("T0_cohort_stats.csv",       t0_stats_path),
    ("T0_per_school_counts.csv",  t0_school_path),
    ("T0_cv_fold_summary.csv",    fold_summary_path),
]

all_ok = True
for name, path in CHECKLIST:
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  [OK]  {name:<30}  {size:>10,} bytes")
    else:
        print(f"  [!!]  MISSING: {name}")
        all_ok = False

print("=" * 72)
print(f"\n  FINAL SPLIT:")
print(f"    Train : N={N_DEV:4d}   ASD={dev_asd:3d}   Normal={dev_norm:3d}   "
      f"({N_DEV/(N_DEV+N_TEST)*100:.1f}%)")
print(f"    Test  : N={N_TEST:4d}   ASD={test_asd:3d}   Normal={test_norm:3d}   "
      f"({N_TEST/(N_DEV+N_TEST)*100:.1f}%)")
print(f"\n  Schools contributing to test : {n_schools_test} / {N_SCHOOLS}")
print(f"  Participant ID leakage       : 0              [PASS]")
print(f"  Total IDs accounted          : {N_DEV+N_TEST} / {N_FINAL}  [PASS]")
print(f"  Encoding NaN values          : {encoding_na}              [PASS]")
print(f"  CV folds pre-sealed          : {N_OUTER_FOLDS}              [PASS]")
print(f"  MD5 hashes written           : {len(hashes)}              [PASS]")

print("\n" + "=" * 72)
if all_ok:
    print("  STATUS: PHASE 0 FINAL-5.0 — COMPLETE & CRYPTOGRAPHICALLY LOCKED")
    print("  READY : RQ1 (5-fold CV hyperparameter search on train.csv)")
else:
    print("  STATUS: ERROR — CHECK MISSING FILES ABOVE")
print("=" * 72)

