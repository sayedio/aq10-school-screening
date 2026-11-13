# aq10-school-screening-code

Analysis code for the paper _Machine Learning on AQ-10 Child Responses for School-Based Autism Screening: Validation in Unseen Schools Against a Score-Independent Label_.

The study compares a support vector machine trained on the ten graded AQ-10 Child answers with the published cut-off (score ≥ 6) in 843 children aged 4 to 11 years from five special and seven mainstream schools in Bangladesh. The reference label is school placement. The model is tested once on a held-out set and then with leave-one-school-out validation.

**To check a number in the paper, open [`results/paper_numbers.md`](results/paper_numbers.md).**

**This repository holds code and the locked split only. It contains no questionnaire answers, no school names and no trained models.**

## Data

The questionnaire data are on Zenodo: **BD-AQ10**, version 1.0.0, [doi:10.5281/zenodo.23988060](https://doi.org/10.5281/zenodo.23988060). The files are shared on request for non-commercial research. Approved users receive `full_dataset.csv`, which `pipeline/prepare_from_zenodo.py` turns into the exact inputs of the locked run.

## Contents

| File                                                   | Purpose                                                                                                                                                                                             |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pipeline/prepare_from_zenodo.py`                      | Rebuilds the analysis files from the Zenodo file, in the locked row order and split                                                                                                                 |
| `pipeline/phase0_qc.py`                                | Original Phase 0 (cleaning, exclusions, aligned 0–3 encoding, 80/20 split, five folds). Needs the original school spreadsheets, which are not shared; use `pipeline/prepare_from_zenodo.py` instead |
| `pipeline/rq1_baseline.py`                             | Five learners, 86 configurations, locked SVM, one hold-out test against the AQ-10 rule                                                                                                              |
| `pipeline/rq2_loso.py`                                 | Leave-one-school-out validation of the locked SVM                                                                                                                                                   |
| `pipeline/rq3_augmentation.py`                         | Ten items versus fourteen items under the same scheme                                                                                                                                               |
| `pipeline/rq4_explanation.py`                          | Kernel SHAP, permutation importance and rank stability                                                                                                                                              |
| `pipeline/step5_school_effects.py`                     | Run after RQ4. Whether the model learns school-specific answer patterns: school fingerprints in the answers, seen versus unseen schools, and why school S06 had false positives                     |
| `pipeline/step6_robustness.py`                         | Run after RQ4. Paired tests and intervals with the school as the unit, SVM settings re-tuned inside every fold, and AQ-10 cut-offs 4 to 8                                                           |
| `analysis/paper_numbers.py`                            | Writes `results/paper_numbers.md` from the locked-run outputs, including the post hoc analyses                                                                                                      |
| `analysis/make_figure1.py`                             | Draws Fig. 1 (study design)                                                                                                                                                                         |
| `config/`                                              | Locked split by coded child ID (`split_assignment.csv`), fold indices and Phase-0 settings                                                                                                          |
| [`results/paper_numbers.md`](results/paper_numbers.md) | The numbers behind each table, figure and key claim in the paper, from the locked run                                                                                                               |
| `results/report_step5.md`, `results/report_step6.md`   | Reports of steps 5 and 6 from the verified rerun (school codes only)                                                                                                                                |

## Reproducing the paper

```bash
pip install -r requirements.txt
```

Set `AQ10_BASE` to an empty working folder outside this repository, then run:

```bash
python pipeline/prepare_from_zenodo.py path/to/full_dataset.csv
python pipeline/rq1_baseline.py
python pipeline/rq2_loso.py
python pipeline/rq3_augmentation.py
python pipeline/rq4_explanation.py
python pipeline/step5_school_effects.py
python pipeline/step6_robustness.py
python analysis/paper_numbers.py > my_paper_numbers.md
```

If you leave out the path, `prepare_from_zenodo.py` asks for it; in a Colab notebook, `%run pipeline/prepare_from_zenodo.py` shows an upload button instead. Each step writes into `$AQ10_BASE/aq_10/` and checks that the previous step finished. Steps 5 and 6 were added after the locked run to examine the school effects seen in RQ2; they use the locked SVM settings and predictions and write `step5_school_effects/` and `step6_robustness/`, each with a `report_step5.md` or `report_step6.md`. Compare `my_paper_numbers.md` with [`results/paper_numbers.md`](results/paper_numbers.md). Never commit the working folder.

The locked run used Google Colab with scikit-learn 1.6.1, numpy 2.0.2, pandas 2.2.2 and shap 0.52.0. A rerun of the whole pipeline from `full_dataset.csv` on Google Colab (Python 3.13, scikit-learn 1.6.1, numpy 2.4.6, pandas 2.2.3, shap 0.54.1) reproduced every locked prediction of RQ1 to RQ3 and every SHAP value to within 10<sup>-14</sup>, and the refits in steps 5 and 6 and in `analysis/paper_numbers.py` reproduced all 843 locked leave-one-school-out decisions. scikit-learn is the version that matters; a different one can move a result in the last decimal place.

## Licence

Code: MIT (see `LICENSE`). The English AQ-10 Child items are copyrighted by Allison, Auyeung and Baron-Cohen (2012).

## How to cite

Paper: to be added after acceptance.
Dataset: Sheikh, M. S., Munna, M. M. M., Habib, M. H. R., & Ahamed, M. S. (2026). _BD-AQ10: School-based AQ-10 Child data from special and mainstream schools in Bangladesh_ (Version 1.0.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23988060
