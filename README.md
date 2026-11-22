# aq10-school-screening-code

Analysis code for the study _Machine Learning on AQ-10 Child Responses: Agreement With ASD-School Placement and School-Level Response Patterns in Unseen Schools_.

The study compares a support vector machine (SVM), trained on the ten graded answers of the AQ-10 Child, with the published AQ-10 cut-off (a total score of 6 or more). It uses 843 children aged 4 to 11 years from five ASD schools and seven mainstream schools in Bangladesh, rated by parents or teachers. The reference label is placement in an ASD school, not a clinical diagnosis. The model is chosen by cross-validation, tested once on a test set, and then validated with each school left out in turn.

**This repository holds code and the locked data split only. It contains no questionnaire answers, no school names and no trained models.**

## Analysis steps

| Step    | Script                             | What it does                                                                                                                                                                                         |
| ------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Prepare | `pipeline/prepare_from_zenodo.py`  | Builds the analysis files from the Zenodo file: leaves out the 20 children outside the age range of 4 to 11, puts the rest in the locked row order and split, and draws the study design             |
| RQ1     | `pipeline/rq1_baseline.py`         | Compares five learners (86 settings) by five-fold cross-validation on the development set of 674 children, then tests the selected model once on the test set of 169 children against the AQ-10 rule |
| RQ2     | `pipeline/rq2_loso.py`             | Leave-one-school-out validation of the selected model, with its settings fixed, against the AQ-10 rule                                                                                               |
| RQ3     | `pipeline/rq3_augmentation.py`     | Ten items against fourteen items (the AQ-10 plus four supplementary items) under the same validation, with an adoption rule set in advance                                                           |
| RQ4     | `pipeline/rq4_explanation.py`      | Kernel SHAP, permutation importance, stability of the item ranking across schools, and subgroups                                                                                                     |
| Step 5  | `pipeline/step5_school_effects.py` | Post hoc. School-specific answer patterns: schools seen and unseen in training, local training data, and the false positives in school S06                                                           |
| Step 6  | `pipeline/step6_robustness.py`     | Post hoc. Paired tests and intervals with the school as the unit, model settings re-tuned inside every fold, and AQ-10 cut-offs 4 to 8                                                               |
| Step 7  | `pipeline/step7_summary.py`        | Collects the main results of all steps in CSV files and adds post hoc summaries: school-clustered intervals, lower cut-offs, refits, subgroups and screening prevalence                              |

RQ1 to RQ4 follow the analysis plan: the test split, the five folds and the decision threshold of 0.50 were fixed before any model was trained. Steps 5 and 6, and the post hoc parts of step 7, were added after the first full run to examine the school effects seen in RQ2. They use the locked model settings and predictions and do not change any result of RQ1 to RQ4.

## Data

The questionnaire data are on Zenodo: **BD-AQ10**, version 1.0.0, [doi:10.5281/zenodo.23988059](https://doi.org/10.5281/zenodo.23988059) (this DOI always leads to the latest version). The files are shared on request for non-commercial research. Approved users receive `full_dataset.csv`, which `pipeline/prepare_from_zenodo.py` turns into the exact inputs of the analysis. Its `group` column is `ASD` or `mainstream` (the first copies shared said `special` instead of `ASD`); the script accepts both.

`full_dataset.csv` holds all 863 forms received. The script leaves out the 20 children aged 3 or 12 (16 from ASD schools, 4 from mainstream schools), the same exclusion as in the original cleaning step, and keeps the 843 children aged 4 to 11. Version 1.0.0 of the file held only these 843 children. It gives the same results, but the study design figure then shows no exclusions.

## Run on Google Colab

You need only `full_dataset.csv`. Open a new notebook at [colab.research.google.com](https://colab.research.google.com) (a CPU runtime is enough), paste each block below into its own cell and run the cells in order. Each cell prints its progress and shows its figures below it.

Nothing is written to Google Drive. The notebook never mounts Drive: the code, the uploaded file and every output stay on the temporary Colab machine, which deletes them when the runtime ends. If the runtime disconnects before the last cell, start again from cell 1.

**1. Get the code and the package versions.** The last line must print `scikit-learn 1.6.1`. The cell uses fixed paths, so running it again does no harm.

```python
!test -d /content/aq10-school-screening || git clone https://github.com/sayedio/aq10-school-screening.git /content/aq10-school-screening
%cd /content/aq10-school-screening
!pip install -q -r requirements.txt
import run_pipeline as rp
!python -c "import sklearn; print('scikit-learn', sklearn.__version__)"
```

**2. Upload `full_dataset.csv`.** An upload button appears. The cell then leaves out the children outside the age range, builds the analysis files and shows the study design.

```python
%run pipeline/prepare_from_zenodo.py
```

**3. RQ1: model selection and the test set**

```python
rp.run("rq1")
```

**4. RQ2: leave-one-school-out validation**

```python
rp.run("rq2")
```

**5. RQ3: ten against fourteen items**

```python
rp.run("rq3")
```

**6. RQ4: item attribution and subgroups** (the longest step)

```python
rp.run("rq4")
```

**7. Step 5: school effects**

```python
rp.run("step5")
```

**8. Step 6: school-level tests, re-tuned settings and AQ-10 cut-offs**

```python
rp.run("step6")
```

**9. Step 7: summary, and all results in one zip file**

```python
rp.run("step7", "collect")
```

**10. Download the zip file**

```python
from google.colab import files
files.download("/content/aq10-school-screening/work/aq10_results.zip")
```

Then choose Runtime > Disconnect and delete runtime. This removes the uploaded data from the Colab machine at once.

`rp.run("all")` runs cells 3 to 9 in one cell instead.

**How long it takes.** About an hour in all on a free Colab CPU runtime; RQ4 takes about half of it. Each step prints how long it usually takes. During a long calculation that prints nothing, a line such as `rq4 is still running: 12 min so far` appears every two minutes. As a safeguard, a step is stopped with a message if it runs about five times longer than usual; the results of the earlier steps are kept, and `rp.run("rq4", max_minutes=300)` gives a slow machine more time. Interrupting a cell also stops the step it started, so nothing keeps running in the background.

## Run on your own computer

Use Python 3.10 to 3.13. From the repository folder:

```bash
pip install -r requirements.txt
python run_pipeline.py all --data path/to/full_dataset.csv
```

This prepares the data, runs every step and writes `work/aq10_results.zip`. To run a single step, give its name, for example `python run_pipeline.py rq2`. The `work/` folder is ignored by git; never commit it.

## Output

Everything is written to the working folder `work/` (or to the folder in the environment variable `AQ10_BASE`, if set). The zip file `work/aq10_results.zip` holds:

| Folder or file    | Content                                                                                                                                                                                                 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `figures/`        | All figures, at 600 dpi (listed below)                                                                                                                                                                  |
| `summary/`        | The CSV files of step 7: cohort, model selection, test set, leave-one-school-out results, cut-offs, refits, ten against fourteen items, item importance, school S06, subgroups and screening prevalence |
| `reports/`        | Full reports of steps 5 and 6                                                                                                                                                                           |
| `tables/`         | All result tables of each step (tables with one row per child are left out)                                                                                                                             |
| `step_summaries/` | The settings and main results of RQ1 to RQ4, as JSON                                                                                                                                                    |
| `logs/`           | Everything each step printed                                                                                                                                                                            |
| `versions.txt`    | Python and package versions, the code commit and the date                                                                                                                                               |

Figures:

| File                                           | Content                                                                                                                                                                                        |
| ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `study_design.png`                             | Study design, with the counts of the analysis cohort                                                                                                                                           |
| `rq1_model_selection_cv_auroc.png`             | Cross-validated AUROC of every setting of the five learners                                                                                                                                    |
| `rq1_test_set_roc_and_svm_vs_rule.png`         | Test set: ROC curves of the five learners, and the selected model against the AQ-10 rule with 95% bootstrap intervals                                                                          |
| `rq1_test_set_calibration_and_net_benefit.png` | Test set: calibration of the selected model, and net benefit across threshold probabilities                                                                                                    |
| `rq2_aq10_score_and_svm_flags.png`             | AQ-10 total score in each group, and children flagged by the SVM at each score when their school was left out                                                                                  |
| `rq2_pooled_roc_and_calibration.png`           | Leave-one-school-out: pooled ROC curves of the SVM and the AQ-10 total score, and calibration of the SVM                                                                                       |
| `rq2_svm_probability_by_school.png`            | Leave-one-school-out: SVM probability of every child, by school                                                                                                                                |
| `rq3_flags_by_school_and_roc.png`              | Children flagged in each left-out school by the SVM and the AQ-10 rule, and pooled ROC curves of the ten-item and fourteen-item SVMs and the AQ-10 total score                                 |
| `rq3_14_vs_10_items.png`                       | Paired differences between fourteen and ten items, and the share flagged in each school with both item sets                                                                                    |
| `rq4_item_attribution.png`                     | Mean absolute SHAP value and permutation importance of each item, and the rank of each item in each left-out school                                                                            |
| `rq4_shap_values_by_answer.png`                | SHAP value of every child for every item, by answer                                                                                                                                            |
| `rq4_subgroup_sensitivity_specificity.png`     | Sensitivity and specificity by age group, gender and family history, with 95% Wilson intervals                                                                                                 |
| `step5_item1_and_school_effects.png`           | Answers to item 1 by school, the SHAP value of item 1 by answer, and children flagged in each mainstream school when it was seen, unseen, or unseen with ten of its children added to training |
| `step5_training_school_influence.png`          | Change in the share flagged in each left-out school when one training school is removed                                                                                                        |
| `step6_school_level_tests_and_cutoffs.png`     | SVM minus AQ-10 rule in each school with child-level and school-level tests, and AQ-10 cut-offs 4 to 8 with school-clustered intervals                                                         |

## Reproducibility

- Every random step uses the seed 42. The test split and the five folds are stored by coded child ID in `config/split_assignment.csv` and `config/cv_fold_indices.json`, so every run uses the same rows in the same order.
- scikit-learn decides the model results and is pinned to 1.6.1, the version of the original run. A different version can move a result in the last decimal place. The original run used Google Colab with scikit-learn 1.6.1, numpy 2.0.2, pandas 2.2.2 and shap 0.52.0. A rerun on Google Colab with Python 3.13, scikit-learn 1.6.1, numpy 2.4.6, pandas 2.2.3 and shap 0.54.1 reproduced every prediction of RQ1 to RQ3 and every SHAP value to within 10<sup>-14</sup>. Steps 5 and 7 refit the leave-one-school-out model and report how many of the 843 decisions of RQ2 they reproduce; in that rerun, all 843 were reproduced.
- `versions.txt` in the zip file records the versions and the code commit of each run.
- `pipeline/phase0_qc.py` is the original cleaning step (exclusions, aligned 0 to 3 coding, split and folds). It needs the original school spreadsheets, which are not shared because they contain school names; `pipeline/prepare_from_zenodo.py` rebuilds its outputs from the Zenodo file.

## Repository contents

| File               | Purpose                                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `run_pipeline.py`  | Runs the steps in order, keeps a log of each, shows the figures in a notebook and collects the results in one zip file |
| `pipeline/`        | The scripts of the analysis steps listed above                                                                         |
| `config/`          | The locked split by coded child ID, the fold indices and the settings of the cleaning step                             |
| `requirements.txt` | Package requirements                                                                                                   |

## Licence

Code: MIT (see `LICENSE`). The English AQ-10 Child items are copyrighted by Allison, Auyeung and Baron-Cohen (2012).

## How to cite for dataset only

Study: to be added after publication.
Dataset: Sheikh, M. S., Munna, M. M. M., Habib, M. H. R., & Ahamed, M. S. (2026). _BD-AQ10: School-based AQ-10 Child data from ASD and mainstream schools in Bangladesh_ (Version 1.1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.23988059
