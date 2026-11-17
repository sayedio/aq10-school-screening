# School-level tests: numbers behind the clustered analyses

Excerpt of the step 6 report from the verified rerun (scikit-learn 1.6.1, seed 42), limited to numbers that the paper reports. `pipeline/step6_robustness.py` writes the full report for anyone with approved access to the data.

## Paired decisions with the school as the unit (Sections IV-B and IV-C)

b, children flagged by the SVM only; c, by the comparator only. The exact sign-flip test uses school-level differences; with five ASD schools its smallest possible p is 0.0625.

| Comparison | Children | b vs c | p, child level | p, clustered (Durkalski) | p, exact sign-flip |
|---|---|---|---|---|---|
| Hold-out: SVM vs rule (S >= 6) | all | 20 vs 1 | 8.6e-05 | 0.042 | 0.031 |
| Unseen schools: SVM vs rule (S >= 6) | all | 111 vs 19 | 1.4e-15 | 0.027 | 0.023 |
| Unseen schools: SVM vs rule (S >= 6) | ASD schools | 81 vs 3 | 4.4e-17 | 0.032 | 0.062 |
| Unseen schools: SVM vs S >= 5 | ASD schools | 33 vs 7 | 7.7e-05 | 0.037 | 0.062 |

## Hold-out set, school-clustered interval (Section IV-B)

- Sensitivity difference, SVM minus rule: 0.222 [0.127, 0.317]

## AQ-10 cut-offs (Sections III-B and IV-B, Table V)

| Cut-off | Hold-out sensitivity | Hold-out specificity | All 843: sensitivity | All 843: specificity |
|---|---|---|---|---|
| S >= 4 | 0.963 | 0.909 | 0.970 | 0.861 |
| S >= 5 | 0.864 | 0.943 | 0.876 | 0.920 |
| S >= 6 (published cut-off) | 0.728 | 0.977 | 0.747 | 0.955 |
| S >= 7 ('more than 6' wording) | 0.568 | 0.977 | 0.593 | 0.986 |

The rule needs no training, so its results on all 843 children are its results in unseen schools.

## SVM settings re-tuned inside every fold (Table VI)

- AUROC 0.983, sensitivity 0.945, specificity 0.902; S06 flagged 18 of 63.
- AUROC difference from the locked run: -0.001 [-0.004, 0.004].
