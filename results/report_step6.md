# Step 6 - Robustness

scikit-learn 1.6.1, numpy 2.4.6, pandas 2.2.3; seed 42. Added after the locked RQ1-RQ4 run; uses the locked predictions.

## A. Paired decisions with the school as the unit (T_s6_A_clustered_mcnemar.csv)

b = flagged by the first method only, c = by the second only. The exact sign-flip test uses school-level differences (with 5 special schools its smallest possible p is 0.0625).

| Comparison | Children | b vs c | Schools b>c / b<c | p child-level | p clustered (Durkalski) | p exact sign-flip |
|---|---|---|---|---|---|---|
| Unseen schools: SVM vs rule (S>=6) | all children | 111 vs 19 | 8 / 4 | 1.4e-15 | 0.027 | 0.023 |
| Unseen schools: SVM vs rule (S>=6) | special (sensitivity) | 81 vs 3 | 5 / 0 | 4.4e-17 | 0.032 | 0.062 |
| Unseen schools: SVM vs rule (S>=6) | mainstream (specificity) | 30 vs 16 | 3 / 4 | 0.055 | 0.487 | 0.719 |
| Unseen schools: SVM vs S>=5 | all children | 56 vs 31 | 7 / 5 | 0.010 | 0.289 | 0.345 |
| Unseen schools: SVM vs S>=5 | special (sensitivity) | 33 vs 7 | 5 / 0 | 7.7e-05 | 0.037 | 0.062 |
| Unseen schools: SVM vs S>=5 | mainstream (specificity) | 23 vs 24 | 2 / 5 | 1.000 | 0.960 | 1.000 |
| Unseen schools: SVM vs S>=4 | all children | 21 vs 60 | 2 / 10 | 2.4e-05 | 0.096 | 0.108 |
| Unseen schools: SVM vs S>=4 | special (sensitivity) | 3 vs 15 | 1 / 4 | 0.010 | 0.109 | 0.188 |
| Unseen schools: SVM vs S>=4 | mainstream (specificity) | 18 vs 45 | 1 / 6 | 0.001 | 0.224 | 0.266 |
| Unseen schools: 14 items vs 10 items | all children | 22 vs 12 | 5 / 1 | 0.123 | 0.077 | 0.125 |
| Unseen schools: 14 items vs 10 items | special (sensitivity) | 12 vs 4 | 4 / 0 | 0.080 | 0.088 | 0.125 |
| Unseen schools: 14 items vs 10 items | mainstream (specificity) | 10 vs 8 | 1 / 1 | 0.814 | 0.527 | 1.000 |
| Hold-out: SVM vs rule (S>=6) | all children | 20 vs 1 | 6 / 0 | 8.6e-05 | 0.042 | 0.031 |
| Hold-out: SVM vs rule (S>=6) | special (sensitivity) | 18 vs 0 | 5 / 0 | 6.2e-05 | 0.052 | 0.062 |
| Hold-out: SVM vs rule (S>=6) | mainstream (specificity) | 2 vs 1 | 1 / 0 | 1.000 | 0.317 | 1.000 |

## B. Hold-out set with school-clustered intervals (5 special and 7 mainstream schools; T_s6_B_*.csv)

- Sensitivity, SVM: 0.951 [0.897, 0.989]
- Specificity, SVM: 0.966 [0.915, 1.000]
- Sensitivity, rule: 0.728 [0.616, 0.835]
- Specificity, rule: 0.977 [0.930, 1.000]
- Sensitivity difference: 0.222 [0.127, 0.317]
- AUROC, SVM: 0.992 [0.982, 0.999]
- AUROC, total score: 0.976 [0.947, 0.996]

## C. Fourteen vs ten items, school-clustered intervals

- Sensitivity difference (14 - 10): +0.020 [+0.007, +0.035]
- Specificity difference (14 - 10): -0.005 [-0.021, +0.006]
- AUROC difference (14 - 10): +0.000 [-0.006, +0.005]

## D. AQ-10 cut-offs (the rule needs no training, so 'all 843' is the same as unseen schools)

| Cut-off | Note | Hold-out sens | Hold-out spec | All 843 sens [school CI] | All 843 spec [school CI] |
|---|---|---|---|---|---|
| S >= 4 |  | 0.963 [0.897, 0.987] | 0.909 [0.831, 0.953] | 0.970 [0.955, 0.985] | 0.861 [0.816, 0.893] |
| S >= 5 |  | 0.864 [0.773, 0.922] | 0.943 [0.874, 0.975] | 0.876 [0.860, 0.901] | 0.920 [0.886, 0.951] |
| S >= 6 | Allison et al. 2012 cut-off (6 or more) | 0.728 [0.623, 0.813] | 0.977 [0.921, 0.994] | 0.747 [0.719, 0.774] | 0.955 [0.938, 0.975] |
| S >= 7 | scoring-sheet wording (more than 6) | 0.568 [0.459, 0.670] | 0.977 [0.921, 0.994] | 0.593 [0.552, 0.633] | 0.986 [0.972, 0.996] |
| S >= 8 |  | 0.444 [0.341, 0.553] | 0.989 [0.938, 0.998] | 0.449 [0.382, 0.530] | 0.998 [0.992, 1.000] |

## E. Nested leave-one-school-out: the SVM grid (12 settings) re-tuned on the 11 training schools of every fold

Inner selection: pooled inner leave-one-school-out AUROC (tie-break: sensitivity).

- Settings chosen: S01 C=0.1, rbf, gamma=scale; S02 C=0.1, rbf, gamma=scale; S03 C=0.1, rbf, gamma=scale; S04 C=0.1, rbf, gamma=scale; S05 C=1.0, rbf, gamma=scale; S06 C=1.0, rbf, gamma=scale; S07 C=0.1, rbf, gamma=scale; S08 C=0.1, rbf, gamma=scale; S09 C=0.1, rbf, gamma=scale; S10 C=0.1, rbf, gamma=scale; S11 C=0.1, rbf, gamma=scale; S12 C=0.1, rbf, gamma=scale.
- Nested: AUROC 0.983 [0.971, 0.992], sensitivity 0.945 [0.934, 0.961], specificity 0.902 [0.825, 0.961].
- Locked: AUROC 0.984, sensitivity 0.940, specificity 0.923. AUROC difference (nested - locked) -0.001 [-0.004, 0.004].
- Same decision for 822 of 843 children; S06 flagged nested 18 vs locked 18.

(E took 79 s; step 6 total 95 s)

