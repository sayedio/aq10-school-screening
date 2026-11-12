# Step 5 - School effects

scikit-learn 1.6.1, numpy 2.4.6, pandas 2.2.3; seed 42. Added after the locked RQ1-RQ4 run; uses the locked SVM settings and predictions.

## A. Reproducing the locked leave-one-school-out run

- The refit reproduced 843 of 843 locked decisions (largest probability difference 0.0000; schools with a changed decision: none).
- Pooled numbers use the locked predictions; analyses that refit the model are compared with this refit.

## B. Do schools leave a fingerprint in the answers?

### B1. School profiles (T_s5_B1_school_profile.csv, F_s5_B1_endorsement_by_school.png)

| School | Group | n | Age | Boys % | AQ total | Rule + % | SVM flagged % | ERS | Q1 definitely disagree % |
|---|---|---|---|---|---|---|---|---|---|
| S01 | special | 58 | 9.0 | 72 | 6.81 | 74.1 | 94.8 | 0.70 | 56.9 |
| S04 | special | 68 | 8.0 | 66 | 6.63 | 77.9 | 97.1 | 0.55 | 30.9 |
| S08 | special | 97 | 7.4 | 71 | 7.15 | 70.1 | 93.8 | 0.63 | 7.2 |
| S09 | special | 97 | 8.5 | 64 | 6.76 | 74.2 | 93.8 | 0.63 | 26.8 |
| S10 | special | 83 | 6.6 | 70 | 7.66 | 78.3 | 91.6 | 0.58 | 2.4 |
| S02 | mainstream | 116 | 8.1 | 50 | 2.16 | 6.0 | 0.9 | 0.35 | 0.0 |
| S03 | mainstream | 51 | 8.6 | 69 | 1.92 | 3.9 | 0.0 | 0.49 | 0.0 |
| S05 | mainstream | 60 | 8.8 | 53 | 2.45 | 3.3 | 5.0 | 0.36 | 0.0 |
| S06 | mainstream | 63 | 7.9 | 59 | 1.94 | 0.0 | 28.6 | 0.43 | 30.2 |
| S07 | mainstream | 58 | 9.0 | 45 | 2.47 | 8.6 | 19.0 | 0.46 | 1.7 |
| S11 | mainstream | 55 | 8.9 | 73 | 1.89 | 3.6 | 0.0 | 0.41 | 0.0 |
| S12 | mainstream | 37 | 8.9 | 70 | 1.68 | 5.4 | 2.7 | 0.47 | 0.0 |

### B2. Does each answer depend on the school within each group? (T_s5_B2_item_by_school.csv)

Cramer's V for answer (4 levels) by school within a group; p from 2,000 shuffles of school within the group, Holm-adjusted over the ten items. V for answer by group (special vs mainstream) is shown for comparison.

| Item | V by group | V school (special) | p Holm | V school (mainstream) | p Holm | V school (mainstream without S06) | p Holm |
|---|---|---|---|---|---|---|---|
| Q1 | 0.30 | 0.34 | 0.005 | 0.32 | 0.005 | 0.17 | 0.012 |
| Q2 | 0.47 | 0.21 | 0.005 | 0.18 | 0.006 | 0.19 | 0.012 |
| Q3 | 0.71 | 0.18 | 0.005 | 0.18 | 0.013 | 0.18 | 0.013 |
| Q4 | 0.59 | 0.21 | 0.005 | 0.25 | 0.005 | 0.21 | 0.005 |
| Q5 | 0.76 | 0.18 | 0.005 | 0.29 | 0.005 | 0.29 | 0.005 |
| Q6 | 0.71 | 0.20 | 0.005 | 0.24 | 0.005 | 0.19 | 0.005 |
| Q7 | 0.54 | 0.20 | 0.005 | 0.21 | 0.005 | 0.22 | 0.005 |
| Q8 | 0.59 | 0.19 | 0.005 | 0.23 | 0.005 | 0.24 | 0.005 |
| Q9 | 0.58 | 0.21 | 0.005 | 0.11 | 0.655 | 0.11 | 0.509 |
| Q10 | 0.74 | 0.17 | 0.005 | 0.30 | 0.005 | 0.32 | 0.005 |

### B3. Share of variance between schools, within each group (ICC(1); T_s5_B3_icc_by_school.csv)

| Variable | ICC special | p | ICC mainstream | p | ICC mainstream without S06 | p |
|---|---|---|---|---|---|---|
| Q1 | 0.343 | 0.0005 | 0.130 | 0.0005 | 0.040 | 0.004 |
| Q2 | 0.065 | 0.0005 | 0.033 | 0.008 | 0.041 | 0.003 |
| Q3 | 0.010 | 0.126 | 0.047 | 0.001 | 0.051 | 0.001 |
| Q4 | 0.080 | 0.0005 | 0.078 | 0.0005 | 0.028 | 0.019 |
| Q5 | 0.063 | 0.001 | 0.193 | 0.0005 | 0.188 | 0.0005 |
| Q6 | 0.077 | 0.0005 | 0.136 | 0.0005 | 0.082 | 0.0005 |
| Q7 | 0.105 | 0.0005 | 0.024 | 0.019 | 0.031 | 0.017 |
| Q8 | 0.072 | 0.0005 | 0.077 | 0.0005 | 0.089 | 0.0005 |
| Q9 | 0.069 | 0.0005 | 0.001 | 0.398 | 0.004 | 0.291 |
| Q10 | 0.045 | 0.001 | 0.213 | 0.0005 | 0.254 | 0.0005 |
| AQ-10 total | 0.029 | 0.013 | 0.023 | 0.022 | 0.024 | 0.028 |
| Ordinal sum | 0.008 | 0.166 | 0.027 | 0.013 | 0.026 | 0.020 |
| Extreme responding | 0.036 | 0.006 | 0.037 | 0.005 | 0.043 | 0.003 |
| SVM output (logit) | 0.041 | 0.004 | 0.135 | 0.0005 | 0.123 | 0.0005 |
| SVM flag | -0.006 | 0.727 | 0.163 | 0.0005 | 0.111 | 0.0005 |
| Rule flag | -0.006 | 0.740 | 0.001 | 0.393 | -0.009 | 0.807 |

### B4. Rater response style: share of 'definitely' answers (ERS)

- ERS mean: special 0.615, mainstream 0.410 (Mann-Whitney p = 2e-29).
- ERS alone as a score separates the groups with AUROC 0.723 [0.654, 0.777] (school-clustered interval).
- Mainstream children: false positives 0.397 (n = 34) vs true negatives 0.411 (p = 0.962); S06 0.430 vs other mainstream 0.407 (p = 0.185).
- Special-school children scoring below 6: found by the SVM only 0.599 (n = 81) vs missed by both 0.419 (n = 21; p = 0.002).

### B5. Can the ten answers identify the school within each group? (T_s5_B5_school_identifiability.csv)

Same RBF SVM settings, multiclass, 5-fold CV repeated 5 times; p from 200 shuffles of school labels.

- special, 10 items: balanced accuracy 0.500 (chance 0.200; p = 0.005). Recall by school: S08 0.63, S04 0.52, S10 0.46, S01 0.40, S09 0.48.
- special, 9 items without Q1: balanced accuracy 0.472 (chance 0.200; p = n/a). Recall by school: S08 0.60, S04 0.56, S10 0.45, S01 0.24, S09 0.52.
- mainstream, 10 items: balanced accuracy 0.356 (chance 0.143; p = 0.005). Recall by school: S02 0.86, S07 0.48, S03 0.15, S06 0.40, S05 0.28, S12 0.02, S11 0.31.
- mainstream, 9 items without Q1: balanced accuracy 0.349 (chance 0.143; p = n/a). Recall by school: S02 0.86, S07 0.41, S03 0.13, S06 0.43, S05 0.29, S12 0.01, S11 0.31.

### B6. Answer-distribution check per school and item (school vs the other schools of its group; T_s5_B6_item_drift_by_school.csv)

- 30 of 120 school-item pairs differ after Bonferroni correction. Largest total variation distances:
  - S01 (special) Q1: TVD 0.43, Bonferroni p = 1.2e-09
  - S01 (special) Q9: TVD 0.39, Bonferroni p = 8.7e-05
  - S06 (mainstream) Q6: TVD 0.38, Bonferroni p = 4e-05
  - S02 (mainstream) Q10: TVD 0.38, Bonferroni p = 1e-09
  - S10 (special) Q1: TVD 0.37, Bonferroni p = 1.5e-08
  - S02 (mainstream) Q5: TVD 0.33, Bonferroni p = 1.6e-07
  - S02 (mainstream) Q6: TVD 0.33, Bonferroni p = 1.6e-06
  - S01 (special) Q4: TVD 0.33, Bonferroni p = 0.004
  - S07 (mainstream) Q10: TVD 0.33, Bonferroni p = 0.002
  - S02 (mainstream) Q8: TVD 0.32, Bonferroni p = 1.8e-06

(A and B took 42 s)

## C. Seen versus unseen schools

### C1. Random child-level CV (schools seen) vs leave-one-school-out (school unseen)

| Validation | AUROC | Sensitivity | Specificity |
|---|---|---|---|
| Leave-one-school-out (locked) | 0.984 | 0.940 | 0.923 |
| Random 5-fold CV, 10 repeats: mean [range] | 0.993 [0.992 to 0.994] | 0.957 [0.953 to 0.960] | 0.972 [0.964 to 0.977] |
| Hold-out test set (schools seen in development) | 0.992 | 0.951 | 0.966 |

| School | Group | n | Flagged %, unseen (LOSO) | Flagged %, seen (random CV) | Hold-out flagged / n |
|---|---|---|---|---|---|
| S01 | special | 58 | 94.8 | 95.7 | 10 / 12 |
| S04 | special | 68 | 97.1 | 97.4 | 16 / 16 |
| S08 | special | 97 | 93.8 | 96.5 | 17 / 18 |
| S09 | special | 97 | 93.8 | 94.7 | 15 / 16 |
| S10 | special | 83 | 91.6 | 94.5 | 19 / 19 |
| S02 | mainstream | 116 | 0.9 | 0.2 | 0 / 21 |
| S03 | mainstream | 51 | 0.0 | 0.0 | 0 / 11 |
| S05 | mainstream | 60 | 5.0 | 3.2 | 1 / 8 |
| S06 | mainstream | 63 | 28.6 | 3.2 | 0 / 13 |
| S07 | mainstream | 58 | 19.0 | 11.6 | 2 / 15 |
| S11 | mainstream | 55 | 0.0 | 0.5 | 0 / 8 |
| S12 | mainstream | 37 | 2.7 | 3.2 | 0 / 12 |

### C2. Local adaptation: k children of the held-out school added to training (10 random draws each)

| k | Sensitivity, mean [range] | Specificity, mean [range] | S06 flagged %, mean |
|---|---|---|---|
| 0 (LOSO) | 0.940 | 0.923 | 28.6 |
| 5 | 0.943 [0.939 to 0.952] | 0.953 [0.941 to 0.963] | 10.0 |
| 10 | 0.944 [0.938 to 0.952] | 0.957 [0.949 to 0.968] | 6.4 |
| 20 | 0.941 [0.934 to 0.954] | 0.966 [0.957 to 0.980] | 5.6 |

### C3. Leave one special and one mainstream school out together (35 test folds with both groups)

- AUROC SVM: 0.992 [IQR 0.984 to 0.995; range 0.957 to 1.000]; AUROC total score: 0.968 [IQR 0.956 to 0.973; range 0.922 to 0.991]; SVM higher in 32 of 35.
- Sensitivity SVM: 0.948 [IQR 0.938 to 0.964; range 0.916 to 0.985]; rule: 0.742 [IQR 0.741 to 0.779; range 0.701 to 0.783]; SVM higher in 35 of 35.
- Specificity SVM: 0.991 [IQR 0.860 to 1.000; range 0.683 to 1.000]; rule: 0.961 [IQR 0.940 to 0.967; range 0.914 to 1.000]; SVM lower in 12 of 35.
- Folds containing S06: SVM specificity 0.714 [IQR 0.698 to 0.810; range 0.683 to 0.857]; other folds 1.000 [IQR 0.967 to 1.000; range 0.810 to 1.000].

(C took 15 s)

## D. Mechanism of the S06 false positives

- In this refit the model flagged 18 of 63 children in S06 (locked run 18).

### D1. Counterfactual: every child's answer to item 1 replaced, all other answers kept

| Item 1 answer | Still flagged, of the 18 S06 children | S06 flagged, of 63 | Other mainstream flagged % | Special flagged % |
|---|---|---|---|---|
| as given | 18 | 18 | 4.2 | 94.0 |
| definitely agree | 6 | 8 | 11.4 | 96.0 |
| slightly agree | 5 | 5 | 5.0 | 94.0 |
| slightly disagree | 6 | 7 | 2.9 | 93.1 |
| definitely disagree | 18 | 42 | 6.1 | 94.8 |

### D2. Does a higher trait value on each item always raise the prediction? (T_s5_D2_*.csv)

- Items whose average effect is not monotone (all children): Q1, Q2, Q4, Q8, Q9.
| Item | Mean p at 0 / 1 / 2 / 3 (all children) | Children with any decrease % |
|---|---|---|
| Q1 | 0.546 / 0.476 / 0.482 / 0.526 | 55.9 |
| Q2 | 0.502 / 0.480 / 0.514 / 0.647 | 30.1 |
| Q3 | 0.465 / 0.472 / 0.529 / 0.711 | 7.9 |
| Q4 | 0.485 / 0.476 / 0.543 / 0.790 | 17.4 |
| Q5 | 0.456 / 0.467 / 0.519 / 0.628 | 9.1 |
| Q6 | 0.475 / 0.485 / 0.549 / 0.733 | 9.0 |
| Q7 | 0.460 / 0.456 / 0.498 / 0.596 | 28.2 |
| Q8 | 0.504 / 0.485 / 0.512 / 0.649 | 22.3 |
| Q9 | 0.526 / 0.483 / 0.492 / 0.620 | 44.0 |
| Q10 | 0.445 / 0.464 / 0.531 / 0.684 | 3.1 |

### D3. What the training data taught about item 1 (T_s5_D3_training_composition.csv)

- When S06 was held out, 'definitely disagree' to item 1 appeared for 89 special-school and 1 mainstream children in training, so P(special | that answer) = 0.989.
- 'Definitely disagree' to item 1 by special school: {'S01': 33, 'S04': 21, 'S08': 7, 'S09': 26, 'S10': 2}.

### D4. Which training school taught the pattern? (one training school removed at a time; T_s5_D4_*.csv)

Entries are the change in the held-out school's flag rate, in percentage points, when one training school is removed.

- S06 (baseline 18 of 63 flagged): without S01 19, without S02 18, without S03 21, without S04 9, without S05 13, without S07 23, without S08 20, without S09 12, without S10 18, without S11 19, without S12 18.
- Without the three special schools with the most 'definitely disagree' answers to item 1 (S01, S04, S09): 4 of 63 flagged in S06.
- Largest changes anywhere (held-out school, removed school, change in pp):
  - S06, without S04: -14.3
  - S06, without S09: -9.5
  - S06, without S05: -7.9
  - S06, without S07: +7.9
  - S12, without S07: +5.4
  - S07, without S06: +5.2
  - S07, without S09: -5.2
  - S10, without S08: -4.8

### D5. Each item removed in turn (LOSO refit; T_s5_D5_drop_one_item.csv)

| Item removed | AUROC | Change [school-clustered 95% interval] | Sens | Spec | S06 flagged | S07 flagged | FP other mainstream |
|---|---|---|---|---|---|---|---|
| none | 0.984 | +0.000 [+0.000, +0.000] | 0.940 | 0.923 | 18 | 11 | 16 |
| Q1 | 0.986 | +0.002 [-0.004, +0.008] | 0.943 | 0.950 | 6 | 9 | 16 |
| Q2 | 0.983 | -0.001 [-0.004, +0.002] | 0.938 | 0.909 | 21 | 14 | 19 |
| Q3 | 0.980 | -0.004 [-0.009, +0.001] | 0.933 | 0.920 | 17 | 11 | 18 |
| Q4 | 0.978 | -0.006 [-0.014, -0.000] | 0.933 | 0.916 | 18 | 13 | 19 |
| Q5 | 0.986 | +0.002 [-0.001, +0.006] | 0.935 | 0.948 | 12 | 8 | 11 |
| Q6 | 0.980 | -0.003 [-0.009, +0.001] | 0.940 | 0.907 | 22 | 12 | 19 |
| Q7 | 0.976 | -0.008 [-0.017, -0.001] | 0.931 | 0.905 | 21 | 14 | 21 |
| Q8 | 0.981 | -0.003 [-0.006, +0.001] | 0.935 | 0.909 | 22 | 11 | 18 |
| Q9 | 0.981 | -0.003 [-0.006, +0.000] | 0.943 | 0.920 | 19 | 11 | 16 |
| Q10 | 0.981 | -0.003 [-0.007, +0.001] | 0.935 | 0.923 | 19 | 11 | 15 |

### D6. S06 compared with the other mainstream schools (T_s5_D6_*.csv)

- Boys: S06 59% vs 58% (Fisher p = 0.891).
| Variable | S06 mean | Other mainstream mean | Rank-biserial r | p (Holm) |
|---|---|---|---|---|
| Age | 7.873 | 8.645 | -0.25 | 0.012 |
| AQ-10 total | 1.937 | 2.135 | -0.01 | 1.000 |
| Ordinal sum | 8.302 | 8.931 | -0.10 | 1.000 |
| Extreme responding | 0.430 | 0.407 | +0.10 | 1.000 |
| Q1 | 1.556 | 2.236 | -0.31 | 0.00014 |
| Q2 | 0.841 | 0.788 | +0.05 | 1.000 |
| Q3 | 0.873 | 0.737 | +0.12 | 0.715 |
| Q4 | 0.413 | 0.801 | -0.34 | 1.8e-05 |
| Q5 | 0.921 | 0.390 | +0.30 | 6.1e-05 |
| Q6 | 0.317 | 0.764 | -0.39 | 5.2e-07 |
| Q7 | 1.127 | 1.151 | -0.01 | 1.000 |
| Q8 | 0.651 | 0.698 | -0.01 | 1.000 |
| Q9 | 0.889 | 0.918 | -0.02 | 1.000 |
| Q10 | 0.714 | 0.448 | +0.17 | 0.088 |

### D7. AQ-10 totals of children by decision (T_s5_D7_who_the_svm_adds.csv)

- special, both flag (n = 298): totals 6: 61, 7: 56, 8: 72, 9: 51, 10: 58
- special, SVM only (n = 81): totals 3: 3, 4: 30, 5: 48
- special, rule only (n = 3): totals 6: 1, 7: 2
- special, neither (n = 21): totals 0: 1, 1: 2, 2: 2, 3: 4, 4: 8, 5: 4
- mainstream, both flag (n = 4): totals 6: 2, 7: 1, 8: 1
- mainstream, SVM only (n = 30): totals 1: 3, 2: 7, 3: 8, 4: 5, 5: 7
- mainstream, rule only (n = 16): totals 6: 12, 7: 4
- mainstream, neither (n = 390): totals 0: 4, 1: 169, 2: 153, 3: 35, 4: 21, 5: 8

(D took 76 s; step 5 total 134 s)

