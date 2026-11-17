# School effects: numbers behind Sections IV-G and IV-H

Excerpt of the step 5 report from the verified rerun (scikit-learn 1.6.1, seed 42), limited to numbers that the paper reports. School codes only. `pipeline/step5_school_effects.py` writes the full report for anyone with approved access to the data.

## Reproduction

- The refit reproduced 843 of 843 locked leave-one-school-out decisions.

## Seen versus unseen schools (Table VI, Fig. 5(c))

| Validation | AUROC | Sensitivity | Specificity | S06 flagged % |
|---|---|---|---|---|
| Random child-level CV, 5 folds x 10 repeats (school seen), mean | 0.993 | 0.957 | 0.972 | 3.2 |
| Hold-out test set (school seen) | 0.992 | 0.951 | 0.966 | 0 (0 of 13) |
| Leave one school out (school unseen) | 0.984 | 0.940 | 0.923 | 28.6 |

Children flagged in each mainstream school (%):

| School | Unseen (leave one school out) | Seen (random CV) |
|---|---|---|
| S02 | 0.9 | 0.2 |
| S03 | 0.0 | 0.0 |
| S05 | 5.0 | 3.2 |
| S06 | 28.6 | 3.2 |
| S07 | 19.0 | 11.6 |
| S11 | 0.0 | 0.5 |
| S12 | 2.7 | 3.2 |

## Local adaptation (Table VI)

Children of the held-out school added to training; means of ten random draws.

| Local children added | Sensitivity | Specificity | S06 flagged % |
|---|---|---|---|
| 0 (leave one school out) | 0.940 | 0.923 | 28.6 |
| 5 | 0.943 | 0.953 | 10.0 |
| 10 | 0.944 | 0.957 | 6.4 |
| 20 | 0.941 | 0.966 | 5.6 |

## One ASD and one mainstream school left out together (Table VI, 35 folds)

- Medians: AUROC 0.992 (AQ-10 total 0.968), sensitivity 0.948, specificity 0.991.
- The SVM's AUROC exceeded the total score's in 32 of 35 folds and its sensitivity exceeded the rule's in 35 of 35; its specificity fell below the rule's in 12 of 35.

## School signature in the answers (Section IV-H)

- Answer by school, within each group (Cramer's V, Holm-adjusted permutation p): all ten items differed between ASD schools and nine of ten between mainstream schools (not Q9); V was 0.17 to 0.34 for the items that differed.
- The ten answers identified a child's school with a balanced accuracy of 0.500 among ASD schools (chance 0.200) and 0.356 among mainstream schools (chance 0.143); p = 0.005 for both (200 label permutations).
- 30 of 120 school-item answer distributions differed from those of the other schools of the same type (Bonferroni).
- Share of variance between mainstream schools, ICC(1): SVM output 0.135 (0.123 without S06), AQ-10 total 0.023, SVM decision 0.163, rule decision 0.001.
- Share of answers in the two 'definitely' categories: 0.615 for ASD-school and 0.410 for mainstream children. ASD-school children scoring below 6: found by the SVM only 0.599 (n = 81) against missed by both 0.419 (n = 21; p = 0.002). Mainstream false positives 0.397 against true negatives 0.411.

## False positives in S06 (Section IV-G)

- 'Definitely disagree' to item 1: S01 56.9%, S04 30.9%, S08 7.2%, S09 26.8%, S10 2.4%; S06 30.2%; other mainstream schools 0.0% to 1.7%.
- With S06 held out, 89 of the 90 training children who gave this answer were in ASD schools.
- Item 1 answer of the 18 flagged S06 children replaced by another answer: 6, 5 and 6 of them still flagged (definitely agree, slightly agree, slightly disagree).
- One training school removed: S06 flagged 9 without S04 and 12 without S09, and 4 without S01, S04 and S09 together. No other removal changed a held-out school's flag rate by more than 7.9 percentage points.
- One item removed: S06 flagged 12 to 22 with any item other than item 1 removed, and 6 without item 1.
- S06 against the other mainstream schools: answers to items 1, 4, 5 and 6 differed (Holm p <= 0.00014); mean AQ-10 total 1.94 against 2.14.
