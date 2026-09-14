# Why the model stops near ROC-AUC 0.65 — Step 13b write-up

> **Status: preliminary.** Numbers below come from 2,100 PRs in the 7 repositories
> fully collected on 2026-09-14. `backend/ml/data/experiments_summary.txt` is
> regenerated automatically on the full 16-repository dataset; update the tables
> from it before this goes into the report.

## Result

No configuration reached the PRD target of ROC-AUC 0.75. The best model is the
logistic-regression baseline, not XGBoost.

All scores use **leave-repositories-out cross-validation**: every repository is
scored by a model that never saw any of its PRs. "Pooled" ranks all
out-of-fold predictions together, so every risky PR counts.

| Configuration | Risky PRs | Baseline ROC-AUC | XGBoost ROC-AUC | Best PR-AUC (random) |
|---|---|---|---|---|
| Current rules, 7-day hotfix window | 101 (4.8%) | **0.644** | 0.533 | 0.116 (0.048) |
| Fix 1: 14-day hotfix window | 128 (6.1%) | **0.652** | 0.539 | 0.136 (0.061) |
| Fix 3: size features as within-repo percentiles | 101 (4.8%) | 0.635 | 0.567 | 0.114 (0.048) |
| Fixes 1 + 3 | 128 (6.1%) | 0.649 | 0.574 | 0.142 (0.061) |
| Fix 5: reverts only | 5 (0.2%) | too few to score | — | — |

Without `author_pr_count` (recommended, see below) the baseline scores
0.588 / 0.608 / 0.589 / 0.619 for the same four rows.

The playbook's five fixes, in order:

1. **Widen the hotfix window to 14 days** — +0.008. More risky PRs (101 → 128),
   no real gain in ranking.
2. **Drop Signal C** — already done; bug-issue data was never collected, so the
   signal most likely to point the wrong way was never used.
3. **Repository-relative size features** — helps XGBoost (+0.03 to +0.05),
   neutral for the baseline. Raw PR size does partly encode repository size, but
   removing that is not the bottleneck.
4. **More repositories** — pending the full 16-repository run (≈4,700 PRs, roughly
   twice the risky examples). The effect of 2× data on the fixes above has been
   small, so a jump to 0.75 is not expected.
5. **Reverts only** — only 5 reverted PRs in 2,100. Reverts are rare (~1% of
   merges) and the PR sample was spread evenly over two years rather than
   targeted at reverted PRs, so this label cannot be trained on without a
   separate, targeted collection.

## Why there is a ceiling

The label is a proxy. A PR is marked risky when public history suggests a
follow-up fix or a revert. Reading labelled examples by hand showed that only
about **35–50% of "risky" labels are real deployment problems**; the rest are
planned follow-up work that happens to look like a hotfix (for example, one
author landing a series of related fixes minutes apart).

That noise sets a hard limit on the score, independent of the model. Suppose a
fraction *p* of the risky labels are real and the rest behave like ordinary PRs.
Even a perfect model — one that ranks every truly risky PR above every safe one —
can only score:

> **ROC-AUC ≤ p × 1.0 + (1 − p) × 0.5 = 0.5 + p / 2**

because it ranks the real risky PRs correctly (AUC 1.0) and the mislabelled ones
no better than chance (AUC 0.5). With *p* between 0.35 and 0.50 the ceiling is
**about 0.68–0.75**, before counting the risky PRs that were never labelled at
all (reverts of PRs outside the sample, fixes by other people with no link to the
PR, incidents that never reached git history), which lower it further.

The observed **0.65** sits just below that ceiling. The model is not failing to
learn available signal; the proxy labels do not contain much more.

## What this means

- The honest headline is **ROC-AUC ≈ 0.65 with leave-repositories-out
  validation**, and **PR-AUC about 2.4× the random baseline** (0.116 vs 0.048):
  the model's top-ranked PRs are risky more than twice as often as a random pick.
- A random train/test split was deliberately avoided. It would let the model
  learn repository quirks (for example, one repository's commit-message style)
  and report a much higher, meaningless number.
- XGBoost underperforms the linear baseline because there are only ~100–230
  risky examples and the labels are noisy; trees overfit the noise.
- Reaching 0.75 needs better labels, not a better model: real incident data
  (deployment rollbacks, error-tracker spikes linked to releases) — exactly what
  this project deliberately excludes by using only public data.

## Checks behind these numbers

- The strongest feature, `review_comment_count`, was checked for leakage: only 2%
  of review comments were posted after merge, and scores were identical with them
  removed (0.679 vs 0.679 on a 40-PR sample).
- `author_pr_count` adds about +0.05 but is biased by the labelling rule (the five
  authors with most risky PRs hold 73.5% of all positives) and is computed
  differently in training and in the live API. Recommended: drop it and report
  the lower, cleaner number.
- Label rates vary by repository culture (0.3% to 8.7%) because commit-message
  conventions differ, which also limits cross-repository transfer.
