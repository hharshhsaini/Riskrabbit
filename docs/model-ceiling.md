# Why the model stops near ROC-AUC 0.65 — Step 13b write-up

Final numbers: 4,594 merged PRs from 16 public repositories, collected
2026-09-14, labelled with the rules in `backend/ml/label.py`. Reproduce with, from
`backend/`: `python -m ml.label`, `python -m ml.audit`, `python -m ml.train`,
`python -m ml.experiments`.

## Result

**The honest headline is ROC-AUC ≈ 0.65**, measured with leave-repositories-out
cross-validation: every repository is scored by a model that never saw any of its
PRs. No configuration reached the PRD target of 0.75 under that test.

| Evaluation | Baseline (logistic regression) | XGBoost |
|---|---|---|
| 5-fold leave-repos-out CV, mean ± std | **0.667 ± 0.060** | 0.650 ± 0.068 |
| Pooled over all out-of-fold predictions | 0.640 | 0.624 |
| Fixed held-out repos (aiohttp, pytest, wagtail) | 0.734 | 0.795 |
| PR-AUC on the fixed held-out repos (random = 0.029) | 0.066 | 0.119 |

The fixed held-out split scores much higher than cross-validation. It contains
only **26 risky PRs**, so one split can land well above or below the true
average; the five CV folds (20–45 risky PRs each) range from 0.56 to 0.74. The
0.795 is reported, but it is not used to claim the target.

On the held-out split, XGBoost's top-ranked PRs are risky about **4× more often
than a random pick** (PR-AUC 0.119 vs 0.029).

## The playbook's five fixes

Pooled leave-repos-out ROC-AUC, all 16 features. `*` = changes the label, i.e.
what counts as risky, not only the model.

| Configuration | Risky PRs | Baseline | XGBoost |
|---|---|---|---|
| Current rules, 7-day hotfix window | 163 (3.5%) | **0.640** | 0.624 |
| Fix 1: 14-day hotfix window `*` | 220 (4.8%) | 0.635 | 0.660 |
| Fix 3: size as within-repo percentile | 163 (3.5%) | 0.638 | 0.633 |
| Fixes 1 + 3 `*` | 220 (4.8%) | 0.649 | **0.663** |
| Fix 5: reverts only `*` | 13 (0.3%) | 0.523 | 0.530 |

Without `author_pr_count` the same rows score 0.618 / 0.600, 0.608 / 0.634,
0.628 / 0.625, 0.637 / 0.621.

1. **14-day hotfix window** — 35% more risky PRs (163 → 220) but no better
   ranking for the baseline; +0.04 for XGBoost.
2. **Drop Signal C** — already done; bug-issue data was never collected.
3. **Repository-relative size** — no meaningful change (±0.01). Raw size is not
   what holds the model back.
4. **More repositories** — going from 7 to 16 repositories (2,100 → 4,594 PRs)
   did not raise the ceiling: 0.644 → 0.640 for the baseline.
5. **Reverts only** — just 13 reverted PRs in 4,594 (0.3%). Unlearnable: the
   scores are close to random.

Best of all: **0.663** (fixes 1 + 3, XGBoost), +0.023 over the current rules.

## Why there is a ceiling

The label is a proxy. A PR is marked risky when public history shows a revert, or
a follow-up fix by the same author touching the same source files within 7 days.
Reading labelled examples by hand showed that only about **45–50% of risky labels
are real deployment problems**. The rest are planned follow-up work that looks like
a hotfix — for example one maintainer landing a string of related fixes in quick
succession.

That noise caps the score regardless of the model. If a fraction *p* of risky
labels are real and the rest behave like ordinary PRs, then even a perfect model —
one that ranks every truly risky PR first — can only reach:

> **ROC-AUC ≤ p × 1.0 + (1 − p) × 0.5 = 0.5 + p / 2**

It ranks the real risky PRs correctly (AUC 1.0) and the mislabelled ones at chance
(AUC 0.5). With *p* ≈ 0.45–0.50 the ceiling is **about 0.72–0.75**, before
counting risky PRs that were never labelled at all (reverts of PRs outside the
sample, incidents that never reached git history), which lower it further.

The observed **0.65** sits below that ceiling. Getting higher needs better labels
— real incident data such as deployment rollbacks or error-tracker spikes linked to
releases — which this project deliberately does not use.

## Labelling bug found and fixed on the full dataset

Inspecting labels across all 16 repositories showed that **backports and
cherry-picks of a PR were counted as hotfixes for that same PR**. For example,
aiohttp#11290 was marked risky only because of
`[PR #11290/16703bb9 backport][3.13] Fix file uploads…`, a copy of the change onto
a release branch. Ignoring backports, cherry-picks and commits repeating the PR's
own title removed 355 copied commits and cut risky PRs from 211 to 163
(aiohttp 41 → 15, pylint 24 → 2). All numbers in this document use the fixed labels.

## Limitations to state in the report

- **Label precision ~45–50%** (hand-checked). Every metric is measured against
  these noisy labels.
- **Positive rate 3.5%**, below the playbook's 5% guideline, and it varies from
  0.3% to 13.3% by repository because commit-message conventions and team
  structure differ. That limits how well a model transfers between repositories.
- **Author bias.** The same-author hotfix rule gives prolific maintainers more
  risky labels: the five authors with most risky PRs hold 54.6% of all positives,
  and one tox maintainer alone accounts for 39. `author_pr_count` is the strongest
  single feature (0.687) largely for that reason; it is recommended to drop it and
  report the lower, cleaner score.
- **Quickly merged PRs look riskier.** Median time open is 3.4 hours for risky
  PRs versus 17.5 for safe ones. This may be real (less review) or partly the same
  author bias (maintainers merge their own PRs fast).
- **XGBoost's tree count is not reliable.** Early stopping chose 29 trees, but its
  validation PR-AUC (0.028) was no better than random (0.026).
- **No leakage from review comments.** Only 2% of review comments were posted after
  merge, and a 40-PR check scored the same with them removed (0.679 vs 0.679).
