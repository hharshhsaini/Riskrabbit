"""Step 13b: try the playbook's fixes for a model below ROC-AUC 0.75.

Run from backend/ once collection has finished (it re-labels from the raw data):
    ./venv/bin/python -m ml.experiments
    ./venv/bin/python -m ml.experiments --drop author_pr_count

The five fixes, in playbook order, and how each is handled:
  1. Widen the hotfix window from 7 to 14 days       -> tested
  2. Drop Signal C                                   -> already done: never collected
  3. Replace files_changed and total_churn with their
     percentile within the PR's own repository        -> tested (experiment only, see below)
  4. Collect more repositories                        -> judged on the full 16-repo run;
                                                         going beyond 16 means more collection
  5. Use reverts (Signal A) only                      -> tested

Every configuration is scored identically: leave-repositories-out cross-validation.
"Pooled" ROC-AUC ranks all out-of-fold predictions together, so it uses every risky
PR and is steadier than a mean over a few small folds.

Two cautions when reading results:
  - Fixes 1 and 5 change the LABEL, i.e. what is being predicted. A higher score
    for a different target is not automatically a better risk model.
  - Repo percentiles here are an experiment computed outside features.py. Shipping
    them means moving the computation into features.build_feature_vector with a
    per-repository reference distribution available at serving time.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from features import FEATURE_ORDER
from ml import audit, label, train

TARGET_ROC_AUC = 0.75
CV_XGBOOST_TREES = 100
MIN_POSITIVES_TO_SCORE = 10
REPO_PERCENTILE_FEATURES = ("files_changed", "total_churn")
RESULTS_FILENAME = "experiments_13b.json"

CONFIGURATIONS = [
    {"id": "0", "name": "current rules (7-day hotfix window)", "window": 7, "signals": ("reverted", "hotfix"), "repo_percentiles": False, "changes_label": False},
    {"id": "1", "name": "fix 1: 14-day hotfix window", "window": 14, "signals": ("reverted", "hotfix"), "repo_percentiles": False, "changes_label": True},
    {"id": "3", "name": "fix 3: repo percentiles, 7-day window", "window": 7, "signals": ("reverted", "hotfix"), "repo_percentiles": True, "changes_label": False},
    {"id": "1+3", "name": "fixes 1+3: repo percentiles, 14-day window", "window": 14, "signals": ("reverted", "hotfix"), "repo_percentiles": True, "changes_label": True},
    {"id": "5", "name": "fix 5: reverts only", "window": 7, "signals": ("reverted",), "repo_percentiles": False, "changes_label": True},
    {"id": "5+3", "name": "fixes 5+3: reverts only, repo percentiles", "window": 7, "signals": ("reverted",), "repo_percentiles": True, "changes_label": True},
]


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    window = json.loads((args.data_dir / "cache" / "window.json").read_text())
    prs_by_repo = label._group_by_repo(args.data_dir / "raw_prs.jsonl")
    commits_by_repo = label._group_by_repo(args.data_dir / "raw_commits.jsonl")
    repos = [
        repo for repo in sorted(prs_by_repo)
        if label._commits_complete(args.data_dir, repo, len(commits_by_repo.get(repo, [])))
    ]
    features = [name for name in FEATURE_ORDER if name not in set(args.drop)]

    print(f"Step 13b experiments on {sum(len(prs_by_repo[r]) for r in repos)} PRs from {len(repos)} fully collected repositories")
    print(f"features: {len(features)}" + (f" (dropped {', '.join(args.drop)})" if args.drop else ""))
    print("fix 2 (drop Signal C): already done, issues were never collected")
    print(f"fix 4 (more repositories): judged on the full run; this run has {len(repos)} repositories")

    results = []
    for config in CONFIGURATIONS:
        if config["window"] > window["label_maturity_days"]:
            print(f"skipping {config['name']}: commit file lists only cover {window['label_maturity_days']} days")
            continue
        records = []
        for repo in repos:
            labeled, _ = label.label_repository(
                prs_by_repo[repo], commits_by_repo[repo],
                hotfix_window_days=config["window"], signals=config["signals"],
            )
            records.extend(labeled)
        dataset = audit.build_dataset(records)
        if config["repo_percentiles"]:
            dataset = with_repo_percentiles(dataset)
        results.append({**config, "signals": list(config["signals"]), **score(dataset, features)})

    _print_table(results)
    _print_verdict(results)

    output = args.data_dir / (RESULTS_FILENAME if not args.drop else RESULTS_FILENAME.replace(".json", f"_without_{'_'.join(args.drop)}.json"))
    output.write_text(json.dumps({"features": features, "repositories": repos, "results": results}, indent=2) + "\n")
    print(f"\nSaved {output}")


def with_repo_percentiles(dataset: pd.DataFrame) -> pd.DataFrame:
    """Replace size counts with their percentile rank inside the PR's own repository."""
    ranked = dataset.copy()
    for name in REPO_PERCENTILE_FEATURES:
        ranked[name] = ranked.groupby(train.REPO_COLUMN)[name].rank(pct=True, method="average")
    return ranked


def score(dataset: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    y = dataset[train.LABEL_COLUMN].to_numpy()
    summary: dict[str, Any] = {
        "rows": len(dataset),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
    }
    repos = dataset[train.REPO_COLUMN]
    if summary["positives"] < MIN_POSITIVES_TO_SCORE or repos.nunique() < 2:
        return {**summary, "skipped": f"fewer than {MIN_POSITIVES_TO_SCORE} risky PRs, too few to score"}

    predictions = {"baseline": np.full(len(dataset), np.nan), "xgboost": np.full(len(dataset), np.nan)}
    folds = []
    for train_index, test_index in GroupKFold(n_splits=min(train.CV_MAX_FOLDS, repos.nunique())).split(dataset, groups=repos):
        fit, held = dataset.iloc[train_index], dataset.iloc[test_index]
        if fit[train.LABEL_COLUMN].nunique() < 2:
            continue
        models = {
            "baseline": train.build_baseline(),
            "xgboost": train.build_xgboost(fit[train.LABEL_COLUMN], n_estimators=CV_XGBOOST_TREES),
        }
        fold = {"repos": sorted(held[train.REPO_COLUMN].unique()), "positives": int(held[train.LABEL_COLUMN].sum())}
        for name, model in models.items():
            model.fit(fit[features], fit[train.LABEL_COLUMN])
            probabilities = model.predict_proba(held[features])[:, 1]
            predictions[name][test_index] = probabilities
            fold[name] = (
                float(roc_auc_score(held[train.LABEL_COLUMN], probabilities))
                if held[train.LABEL_COLUMN].nunique() == 2 else float("nan")
            )
        folds.append(fold)

    for name, values in predictions.items():
        scored = ~np.isnan(values)
        trusted = [f[name] for f in folds if f["positives"] >= train.MIN_POSITIVES_PER_FOLD and not np.isnan(f[name])]
        summary[name] = {
            "pooled_roc_auc": float(roc_auc_score(y[scored], values[scored])),
            "pooled_pr_auc": float(average_precision_score(y[scored], values[scored])),
            "fold_mean_roc_auc": float(np.mean(trusted)) if trusted else float("nan"),
            "fold_std_roc_auc": float(np.std(trusted)) if trusted else float("nan"),
            "trusted_folds": len(trusted),
        }
    summary["folds"] = folds
    return summary


def _print_table(results: list[dict[str, Any]]) -> None:
    print(
        f"\n{'config':<46}{'PRs':>6}{'risky':>6}{'rate':>7}"
        f"{'base pooled':>12}{'xgb pooled':>11}{'base folds':>12}{'xgb folds':>11}{'best PR-AUC':>13}"
    )
    for r in results:
        name = r["name"] + (" *" if r["changes_label"] else "")
        head = f"{name:<46}{r['rows']:>6}{r['positives']:>6}{r['positive_rate']:>7.1%}"
        if "skipped" in r:
            print(f"{head}   {r['skipped']}")
            continue
        b, x = r["baseline"], r["xgboost"]
        best_pr = max(b["pooled_pr_auc"], x["pooled_pr_auc"])
        print(
            f"{head}{b['pooled_roc_auc']:>12.3f}{x['pooled_roc_auc']:>11.3f}"
            f"{b['fold_mean_roc_auc']:>12.3f}{x['fold_mean_roc_auc']:>11.3f}"
            f"{best_pr:>8.3f} vs {r['positive_rate']:.3f}"
        )
    print("* changes the label (what counts as risky), not only the model")
    print("pooled = one ROC-AUC over all out-of-fold predictions; folds = mean over folds with >= "
          f"{train.MIN_POSITIVES_PER_FOLD} risky PRs; PR-AUC is compared with the positive rate (random)")


def _print_verdict(results: list[dict[str, Any]]) -> None:
    scored = [r for r in results if "skipped" not in r]
    if not scored:
        print("\nVerdict: nothing could be scored")
        return
    best = max(
        ((r, model, r[model]["pooled_roc_auc"]) for r in scored for model in ("baseline", "xgboost")),
        key=lambda item: item[2],
    )
    config, model, auc = best
    reference = next((r for r in scored if r["id"] == "0"), None)
    print(f"\nBest: {config['name']} with {model}, pooled ROC-AUC {auc:.3f}")
    if reference:
        ref = max(reference["baseline"]["pooled_roc_auc"], reference["xgboost"]["pooled_roc_auc"])
        print(f"Current rules best: {ref:.3f} (change {auc - ref:+.3f})")
    if auc >= TARGET_ROC_AUC:
        caution = " — but this configuration changes the label; justify the new target before adopting it" if config["changes_label"] else ""
        print(f"Verdict: clears {TARGET_ROC_AUC}{caution}")
    else:
        print(f"Verdict: no configuration clears {TARGET_ROC_AUC}. Gate option 2: write up the ceiling with the baseline comparison.")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=train.DATA_DIR)
    parser.add_argument("--drop", nargs="*", default=[], choices=FEATURE_ORDER)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
