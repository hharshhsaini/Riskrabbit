import numpy as np
import pandas as pd

from features import FEATURE_ORDER
from ml import experiments


def make_dataset(repos=("a/one", "b/two", "c/three", "d/four"), rows_per_repo=60, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for index, repo in enumerate(repos):
        data = {name: rng.normal(loc=(index + 1) * 10, scale=3, size=rows_per_repo).clip(min=0) for name in FEATURE_ORDER}
        risky = (data["files_changed"] > np.quantile(data["files_changed"], 0.8)).astype(int)
        frames.append(pd.DataFrame({"repo": repo, "is_risky": risky, **data}))
    return pd.concat(frames, ignore_index=True)


def test_repo_percentiles_rank_within_each_repository_only():
    dataset = make_dataset()

    ranked = experiments.with_repo_percentiles(dataset)

    for repo, group in ranked.groupby("repo"):
        assert group["files_changed"].between(0, 1).all()
        assert group["files_changed"].max() == 1.0
        original = dataset.loc[group.index, "files_changed"]
        assert (group["files_changed"].rank() == original.rank()).all()
    assert (ranked["lines_added"] == dataset["lines_added"]).all()
    assert dataset["files_changed"].max() > 1


def test_repo_percentiles_remove_repo_size_but_keep_within_repo_signal():
    dataset = make_dataset()

    raw = experiments.score(dataset, FEATURE_ORDER)
    ranked = experiments.score(experiments.with_repo_percentiles(dataset), FEATURE_ORDER)

    assert raw["positives"] == ranked["positives"] == int(dataset["is_risky"].sum())
    assert ranked["baseline"]["pooled_roc_auc"] > 0.9
    assert {"pooled_roc_auc", "pooled_pr_auc", "fold_mean_roc_auc", "trusted_folds"} <= set(ranked["xgboost"])
    assert all(len(fold["repos"]) == 1 for fold in ranked["folds"])


def test_score_skips_configurations_with_too_few_positives():
    dataset = make_dataset().assign(is_risky=0)
    dataset.loc[:3, "is_risky"] = 1

    result = experiments.score(dataset, FEATURE_ORDER)

    assert "skipped" in result
