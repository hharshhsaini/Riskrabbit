import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from features import FEATURE_ORDER
from ml import train


def make_dataset(repos=("a/one", "b/two", "c/three", "d/four", "e/five", "f/six"), rows_per_repo=60, seed=0):
    rng = np.random.default_rng(seed)
    frames = []
    for index, repo in enumerate(repos):
        data = {name: rng.normal(loc=index + 1, size=rows_per_repo).clip(min=0) for name in FEATURE_ORDER}
        risky = (data["files_changed"] > np.quantile(data["files_changed"], 0.8)).astype(int)
        frames.append(pd.DataFrame({"repo": repo, "is_risky": risky, **data}))
    return pd.concat(frames, ignore_index=True)


def test_split_by_repository_keeps_repos_on_one_side():
    dataset = make_dataset()

    training, testing = train.split_by_repository(dataset, ["b/two", "d/four"])

    assert set(training["repo"]) == {"a/one", "c/three", "e/five", "f/six"}
    assert set(testing["repo"]) == {"b/two", "d/four"}
    assert len(training) + len(testing) == len(dataset)


def test_split_rejects_missing_test_repo_and_empty_training_set():
    dataset = make_dataset(repos=("a/one", "b/two"))

    with pytest.raises(SystemExit, match="not in the dataset"):
        train.split_by_repository(dataset, ["z/missing"])
    with pytest.raises(SystemExit, match="No repositories left"):
        train.split_by_repository(dataset, ["a/one", "b/two"])


def test_baseline_is_standardised_balanced_logistic_regression_fit_on_training_rows_only():
    dataset = make_dataset()
    training, _ = train.split_by_repository(dataset, ["c/three", "d/four"])

    model = train.build_baseline().fit(training[FEATURE_ORDER], training["is_risky"])

    scaler, classifier = model[0], model[-1]
    assert isinstance(scaler, StandardScaler)
    assert isinstance(classifier, LogisticRegression)
    assert classifier.class_weight == "balanced"
    assert classifier.max_iter == 1000
    assert np.allclose(scaler.mean_, training[FEATURE_ORDER].mean().to_numpy())


def test_evaluate_reports_all_metrics_and_handles_single_class_test_set():
    dataset = make_dataset()
    training, testing = train.split_by_repository(dataset, ["d/four"])
    model = train.build_baseline(log_counts=True).fit(training[FEATURE_ORDER], training["is_risky"])

    metrics = train.evaluate(model, testing[FEATURE_ORDER], testing["is_risky"])
    assert {"accuracy", "precision", "recall", "roc_auc", "pr_auc", "confusion_matrix"} <= set(metrics)
    assert metrics["confusion_matrix"].sum() == len(testing)

    only_safe = testing.assign(is_risky=0)
    assert np.isnan(train.evaluate(model, only_safe[FEATURE_ORDER], only_safe["is_risky"])["roc_auc"])


def test_grouped_cv_never_puts_a_repo_in_train_and_test(monkeypatch):
    dataset = make_dataset(repos=("a/one", "b/two", "c/three", "d/four"))
    seen = []
    real_fit = train.Pipeline.fit

    def spy_fit(self, X, y, **kw):
        seen.append(set(dataset.loc[X.index, "repo"]))
        return real_fit(self, X, y, **kw)

    monkeypatch.setattr(train.Pipeline, "fit", spy_fit)
    folds = train.grouped_cv_roc_auc(dataset, FEATURE_ORDER, lambda y: train.build_baseline())

    assert len(folds) == 4
    assert all(len(fold["repos"]) == 1 and fold["repos"][0] not in trained for fold, trained in zip(folds, seen))
    assert all(fold["positives"] == int(dataset.loc[dataset.repo == fold["repos"][0], "is_risky"].sum()) for fold in folds)


def test_validation_repos_are_carved_from_training_repos_only():
    dataset = make_dataset()
    training, testing = train.split_by_repository(dataset, ["f/six"])

    fit, validation = train.carve_validation_repos(training, seed=1)

    assert set(fit["repo"]).isdisjoint(validation["repo"])
    assert set(validation["repo"]) <= set(training["repo"])
    assert set(validation["repo"]).isdisjoint(testing["repo"])
    assert fit["is_risky"].nunique() == 2 and validation["is_risky"].nunique() == 2
    again_fit, again_validation = train.carve_validation_repos(training, seed=1)
    assert set(again_validation["repo"]) == set(validation["repo"])


def test_carving_needs_at_least_two_training_repos():
    dataset = make_dataset(repos=("a/one",))
    with pytest.raises(SystemExit, match="at least 2 training repositories"):
        train.carve_validation_repos(dataset)


def test_xgboost_uses_prompt_hyperparameters_and_imbalance_weight():
    y = pd.Series([1] * 10 + [0] * 90)

    model = train.build_xgboost(y)

    params = model.get_params()
    assert params["max_depth"] == 4
    assert params["n_estimators"] == 300
    assert params["learning_rate"] == 0.05
    assert params["subsample"] == 0.8
    assert params["colsample_bytree"] == 0.8
    assert params["scale_pos_weight"] == 9.0


def test_train_xgboost_refits_with_the_early_stopped_tree_count():
    dataset = make_dataset()
    training, _ = train.split_by_repository(dataset, ["f/six"])

    model, info = train.train_xgboost(training, FEATURE_ORDER, seed=3)

    assert 1 <= info["best_n_estimators"] <= 300
    assert model.get_params()["n_estimators"] == info["best_n_estimators"]
    assert model.get_booster().num_boosted_rounds() == info["best_n_estimators"]
    assert set(info["validation_repositories"]) <= set(training["repo"])


def test_main_saves_reloadable_artifacts_with_feature_order_and_metrics(tmp_path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dataset = make_dataset()
    dataset.to_csv(data_dir / "dataset.csv", index=False)
    out = tmp_path / "ml"

    train.main(["--data-dir", str(data_dir), "--output-dir", str(out), "--test-repos", "f/six"])

    order = json.loads((out / "feature_order.json").read_text())
    metrics = json.loads((out / "metrics.json").read_text())
    assert order == FEATURE_ORDER
    assert metrics["model_version"] == "xgb-v1"
    assert metrics["held_out_repositories"] == ["f/six"]
    assert metrics["rows"] == len(dataset)
    for model in ("baseline", "xgboost"):
        assert {"accuracy", "precision", "recall", "roc_auc", "pr_auc", "confusion_matrix"} <= set(metrics[model]["held_out"])
    assert set(metrics["feature_means"]) == set(FEATURE_ORDER)

    loaded = XGBClassifier()
    loaded.load_model(out / "model.json")
    test_rows = dataset[dataset.repo == "f/six"][FEATURE_ORDER]
    assert loaded.get_booster().num_boosted_rounds() == metrics["xgboost"]["hyperparameters"]["n_estimators"]
    assert loaded.predict_proba(test_rows).shape == (len(test_rows), 2)


def test_main_refuses_to_save_when_features_are_dropped(tmp_path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    make_dataset().to_csv(data_dir / "dataset.csv", index=False)
    out = tmp_path / "ml"

    train.main(["--data-dir", str(data_dir), "--output-dir", str(out), "--test-repos", "f/six", "--drop", "author_pr_count"])

    assert not out.exists() or not any(out.iterdir())
    assert "Not saved" in capsys.readouterr().out
