"""Train and evaluate the deployment-risk models.

Run from backend/ after ml/audit.py:
    ./venv/bin/python -m ml.train

Trains a logistic-regression baseline (Step 12) and an XGBoost model (Step 13),
evaluates both on repositories held out entirely from training, and saves the
XGBoost model, its feature order and the metrics to backend/ml/.

Every split is by repository: the test set, the early-stopping validation set and
the cross-validation folds. A random split would put PRs from the same repository
on both sides, letting a model memorise repository quirks and inflating every
metric.
"""

import argparse
import json
import random
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from features import FEATURE_ORDER

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data"
DATASET_FILENAME = "dataset.csv"
MODEL_FILENAME = "model.json"
FEATURE_ORDER_FILENAME = "feature_order.json"
METRICS_FILENAME = "metrics.json"
MODEL_VERSION = "xgb-v1"

LABEL_COLUMN = "is_risky"
REPO_COLUMN = "repo"

# The held-out test set. Chosen for variety — a test framework, a web CMS and an
# async HTTP library — not for how well the model scores on them. Keep it fixed
# so results stay comparable between runs.
TEST_REPOS = ("pytest-dev/pytest", "wagtail/wagtail", "aio-libs/aiohttp")
DECISION_THRESHOLD = 0.5
CV_MAX_FOLDS = 5
# ROC-AUC from a handful of positives swings wildly (one positive gives 0.0 or 1.0),
# so such folds are reported but left out of the mean.
MIN_POSITIVES_PER_FOLD = 5

# Small dataset: shallow trees and a low learning rate, or the model memorises it.
XGB_PARAMS = {
    "max_depth": 4,
    "n_estimators": 300,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}
EARLY_STOPPING_ROUNDS = 30
VALIDATION_REPO_SHARE = 0.2
RANDOM_STATE = 42
MAX_VALIDATION_DRAWS = 50


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    dataset = load_dataset(args.data_dir / DATASET_FILENAME)
    features = [name for name in FEATURE_ORDER if name not in set(args.drop)]
    train, test = split_by_repository(dataset, args.test_repos)
    y_train, y_test = train[LABEL_COLUMN], test[LABEL_COLUMN]

    _heading("Data")
    print(f"{len(dataset)} PRs from {dataset[REPO_COLUMN].nunique()} repositories, {len(features)} features")
    if args.drop:
        print(f"dropped: {', '.join(args.drop)}")
    for name, part in (("train", train), ("test", test)):
        print(f"{name:<5} {_describe(part)}")

    baseline = build_baseline().fit(train[features], y_train)
    baseline_metrics = evaluate(baseline, test[features], y_test)

    xgboost, early_stopping = train_xgboost(train, features, args.seed)
    xgboost_metrics = evaluate(xgboost, test[features], y_test)
    print(
        f"valid {_describe(train[train[REPO_COLUMN].isin(early_stopping['validation_repositories'])])}"
        "  (carved from train, used only to choose the number of trees)"
    )

    _heading("Held-out repositories: baseline vs XGBoost")
    _print_side_by_side(baseline_metrics, xgboost_metrics, y_test.mean())
    print(
        f"\nXGBoost early stopping: best at {early_stopping['best_n_estimators']} trees of "
        f"{XGB_PARAMS['n_estimators']} (validation PR-AUC {early_stopping['validation_pr_auc']:.3f}); "
        "final model refit on all training repos with that many trees"
    )
    _print_confusion("baseline", baseline_metrics["confusion_matrix"])
    _print_confusion("XGBoost", xgboost_metrics["confusion_matrix"])
    _print_coefficients(baseline, features)
    _print_importances(xgboost)

    _heading("Robustness: leave-repositories-out cross-validation (every repo tested once)")
    baseline_folds = grouped_cv_roc_auc(dataset, features, lambda y: build_baseline())
    xgboost_folds = grouped_cv_roc_auc(
        dataset, features, lambda y: build_xgboost(y, n_estimators=early_stopping["best_n_estimators"])
    )
    _print_folds(baseline_folds, xgboost_folds)

    _heading("Variants on the same held-out repositories (for the open feature decisions)")
    _print_variants(train, test, features)

    metrics = build_metrics(
        dataset, train, test, features, baseline_metrics, xgboost_metrics,
        early_stopping, xgboost, baseline_folds, xgboost_folds, args.data_dir,
    )
    _heading("Artifacts")
    if args.drop:
        print(
            "Not saved: features were dropped on the command line, so the saved feature order would "
            "not match features.FEATURE_ORDER and services/model.py would refuse to load it. To ship "
            "a smaller feature set, remove the features from FEATURE_ORDER in features.py instead."
        )
        return
    paths = save_artifacts(xgboost, features, metrics, args.output_dir)
    for path in paths:
        print(f"saved {path}")


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run ml.label and then ml.audit first.")
    dataset = pd.read_csv(path)
    missing = [c for c in [REPO_COLUMN, LABEL_COLUMN, *FEATURE_ORDER] if c not in dataset.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns {missing}. Re-run ml.audit.")
    return dataset


def split_by_repository(
    dataset: pd.DataFrame, test_repos: Sequence[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available = set(dataset[REPO_COLUMN].unique())
    absent = [repo for repo in test_repos if repo not in available]
    if absent:
        raise SystemExit(
            f"Test repositories not in the dataset: {absent}. Available: {sorted(available)}. "
            "Finish collection, or pass --test-repos."
        )
    is_test = dataset[REPO_COLUMN].isin(test_repos)
    train, test = dataset[~is_test], dataset[is_test]
    if train.empty:
        raise SystemExit("No repositories left for training after holding out the test set.")
    return train, test


def carve_validation_repos(
    train: pd.DataFrame, seed: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split training repositories into fit and validation repositories, both with risky PRs."""
    repos = sorted(train[REPO_COLUMN].unique())
    n_validation = max(1, round(VALIDATION_REPO_SHARE * len(repos)))
    if len(repos) - n_validation < 1:
        raise SystemExit(
            f"Need at least 2 training repositories to carve a validation set; have {len(repos)}."
        )
    rng = random.Random(seed)
    for _ in range(MAX_VALIDATION_DRAWS):
        chosen = rng.sample(repos, n_validation)
        is_validation = train[REPO_COLUMN].isin(chosen)
        fit, validation = train[~is_validation], train[is_validation]
        if fit[LABEL_COLUMN].nunique() == 2 and validation[LABEL_COLUMN].nunique() == 2:
            return fit, validation
    raise SystemExit("Could not find validation repositories that contain risky PRs.")


def build_baseline(log_counts: bool = False) -> Pipeline:
    steps: list[Any] = []
    if log_counts:
        # Every feature is non-negative; log1p tames very large PRs without reordering them.
        steps.append(FunctionTransformer(np.log1p, feature_names_out="one-to-one"))
    steps += [StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)]
    return make_pipeline(*steps)


def build_xgboost(
    y: pd.Series,
    n_estimators: int = XGB_PARAMS["n_estimators"],
    early_stopping: bool = False,
) -> XGBClassifier:
    positives = int(y.sum())
    if positives == 0:
        raise SystemExit("Cannot train XGBoost: the training data has no risky PRs.")
    return XGBClassifier(
        **{**XGB_PARAMS, "n_estimators": n_estimators},
        scale_pos_weight=(len(y) - positives) / positives,
        eval_metric="aucpr",
        early_stopping_rounds=EARLY_STOPPING_ROUNDS if early_stopping else None,
        random_state=RANDOM_STATE,
    )


def train_xgboost(
    train: pd.DataFrame, features: list[str], seed: int = RANDOM_STATE
) -> tuple[XGBClassifier, dict[str, Any]]:
    """Choose the tree count by early stopping on held-out training repos, then refit on all of them.

    Refitting with a fixed tree count means the saved model.json contains exactly
    the trees it should use, with no best_iteration setting to honour at load time.
    """
    fit, validation = carve_validation_repos(train, seed)
    probe = build_xgboost(fit[LABEL_COLUMN], early_stopping=True)
    probe.fit(
        fit[features], fit[LABEL_COLUMN],
        eval_set=[(validation[features], validation[LABEL_COLUMN])],
        verbose=False,
    )
    best_n = int(probe.best_iteration) + 1

    final = build_xgboost(train[LABEL_COLUMN], n_estimators=best_n)
    final.fit(train[features], train[LABEL_COLUMN], verbose=False)
    return final, {
        "validation_repositories": sorted(validation[REPO_COLUMN].unique()),
        "best_n_estimators": best_n,
        "validation_pr_auc": float(probe.best_score),
        "patience_rounds": EARLY_STOPPING_ROUNDS,
    }


def evaluate(model: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    both_classes = y.nunique() == 2
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities) if both_classes else float("nan"),
        "pr_auc": average_precision_score(y, probabilities) if both_classes else float("nan"),
        "confusion_matrix": confusion_matrix(y, predictions, labels=[0, 1]),
    }


def grouped_cv_roc_auc(
    dataset: pd.DataFrame, features: list[str], make_model: Callable[[pd.Series], Any]
) -> list[dict[str, Any]]:
    groups = dataset[REPO_COLUMN]
    n_folds = min(CV_MAX_FOLDS, groups.nunique())
    if n_folds < 2:
        return []
    folds = []
    for train_index, test_index in GroupKFold(n_splits=n_folds).split(dataset, groups=groups):
        train, test = dataset.iloc[train_index], dataset.iloc[test_index]
        if train[LABEL_COLUMN].nunique() < 2 or test[LABEL_COLUMN].nunique() < 2:
            continue
        model = make_model(train[LABEL_COLUMN])
        model.fit(train[features], train[LABEL_COLUMN])
        folds.append(
            {
                "repos": sorted(test[REPO_COLUMN].unique()),
                "rows": len(test),
                "positives": int(test[LABEL_COLUMN].sum()),
                "roc_auc": roc_auc_score(test[LABEL_COLUMN], model.predict_proba(test[features])[:, 1]),
            }
        )
    return folds


def trusted_mean(folds: list[dict[str, Any]]) -> tuple[float, float, int]:
    scores = [f["roc_auc"] for f in folds if f["positives"] >= MIN_POSITIVES_PER_FOLD]
    if not scores:
        return float("nan"), float("nan"), 0
    return float(np.mean(scores)), float(np.std(scores)), len(scores)


def build_metrics(
    dataset: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    baseline_metrics: dict[str, Any],
    xgboost_metrics: dict[str, Any],
    early_stopping: dict[str, Any],
    xgboost: XGBClassifier,
    baseline_folds: list[dict[str, Any]],
    xgboost_folds: list[dict[str, Any]],
    data_dir: Path,
) -> dict[str, Any]:
    window_path = data_dir / "cache" / "window.json"
    return {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "rows": len(dataset),
        "positive_rate": float(dataset[LABEL_COLUMN].mean()),
        "repositories": sorted(dataset[REPO_COLUMN].unique()),
        "held_out_repositories": sorted(test[REPO_COLUMN].unique()),
        "train": _summary(train),
        "test": _summary(test),
        "collection_window": json.loads(window_path.read_text()) if window_path.exists() else None,
        "features": features,
        # Training-set means, used by the explanation prompt to find unusual features.
        "feature_means": {name: float(train[name].mean()) for name in features},
        "decision_threshold": DECISION_THRESHOLD,
        "xgboost": {
            "hyperparameters": {
                **XGB_PARAMS,
                "n_estimators": early_stopping["best_n_estimators"],
                "scale_pos_weight": float(xgboost.get_params()["scale_pos_weight"]),
                "random_state": RANDOM_STATE,
            },
            "early_stopping": {**early_stopping, "max_estimators": XGB_PARAMS["n_estimators"]},
            "held_out": _metrics_json(xgboost_metrics),
            "cross_validation": _folds_json(xgboost_folds),
        },
        "baseline": {
            "model": "StandardScaler + LogisticRegression(class_weight='balanced', max_iter=1000)",
            "held_out": _metrics_json(baseline_metrics),
            "cross_validation": _folds_json(baseline_folds),
        },
    }


def save_artifacts(
    model: XGBClassifier, features: list[str], metrics: dict[str, Any], output_dir: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / MODEL_FILENAME
    order_path = output_dir / FEATURE_ORDER_FILENAME
    metrics_path = output_dir / METRICS_FILENAME
    model.save_model(model_path)
    order_path.write_text(json.dumps(features, indent=2) + "\n")
    metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
    return [model_path, order_path, metrics_path]


def _summary(part: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(part),
        "positives": int(part[LABEL_COLUMN].sum()),
        "positive_rate": float(part[LABEL_COLUMN].mean()),
        "repositories": sorted(part[REPO_COLUMN].unique()),
    }


def _metrics_json(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else float(value))
        for key, value in metrics.items()
    }


def _folds_json(folds: list[dict[str, Any]]) -> dict[str, Any]:
    mean, std, count = trusted_mean(folds)
    return {
        "folds": [{**fold, "roc_auc": float(fold["roc_auc"])} for fold in folds],
        "min_positives_per_fold": MIN_POSITIVES_PER_FOLD,
        "roc_auc_mean": mean,
        "roc_auc_std": std,
        "trusted_folds": count,
    }


def _describe(part: pd.DataFrame) -> str:
    repos = sorted(part[REPO_COLUMN].unique())
    return (
        f"{len(part):>5} PRs, {int(part[LABEL_COLUMN].sum()):>4} risky ({part[LABEL_COLUMN].mean():.1%}) "
        f"from {len(repos)} repos: {', '.join(repos)}"
    )


def _print_side_by_side(baseline: dict[str, Any], xgboost: dict[str, Any], positive_rate: float) -> None:
    rows = [
        ("accuracy", "accuracy", f"all-safe scores {1 - positive_rate:.3f}"),
        ("precision", "precision", "of PRs flagged risky, share actually risky"),
        ("recall", "recall", "of risky PRs, share flagged"),
        ("ROC-AUC", "roc_auc", "0.5 = random ranking"),
        ("PR-AUC", "pr_auc", f"random = {positive_rate:.3f}"),
    ]
    print(f"{'metric':<11}{'baseline':>10}{'XGBoost':>10}   note")
    for label, key, note in rows:
        print(f"{label:<11}{baseline[key]:>10.3f}{xgboost[key]:>10.3f}   {note}")


def _print_confusion(name: str, matrix: np.ndarray) -> None:
    (tn, fp), (fn, tp) = matrix
    print(f"\n{name} confusion matrix (threshold {DECISION_THRESHOLD}):")
    print(f"{'':<16}{'predicted safe':>16}{'predicted risky':>17}")
    print(f"{'actually safe':<16}{tn:>16}{fp:>17}")
    print(f"{'actually risky':<16}{fn:>16}{tp:>17}")


def _print_coefficients(model: Pipeline, features: list[str]) -> None:
    print("\nbaseline standardised coefficients (positive = pushes towards risky):")
    for name, value in sorted(zip(features, model[-1].coef_[0]), key=lambda item: -abs(item[1]))[:8]:
        print(f"  {name:<22}{value:>+8.3f}")


def _print_importances(model: XGBClassifier) -> None:
    gains = model.get_booster().get_score(importance_type="gain")
    total = sum(gains.values()) or 1.0
    print("\nXGBoost feature importance (share of total gain):")
    for name, gain in sorted(gains.items(), key=lambda item: -item[1])[:8]:
        print(f"  {name:<22}{gain / total:>7.1%}")


def _print_folds(baseline_folds: list[dict[str, Any]], xgboost_folds: list[dict[str, Any]]) -> None:
    xgb_by_repos = {tuple(f["repos"]): f for f in xgboost_folds}
    print(f"{'test repos':<44}{'risky':>7}{'baseline':>10}{'XGBoost':>10}")
    for fold in baseline_folds:
        other = xgb_by_repos.get(tuple(fold["repos"]))
        note = "" if fold["positives"] >= MIN_POSITIVES_PER_FOLD else "  too few risky PRs, not in mean"
        xgb_auc = f"{other['roc_auc']:>10.3f}" if other else f"{'-':>10}"
        print(f"{', '.join(fold['repos'])[:43]:<44}{fold['positives']:>7}{fold['roc_auc']:>10.3f}{xgb_auc}{note}")
    for name, folds in (("baseline", baseline_folds), ("XGBoost", xgboost_folds)):
        mean, std, count = trusted_mean(folds)
        print(f"{name} mean ROC-AUC {mean:.3f} ± {std:.3f} over {count} folds with >= {MIN_POSITIVES_PER_FOLD} risky PRs")


def _print_variants(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> None:
    without_author = [f for f in features if f != "author_pr_count"]
    variants: list[tuple[str, list[str], Callable[[pd.Series], Any]]] = [
        ("baseline, log1p counts", features, lambda y: build_baseline(log_counts=True)),
        ("baseline without author_pr_count", without_author, lambda y: build_baseline()),
        ("XGBoost without author_pr_count", without_author, lambda y: build_xgboost(y)),
    ]
    print(f"{'variant':<40}{'ROC-AUC':>9}{'PR-AUC':>9}")
    for name, columns, factory in variants:
        if columns == features and "without" in name:
            continue
        model = factory(train[LABEL_COLUMN]).fit(train[columns], train[LABEL_COLUMN])
        result = evaluate(model, test[columns], test[LABEL_COLUMN])
        print(f"{name:<40}{result['roc_auc']:>9.3f}{result['pr_auc']:>9.3f}")


def _heading(title: str) -> None:
    print(f"\n== {title} ==")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=ML_DIR, help="where model.json and friends go")
    parser.add_argument("--test-repos", nargs="+", default=list(TEST_REPOS))
    parser.add_argument("--drop", nargs="*", default=[], choices=FEATURE_ORDER, help="features to leave out (not saved)")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE, help="seed for choosing validation repos")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
