import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

import config
from features import FEATURE_ORDER, build_feature_vector

BACKEND_DIR = Path(__file__).resolve().parent.parent
REAL_MODEL = BACKEND_DIR / "ml" / "model.json"
UTC = timezone.utc
WEDNESDAY_NOON = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
SATURDAY_NOON = datetime(2024, 1, 6, 12, 0, tzinfo=UTC)


def iso(moment):
    return moment.isoformat().replace("+00:00", "Z")


def obviously_risky():
    files = [{"filename": f"src/module_{i}.py", "additions": 40, "deletions": 10} for i in range(49)]
    files.append({"filename": "deploy/config.yml", "additions": 20, "deletions": 5})
    pr = {
        "changed_files": 50, "additions": 1980, "deletions": 495, "commits": 12, "review_comments": 2,
        "created_at": iso(SATURDAY_NOON - timedelta(hours=30)), "merged_at": iso(SATURDAY_NOON),
    }
    commits = [{"commit": {"message": "wip"}}] * 12
    return build_feature_vector(pr, files, commits, author_pr_count=5, now=SATURDAY_NOON)


def obviously_safe():
    files = [
        {"filename": "src/utils.py", "additions": 6, "deletions": 2},
        {"filename": "tests/test_utils.py", "additions": 12, "deletions": 0},
    ]
    pr = {
        "changed_files": 2, "additions": 18, "deletions": 2, "commits": 1, "review_comments": 2,
        "created_at": iso(WEDNESDAY_NOON - timedelta(hours=4)), "merged_at": iso(WEDNESDAY_NOON),
    }
    commits = [{"commit": {"message": "Add a helper for parsing durations"}}]
    return build_feature_vector(pr, files, commits, author_pr_count=5, now=WEDNESDAY_NOON)


def write_artifacts(directory, feature_order=FEATURE_ORDER, version="xgb-test"):
    """A tiny model that scores larger PRs as riskier, for wiring tests only."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((400, len(FEATURE_ORDER))), columns=FEATURE_ORDER)
    X["files_changed"] = rng.integers(1, 60, len(X))
    y = (X["files_changed"] > 25).astype(int)
    XGBClassifier(n_estimators=20, max_depth=2).fit(X, y).save_model(directory / "model.json")
    (directory / "feature_order.json").write_text(json.dumps(feature_order))
    (directory / "metrics.json").write_text(json.dumps({"model_version": version}))
    return directory / "model.json"


def import_model_module(monkeypatch, model_path):
    monkeypatch.setattr(config.settings, "MODEL_PATH", str(model_path))
    sys.modules.pop("services.model", None)
    return importlib.import_module("services.model")


@pytest.fixture(autouse=True)
def forget_model_module():
    yield
    sys.modules.pop("services.model", None)


@pytest.fixture
def model_module(tmp_path, monkeypatch):
    return import_model_module(monkeypatch, write_artifacts(tmp_path))


@pytest.mark.parametrize(
    ("score", "label"),
    [(0.0, "Low"), (0.3399, "Low"), (0.34, "Medium"), (0.6699, "Medium"), (0.67, "High"), (1.0, "High")],
)
def test_score_to_label_boundaries(model_module, score, label):
    assert model_module.LOW_RISK_BELOW == 0.34
    assert model_module.MEDIUM_RISK_BELOW == 0.67
    assert model_module.score_to_label(score) == label


def test_import_loads_model_version_and_feature_order(model_module):
    assert model_module.MODEL_VERSION == "xgb-test"
    assert model_module.FEATURE_ORDER == FEATURE_ORDER
    assert isinstance(model_module.classifier, XGBClassifier)


def test_predict_returns_float_probability_and_label(model_module):
    score, label = model_module.predict(obviously_safe())

    assert type(score) is float
    assert 0.0 <= score <= 1.0
    assert label == model_module.score_to_label(score)


def test_contributions_are_per_feature_and_add_up_to_the_score(model_module):
    import math

    import xgboost as xgb

    vector = obviously_risky()
    effects = model_module.contributions(vector)
    row = pd.DataFrame([[vector[n] for n in FEATURE_ORDER]], columns=FEATURE_ORDER)
    full = model_module.classifier.get_booster().predict(xgb.DMatrix(row), pred_contribs=True)[0]

    assert list(effects) == FEATURE_ORDER
    assert effects["files_changed"] > 0
    assert math.isclose(1 / (1 + math.exp(-float(full.sum()))), model_module.predict(vector)[0], rel_tol=1e-5)


def test_predict_never_reloads_the_model(model_module, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("model.json was reloaded during predict()")

    monkeypatch.setattr(XGBClassifier, "load_model", fail)

    for _ in range(3):
        model_module.predict(obviously_risky())


def test_predict_rejects_a_vector_missing_a_feature(model_module):
    vector = obviously_safe()
    del vector["has_tests"]

    with pytest.raises(KeyError, match="has_tests"):
        model_module.predict(vector)


def test_wiring_risky_vector_scores_higher_on_a_size_based_model(model_module):
    assert model_module.predict(obviously_risky())[0] > model_module.predict(obviously_safe())[0]


@pytest.mark.parametrize(
    ("order", "message"),
    [
        (FEATURE_ORDER[:-1], "unknown to the model: \\['author_pr_count'\\]"),
        (FEATURE_ORDER + ["fake_feature"], "not computed by features.py: \\['fake_feature'\\]"),
        ([FEATURE_ORDER[1], FEATURE_ORDER[0], *FEATURE_ORDER[2:]], "position 0: model has 'lines_added', features.py has 'files_changed'"),
    ],
)
def test_import_fails_loudly_on_feature_order_mismatch(tmp_path, monkeypatch, order, message):
    model_path = write_artifacts(tmp_path, feature_order=order)

    with pytest.raises(RuntimeError, match=message):
        import_model_module(monkeypatch, model_path)


def test_import_fails_clearly_when_artifacts_are_missing(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="Model artifacts not found"):
        import_model_module(monkeypatch, tmp_path / "model.json")


@pytest.mark.skipif(not REAL_MODEL.exists(), reason="backend/ml/model.json not trained yet (runs after ml.train)")
def test_trained_model_scores_obviously_risky_above_obviously_safe(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # the relative MODEL_PATH must resolve from backend/, not the cwd
    module = import_model_module(monkeypatch, "ml/model.json")

    risky_score, _ = module.predict(obviously_risky())
    safe_score, _ = module.predict(obviously_safe())

    assert risky_score > safe_score, f"risky {risky_score:.3f} should exceed safe {safe_score:.3f}"
