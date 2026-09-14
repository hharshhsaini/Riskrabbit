import logging
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import func, select

from exceptions import GitHubAPIError, NotFoundError
from models import Prediction
from services import explain, github, model, pipeline

UTC = timezone.utc
OPENED = datetime(2025, 6, 10, 12, 0, tzinfo=UTC)
STAGES = ["load pull request", "fetch from GitHub", "count author history", "build features", "score", "explain", "save"]


def iso(moment):
    return moment.isoformat().replace("+00:00", "Z")


def fake_github(monkeypatch, calls, pr_overrides=None):
    pr = {"number": 42, "user": {"login": "alice"}, "created_at": iso(OPENED), "merged_at": None,
          "changed_files": 14, "additions": 600, "deletions": 90, "commits": 5, "review_comments": 3,
          **(pr_overrides or {})}

    def get_pr(owner, name, number):
        calls.append(("get_pr", owner, name, number))
        return pr

    def get_pr_files(owner, name, number):
        calls.append(("get_pr_files", owner, name, number))
        return [{"filename": f"src/w{i}.py", "additions": 40, "deletions": 6} for i in range(14)]

    def get_pr_commits(owner, name, number):
        calls.append(("get_pr_commits", owner, name, number))
        return [{"commit": {"message": "Refactor widget loading"}}] * 5

    monkeypatch.setattr(github, "get_pr", get_pr)
    monkeypatch.setattr(github, "get_pr_files", get_pr_files)
    monkeypatch.setattr(github, "get_pr_commits", get_pr_commits)


def spy_on_order(monkeypatch, calls):
    real_build, real_predict = pipeline.features.build_feature_vector, model.predict

    def build(*args, **kwargs):
        calls.append(("build_feature_vector", kwargs.get("author_pr_count")))
        return real_build(*args, **kwargs)

    def predict(vector):
        calls.append(("predict",))
        return real_predict(vector)

    def explanation(vector, score, label):
        calls.append(("generate_explanation", score, label))
        return f"Explained: {label} at {score:.2f}."

    monkeypatch.setattr(pipeline.features, "build_feature_vector", build)
    monkeypatch.setattr(model, "predict", predict)
    monkeypatch.setattr(explain, "generate_explanation", explanation)


def test_run_prediction_saves_one_row_from_the_full_pipeline(db, pull_request, monkeypatch, caplog):
    calls = []
    fake_github(monkeypatch, calls)
    spy_on_order(monkeypatch, calls)
    before = db.scalar(select(func.count()).select_from(Prediction))

    with caplog.at_level(logging.INFO, logger="services.pipeline"):
        prediction = pipeline.run_prediction(db, pull_request.id)

    assert db.scalar(select(func.count()).select_from(Prediction)) == before + 1
    saved = db.get(Prediction, prediction.id)
    assert saved.pull_request_id == pull_request.id
    assert saved.model_version == model.MODEL_VERSION
    assert saved.features["files_changed"] == 14.0
    assert saved.features["author_pr_count"] == 2.0
    assert (saved.risk_score, saved.risk_label) == model.predict(saved.features)
    assert saved.explanation == f"Explained: {saved.risk_label} at {saved.risk_score:.2f}."

    names = [c[0] for c in calls]
    assert names == ["get_pr", "get_pr_files", "get_pr_commits", "build_feature_vector", "predict", "generate_explanation", "predict"]
    assert calls[0] == ("get_pr", "acme", "widgets", 42)

    messages = [r.getMessage() for r in caplog.records]
    for stage in STAGES:
        assert sum(f": {stage} took " in m and m.endswith(" ms") for m in messages) == 1, stage
    assert any("without the explanation" in m for m in messages)


def test_unknown_pull_request_is_not_found_and_calls_nothing(db, monkeypatch):
    calls = []
    fake_github(monkeypatch, calls)

    with pytest.raises(NotFoundError, match="Pull request not found"):
        pipeline.run_prediction(db, uuid.uuid4())

    assert calls == []


def test_pull_request_deleted_on_github_is_not_found_and_saves_nothing(db, pull_request, monkeypatch):
    def gone(owner, name, number):
        raise GitHubAPIError("Not Found", status=404)

    monkeypatch.setattr(github, "get_pr", gone)
    before = db.scalar(select(func.count()).select_from(Prediction))

    with pytest.raises(NotFoundError, match="no longer exists in acme/widgets"):
        pipeline.run_prediction(db, pull_request.id)

    assert db.scalar(select(func.count()).select_from(Prediction)) == before


def test_github_outage_propagates_as_github_error_and_saves_nothing(db, pull_request, monkeypatch):
    calls = []
    fake_github(monkeypatch, calls)

    def outage(owner, name, number):
        raise GitHubAPIError("GitHub returned 502", status=502)

    monkeypatch.setattr(github, "get_pr_commits", outage)
    before = db.scalar(select(func.count()).select_from(Prediction))

    with pytest.raises(GitHubAPIError):
        pipeline.run_prediction(db, pull_request.id)

    assert db.scalar(select(func.count()).select_from(Prediction)) == before


def test_pipeline_module_is_the_only_caller_of_the_prediction_services():
    import pathlib

    backend = pathlib.Path(pipeline.__file__).resolve().parent.parent
    sources = [p for p in backend.rglob("*.py") if "venv" not in p.parts and "tests" not in p.parts and "ml" not in p.parts]
    callers = [p.relative_to(backend).as_posix() for p in sources if "generate_explanation(" in p.read_text() and "model.predict(" in p.read_text()]

    assert callers == ["services/pipeline.py"]
