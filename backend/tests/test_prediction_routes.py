import uuid
from datetime import datetime, timedelta, timezone

import pytest

from exceptions import GitHubAPIError, NotFoundError
from models import Prediction, PullRequest
import main

T0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


def add_prediction(db, pull_request, minutes, score=0.5, label="Medium"):
    prediction = Prediction(
        pull_request_id=pull_request.id, risk_score=score, risk_label=label,
        features={"files_changed": 3.0}, explanation="x", model_version="xgb-v1",
        created_at=T0 + timedelta(minutes=minutes),
    )
    db.add(prediction)
    db.flush()
    return prediction


def test_post_predictions_delegates_to_the_pipeline_and_returns_prediction_out(client, db, pull_request, monkeypatch):
    received = []

    def fake_run(session, pull_request_id):
        received.append(pull_request_id)
        return add_prediction(db, pull_request, minutes=0, score=0.71, label="High")

    monkeypatch.setattr(main.pipeline, "run_prediction", fake_run)

    response = client.post("/predictions", json={"pull_request_id": str(pull_request.id)})

    assert response.status_code == 200
    body = response.json()
    assert received == [pull_request.id]
    assert body["pull_request_id"] == str(pull_request.id)
    assert (body["risk_score"], body["risk_label"], body["model_version"]) == (0.71, "High", "xgb-v1")
    assert set(body) == {"id", "pull_request_id", "risk_score", "risk_label", "explanation", "features", "model_version", "created_at"}


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (NotFoundError("Pull request not found"), 404, "Pull request not found"),
        (GitHubAPIError("GitHub rate limit exhausted.", status=403, retry_after=60), 502, "GitHub rate limit exhausted."),
    ],
)
def test_post_predictions_maps_pipeline_errors(client, monkeypatch, error, status, detail):
    def fail(session, pull_request_id):
        raise error

    monkeypatch.setattr(main.pipeline, "run_prediction", fail)

    response = client.post("/predictions", json={"pull_request_id": str(uuid.uuid4())})

    assert (response.status_code, response.json()) == (status, {"detail": detail})


def test_post_predictions_rejects_a_malformed_id(client):
    assert client.post("/predictions", json={"pull_request_id": "not-a-uuid"}).status_code == 422
    assert client.post("/predictions", json={}).status_code == 422


def test_pull_request_history_is_newest_first(client, db, pull_request):
    oldest = add_prediction(db, pull_request, minutes=1)
    newest = add_prediction(db, pull_request, minutes=30)
    middle = add_prediction(db, pull_request, minutes=10)

    response = client.get(f"/predictions/pull-request/{pull_request.id}")

    assert response.status_code == 200
    assert [p["id"] for p in response.json()] == [str(newest.id), str(middle.id), str(oldest.id)]


def test_pull_request_history_unknown_or_malformed_id(client):
    assert client.get(f"/predictions/pull-request/{uuid.uuid4()}").json() == {"detail": "Pull request not found"}
    assert client.get(f"/predictions/pull-request/{uuid.uuid4()}").status_code == 404
    assert client.get("/predictions/pull-request/abc").json() == {"detail": "Pull request not found"}


def test_repository_history_spans_prs_newest_first_limited_to_50(client, db, repository, pull_request):
    other_pr = PullRequest(repository_id=repository.id, github_pr_number=99, author="carol", state="open")
    db.add(other_pr)
    db.flush()
    for minute in range(30):
        add_prediction(db, pull_request, minutes=minute)
    newest_overall = [add_prediction(db, other_pr, minutes=100 + minute) for minute in range(25)]

    response = client.get(f"/repositories/{repository.id}/predictions")

    body = response.json()
    assert response.status_code == 200
    assert len(body) == 50
    assert [p["id"] for p in body[:25]] == [str(p.id) for p in reversed(newest_overall)]
    assert {p["pull_request_id"] for p in body} == {str(pull_request.id), str(other_pr.id)}
    created = [p["created_at"] for p in body]
    assert created == sorted(created, reverse=True)


def test_repository_history_excludes_other_repositories(client, db, repository, pull_request):
    from models import Repository
    from database import get_default_user

    elsewhere = Repository(user_id=get_default_user(db).id, github_repo_id=800_000_000 + uuid.uuid4().int % 99_999, owner="other", name="repo")
    db.add(elsewhere)
    db.flush()
    foreign_pr = PullRequest(repository_id=elsewhere.id, github_pr_number=1, author="dave", state="open")
    db.add(foreign_pr)
    db.flush()
    mine = add_prediction(db, pull_request, minutes=1)
    add_prediction(db, foreign_pr, minutes=2)

    body = client.get(f"/repositories/{repository.id}/predictions").json()

    assert [p["id"] for p in body] == [str(mine.id)]


def test_repository_history_unknown_repository(client):
    response = client.get(f"/repositories/{uuid.uuid4()}/predictions")
    assert (response.status_code, response.json()) == (404, {"detail": "Repository not found"})


def test_debug_route_and_module_are_gone(client):
    import pathlib

    assert client.get(f"/debug/features/{uuid.uuid4()}/1").status_code == 404
    assert not (pathlib.Path(main.__file__).parent / "services" / "debug_features.py").exists()
    assert not any(route.path.startswith("/debug") for route in main.app.routes)


def test_importing_the_app_loads_the_model():
    # Checked through the module main imported: test_model.py removes services.model
    # from sys.modules between its own tests, so sys.modules is not reliable here.
    loaded = main.pipeline.model
    assert loaded.MODEL_VERSION
    assert loaded.classifier.get_booster().num_boosted_rounds() > 0


def test_get_pull_request_returns_metadata_with_its_repository(client, repository, pull_request):
    response = client.get(f"/pull-requests/{pull_request.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == str(pull_request.id)
    assert (body["github_pr_number"], body["author"], body["state"]) == (42, "alice", "open")
    assert body["repository"]["id"] == str(repository.id)
    assert (body["repository"]["owner"], body["repository"]["name"]) == ("acme", "widgets")


def test_get_pull_request_unknown_or_malformed_id(client):
    response = client.get(f"/pull-requests/{uuid.uuid4()}")
    assert (response.status_code, response.json()) == (404, {"detail": "Pull request not found"})
    assert client.get("/pull-requests/not-a-uuid").json() == {"detail": "Pull request not found"}
