"""API tests through FastAPI's TestClient, end to end.

Real routes, the real model and a real database session that conftest rolls back
after each test. GitHub and the LLM explanation are mocked with unittest.mock.patch,
and conftest blocks any other network access, so no test reaches a real service.
"""

import uuid
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests
from sqlalchemy import func, select

from exceptions import GitHubAPIError
from models import Prediction, Repository
from services import model

MOCK_EXPLANATION = "Mocked explanation: 14 files changed raised the score."
MOCK_PR = {
    "number": 42,
    "user": {"login": "alice"},
    "created_at": "2025-06-10T12:00:00Z",
    "merged_at": None,
    "changed_files": 14,
    "additions": 600,
    "deletions": 90,
    "commits": 5,
    "review_comments": 3,
}
MOCK_FILES = [{"filename": f"src/widget_{i}.py", "additions": 40, "deletions": 6} for i in range(14)]
MOCK_COMMITS = [{"commit": {"message": "Refactor widget loading"}}] * 5


def github_repository(github_repo_id, owner="acme", name="widgets"):
    return {
        "id": github_repo_id,
        "private": False,
        "owner": {"login": owner},
        "name": name,
        "full_name": f"{owner}/{name}",
    }


def unique_github_repo_id():
    return 700_000_000 + uuid.uuid4().int % 99_999


def row_count(db, table, *conditions):
    return db.scalar(select(func.count()).select_from(table).where(*conditions))


@contextmanager
def mocked_github_and_llm():
    with (
        patch("services.github.get_pr", return_value=MOCK_PR) as get_pr,
        patch("services.github.get_pr_files", return_value=MOCK_FILES) as get_pr_files,
        patch("services.github.get_pr_commits", return_value=MOCK_COMMITS) as get_pr_commits,
        patch("services.explain.generate_explanation", return_value=MOCK_EXPLANATION) as generate_explanation,
    ):
        yield SimpleNamespace(
            get_pr=get_pr,
            get_pr_files=get_pr_files,
            get_pr_commits=get_pr_commits,
            generate_explanation=generate_explanation,
        )


def test_health_returns_200(client):
    response = client.get("/health")

    assert (response.status_code, response.json()) == (200, {"status": "ok"})


def test_connect_repository_with_github_mocked_creates_one_row(client, db):
    github_repo_id = unique_github_repo_id()

    with patch("services.github.get_repo", return_value=github_repository(github_repo_id)) as get_repo:
        response = client.post("/repositories", json={"owner": "acme", "name": "widgets"})

    assert response.status_code == 200
    body = response.json()
    assert (body["owner"], body["name"], body["github_repo_id"]) == ("acme", "widgets", github_repo_id)
    get_repo.assert_called_once_with("acme", "widgets")
    assert row_count(db, Repository, Repository.github_repo_id == github_repo_id) == 1


def test_connecting_the_same_repository_twice_keeps_one_row(client, db):
    github_repo_id = unique_github_repo_id()

    with patch("services.github.get_repo", return_value=github_repository(github_repo_id)) as get_repo:
        first = client.post("/repositories", json={"owner": "acme", "name": "widgets"})
        second = client.post("/repositories", json={"owner": "acme", "name": "widgets"})

    assert first.status_code == second.status_code == 200
    assert first.json()["id"] == second.json()["id"]
    assert get_repo.call_count == 2
    assert row_count(db, Repository, Repository.github_repo_id == github_repo_id) == 1


def test_repository_not_found_on_github_returns_404_with_detail(client, db):
    with patch("services.github.get_repo", side_effect=GitHubAPIError("Not Found", status=404)):
        response = client.post("/repositories", json={"owner": "acme", "name": "does-not-exist"})

    assert response.status_code == 404
    assert response.json() == {"detail": "Repository not found or not public"}
    assert row_count(db, Repository, Repository.name == "does-not-exist") == 0


def test_create_prediction_with_github_and_llm_mocked_inserts_one_prediction(client, db, pull_request):
    before = row_count(db, Prediction, Prediction.pull_request_id == pull_request.id)

    with mocked_github_and_llm() as mocks:
        response = client.post("/predictions", json={"pull_request_id": str(pull_request.id)})

    assert response.status_code == 200
    body = response.json()
    assert row_count(db, Prediction, Prediction.pull_request_id == pull_request.id) == before + 1
    assert body["pull_request_id"] == str(pull_request.id)
    assert body["explanation"] == MOCK_EXPLANATION
    assert body["model_version"] == model.MODEL_VERSION
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_label"] == model.score_to_label(body["risk_score"])
    assert body["features"]["files_changed"] == 14.0
    assert body["features"]["author_pr_count"] == 2.0
    mocks.get_pr.assert_called_once_with("acme", "widgets", 42)
    mocks.get_pr_files.assert_called_once_with("acme", "widgets", 42)
    mocks.get_pr_commits.assert_called_once_with("acme", "widgets", 42)
    mocks.generate_explanation.assert_called_once()


def test_both_history_endpoints_return_the_inserted_prediction(client, repository, pull_request):
    with mocked_github_and_llm():
        created = client.post("/predictions", json={"pull_request_id": str(pull_request.id)}).json()

    pr_history = client.get(f"/predictions/pull-request/{pull_request.id}")
    repo_history = client.get(f"/repositories/{repository.id}/predictions")

    assert pr_history.status_code == repo_history.status_code == 200
    assert pr_history.json() == [created]
    assert repo_history.json() == [created]


def test_the_suite_cannot_reach_the_network():
    with pytest.raises(requests.ConnectionError, match="network access is blocked in tests"):
        requests.get("https://api.github.com", timeout=5)
