"""Step 24 hardening: 404s for every id route, contained GitHub errors, accurate /docs."""

import uuid
from unittest.mock import patch

import pytest
import requests
from fastapi.testclient import TestClient

import main
from exceptions import GitHubAPIError
from services import github

ID_ROUTES = [
    ("GET", "/repositories/{id}/pull-requests", "Repository not found"),
    ("GET", "/pull-requests/{id}", "Pull request not found"),
    ("GET", "/predictions/pull-request/{id}", "Pull request not found"),
    ("GET", "/repositories/{id}/predictions", "Repository not found"),
]


@pytest.fixture
def offline_client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.mark.parametrize("malformed", ["abc", "123", "not-a-uuid-at-all"])
@pytest.mark.parametrize(("method", "path", "detail"), ID_ROUTES)
def test_a_malformed_path_id_is_a_404_not_a_validation_error(offline_client, method, path, detail, malformed):
    response = offline_client.request(method, path.format(id=malformed))

    assert (response.status_code, response.json()) == (404, {"detail": detail})


@pytest.mark.parametrize(("method", "path", "detail"), ID_ROUTES)
def test_an_unknown_path_id_is_a_404(client, method, path, detail):
    response = client.request(method, path.format(id=uuid.uuid4()))

    assert (response.status_code, response.json()) == (404, {"detail": detail})


def test_an_unknown_pull_request_id_in_the_prediction_body_is_a_404(client):
    response = client.post("/predictions", json={"pull_request_id": str(uuid.uuid4())})

    assert (response.status_code, response.json()) == (404, {"detail": "Pull request not found"})


def test_a_malformed_body_id_is_still_a_validation_error(offline_client):
    response = offline_client.post("/predictions", json={"pull_request_id": "abc"})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "pull_request_id"]


def test_a_repository_deleted_on_github_lists_as_404_not_502(client, repository):
    with patch("services.github.list_pull_requests", side_effect=GitHubAPIError("Not Found", status=404)):
        response = client.get(f"/repositories/{repository.id}/pull-requests")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Repository acme/widgets no longer exists on GitHub or is no longer public"
    }


def test_other_github_failures_while_listing_stay_502(client, repository):
    with patch("services.github.list_pull_requests", side_effect=GitHubAPIError("GitHub is down", status=503)):
        response = client.get(f"/repositories/{repository.id}/pull-requests")

    assert (response.status_code, response.json()) == (502, {"detail": "GitHub is down"})


@pytest.mark.parametrize("body", [b"[1, 2]", b'"just a string"', b"null", b"<html>proxy error</html>"])
def test_a_github_error_body_that_is_not_a_json_object_still_raises_github_api_error(body):
    response = requests.Response()
    response.status_code, response.reason, response._content = 400, "Bad Request", body
    response.url = "https://api.github.com/repos/acme/widgets"

    with patch.object(github._session, "request", return_value=response):
        with pytest.raises(GitHubAPIError) as caught:
            github.get_repo("acme", "widgets")

    assert caught.value.status == 400
    assert caught.value.message.endswith(": Bad Request")


def _operations():
    for path, operations in main.app.openapi()["paths"].items():
        for method, operation in operations.items():
            yield method.upper(), path, operation


def _schema(operation, status):
    return operation["responses"][status]["content"]["application/json"]["schema"]


def test_docs_list_every_route():
    documented = {(method, path) for method, path, _ in _operations()}
    routed = {
        (method, route.path)
        for route in main.app.routes
        if route.path not in {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
        for method in route.methods
    }
    assert documented == routed


def test_docs_give_every_success_response_a_named_model():
    for method, path, operation in _operations():
        schema = _schema(operation, "200")
        ref = schema.get("$ref") or schema.get("items", {}).get("$ref")
        assert ref, f"{method} {path} has no response model"


@pytest.mark.parametrize(
    ("method", "path", "body", "errors"),
    [
        ("GET", "/health", None, set()),
        ("GET", "/repositories", None, set()),
        ("POST", "/repositories", "RepositoryCreate", {"404", "422", "502"}),
        ("GET", "/repositories/{repo_id}/pull-requests", None, {"404", "502"}),
        ("GET", "/pull-requests/{pr_id}", None, {"404"}),
        ("POST", "/predictions", "PredictionCreate", {"404", "422", "500", "502"}),
        ("GET", "/predictions/pull-request/{pr_id}", None, {"404"}),
        ("GET", "/repositories/{repo_id}/predictions", None, {"404"}),
    ],
)
def test_docs_show_the_real_request_body_and_error_responses(method, path, body, errors):
    operation = main.app.openapi()["paths"][path][method.lower()]

    documented_body = operation.get("requestBody", {}).get("content", {}).get("application/json", {}).get("schema", {})
    assert documented_body.get("$ref", "").rsplit("/", 1)[-1] == (body or "")
    assert set(operation["responses"]) - {"200"} == errors
    for status in errors - {"422"}:
        assert _schema(operation, status)["$ref"].endswith("/ErrorOut")
