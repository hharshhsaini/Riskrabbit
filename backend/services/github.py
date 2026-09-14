import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

import requests

from config import settings
from exceptions import GitHubAPIError

API_ROOT = "https://api.github.com"
TIMEOUT_SECONDS = 15
RETRY_DELAY_SECONDS = 1
PAGE_SIZE = 100
SECONDARY_RATE_LIMIT_WAIT_SECONDS = 60

_session = requests.Session()
_session.headers.update(
    {
        "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
)


def get_repo(owner: str, name: str) -> dict[str, Any]:
    return _json(_request("GET", f"/repos/{owner}/{name}"))


def list_pull_requests(
    owner: str, name: str, state: str = "all", limit: int = 30
) -> list[dict[str, Any]]:
    pulls: list[dict[str, Any]] = []
    params = {"state": state, "per_page": min(limit, PAGE_SIZE)}
    for page in iter_pages(f"/repos/{owner}/{name}/pulls", params):
        pulls.extend(page)
        if len(pulls) >= limit:
            break
    return pulls[:limit]


def get_pr(owner: str, name: str, number: int) -> dict[str, Any]:
    return _json(_request("GET", f"/repos/{owner}/{name}/pulls/{number}"))


def get_pr_files(owner: str, name: str, number: int) -> list[dict[str, Any]]:
    # GitHub stops listing after 3,000 files per PR.
    return _paginate(f"/repos/{owner}/{name}/pulls/{number}/files")


def get_pr_commits(owner: str, name: str, number: int) -> list[dict[str, Any]]:
    # GitHub stops listing after 250 commits per PR; get_pr()["commits"] has the true count.
    return _paginate(f"/repos/{owner}/{name}/pulls/{number}/commits")


def get_commit(owner: str, name: str, sha: str) -> dict[str, Any]:
    return _json(_request("GET", f"/repos/{owner}/{name}/commits/{sha}"))


def get_rate_limit() -> dict[str, Any]:
    """Core REST rate-limit status. Calling this does not count against the limit."""
    return _json(_request("GET", "/rate_limit"))["resources"]["core"]


def fetch_page(
    path: str, params: dict[str, Any] | None = None
) -> tuple[list[dict[str, Any]], str | None]:
    """Fetch one page of a list endpoint. Returns the items and the next page's URL."""
    response = _request("GET", path, params=params)
    return _json(response), response.links.get("next", {}).get("url")


def iter_pages(
    path: str, params: dict[str, Any] | None = None
) -> Iterator[list[dict[str, Any]]]:
    """Yield one page of results at a time, following the Link header."""
    page, next_url = fetch_page(path, {"per_page": PAGE_SIZE, **(params or {})})
    yield page
    while next_url:
        # The "next" URL already carries the query string.
        page, next_url = fetch_page(next_url)
        yield page


def _paginate(path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    return [item for page in iter_pages(path, params) for item in page]


def _request(method: str, path: str, **kw: Any) -> requests.Response:
    url = path if path.startswith("http") else f"{API_ROOT}{path}"
    kw.setdefault("timeout", TIMEOUT_SECONDS)

    for attempt in (1, 2):
        try:
            response = _session.request(method, url, **kw)
        except requests.ConnectionError as exc:
            if attempt == 1:
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise GitHubAPIError(f"Could not reach GitHub: {exc}") from exc
        except requests.RequestException as exc:
            raise GitHubAPIError(f"GitHub request failed: {exc}") from exc

        if response.status_code >= 500 and attempt == 1:
            time.sleep(RETRY_DELAY_SECONDS)
            continue
        break

    if response.ok:
        return response

    if response.status_code in (403, 429):
        if response.headers.get("X-RateLimit-Remaining") == "0":
            message, retry_after = _primary_rate_limit(response)
            raise GitHubAPIError(message, status=response.status_code, retry_after=retry_after)
        error = _error_message(response)
        if "Retry-After" in response.headers or "secondary rate limit" in error.lower():
            retry_after = _seconds(response.headers.get("Retry-After"))
            raise GitHubAPIError(
                f"GitHub secondary rate limit: {error}",
                status=response.status_code,
                retry_after=retry_after or SECONDARY_RATE_LIMIT_WAIT_SECONDS,
            )

    raise GitHubAPIError(
        f"GitHub returned {response.status_code} for {method} {url}: {_error_message(response)}",
        status=response.status_code,
    )


def _json(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise GitHubAPIError(
            f"GitHub returned invalid JSON for {response.url}", status=response.status_code
        ) from exc


def _error_message(response: requests.Response) -> str:
    try:
        return response.json().get("message") or response.reason
    except ValueError:
        return response.reason or "no error message"


def _seconds(value: str | None) -> float | None:
    try:
        return max(0.0, float(value)) if value is not None else None
    except ValueError:
        return None


def _primary_rate_limit(response: requests.Response) -> tuple[str, float]:
    reset = response.headers.get("X-RateLimit-Reset")
    if reset and reset.isdigit():
        reset_at = datetime.fromtimestamp(int(reset), tz=timezone.utc)
        wait = max(0.0, (reset_at - datetime.now(timezone.utc)).total_seconds())
        return (
            f"GitHub rate limit exhausted. It resets at {reset_at:%Y-%m-%d %H:%M:%S} UTC "
            f"(in about {round(wait / 60)} min).",
            wait,
        )
    return (
        "GitHub rate limit exhausted. Reset time was not provided.",
        float(SECONDARY_RATE_LIMIT_WAIT_SECONDS),
    )
