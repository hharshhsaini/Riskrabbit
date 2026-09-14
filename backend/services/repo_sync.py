import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from database import get_default_user
from exceptions import GitHubAPIError, NotFoundError
from models import PullRequest, Repository
from services import github

PR_SYNC_LIMIT = 30


def connect_repository(db: Session, owner: str, name: str) -> Repository:
    """Validate that a GitHub repository is public, then upsert it for the default user."""
    try:
        repo = github.get_repo(owner, name)
    except GitHubAPIError as exc:
        if exc.status == 404:
            raise NotFoundError("Repository not found or not public") from exc
        raise
    if repo.get("private"):
        raise NotFoundError("Repository not found or not public")

    user = get_default_user(db)
    stmt = insert(Repository).values(
        user_id=user.id,
        github_repo_id=repo["id"],
        # GitHub resolves names case-insensitively; store its canonical spelling.
        owner=repo["owner"]["login"],
        name=repo["name"],
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_repositories_user_id_github_repo_id",
        set_={"owner": stmt.excluded.owner, "name": stmt.excluded.name},
    ).returning(Repository)

    repository = db.scalars(stmt, execution_options={"populate_existing": True}).one()
    db.commit()
    db.refresh(repository)
    return repository


def list_repositories(db: Session) -> list[Repository]:
    user = get_default_user(db)
    return list(
        db.scalars(
            select(Repository)
            .where(Repository.user_id == user.id)
            .order_by(Repository.connected_at.desc())
        )
    )


def get_repository(db: Session, repository_id: uuid.UUID) -> Repository:
    user = get_default_user(db)
    repository = db.scalars(
        select(Repository).where(
            Repository.id == repository_id, Repository.user_id == user.id
        )
    ).one_or_none()
    if repository is None:
        raise NotFoundError("Repository not found")
    return repository


def get_pull_request(db: Session, pull_request_id: uuid.UUID) -> PullRequest:
    pull_request = db.scalars(
        select(PullRequest)
        .join(Repository, PullRequest.repository_id == Repository.id)
        .where(PullRequest.id == pull_request_id, Repository.user_id == get_default_user(db).id)
    ).one_or_none()
    if pull_request is None:
        raise NotFoundError("Pull request not found")
    return pull_request


def count_prior_merged_prs(
    db: Session, repository_id: uuid.UUID, author: str | None, before: Any
) -> int:
    """Merged PRs by `author` in this repository, stored locally, merged before `before`."""
    if not author:
        return 0
    query = (
        select(func.count())
        .select_from(PullRequest)
        .where(
            PullRequest.repository_id == repository_id,
            PullRequest.author == author,
            PullRequest.merged_at.is_not(None),
        )
    )
    cutoff = before if isinstance(before, datetime) else _parse_github_datetime(before)
    if cutoff is not None:
        query = query.where(PullRequest.merged_at < cutoff)
    return db.scalar(query) or 0


def sync_pull_requests(db: Session, repository_id: uuid.UUID) -> list[PullRequest]:
    """Fetch recent PRs from GitHub, upsert them, and return the repository's stored PRs."""
    repository = get_repository(db, repository_id)

    try:
        pulls = github.list_pull_requests(
            repository.owner, repository.name, state="all", limit=PR_SYNC_LIMIT
        )
    except GitHubAPIError as exc:
        if exc.status == 404:
            raise NotFoundError(
                f"Repository {repository.owner}/{repository.name} no longer exists on GitHub or is no longer public"
            ) from exc
        raise
    # Keyed by number: Postgres rejects an upsert that touches the same row twice,
    # which can happen if a PR is opened while pages are being fetched.
    rows = {p["number"]: _pull_request_row(repository.id, p) for p in pulls}

    if rows:
        stmt = insert(PullRequest).values(list(rows.values()))
        stmt = stmt.on_conflict_do_update(
            constraint="uq_pull_requests_repository_id_github_pr_number",
            set_={
                column: stmt.excluded[column]
                for column in ("title", "author", "state", "opened_at", "merged_at")
            },
        )
        db.execute(stmt)
        db.commit()

    return list(
        db.scalars(
            select(PullRequest)
            .where(PullRequest.repository_id == repository.id)
            .order_by(
                PullRequest.opened_at.desc().nulls_last(),
                PullRequest.github_pr_number.desc(),
            )
        )
    )


def _pull_request_row(repository_id: uuid.UUID, pull: dict[str, Any]) -> dict[str, Any]:
    merged_at = _parse_github_datetime(pull.get("merged_at"))
    return {
        "repository_id": repository_id,
        "github_pr_number": pull["number"],
        "title": pull.get("title"),
        "author": (pull.get("user") or {}).get("login"),
        "state": "merged" if merged_at else pull.get("state"),
        "opened_at": _parse_github_datetime(pull.get("created_at")),
        "merged_at": merged_at,
    }


def _parse_github_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
