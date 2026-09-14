"""Prediction history for a pull request or a repository, newest first."""

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from models import Prediction, PullRequest
from services import repo_sync

REPOSITORY_HISTORY_LIMIT = 50


def pull_request_predictions(db: Session, pull_request_id: uuid.UUID) -> list[Prediction]:
    repo_sync.get_pull_request(db, pull_request_id)  # 404 for an unknown or foreign PR
    return list(
        db.scalars(
            select(Prediction)
            .where(Prediction.pull_request_id == pull_request_id)
            .order_by(Prediction.created_at.desc(), Prediction.id.desc())
        )
    )


def repository_predictions(
    db: Session, repository_id: uuid.UUID, limit: int = REPOSITORY_HISTORY_LIMIT
) -> list[Prediction]:
    repository = repo_sync.get_repository(db, repository_id)
    return list(
        db.scalars(
            select(Prediction)
            .join(PullRequest, Prediction.pull_request_id == PullRequest.id)
            .where(PullRequest.repository_id == repository.id)
            .order_by(Prediction.created_at.desc(), Prediction.id.desc())
            .limit(limit)
        )
    )
