"""The prediction pipeline: GitHub, features, model and explanation, in that order.

run_prediction is the only place these services are called together. Routes call it;
nothing else runs them one after another.
"""

import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import features
from database import get_default_user
from exceptions import GitHubAPIError, NotFoundError, PredictionError
from models import Prediction, PullRequest, Repository
from services import explain, github, model, repo_sync

logger = logging.getLogger(__name__)


def run_prediction(db: Session, pull_request_id: uuid.UUID) -> Prediction:
    started = time.perf_counter()
    timings: dict[str, float] = {}

    with _stage("load pull request", pull_request_id, timings):
        pull_request, repository = _load_pull_request(db, pull_request_id)
    owner, name, number = repository.owner, repository.name, pull_request.github_pr_number

    with _stage("fetch from GitHub", pull_request_id, timings):
        # The full PR object, not a list item: only it carries changed_files,
        # additions, deletions, commits and review_comments, as in training.
        try:
            pr = github.get_pr(owner, name, number)
        except GitHubAPIError as exc:
            if exc.status == 404:
                raise NotFoundError(f"Pull request #{number} no longer exists in {owner}/{name}") from exc
            raise
        files = github.get_pr_files(owner, name, number)
        commits = github.get_pr_commits(owner, name, number)

    with _stage("count author history", pull_request_id, timings):
        author = (pr.get("user") or {}).get("login")
        author_pr_count = repo_sync.count_prior_merged_prs(
            db, repository.id, author, before=pr.get("created_at")
        )

    with _stage("build features", pull_request_id, timings):
        feature_vector = features.build_feature_vector(pr, files, commits, author_pr_count=author_pr_count)

    with _stage("score", pull_request_id, timings):
        score, label = model.predict(feature_vector)

    with _stage("explain", pull_request_id, timings):
        explanation = explain.generate_explanation(feature_vector, score, label)

    with _stage("save", pull_request_id, timings):
        prediction = Prediction(
            pull_request_id=pull_request.id,
            risk_score=score,
            risk_label=label,
            features=feature_vector,
            explanation=explanation,
            model_version=model.MODEL_VERSION,
        )
        db.add(prediction)
        try:
            db.commit()
        except SQLAlchemyError as exc:
            db.rollback()
            raise PredictionError(f"Could not save the prediction for {owner}/{name}#{number}") from exc
        db.refresh(prediction)

    total_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "prediction %s for %s/%s#%s: %s risk %.3f in %.0f ms (%.0f ms without the explanation)",
        pull_request_id, owner, name, number, label, score, total_ms, total_ms - timings["explain"],
    )
    return prediction


def _load_pull_request(db: Session, pull_request_id: uuid.UUID) -> tuple[PullRequest, Repository]:
    row = db.execute(
        select(PullRequest, Repository)
        .join(Repository, PullRequest.repository_id == Repository.id)
        .where(PullRequest.id == pull_request_id, Repository.user_id == get_default_user(db).id)
    ).one_or_none()
    if row is None:
        raise NotFoundError("Pull request not found")
    return row[0], row[1]


@contextmanager
def _stage(name: str, pull_request_id: uuid.UUID, timings: dict[str, float]) -> Iterator[None]:
    started = time.perf_counter()
    yield
    timings[name] = (time.perf_counter() - started) * 1000
    logger.info("prediction %s: %s took %.0f ms", pull_request_id, name, timings[name])
