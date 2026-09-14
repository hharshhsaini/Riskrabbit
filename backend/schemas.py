from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

REPO_SEGMENT_PATTERN = r"^[A-Za-z0-9._-]+$"


class RepositoryCreate(BaseModel):
    owner: str = Field(min_length=1, max_length=100, pattern=REPO_SEGMENT_PATTERN)
    name: str = Field(min_length=1, max_length=100, pattern=REPO_SEGMENT_PATTERN)


class RepositoryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    owner: str
    name: str
    github_repo_id: int
    connected_at: datetime


class PullRequestOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    github_pr_number: int
    title: str | None
    author: str | None
    state: str | None
    opened_at: datetime | None
    merged_at: datetime | None


class PredictionCreate(BaseModel):
    pull_request_id: UUID


class PredictionOut(BaseModel):
    # protected_namespaces is cleared so `model_version` does not collide with
    # pydantic's reserved `model_` prefix.
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: UUID
    pull_request_id: UUID
    risk_score: float
    risk_label: str
    explanation: str | None
    features: dict[str, Any]
    model_version: str
    created_at: datetime
