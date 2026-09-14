import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from config import settings
from database import Base, SessionLocal, engine, get_db
from exceptions import GitHubAPIError, NotFoundError, PredictionError
from models import PullRequest, Repository, User
from schemas import PullRequestOut, RepositoryCreate, RepositoryOut
from services import debug_features, repo_sync


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        if db.execute(select(User.id).limit(1)).first() is None:
            db.add(User(email=settings.DEFAULT_USER_EMAIL))
            db.commit()
    yield


app = FastAPI(
    title="Predictive Deployment Risk Assessment System", lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(NotFoundError)
def handle_not_found(request: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": exc.message})


@app.exception_handler(GitHubAPIError)
def handle_github_error(request: Request, exc: GitHubAPIError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": exc.message})


@app.exception_handler(PredictionError)
def handle_prediction_error(request: Request, exc: PredictionError) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": exc.message})


@app.exception_handler(Exception)
def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    # The traceback still goes to the server log; only the client response is sanitised.
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/repositories", response_model=RepositoryOut)
def create_repository(
    payload: RepositoryCreate, db: Session = Depends(get_db)
) -> Repository:
    return repo_sync.connect_repository(db, payload.owner, payload.name)


@app.get("/repositories", response_model=list[RepositoryOut])
def list_repositories(db: Session = Depends(get_db)) -> list[Repository]:
    return repo_sync.list_repositories(db)


@app.get(
    "/repositories/{repo_id}/pull-requests", response_model=list[PullRequestOut]
)
def list_pull_requests(
    repo_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[PullRequest]:
    return repo_sync.sync_pull_requests(db, repo_id)


# TODO: remove before deployment
@app.get("/debug/features/{repo_id}/{pr_number}")
def debug_pr_features(
    repo_id: uuid.UUID, pr_number: int, db: Session = Depends(get_db)
) -> dict[str, Any]:
    return debug_features.inspect_pr_features(db, repo_id, pr_number)
