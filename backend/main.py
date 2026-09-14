import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from config import settings
from database import Base, SessionLocal, engine, get_db, seed_default_user
from exceptions import GitHubAPIError, NotFoundError, PredictionError
from models import Prediction, PullRequest, Repository
from schemas import (
    ErrorOut,
    HealthOut,
    PredictionCreate,
    PredictionOut,
    PullRequestDetailOut,
    PullRequestOut,
    RepositoryCreate,
    RepositoryOut,
)
# Importing the pipeline loads the model, so the app refuses to start on bad model artifacts.
from services import history, pipeline, repo_sync

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# uvicorn installs its own handlers, without timestamps; route its lines through the one above.
for uvicorn_logger in ("uvicorn", "uvicorn.access"):
    logging.getLogger(uvicorn_logger).handlers.clear()
    logging.getLogger(uvicorn_logger).propagate = True

# Path parameter -> the 404 message for an id that cannot exist because it is not a UUID.
PATH_ID_NOT_FOUND = {"repo_id": "Repository not found", "pr_id": "Pull request not found"}

NOT_FOUND = {404: {"model": ErrorOut, "description": "No such repository or pull request"}}
GITHUB_FAILED = {502: {"model": ErrorOut, "description": "GitHub could not be reached or returned an error"}}
SAVE_FAILED = {500: {"model": ErrorOut, "description": "The prediction could not be saved"}}


def cors_origins(value: str) -> list[str]:
    # Browsers send Origin without a trailing slash, so a pasted "https://x.vercel.app/" would never match.
    return [origin.strip().rstrip("/") for origin in value.split(",") if origin.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_default_user(db)
    yield


app = FastAPI(
    title="Predictive Deployment Risk Assessment System", lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(settings.CORS_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # A path id that is not a UUID cannot name a row, so it is as unknown as any other id.
    locations = [tuple(error["loc"]) for error in exc.errors()]
    if locations and all(loc[0] == "path" and loc[-1] in PATH_ID_NOT_FOUND for loc in locations):
        return JSONResponse(status_code=404, content={"detail": PATH_ID_NOT_FOUND[locations[0][-1]]})
    return await request_validation_exception_handler(request, exc)


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


@app.get("/health", response_model=HealthOut)
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/repositories", response_model=RepositoryOut, responses={**NOT_FOUND, **GITHUB_FAILED})
def create_repository(
    payload: RepositoryCreate, db: Session = Depends(get_db)
) -> Repository:
    return repo_sync.connect_repository(db, payload.owner, payload.name)


@app.get("/repositories", response_model=list[RepositoryOut])
def list_repositories(db: Session = Depends(get_db)) -> list[Repository]:
    return repo_sync.list_repositories(db)


@app.get(
    "/repositories/{repo_id}/pull-requests",
    response_model=list[PullRequestOut],
    responses={**NOT_FOUND, **GITHUB_FAILED},
)
def list_pull_requests(
    repo_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[PullRequest]:
    return repo_sync.sync_pull_requests(db, repo_id)


@app.get("/pull-requests/{pr_id}", response_model=PullRequestDetailOut, responses=NOT_FOUND)
def get_pull_request(
    pr_id: uuid.UUID, db: Session = Depends(get_db)
) -> PullRequest:
    return repo_sync.get_pull_request(db, pr_id)


@app.post(
    "/predictions",
    response_model=PredictionOut,
    responses={**NOT_FOUND, **GITHUB_FAILED, **SAVE_FAILED},
)
def create_prediction(
    payload: PredictionCreate, db: Session = Depends(get_db)
) -> Prediction:
    return pipeline.run_prediction(db, payload.pull_request_id)


@app.get(
    "/predictions/pull-request/{pr_id}", response_model=list[PredictionOut], responses=NOT_FOUND
)
def pull_request_history(
    pr_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[Prediction]:
    return history.pull_request_predictions(db, pr_id)


@app.get(
    "/repositories/{repo_id}/predictions", response_model=list[PredictionOut], responses=NOT_FOUND
)
def repository_history(
    repo_id: uuid.UUID, db: Session = Depends(get_db)
) -> list[Prediction]:
    return history.repository_predictions(db, repo_id)


def openapi() -> dict[str, Any]:
    if app.openapi_schema is None:
        schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
        # FastAPI documents a 422 for every route with a parameter. Here every path parameter
        # is an id whose malformed value gets a 404 (handle_validation_error) and no route has
        # query parameters, so only routes with a request body can actually return 422.
        for operations in schema["paths"].values():
            for operation in operations.values():
                if "requestBody" not in operation:
                    operation["responses"].pop("422", None)
        app.openapi_schema = schema
    return app.openapi_schema


app.openapi = openapi
