"""Shared test fixtures.

Database tests run against the configured Postgres inside a transaction that is
rolled back after each test, so nothing is written. All other network access is
blocked for the whole run: a test that forgets to mock GitHub or the LLM fails
loudly instead of calling the real service.
"""

import socket
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

import main
from database import engine, get_db, get_default_user
from models import PullRequest, Repository

OPENED = datetime(2025, 6, 10, 12, 0, tzinfo=timezone.utc)
ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1", "testserver"}


def pytest_collection_modifyitems(items):
    # `pytest -m "not db"` then runs only the tests that need no database, and so no network.
    for item in items:
        if "db" in getattr(item, "fixturenames", ()):
            item.add_marker(pytest.mark.db)


@pytest.fixture(autouse=True, scope="session")
def block_network():
    """Refuse Python-level hostname lookups for the whole run.

    requests (GitHub) and httpx (the LLM client) resolve hosts through Python's
    socket module, so they fail here. The database is unaffected: psycopg2's libpq
    resolves the host in C, outside Python.
    """
    real_getaddrinfo = socket.getaddrinfo

    def guarded(host, *args, **kwargs):
        name = host.decode() if isinstance(host, bytes) else str(host)
        if name not in ALLOWED_HOSTS:
            raise socket.gaierror(
                f"network access is blocked in tests (tried to resolve {name!r}); mock the call instead"
            )
        return real_getaddrinfo(host, *args, **kwargs)

    with pytest.MonkeyPatch.context() as patcher:
        patcher.setattr(socket, "getaddrinfo", guarded)
        yield


@pytest.fixture
def db():
    """A real database session whose changes are all rolled back after the test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db):
    """The FastAPI app with its database dependency pointed at the rolled-back session."""
    main.app.dependency_overrides[get_db] = lambda: db
    try:
        yield TestClient(main.app, raise_server_exceptions=False)
    finally:
        main.app.dependency_overrides.clear()


@pytest.fixture
def repository(db):
    repo = Repository(
        user_id=get_default_user(db).id,
        github_repo_id=900_000_000 + uuid.uuid4().int % 99_999,
        owner="acme",
        name="widgets",
    )
    db.add(repo)
    db.flush()
    return repo


@pytest.fixture
def pull_request(db, repository):
    """PR #42 by alice, with two of her PRs merged before it, one merged after, and one by bob."""
    earlier = [
        PullRequest(repository_id=repository.id, github_pr_number=n, author="alice", state="merged",
                    opened_at=OPENED - timedelta(days=10 + n), merged_at=OPENED - timedelta(days=5 + n))
        for n in (1, 2)
    ]
    later = PullRequest(repository_id=repository.id, github_pr_number=3, author="alice", state="merged",
                        opened_at=OPENED + timedelta(days=1), merged_at=OPENED + timedelta(days=2))
    other_author = PullRequest(repository_id=repository.id, github_pr_number=4, author="bob", state="merged",
                               opened_at=OPENED - timedelta(days=9), merged_at=OPENED - timedelta(days=8))
    target = PullRequest(repository_id=repository.id, github_pr_number=42, author="alice", state="open", opened_at=OPENED)
    db.add_all([*earlier, later, other_author, target])
    db.flush()
    return target
