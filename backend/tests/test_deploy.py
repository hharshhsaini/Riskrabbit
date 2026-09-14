"""Step 25 deploy settings: CORS from configuration, and seeding the configured user."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

import main
from config import settings
from database import seed_default_user
from models import User


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://localhost:5173", ["http://localhost:5173"]),
        ("https://riskrabbit.vercel.app/, http://localhost:5173", ["https://riskrabbit.vercel.app", "http://localhost:5173"]),
        (" , ", []),
    ],
)
def test_cors_origins_are_split_trimmed_and_lose_trailing_slashes(value, expected):
    assert main.cors_origins(value) == expected


def test_cors_allows_the_configured_origin_and_no_other():
    client = TestClient(main.app)
    allowed = main.cors_origins(settings.CORS_ORIGINS)[0]
    preflight = {"Access-Control-Request-Method": "POST"}

    ok = client.options("/predictions", headers={"Origin": allowed, **preflight})
    refused = client.options("/predictions", headers={"Origin": "https://evil.example", **preflight})

    assert ok.headers.get("access-control-allow-origin") == allowed
    assert "access-control-allow-origin" not in refused.headers


def test_seeding_creates_the_configured_user_even_when_another_user_exists(db):
    assert db.scalar(select(func.count()).select_from(User)) >= 1

    with patch.object(settings, "DEFAULT_USER_EMAIL", "deploy-check@example.com"):
        seed_default_user(db)
        seed_default_user(db)

    assert db.scalar(select(func.count()).select_from(User).where(User.email == "deploy-check@example.com")) == 1
