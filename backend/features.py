"""Feature engineering shared by the live API and offline training.

Pure: no network, no database, no clock reads except the optional `now` default.
services/pipeline.py and ml/train.py both import build_feature_vector, so training
and serving cannot compute features differently.
"""

import math
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Any

FEATURE_ORDER = [
    "files_changed",
    "lines_added",
    "lines_removed",
    "total_churn",
    "unique_dirs_touched",
    "commit_count",
    "avg_commit_msg_len",
    "has_tests",
    "test_file_ratio",
    "touches_config",
    "touches_ci",
    "review_comment_count",
    "pr_age_hours",
    "is_weekend",
    "is_off_hours",
    "author_pr_count",
]

WORKDAY_START_HOUR = 9
WORKDAY_END_HOUR = 18

CONFIG_SUFFIXES = (".yml", ".yaml")
CONFIG_FILENAMES = ("requirements.txt", "package.json")
CI_PREFIX = ".github/workflows/"


def build_feature_vector(
    pr: dict[str, Any],
    files: list[dict[str, Any]],
    commits: list[dict[str, Any]],
    author_pr_count: int = 0,
    now: datetime | None = None,
) -> dict[str, float]:
    pr = pr or {}
    files = [f for f in (files or []) if isinstance(f, dict)]
    commits = [c for c in (commits or []) if isinstance(c, dict)]
    now = _as_utc(now) if isinstance(now, datetime) else datetime.now(timezone.utc)

    paths = [_path(f) for f in files]
    paths = [p for p in paths if p]

    # PR-level totals are exact; GitHub truncates the file list at 3,000 and the
    # commit list at 250, so the lists are only a fallback.
    files_changed = _count(pr.get("changed_files"), fallback=len(files))
    lines_added = _count(
        pr.get("additions"), fallback=sum(_num(f.get("additions")) for f in files)
    )
    lines_removed = _count(
        pr.get("deletions"), fallback=sum(_num(f.get("deletions")) for f in files)
    )
    commit_count = _count(pr.get("commits"), fallback=len(commits))

    message_lengths = [len(_commit_message(c)) for c in commits]
    test_files = sum(1 for p in paths if _is_test_path(p))

    opened_at = _parse_datetime(pr.get("created_at"))
    merged_at = _parse_datetime(pr.get("merged_at"))
    closed_at = _parse_datetime(pr.get("closed_at"))
    age_end = merged_at or closed_at or now
    pr_age_hours = (
        max(0.0, (age_end - opened_at).total_seconds() / 3600) if opened_at else 0.0
    )

    moment = merged_at or now

    return {
        "files_changed": files_changed,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
        "total_churn": lines_added + lines_removed,
        "unique_dirs_touched": float(len({_top_dirs(p) for p in paths})),
        "commit_count": commit_count,
        "avg_commit_msg_len": _ratio(sum(message_lengths), len(message_lengths)),
        "has_tests": _flag(test_files > 0),
        "test_file_ratio": _ratio(test_files, len(paths)),
        "touches_config": _flag(any(_is_config_path(p) for p in paths)),
        "touches_ci": _flag(any(p.startswith(CI_PREFIX) for p in paths)),
        "review_comment_count": _num(pr.get("review_comments")),
        "pr_age_hours": pr_age_hours,
        "is_weekend": _flag(moment.weekday() >= 5),
        "is_off_hours": _flag(
            not WORKDAY_START_HOUR <= moment.hour < WORKDAY_END_HOUR
        ),
        "author_pr_count": _num(author_pr_count),
    }


def to_array(features: dict[str, float]) -> list[float]:
    """Order feature values by FEATURE_ORDER. Extra keys are ignored."""
    missing = [name for name in FEATURE_ORDER if name not in features]
    if missing:
        raise KeyError(f"Feature vector is missing required features: {missing}")
    return [float(features[name]) for name in FEATURE_ORDER]


def _is_test_path(path: str) -> bool:
    # Anchored to file and directory names, so "latest_version.py" is not a test.
    p = PurePosixPath(path)
    return (
        p.name.startswith("test_")
        or p.stem.endswith("_test")
        or ".test." in p.name
        or ".spec." in p.name
        or "tests" in p.parts[:-1]
    )


def _is_config_path(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        p.name.endswith(CONFIG_SUFFIXES)
        or p.name == ".env"
        or p.name.startswith(".env.")
        or p.name.startswith("dockerfile")
        or p.name in CONFIG_FILENAMES
        or "migrations" in p.parts[:-1]
    )


def _top_dirs(path: str) -> str:
    return "/".join(PurePosixPath(path).parts[:-1][:2]) or "."


def _path(file: dict[str, Any]) -> str:
    name = file.get("filename")
    return name.strip().lower() if isinstance(name, str) else ""


def _commit_message(commit: dict[str, Any]) -> str:
    message = (commit.get("commit") or {}).get("message")
    return message if isinstance(message, str) else ""


def _num(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return 0.0


def _count(value: Any, fallback: float) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return float(fallback)


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _flag(condition: bool) -> float:
    return 1.0 if condition else 0.0


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return _as_utc(value)
    if not isinstance(value, str) or not value:
        return None
    try:
        return _as_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    except ValueError:
        return None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
