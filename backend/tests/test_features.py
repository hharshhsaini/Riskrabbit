import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import features
from features import FEATURE_ORDER, build_feature_vector, to_array

UTC = timezone.utc
WEDNESDAY_NOON = datetime(2024, 1, 3, 12, 0, tzinfo=UTC)
SATURDAY_NOON = datetime(2024, 1, 6, 12, 0, tzinfo=UTC)


def iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def file_entry(path: str, additions: int = 1, deletions: int = 0) -> dict:
    return {"filename": path, "additions": additions, "deletions": deletions}


def commit_entry(message: str) -> dict:
    return {"commit": {"message": message}}


def build(pr=None, files=None, commits=None, author_pr_count=0, now=WEDNESDAY_NOON):
    return build_feature_vector(pr or {}, files or [], commits or [], author_pr_count, now)


def test_empty_pr_returns_all_features_as_zero_floats():
    vector = build()

    assert list(vector) == FEATURE_ORDER
    assert all(type(value) is float for value in vector.values())
    assert all(value == 0.0 for value in vector.values())


def test_pr_with_tests_sets_has_tests_and_ratio():
    files = [
        file_entry("src/app.py"),
        file_entry("tests/test_app.py"),
        file_entry("src/utils.py"),
        file_entry("web/Button.spec.tsx"),
    ]

    vector = build(files=files)

    assert vector["has_tests"] == 1.0
    assert vector["test_file_ratio"] == 0.5


@pytest.mark.parametrize(
    "path",
    [
        "tests/conftest.py",
        "pkg/tests/helpers.py",
        "test_models.py",
        "handler_test.go",
        "src/app.test.js",
        "SRC/APP.SPEC.TS",
    ],
)
def test_test_path_patterns_match(path):
    assert build(files=[file_entry(path)])["has_tests"] == 1.0


@pytest.mark.parametrize(
    "path", ["src/latest_version.py", "src/contest_rules.py", "docs/testing-guide.md"]
)
def test_similar_names_are_not_tests(path):
    assert build(files=[file_entry(path)])["has_tests"] == 0.0


@pytest.mark.parametrize(
    "path",
    [
        "deploy/app.yml",
        "k8s/service.YAML",
        ".env",
        "config/.env.production",
        "Dockerfile",
        "docker/Dockerfile.dev",
        "db/migrations/0001_init.py",
        "requirements.txt",
        "frontend/package.json",
    ],
)
def test_config_touching_pr_sets_touches_config(path):
    vector = build(files=[file_entry("src/app.py"), file_entry(path)])

    assert vector["touches_config"] == 1.0


@pytest.mark.parametrize("path", ["src/environment.py", "docs/migrations.md", "src/app.py"])
def test_similar_names_are_not_config(path):
    assert build(files=[file_entry(path)])["touches_config"] == 0.0


def test_ci_workflow_sets_touches_ci_and_config():
    vector = build(files=[file_entry(".github/workflows/ci.yml")])

    assert vector["touches_ci"] == 1.0
    assert vector["touches_config"] == 1.0
    assert build(files=[file_entry("docs/workflows/ci.md")])["touches_ci"] == 0.0


def test_weekend_merge_uses_merged_at_not_now():
    pr = {"created_at": iso(SATURDAY_NOON - timedelta(hours=5)), "merged_at": iso(SATURDAY_NOON)}

    vector = build(pr=pr, now=WEDNESDAY_NOON)

    assert vector["is_weekend"] == 1.0
    assert vector["is_off_hours"] == 0.0
    assert vector["pr_age_hours"] == 5.0


def test_still_open_pr_measures_age_and_timing_from_now():
    now = SATURDAY_NOON.replace(hour=22)
    pr = {"created_at": iso(now - timedelta(hours=30)), "merged_at": None, "state": "open"}

    vector = build(pr=pr, now=now)

    assert vector["pr_age_hours"] == 30.0
    assert vector["is_weekend"] == 1.0
    assert vector["is_off_hours"] == 1.0


def test_closed_unmerged_pr_age_stops_at_closed_at():
    pr = {"created_at": iso(WEDNESDAY_NOON), "closed_at": iso(WEDNESDAY_NOON + timedelta(hours=2))}

    assert build(pr=pr, now=SATURDAY_NOON)["pr_age_hours"] == 2.0


@pytest.mark.parametrize(
    ("hour", "minute", "off_hours"),
    [(8, 59, 1.0), (9, 0, 0.0), (17, 59, 0.0), (18, 0, 1.0)],
)
def test_off_hours_boundaries(hour, minute, off_hours):
    merged = WEDNESDAY_NOON.replace(hour=hour, minute=minute)

    assert build(pr={"merged_at": iso(merged)})["is_off_hours"] == off_hours


def test_divide_by_zero_cases_return_zero():
    vector = build(pr={"changed_files": 7, "commits": 3}, files=[], commits=[])

    assert vector["test_file_ratio"] == 0.0
    assert vector["avg_commit_msg_len"] == 0.0
    assert vector["files_changed"] == 7.0
    assert vector["commit_count"] == 3.0


def test_missing_and_null_fields_never_raise():
    pr = {
        "changed_files": None,
        "additions": None,
        "deletions": "not a number",
        "commits": float("nan"),
        "review_comments": None,
        "created_at": "not a date",
        "merged_at": None,
        "user": None,
    }
    files = [{"filename": None, "additions": None}, {}, None, {"filename": "src/a.py"}]
    commits = [{"commit": None}, {"commit": {"message": None}}, {}, None]

    vector = build(pr=pr, files=files, commits=commits, author_pr_count=None)

    assert all(type(value) is float for value in vector.values())
    assert vector["files_changed"] == 3.0
    assert vector["lines_added"] == 0.0
    assert vector["lines_removed"] == 0.0
    assert vector["commit_count"] == 3.0
    assert vector["avg_commit_msg_len"] == 0.0
    assert vector["review_comment_count"] == 0.0
    assert vector["pr_age_hours"] == 0.0
    assert vector["author_pr_count"] == 0.0


def test_pr_level_totals_win_over_truncated_lists():
    pr = {"changed_files": 3500, "additions": 90000, "deletions": 10000, "commits": 300}
    files = [file_entry(f"src/file_{i}.py", additions=1, deletions=1) for i in range(3000)]
    commits = [commit_entry("wip")] * 250

    vector = build(pr=pr, files=files, commits=commits)

    assert vector["files_changed"] == 3500.0
    assert vector["commit_count"] == 300.0
    assert vector["total_churn"] == 100000.0


def test_list_fallback_when_pr_totals_absent():
    files = [file_entry("src/a.py", 10, 2), file_entry("src/b.py", 5, 3)]
    commits = [commit_entry("Fix bug"), commit_entry("Add feature flag")]

    vector = build(files=files, commits=commits)

    assert vector["files_changed"] == 2.0
    assert vector["lines_added"] == 15.0
    assert vector["lines_removed"] == 5.0
    assert vector["total_churn"] == 20.0
    assert vector["commit_count"] == 2.0
    assert vector["avg_commit_msg_len"] == 11.5


def test_unique_dirs_counts_top_two_levels():
    files = [
        file_entry("README.md"),
        file_entry("setup.py"),
        file_entry("src/flask/app.py"),
        file_entry("src/flask/json/provider.py"),
        file_entry("src/other.py"),
        file_entry("docs/index.rst"),
    ]

    assert build(files=files)["unique_dirs_touched"] == 4.0


def test_review_comments_and_author_history():
    vector = build(pr={"review_comments": 12}, author_pr_count=41)

    assert vector["review_comment_count"] == 12.0
    assert vector["author_pr_count"] == 41.0


def test_naive_now_is_treated_as_utc_and_default_now_works():
    naive = datetime(2024, 1, 6, 12, 0)

    assert build(now=naive)["is_weekend"] == 1.0
    assert set(build_feature_vector({}, [], [])) == set(FEATURE_ORDER)


def test_to_array_orders_by_feature_order_and_ignores_extras():
    vector = {name: float(index) for index, name in reversed(list(enumerate(FEATURE_ORDER)))}
    vector["_shap"] = {"files_changed": 0.3}

    assert to_array(vector) == [float(i) for i in range(len(FEATURE_ORDER))]


def test_to_array_names_missing_keys():
    vector = build()
    del vector["has_tests"]
    del vector["pr_age_hours"]

    with pytest.raises(KeyError) as excinfo:
        to_array(vector)

    assert "has_tests" in str(excinfo.value)
    assert "pr_age_hours" in str(excinfo.value)


def test_feature_order_has_sixteen_unique_names_matching_output():
    assert len(FEATURE_ORDER) == 16
    assert len(set(FEATURE_ORDER)) == 16
    assert list(build()) == FEATURE_ORDER


def test_features_module_has_no_network_or_database_imports():
    tree = ast.parse(Path(features.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])

    forbidden = {"requests", "httpx", "sqlalchemy", "openai", "config", "database", "models", "services"}
    assert not imported & forbidden
