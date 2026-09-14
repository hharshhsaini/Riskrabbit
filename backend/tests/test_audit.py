import features
from features import FEATURE_ORDER
from ml import audit


def record(pr_id, number, author, created, merged, changed_files=3, is_risky=0, repo="o/r"):
    return {
        "repo": repo,
        "pr": {
            "id": pr_id,
            "number": number,
            "title": f"PR {number}",
            "user": {"login": author},
            "created_at": created,
            "merged_at": merged,
            "changed_files": changed_files,
            "html_url": f"https://github.com/{repo}/pull/{number}",
        },
        "files": [],
        "commits": [],
        "is_risky": is_risky,
        "label_reason": "hotfix" if is_risky else "none",
    }


def test_audit_uses_the_shared_feature_function_not_a_copy():
    assert audit.build_feature_vector is features.build_feature_vector


def test_build_dataset_columns_and_values():
    rows = [
        record(1, 10, "alice", "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z", changed_files=7, is_risky=1),
        record(2, 11, "bob", "2025-01-03T00:00:00Z", "2025-01-04T00:00:00Z"),
    ]

    dataset = audit.build_dataset(rows)

    assert list(dataset.columns) == audit.ID_COLUMNS + [audit.LABEL_COLUMN] + FEATURE_ORDER
    assert dataset.loc[0, "files_changed"] == 7.0
    assert dataset.loc[0, "is_risky"] == 1
    assert dataset.loc[1, "label_reason"] == "none"


def test_prior_merged_counts_only_count_same_author_same_repo_merged_before_opening():
    rows = [
        record(1, 10, "alice", "2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"),
        record(2, 11, "alice", "2025-01-05T00:00:00Z", "2025-01-06T00:00:00Z"),
        record(3, 12, "alice", "2025-01-05T12:00:00Z", "2025-01-09T00:00:00Z"),
        record(4, 13, "bob", "2025-01-10T00:00:00Z", "2025-01-11T00:00:00Z"),
        record(5, 14, "alice", "2025-01-10T00:00:00Z", "2025-01-11T00:00:00Z", repo="o/other"),
    ]

    counts = audit._prior_merged_counts(rows)

    assert counts == {1: 0, 2: 1, 3: 1, 4: 0, 5: 0}
