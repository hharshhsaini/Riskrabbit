# TODO: remove before deployment — temporary Step 08 helper behind GET /debug/features.
import uuid
from typing import Any

from sqlalchemy.orm import Session

from exceptions import GitHubAPIError, NotFoundError
from features import build_feature_vector
from services import github, repo_sync

# feature name -> (GitHub PR total, value recomputed from the paginated lists)
CHECKED_FEATURES = {
    "files_changed": ("changed_files", "files_listed"),
    "lines_added": ("additions", "additions_summed"),
    "lines_removed": ("deletions", "deletions_summed"),
    "commit_count": ("commits", "commits_listed"),
}


def inspect_pr_features(
    db: Session, repository_id: uuid.UUID, pr_number: int
) -> dict[str, Any]:
    repository = repo_sync.get_repository(db, repository_id)
    owner, name = repository.owner, repository.name

    try:
        pr = github.get_pr(owner, name, pr_number)
    except GitHubAPIError as exc:
        if exc.status == 404:
            raise NotFoundError(
                f"Pull request #{pr_number} not found in {owner}/{name}"
            ) from exc
        raise
    files = github.get_pr_files(owner, name, pr_number)
    commits = github.get_pr_commits(owner, name, pr_number)

    author = (pr.get("user") or {}).get("login")
    author_pr_count = repo_sync.count_prior_merged_prs(
        db, repository.id, author, before=pr.get("created_at")
    )
    features = build_feature_vector(pr, files, commits, author_pr_count=author_pr_count)

    from_lists = {
        "files_listed": len(files),
        "additions_summed": sum(f.get("additions") or 0 for f in files),
        "deletions_summed": sum(f.get("deletions") or 0 for f in files),
        "commits_listed": len(commits),
    }
    checks = {
        feature: {
            "feature_value": features[feature],
            "github_pr_total": pr.get(total_key),
            "from_paginated_lists": from_lists[list_key],
            "all_three_match": features[feature]
            == pr.get(total_key)
            == from_lists[list_key],
        }
        for feature, (total_key, list_key) in CHECKED_FEATURES.items()
    }

    html_url = pr.get("html_url")
    return {
        "pull_request": {
            "repository": f"{owner}/{name}",
            "number": pr.get("number"),
            "title": pr.get("title"),
            "author": author,
            "state": "merged" if pr.get("merged_at") else pr.get("state"),
            "created_at": pr.get("created_at"),
            "merged_at": pr.get("merged_at"),
            "verify_files_on_github": f"{html_url}/files" if html_url else None,
            "verify_commits_on_github": f"{html_url}/commits" if html_url else None,
        },
        "checks": checks,
        "raw_counts": {
            "changed_files": pr.get("changed_files"),
            "additions": pr.get("additions"),
            "deletions": pr.get("deletions"),
            "commits": pr.get("commits"),
            "review_comments": pr.get("review_comments"),
            **from_lists,
        },
        "features": features,
    }
