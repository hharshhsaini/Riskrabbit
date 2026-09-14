"""Harvest merged pull requests and default-branch commits for model training.

Run from backend/, keeping the Mac awake while it runs:
    caffeinate -i ./venv/bin/python -m ml.collect

Safe to kill and re-run at any point. PRs and commits already written are
skipped, a half-written last line is repaired, and each repository's PR and
commit listings are cached in ml/data/cache/ after the first pass.
"""

import argparse
import json
import os
import re
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar

from exceptions import GitHubAPIError
from services import github

T = TypeVar("T")

DATA_DIR = Path(__file__).resolve().parent / "data"

# Chosen 2026-09-14 by vetting 36 candidates: public, active, merges most PRs
# through GitHub, and at least ~200 human-authored PRs merged into the default
# branch in the last two years. Excluded: pallets/flask and psf/requests (few PRs
# merged through GitHub), python/typeshed (type stubs only), fastapi/fastapi and
# other very large repos (dominated by bot or translation PRs).
REPOS = [
    "python/mypy",
    "getsentry/sentry-python",
    "netbox-community/netbox",
    "sqlfluff/sqlfluff",
    "getmoto/moto",
    "aio-libs/aiohttp",
    "sphinx-doc/sphinx",
    "pylint-dev/pylint",
    "beetbox/beets",
    "wagtail/wagtail",
    "streamlink/streamlink",
    "pytest-dev/pytest",
    "pypa/pip",
    "fastapi/typer",
    "tox-dev/tox",
    "Kludex/starlette",
]

WINDOW_DAYS = 730
# A PR merged very recently has not had time to be reverted or hotfixed, so its
# label would be a false "safe". Matches the widest hotfix window in Step 13b.
LABEL_MATURITY_DAYS = 14
MAX_PRS_PER_REPO = 300
PROGRESS_EVERY = 25
RATE_LIMIT_PADDING_SECONDS = 5
TRANSIENT_RETRIES = 5
TRANSIENT_BACKOFF_SECONDS = 30
LISTING_LOOKBACK_DAYS = 90

# Commits whose changed files ml/label.py may need. Keep this a superset of the
# hotfix and revert patterns used there.
FIX_LIKE_MESSAGE = re.compile(
    r"^revert\b|\b(fix|fixes|fixed|fixing|hotfix|patch|bug|bugfix|regression)\b",
    re.IGNORECASE,
)


class SkipItem(Exception):
    """The item no longer exists on GitHub; skip it rather than stop the run."""


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cache_dir = args.data_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prs_path = args.data_dir / "raw_prs.jsonl"
    commits_path = args.data_dir / "raw_commits.jsonl"

    window = _load_or_create_window(cache_dir / "window.json", args.window_days)
    _log(f"window: PRs merged {window['since']} to {window['until']}")

    pr_ids, merges = _load_pr_state(prs_path)
    commit_keys = _load_commit_keys(commits_path)
    _log(f"resuming with {len(pr_ids)} PRs and {len(commit_keys)} commits already collected")

    for full_name in args.repos:
        try:
            _collect_repo(
                full_name, args, window, cache_dir,
                prs_path, commits_path, pr_ids, merges, commit_keys,
            )
        except SkipItem as exc:
            _log(f"{full_name}: skipped repository ({exc})")

    _log(f"DONE: {len(pr_ids)} PRs in {prs_path.name}, {len(commit_keys)} commits in {commits_path.name}")


def _collect_repo(
    full_name: str,
    args: argparse.Namespace,
    window: dict[str, Any],
    cache_dir: Path,
    prs_path: Path,
    commits_path: Path,
    pr_ids: set[int],
    merges: dict[str, dict[str, str]],
    commit_keys: set[tuple[str, str]],
) -> None:
    owner, name = full_name.split("/")
    repo = _call(github.get_repo, owner, name)
    canonical, branch = repo["full_name"], repo["default_branch"]
    slug = canonical.replace("/", "__")

    index = _cached(
        cache_dir / f"{slug}__prs.json",
        lambda: _list_eligible_prs(owner, name, branch, window, args.max_prs_per_repo),
    )
    _log(
        f"{canonical}: {len(index['prs'])} PRs selected of {index['eligible']} eligible "
        f"(skipped {index['skipped_bots']} bot, {index['skipped_unmerged']} unmerged, "
        f"{index['skipped_immature']} merged in the last {LABEL_MATURITY_DAYS} days)"
    )

    added = 0
    with prs_path.open("a", encoding="utf-8") as out:
        for position, item in enumerate(index["prs"], start=1):
            if item["id"] in pr_ids:
                continue
            try:
                record = _collect_pr(owner, name, canonical, branch, item["number"])
            except SkipItem as exc:
                _log(f"{canonical}#{item['number']}: skipped ({exc})")
                continue
            _append(out, record)
            pr_ids.add(record["pr"]["id"])
            if record["pr"].get("merge_commit_sha"):
                merges.setdefault(canonical, {})[record["pr"]["merge_commit_sha"]] = record["pr"]["merged_at"]
            added += 1
            if added % PROGRESS_EVERY == 0:
                _progress(f"{canonical}: PR {position}/{len(index['prs'])}", len(pr_ids))
    _log(f"{canonical}: PRs complete ({added} new this run, {len(pr_ids)} total)")

    commits = _cached(
        cache_dir / f"{slug}__commits.json",
        lambda: _list_branch_commits(owner, name, branch, window),
    )
    repo_merges = merges.get(canonical, {})
    merge_times = sorted(t for t in (_parse(v) for v in repo_merges.values()) if t)

    written = fetched = 0
    with commits_path.open("a", encoding="utf-8") as out:
        for summary in commits:
            key = (canonical, summary["sha"])
            if key in commit_keys:
                continue
            files = None
            if _needs_files(summary, repo_merges, merge_times):
                try:
                    detail = _call(github.get_commit, owner, name, summary["sha"])
                except SkipItem:
                    detail = {}
                files = [f["filename"] for f in detail.get("files") or [] if f.get("filename")]
                fetched += 1
                if fetched % PROGRESS_EVERY == 0:
                    _progress(f"{canonical}: commit files {fetched} fetched", len(pr_ids))
            _append(out, {"repo": canonical, **summary, "files": files})
            commit_keys.add(key)
            written += 1
    _log(f"{canonical}: commits complete ({written} new, {fetched} with file lists)")


def _list_eligible_prs(
    owner: str, name: str, branch: str, window: dict[str, Any], max_prs: int
) -> dict[str, Any]:
    since, until = _parse(window["since"]), _parse(window["until"])
    stop_before = since - timedelta(days=LISTING_LOOKBACK_DAYS)
    stats = {"skipped_bots": 0, "skipped_unmerged": 0, "skipped_immature": 0}
    eligible: list[dict[str, Any]] = []

    params = {"state": "closed", "base": branch, "sort": "created", "direction": "desc"}
    for page in _walk_pages(f"/repos/{owner}/{name}/pulls", params):
        reached_end = False
        for pull in page:
            created = _parse(pull.get("created_at"))
            if created and created < stop_before:
                reached_end = True
                break
            merged = _parse(pull.get("merged_at"))
            if merged is None:
                stats["skipped_unmerged"] += 1
            elif merged < since:
                continue
            elif merged > until:
                stats["skipped_immature"] += 1
            elif _is_bot(pull.get("user")):
                stats["skipped_bots"] += 1
            else:
                eligible.append(
                    {"id": pull["id"], "number": pull["number"], "merged_at": pull["merged_at"]}
                )
        if reached_end:
            break

    eligible.sort(key=lambda p: p["merged_at"])
    return {
        "branch": branch,
        "eligible": len(eligible),
        **stats,
        "prs": _evenly_spaced(eligible, max_prs),
    }


def _list_branch_commits(
    owner: str, name: str, branch: str, window: dict[str, Any]
) -> list[dict[str, Any]]:
    params = {"sha": branch, "since": window["since"], "until": window["created_at"]}
    return [
        _trim_branch_commit(commit)
        for page in _walk_pages(f"/repos/{owner}/{name}/commits", params)
        for commit in page
    ]


def _collect_pr(owner: str, name: str, canonical: str, branch: str, number: int) -> dict[str, Any]:
    # The full PR object, not the list item: only it carries changed_files,
    # additions, deletions, commits and review_comments.
    pr = _call(github.get_pr, owner, name, number)
    files = _call(github.get_pr_files, owner, name, number)
    commits = _call(github.get_pr_commits, owner, name, number)
    return {
        "repo": canonical,
        "default_branch": branch,
        "collected_at": _now_iso(),
        "pr": pr,
        "files": [_trim_file(f) for f in files],
        "commits": [_trim_pr_commit(c) for c in commits],
    }


def _needs_files(
    summary: dict[str, Any], repo_merges: dict[str, str], merge_times: list[datetime]
) -> bool:
    if summary["sha"] in repo_merges:
        return False  # a collected PR's merge commit; its files are already stored
    if not FIX_LIKE_MESSAGE.search(summary["message"]):
        return False
    committed = _parse(summary["date"])
    if committed is None:
        return False
    horizon = timedelta(days=LABEL_MATURITY_DAYS)
    return any(timedelta(0) < committed - merged <= horizon for merged in merge_times)


def _walk_pages(path: str, params: dict[str, Any]) -> Iterator[list[dict[str, Any]]]:
    page, next_url = _call(github.fetch_page, path, {"per_page": github.PAGE_SIZE, **params})
    yield page
    while next_url:
        page, next_url = _call(github.fetch_page, next_url)
        yield page


def _call(fn: Callable[..., T], *args: Any) -> T:
    failures = 0
    while True:
        try:
            return fn(*args)
        except GitHubAPIError as exc:
            if exc.retry_after is not None:
                wait = exc.retry_after + RATE_LIMIT_PADDING_SECONDS
                _log(f"rate limited, sleeping {wait / 60:.1f} min: {exc.message}")
                time.sleep(wait)
                continue
            if exc.status in (404, 410, 422):
                raise SkipItem(exc.message) from exc
            if exc.status is None or exc.status >= 500:
                failures += 1
                if failures > TRANSIENT_RETRIES:
                    raise
                wait = TRANSIENT_BACKOFF_SECONDS * failures
                _log(f"GitHub error, retry {failures}/{TRANSIENT_RETRIES} in {wait}s: {exc.message}")
                time.sleep(wait)
                continue
            raise


def _progress(label: str, total_prs: int) -> None:
    try:
        rate = github.get_rate_limit()
        reset = datetime.fromtimestamp(rate["reset"]).strftime("%H:%M")
        limit_text = f"rate limit {rate['remaining']}/{rate['limit']}, resets {reset}"
    except GitHubAPIError:
        limit_text = "rate limit unavailable"
    _log(f"{label} | {total_prs} PRs in raw_prs.jsonl | {limit_text}")


def _trim_file(file: dict[str, Any]) -> dict[str, Any]:
    # Drops "patch": features never read diff text, and it makes the dataset huge.
    keys = ("filename", "status", "additions", "deletions", "changes", "previous_filename")
    return {k: file[k] for k in keys if k in file}


def _trim_pr_commit(commit: dict[str, Any]) -> dict[str, Any]:
    # Keeps the nested shape features.build_feature_vector reads.
    inner = commit.get("commit") or {}
    return {
        "sha": commit.get("sha"),
        "commit": {
            "message": inner.get("message"),
            "author": {"date": (inner.get("author") or {}).get("date")},
            "committer": {"date": (inner.get("committer") or {}).get("date")},
        },
        "author": {"login": (commit.get("author") or {}).get("login")},
        "parents": [{"sha": p.get("sha")} for p in commit.get("parents") or []],
    }


def _trim_branch_commit(commit: dict[str, Any]) -> dict[str, Any]:
    inner = commit.get("commit") or {}
    return {
        "sha": commit["sha"],
        "message": inner.get("message") or "",
        # Committer date is when the change landed on the branch.
        "date": (inner.get("committer") or {}).get("date") or (inner.get("author") or {}).get("date"),
        "author": (commit.get("author") or {}).get("login"),
        "parents": [p.get("sha") for p in commit.get("parents") or []],
    }


def _load_or_create_window(path: Path, window_days: int) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    now = datetime.now(timezone.utc).replace(microsecond=0)
    window = {
        "created_at": _iso(now),
        "since": _iso(now - timedelta(days=window_days)),
        "until": _iso(now - timedelta(days=LABEL_MATURITY_DAYS)),
        "window_days": window_days,
        "label_maturity_days": LABEL_MATURITY_DAYS,
    }
    _write_json_atomic(path, window)
    return window


def _load_pr_state(path: Path) -> tuple[set[int], dict[str, dict[str, str]]]:
    _repair_partial_last_line(path)
    ids: set[int] = set()
    merges: dict[str, dict[str, str]] = {}
    for record in _read_jsonl(path):
        pr = record["pr"]
        ids.add(pr["id"])
        if pr.get("merge_commit_sha"):
            merges.setdefault(record["repo"], {})[pr["merge_commit_sha"]] = pr.get("merged_at")
    return ids, merges


def _load_commit_keys(path: Path) -> set[tuple[str, str]]:
    _repair_partial_last_line(path)
    return {(record["repo"], record["sha"]) for record in _read_jsonl(path)}


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                _log(f"{path.name}: ignoring unreadable line {number}")


def _repair_partial_last_line(path: Path) -> None:
    """Drop a last line left half-written by a crash, so appends start cleanly."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return
        position = f.seek(0, os.SEEK_END)
        while position > 0:
            step = min(65536, position)
            position -= step
            f.seek(position)
            newline = f.read(step).rfind(b"\n")
            if newline != -1:
                f.truncate(position + newline + 1)
                _log(f"{path.name}: removed a partially written last line")
                return
        f.truncate(0)


def _cached(path: Path, build: Callable[[], T]) -> T:
    if path.exists():
        return json.loads(path.read_text())
    data = build()
    _write_json_atomic(path, data)
    return data


def _append(out: Any, record: dict[str, Any]) -> None:
    out.write(json.dumps(record, separators=(",", ":")) + "\n")
    out.flush()
    os.fsync(out.fileno())


def _write_json_atomic(path: Path, data: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data))
    os.replace(temporary, path)


def _evenly_spaced(items: list[T], limit: int) -> list[T]:
    if len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def _is_bot(user: Any) -> bool:
    return isinstance(user, dict) and (
        user.get("type") == "Bot" or str(user.get("login", "")).endswith("[bot]")
    )


def _parse(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def _now_iso() -> str:
    return _iso(datetime.now(timezone.utc).replace(microsecond=0))


def _log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repos", nargs="+", default=REPOS, help="owner/name list")
    parser.add_argument("--max-prs-per-repo", type=int, default=MAX_PRS_PER_REPO)
    parser.add_argument(
        "--window-days", type=int, default=WINDOW_DAYS,
        help="only used on the first run; later runs reuse cache/window.json",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
