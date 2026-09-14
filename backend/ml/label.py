"""Derive a binary deployment-risk label for each collected pull request.

Run from backend/ after (or while) ml/collect.py runs:
    ./venv/bin/python -m ml.label
    ./venv/bin/python -m ml.label --inspect 20     # also print examples to check by hand

A PR is risky when a public signal suggests the change went wrong after merging:
  Signal A, reverted: a later default-branch commit reverts it.
  Signal B, hotfix: within 7 days of the merge, a commit whose subject says
    fix/hotfix/bug/regression touches a SOURCE file the PR also changed.
    Tightened after inspecting labelled examples (Step 10 gate). A fix commit
    does not count when:
      - the only shared files are tests, docs, changelogs or lock files;
      - the shared files are hub files touched by >=5% of the repo's PRs,
        where overlap happens by coincidence;
      - it touches more than 20 files (a repo-wide sweep such as a lint fix);
      - its PR number is lower than this PR's, i.e. it was opened before this
        PR existed and so cannot be fixing it;
      - it is by someone other than this PR's author AND neither cites this PR
        (#N) nor calls itself a regression fix. Hand-checking 8 examples of each
        kind: same-author follow-up fixes were mostly real hotfixes (~6 of 8),
        while other authors' fixes that merely touched a shared file were mostly
        coincidence (~2 of 8).
    "patch" is not a hotfix word here: in these repos it almost always means
    mock-patching or monkeypatching a library.
Signal C (bug issues opened after the merge) is not used. ml/collect.py does not
collect issues, so a bug report about the PR cannot be told apart from a bug the
PR closes, and counting the second as risk would teach the model the opposite of
the target.
"""

import argparse
import json
import os
import random
import re
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"

HOTFIX_WINDOW_DAYS = 7
# Commits landing this soon after the merge are the PR itself reaching the branch
# (a rebase merge re-commits every PR commit at merge time), not a later hotfix.
MERGE_LANDING_GRACE_SECONDS = 120
MAX_EVIDENCE_PER_SIGNAL = 5
SIGNAL_ORDER = ("reverted", "hotfix")

MAX_FIX_COMMIT_FILES = 20
HUB_FILE_SHARE = 0.05
HUB_FILE_MIN_PRS = 3
# GitHub cuts long squash-merge titles with an ellipsis; shorter prefixes are too
# ambiguous to identify a single PR.
MIN_TRUNCATED_TITLE_PREFIX = 20

HOTFIX_SUBJECT = re.compile(r"\b(fix|hotfix|bug|regression)\b", re.IGNORECASE)
REGRESSION_SUBJECT = re.compile(r"\bregression\b", re.IGNORECASE)
# The optional trailing "(#N)" is the revert PR's own number in squash merges:
#   Revert "Add cache layer (#12)" (#15)
REVERT_SUBJECT = re.compile(r'^Revert "(?P<title>.*)"(?:\s*\(#\d+\))?\s*$')
REVERTED_SHA = re.compile(r"\breverts? commit ([0-9a-f]{7,40})\b", re.IGNORECASE)
TRAILING_PR_NUMBER = re.compile(r"\s*\(#(\d+)\)\s*$")
MERGE_PR_SUBJECT = re.compile(r"^Merge pull request #(\d+)\b")

# Files most PRs touch. Sharing only these with a later fix commit is not evidence
# that the fix repaired this PR.
NOISE_FILENAMES = {
    "changelog", "changelog.md", "changelog.rst", "changelog.txt",
    "changes", "changes.md", "changes.rst", "changes.txt",
    "history.md", "history.rst", "news.md", "news.rst",
    "authors", "authors.md", "authors.rst", "contributors.md",
    "uv.lock", "poetry.lock", "pdm.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
}
NOISE_DIRECTORIES = {
    "changelog", "changelog.d", "changes", "news", "newsfragments", "release-notes",
    ".claude", ".cursor", ".idea", ".vscode",
}
# Overlap in tests or docs alone is not evidence the PR's code broke.
TEST_OR_DOCS_DIRECTORIES = {"test", "tests", "testing", "test-data", "test_data", "doc", "docs"}
DOCS_SUFFIXES = (".md", ".rst", ".txt")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    window = json.loads((args.data_dir / "cache" / "window.json").read_text())
    if HOTFIX_WINDOW_DAYS > window["label_maturity_days"]:
        raise SystemExit(
            f"HOTFIX_WINDOW_DAYS={HOTFIX_WINDOW_DAYS} exceeds the {window['label_maturity_days']} days "
            "of commit file lists ml/collect.py fetched. Re-run collection with a larger "
            "LABEL_MATURITY_DAYS first."
        )

    prs_by_repo = _group_by_repo(args.data_dir / "raw_prs.jsonl")
    commits_by_repo = _group_by_repo(args.data_dir / "raw_commits.jsonl")

    labeled: list[dict[str, Any]] = []
    stats: dict[str, Counter] = {}
    skipped: dict[str, int] = {}
    for repo in sorted(prs_by_repo):
        if not _commits_complete(args.data_dir, repo, len(commits_by_repo.get(repo, []))):
            skipped[repo] = len(prs_by_repo[repo])
            continue
        repo_labeled, stats[repo] = label_repository(prs_by_repo[repo], commits_by_repo[repo])
        labeled.extend(repo_labeled)

    output = args.data_dir / "labeled_prs.jsonl"
    _write_jsonl_atomic(output, labeled)
    _print_summary(labeled, stats, skipped, output)
    if args.inspect:
        _print_examples(labeled, args.inspect, args.seed)


def label_repository(
    prs: list[dict[str, Any]],
    commits: list[dict[str, Any]],
    hotfix_window_days: int = HOTFIX_WINDOW_DAYS,
    signals: Sequence[str] = SIGNAL_ORDER,
) -> tuple[list[dict[str, Any]], Counter]:
    """Label one repository's PRs using that repository's default-branch commits.

    hotfix_window_days and signals exist for the Step 13b experiments; the defaults
    are the labelling rules used for training.
    """
    stats: Counter = Counter()
    timeline = sorted(
        ((when, commit) for commit in commits if (when := _parse(commit.get("date")))),
        key=lambda item: item[0],
    )
    dates = [when for when, _ in timeline]

    by_merge_sha = {r["pr"]["merge_commit_sha"]: r for r in prs if r["pr"].get("merge_commit_sha")}
    by_number = {r["pr"]["number"]: r for r in prs}
    by_title: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in prs:
        by_title[_normalize_title(record["pr"].get("title") or "")].append(record)

    path_counts = Counter(p for record in prs for p in _pr_paths(record))
    hub_threshold = max(HUB_FILE_MIN_PRS, HUB_FILE_SHARE * len(prs))
    hub_files = {p for p, count in path_counts.items() if count >= hub_threshold}

    evidence: dict[int, list[dict[str, Any]]] = defaultdict(list)

    if "reverted" in signals:
        for when, commit in timeline:
            for record in _reverted_prs(commit, by_merge_sha, by_number, by_title):
                merged = _parse(record["pr"].get("merged_at"))
                if merged and when > merged:
                    evidence[record["pr"]["id"]].append(_evidence("reverted", commit))

    grace = timedelta(seconds=MERGE_LANDING_GRACE_SECONDS)
    window = timedelta(days=hotfix_window_days)
    for record in prs if "hotfix" in signals else []:
        pr = record["pr"]
        merged = _parse(pr.get("merged_at"))
        if merged is None:
            continue
        pr_paths = _pr_paths(record)
        meaningful = {
            p for p in pr_paths
            if not _is_noise_path(p) and not _is_test_or_docs_path(p) and p not in hub_files
        }

        start = bisect_right(dates, merged + grace)
        end = bisect_right(dates, merged + window)
        for _, commit in timeline[start:end]:
            if commit["sha"] == pr.get("merge_commit_sha"):
                continue
            subject = _subject(commit.get("message"))
            if not HOTFIX_SUBJECT.search(subject):
                continue
            files = _commit_files(commit, by_merge_sha)
            if files is None:
                stats["fix_commits_without_file_list"] += 1
                continue
            if not pr_paths & files:
                continue
            fix_number = _pr_number_in_subject(subject)
            if fix_number is not None and fix_number <= pr["number"]:
                stats["fix_opened_before_this_pr"] += 1
                continue
            if len(files) > MAX_FIX_COMMIT_FILES:
                stats["fix_touching_over_20_files"] += 1
                continue
            shared = meaningful & files
            if not shared:
                stats["overlap_only_in_tests_docs_changelog_or_hub_files"] += 1
                continue
            links = _links_to_pr(commit, subject, pr)
            if not links:
                stats["fix_by_other_author_without_link"] += 1
                continue
            evidence[pr["id"]].append(
                _evidence("hotfix", commit, shared_files=sorted(shared)[:10], linked_by=links)
            )

    labeled = []
    for record in prs:
        found = evidence.get(record["pr"]["id"], [])
        signals = [s for s in SIGNAL_ORDER if any(e["signal"] == s for e in found)]
        kept = [e for s in signals for e in [x for x in found if x["signal"] == s][:MAX_EVIDENCE_PER_SIGNAL]]
        labeled.append(
            {
                **record,
                "is_risky": 1 if signals else 0,
                "label_reason": "+".join(signals) if signals else "none",
                "label_evidence": kept,
            }
        )
    return labeled, stats


def _reverted_prs(
    commit: dict[str, Any],
    by_merge_sha: dict[str, dict[str, Any]],
    by_number: dict[int, dict[str, Any]],
    by_title: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    message = commit.get("message") or ""
    found: dict[int, dict[str, Any]] = {}

    # "This reverts commit <sha>." A plain mention of a sha is not a revert.
    for short_sha in REVERTED_SHA.findall(message):
        short_sha = short_sha.lower()
        for merge_sha, record in by_merge_sha.items():
            if merge_sha.startswith(short_sha):
                found[record["pr"]["id"]] = record

    match = REVERT_SUBJECT.match(_subject(message))
    if match and not match["title"].startswith(('Revert "', "Reapply")):
        quoted = match["title"]
        number = TRAILING_PR_NUMBER.search(quoted) or MERGE_PR_SUBJECT.match(quoted)
        if number:
            record = by_number.get(int(number.group(1)))
            if record:
                found[record["pr"]["id"]] = record
        else:
            normalized = _normalize_title(quoted)
            if normalized.endswith("…"):
                prefix = normalized.rstrip("… ")
                candidates = [
                    r for title, records in by_title.items()
                    if len(prefix) >= MIN_TRUNCATED_TITLE_PREFIX and title.startswith(prefix)
                    for r in records
                ]
            else:
                candidates = by_title.get(normalized, [])
            if len(candidates) == 1:  # an ambiguous title identifies nothing
                found[candidates[0]["pr"]["id"]] = candidates[0]

    return list(found.values())


def _commit_files(
    commit: dict[str, Any], by_merge_sha: dict[str, dict[str, Any]]
) -> set[str] | None:
    if isinstance(commit.get("files"), list):
        return set(commit["files"])
    merged_pr = by_merge_sha.get(commit["sha"])
    if merged_pr is not None:
        return _pr_paths(merged_pr)
    return None


def _pr_paths(record: dict[str, Any]) -> set[str]:
    paths: set[str] = set()
    for file in record.get("files") or []:
        for key in ("filename", "previous_filename"):
            if file.get(key):
                paths.add(file[key])
    return paths


def _links_to_pr(commit: dict[str, Any], subject: str, pr: dict[str, Any]) -> list[str]:
    """Reasons to believe a fix commit responds to this PR rather than coincides with it."""
    links = []
    pr_author = (pr.get("user") or {}).get("login")
    if pr_author and commit.get("author") == pr_author:
        links.append("same_author")
    if re.search(rf"#{pr['number']}\b", commit.get("message") or ""):
        links.append("cites_pr")
    if REGRESSION_SUBJECT.search(subject):
        links.append("regression")
    return links


def _pr_number_in_subject(subject: str) -> int | None:
    number = TRAILING_PR_NUMBER.search(subject) or MERGE_PR_SUBJECT.match(subject)
    return int(number.group(1)) if number else None


def _is_test_or_docs_path(path: str) -> bool:
    p = PurePosixPath(path)
    name = p.name.casefold()
    return (
        any(part.casefold() in TEST_OR_DOCS_DIRECTORIES for part in p.parts[:-1])
        or name.startswith("test_")
        or p.stem.casefold().endswith("_test")
        or ".test." in name
        or ".spec." in name
        or name == "conftest.py"
        or name.endswith(".test")
        or name.endswith(DOCS_SUFFIXES)
    )


def _is_noise_path(path: str) -> bool:
    p = PurePosixPath(path)
    return (
        p.name.casefold() in NOISE_FILENAMES
        or any(part.casefold() in NOISE_DIRECTORIES for part in p.parts[:-1])
    )


def _evidence(signal: str, commit: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "signal": signal,
        "sha": commit["sha"],
        "subject": _subject(commit.get("message")),
        "date": commit.get("date"),
        **extra,
    }


def _subject(message: str | None) -> str:
    return (message or "").split("\n", 1)[0].strip()


def _normalize_title(title: str) -> str:
    title = TRAILING_PR_NUMBER.sub("", title)
    return " ".join(title.casefold().split()).rstrip(".")


def _parse(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _group_by_repo(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return grouped
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # the collector may be mid-write on the last line
            grouped[record["repo"]].append(record)
    return grouped


def _commits_complete(data_dir: Path, repo: str, collected: int) -> bool:
    listing = data_dir / "cache" / f"{repo.replace('/', '__')}__commits.json"
    return listing.exists() and collected == len(json.loads(listing.read_text()))


def _write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _print_summary(
    labeled: list[dict[str, Any]],
    stats: dict[str, Counter],
    skipped: dict[str, int],
    output: Path,
) -> None:
    total = len(labeled)
    positives = sum(r["is_risky"] for r in labeled)
    rate = positives / total * 100 if total else 0.0
    reasons = Counter(r["label_reason"] for r in labeled if r["is_risky"])

    print(f"\nLabelled {total} PRs from {len(stats)} repositories -> {output.name}")
    for repo, count in skipped.items():
        print(f"  skipped {repo} ({count} PRs): commits not fully collected yet")
    print(f"\nPositive (is_risky=1): {positives} of {total} = {rate:.1f}%")
    print("By signal:")
    print(f"  reverted only     {reasons.get('reverted', 0)}")
    print(f"  hotfix only       {reasons.get('hotfix', 0)}")
    print(f"  reverted + hotfix {reasons.get('reverted+hotfix', 0)}")
    print("  Signal C          not used (no issue data collected)")

    print(f"\n{'repository':<28}{'PRs':>6}{'risky':>7}{'rate':>8}{'reverted':>10}{'hotfix':>8}")
    by_repo: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in labeled:
        by_repo[record["repo"]].append(record)
    for repo, records in sorted(by_repo.items()):
        risky = sum(r["is_risky"] for r in records)
        reverted = sum("reverted" in r["label_reason"] for r in records)
        hotfix = sum("hotfix" in r["label_reason"] for r in records)
        print(f"{repo:<28}{len(records):>6}{risky:>7}{risky / len(records) * 100:>7.1f}%{reverted:>10}{hotfix:>8}")

    totals = sum(stats.values(), Counter())
    print("\nHotfix diagnostics:")
    print(f"  no file list available:                           {totals['fix_commits_without_file_list']}")
    print(f"  fix PR opened before this PR (cannot be its fix):   {totals['fix_opened_before_this_pr']}")
    print(f"  fix touches >{MAX_FIX_COMMIT_FILES} files (repo-wide sweep):          {totals['fix_touching_over_20_files']}")
    print(f"  overlap only in tests/docs/changelog/hub files:     {totals['overlap_only_in_tests_docs_changelog_or_hub_files']}")
    print(f"  fix by another author with no link to this PR:      {totals['fix_by_other_author_without_link']}")

    if total == 0:
        verdict = "no PRs labelled yet"
    elif rate < 2 or rate > 40:
        verdict = "HEURISTIC LOOKS BROKEN: inspect 20 labelled examples by hand before continuing"
    elif 5 <= rate <= 15:
        verdict = "within the 5-15% target"
    else:
        verdict = "outside the 5-15% target but not broken: inspect examples before continuing"
    print(f"\nGate: {verdict}")


def _print_examples(labeled: list[dict[str, Any]], count: int, seed: int) -> None:
    rng = random.Random(seed)
    positives = [r for r in labeled if r["is_risky"]]
    negatives = [r for r in labeled if not r["is_risky"]]
    picks = rng.sample(positives, min(len(positives), count // 2))
    picks += rng.sample(negatives, min(len(negatives), count - len(picks)))

    print(f"\n{len(picks)} examples to check by hand (seed {seed}):")
    for record in picks:
        pr = record["pr"]
        print(f"\n[{record['label_reason'].upper()}] {record['repo']}#{pr['number']}: {pr.get('title')}")
        print(f"  {pr.get('html_url')}  merged {pr.get('merged_at')}")
        for item in record["label_evidence"]:
            shared = f"  shared: {', '.join(item['shared_files'])}" if item.get("shared_files") else ""
            linked = f"  [{', '.join(item['linked_by'])}]" if item.get("linked_by") else ""
            print(f"  -> {item['signal']}: {item['sha'][:10]} {item['date']} \"{item['subject']}\"{shared}{linked}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--inspect", type=int, default=0, help="print this many examples to check by hand")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
