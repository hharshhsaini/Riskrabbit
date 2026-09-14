from datetime import datetime, timedelta, timezone

from ml.label import label_repository

UTC = timezone.utc
MERGED = datetime(2025, 3, 3, 12, 0, tzinfo=UTC)


def iso(moment: datetime) -> str:
    return moment.isoformat().replace("+00:00", "Z")


def pr_record(number, title="Add cache layer", files=("src/cache.py",), merged=MERGED, sha=None, pr_id=None, author="alice"):
    return {
        "repo": "o/r",
        "pr": {
            "id": pr_id or 1000 + number,
            "number": number,
            "title": title,
            "user": {"login": author},
            "merged_at": iso(merged),
            "merge_commit_sha": sha or f"{number:040x}",
        },
        "files": [{"filename": f} for f in files],
        "commits": [],
    }


def commit(sha, message, when, files=None, author="alice"):
    return {"repo": "o/r", "sha": sha, "message": message, "date": iso(when), "files": files, "author": author}


def labels(prs, commits):
    labeled, _ = label_repository(prs, commits)
    return {r["pr"]["number"]: r for r in labeled}


def test_revert_by_merge_sha_in_body_labels_pr():
    pr = pr_record(12)
    revert = commit("a" * 40, f'Revert "Something"\n\nThis reverts commit {pr["pr"]["merge_commit_sha"]}.', MERGED + timedelta(days=30))

    result = labels([pr], [revert])[12]

    assert result["is_risky"] == 1
    assert result["label_reason"] == "reverted"
    assert result["label_evidence"][0]["sha"] == "a" * 40


def test_revert_by_abbreviated_sha_labels_pr():
    pr = pr_record(12)
    revert = commit("a" * 40, f"Undo change\n\nThis reverts commit {pr['pr']['merge_commit_sha'][:10]}.", MERGED + timedelta(days=2))

    assert labels([pr], [revert])[12]["label_reason"] == "reverted"


def test_revert_subject_uses_pr_number_not_just_title():
    target = pr_record(12, title="Fix typo")
    same_title = pr_record(13, title="Fix typo")
    revert = commit("a" * 40, 'Revert "Fix typo (#13)"', MERGED + timedelta(days=1))

    result = labels([target, same_title], [revert])

    assert result[12]["is_risky"] == 0
    assert result[13]["label_reason"] == "reverted"


def test_revert_by_unique_title_labels_pr_but_ambiguous_title_does_not():
    unique = pr_record(12, title="Add cache layer")
    dup_a = pr_record(20, title="Bump version", files=("setup.py",))
    dup_b = pr_record(21, title="Bump version", files=("setup.py",))
    commits = [
        commit("a" * 40, 'Revert "Add cache layer"', MERGED + timedelta(days=3)),
        commit("b" * 40, 'Revert "Bump version"', MERGED + timedelta(days=3)),
    ]

    result = labels([unique, dup_a, dup_b], commits)

    assert result[12]["label_reason"] == "reverted"
    assert result[20]["is_risky"] == 0
    assert result[21]["is_risky"] == 0


def test_merge_commit_style_revert_matches_pr_number():
    pr = pr_record(45)
    revert = commit("a" * 40, 'Revert "Merge pull request #45 from alice/cache"', MERGED + timedelta(days=4))

    assert labels([pr], [revert])[45]["label_reason"] == "reverted"


def test_revert_of_a_revert_does_not_label_original_pr():
    pr = pr_record(12, title="Add cache layer")
    reapply = commit("a" * 40, 'Revert "Revert "Add cache layer (#12)""\n\nThis reverts commit ' + "c" * 40 + ".", MERGED + timedelta(days=5))

    assert labels([pr], [reapply])[12]["is_risky"] == 0


def test_revert_dated_before_merge_is_ignored():
    pr = pr_record(12)
    early = commit("a" * 40, 'Revert "Add cache layer (#12)"', MERGED - timedelta(days=1))

    assert labels([pr], [early])[12]["is_risky"] == 0


def test_plain_sha_mention_is_not_a_revert():
    pr = pr_record(12)
    mention = commit("a" * 40, f"Follow-up to {pr['pr']['merge_commit_sha']}", MERGED + timedelta(days=1), files=["docs/x.md"])

    assert labels([pr], [mention])[12]["is_risky"] == 0


def test_hotfix_within_window_touching_same_file_labels_pr():
    pr = pr_record(12, files=("src/cache.py", "tests/test_cache.py"))
    fix = commit("a" * 40, "Fix cache eviction crash", MERGED + timedelta(days=2), files=["src/cache.py"])

    result = labels([pr], [fix])[12]

    assert result["label_reason"] == "hotfix"
    assert result["label_evidence"][0]["shared_files"] == ["src/cache.py"]


def test_hotfix_after_seven_days_does_not_count():
    pr = pr_record(12)
    late = commit("a" * 40, "Fix cache bug", MERGED + timedelta(days=7, minutes=1), files=["src/cache.py"])

    assert labels([pr], [late])[12]["is_risky"] == 0


def test_fix_commit_touching_other_files_does_not_count():
    pr = pr_record(12)
    unrelated = commit("a" * 40, "Fix login bug", MERGED + timedelta(days=1), files=["src/auth.py"])

    assert labels([pr], [unrelated])[12]["is_risky"] == 0


def test_shared_changelog_or_lock_file_alone_does_not_count():
    pr = pr_record(12, files=("src/cache.py", "CHANGELOG.md", "uv.lock", "changelog/12.feature.rst"))
    fix = commit("a" * 40, "Fix docs bug", MERGED + timedelta(days=1), files=["CHANGELOG.md", "uv.lock", "changelog/12.feature.rst"])

    labeled, stats = label_repository([pr], [fix])

    assert labeled[0]["is_risky"] == 0
    assert stats["overlap_only_in_tests_docs_changelog_or_hub_files"] == 1


def test_non_fix_subject_touching_same_file_does_not_count():
    pr = pr_record(12)
    refactor = commit("a" * 40, "Refactor cache internals\n\n* fix lint\n* fix typo", MERGED + timedelta(days=1), files=["src/cache.py"])

    assert labels([pr], [refactor])[12]["is_risky"] == 0


def test_words_containing_fix_or_bug_are_not_hotfixes():
    pr = pr_record(12)
    commits = [
        commit("a" * 40, "Add prefix option", MERGED + timedelta(days=1), files=["src/cache.py"]),
        commit("b" * 40, "Improve debugger output", MERGED + timedelta(days=1), files=["src/cache.py"]),
    ]

    assert labels([pr], commits)[12]["is_risky"] == 0


def test_commits_landing_with_the_merge_are_not_hotfixes():
    pr = pr_record(12)
    rebase_landed = commit("a" * 40, "Fix tests", MERGED + timedelta(seconds=30), files=["src/cache.py"])
    own_merge = commit(pr["pr"]["merge_commit_sha"], "Fix cache bug (#12)", MERGED + timedelta(hours=1), files=["src/cache.py"])

    assert labels([pr], [rebase_landed, own_merge])[12]["is_risky"] == 0


def test_fix_commit_without_file_list_uses_its_merged_pr_files():
    risky = pr_record(12, files=("src/cache.py",))
    fix_pr = pr_record(13, title="Fix cache regression", files=("src/cache.py",), merged=MERGED + timedelta(days=2))
    fix_commit = commit(fix_pr["pr"]["merge_commit_sha"], "Fix cache regression (#13)", MERGED + timedelta(days=2), files=None)

    result = labels([risky, fix_pr], [fix_commit])

    assert result[12]["label_reason"] == "hotfix"
    assert result[13]["is_risky"] == 0


def test_fix_commit_with_unknown_files_is_counted_not_guessed():
    pr = pr_record(12)
    unknown = commit("a" * 40, "Fix crash", MERGED + timedelta(days=1), files=None)

    labeled, stats = label_repository([pr], [unknown])

    assert labeled[0]["is_risky"] == 0
    assert stats["fix_commits_without_file_list"] == 1


def test_both_signals_recorded_in_order():
    pr = pr_record(12)
    commits = [
        commit("a" * 40, "Fix cache crash", MERGED + timedelta(days=1), files=["src/cache.py"]),
        commit("b" * 40, 'Revert "Add cache layer (#12)"', MERGED + timedelta(days=3)),
    ]

    result = labels([pr], commits)[12]

    assert result["label_reason"] == "reverted+hotfix"
    assert type(result["is_risky"]) is int


def test_clean_pr_is_labelled_zero_with_reason_none():
    result = labels([pr_record(12)], [])[12]

    assert result["is_risky"] == 0
    assert result["label_reason"] == "none"
    assert result["label_evidence"] == []


def test_squash_merged_revert_subject_without_sha_labels_pr():
    pr = pr_record(12)
    revert = commit("a" * 40, 'Revert "Add cache layer (#12)" (#15)', MERGED + timedelta(days=2))

    assert labels([pr], [revert])[12]["label_reason"] == "reverted"


def test_truncated_revert_title_matches_unique_long_prefix_only():
    long_title = pr_record(12, title="Fixes #900: Prevent duplicate cable paths when terminations change")
    short_title = pr_record(13, title="Fix cache eviction")
    commits = [
        commit("a" * 40, 'Revert "Fixes #900: Prevent duplicate cable paths when …" (#20)', MERGED + timedelta(days=2)),
        commit("b" * 40, 'Revert "Fix cache …" (#21)', MERGED + timedelta(days=2)),
    ]

    result = labels([long_title, short_title], commits)

    assert result[12]["label_reason"] == "reverted"
    assert result[13]["is_risky"] == 0


def test_patch_in_subject_is_not_a_hotfix():
    pr = pr_record(12)
    mock = commit("a" * 40, "test: Patch time in cache tests", MERGED + timedelta(days=1), files=["src/cache.py"])

    assert labels([pr], [mock])[12]["is_risky"] == 0


def test_overlap_only_in_tests_or_docs_is_not_a_hotfix():
    pr = pr_record(12, files=("src/cache.py", "tests/test_cache.py", "docs/cache.rst", "test-data/unit/cache.test"))
    fix = commit("a" * 40, "Fix flaky test", MERGED + timedelta(days=1), files=["tests/test_cache.py", "docs/cache.rst", "test-data/unit/cache.test"])

    assert labels([pr], [fix])[12]["is_risky"] == 0


def test_sweeping_fix_commit_is_not_a_hotfix():
    pr = pr_record(12)
    sweep = commit("a" * 40, "Fix unused imports", MERGED + timedelta(days=1), files=["src/cache.py"] + [f"src/m{i}.py" for i in range(25)])

    labeled, stats = label_repository([pr], [sweep])

    assert labeled[0]["is_risky"] == 0
    assert stats["fix_touching_over_20_files"] == 1


def test_overlap_in_hub_file_is_ignored_but_other_source_file_counts():
    background = [pr_record(100 + i, title=f"Change {i}", files=("src/core.py",)) for i in range(20)]
    target = pr_record(12, files=("src/core.py", "src/cache.py"))
    hub_only = commit("a" * 40, "Fix core crash", MERGED + timedelta(days=1), files=["src/core.py"])

    assert labels(background + [target], [hub_only])[12]["is_risky"] == 0

    real = commit("b" * 40, "Fix cache crash", MERGED + timedelta(days=1), files=["src/core.py", "src/cache.py"])
    result = labels(background + [target], [real])[12]
    assert result["label_reason"] == "hotfix"
    assert result["label_evidence"][0]["shared_files"] == ["src/cache.py"]


def test_fix_pr_opened_before_this_pr_is_not_its_hotfix():
    pr = pr_record(50)
    older = commit("a" * 40, "Fix cache crash (#41)", MERGED + timedelta(days=1), files=["src/cache.py"])
    newer = commit("b" * 40, "Fix cache crash (#57)", MERGED + timedelta(days=1), files=["src/cache.py"])

    labeled, stats = label_repository([pr], [older])
    assert labeled[0]["is_risky"] == 0
    assert stats["fix_opened_before_this_pr"] == 1

    assert labels([pr], [newer])[50]["label_reason"] == "hotfix"


def test_fix_by_another_author_without_link_is_not_a_hotfix():
    pr = pr_record(12, author="alice")
    coincidence = commit("a" * 40, "Fix unrelated crash (#20)", MERGED + timedelta(days=2), files=["src/cache.py"], author="bob")

    labeled, stats = label_repository([pr], [coincidence])

    assert labeled[0]["is_risky"] == 0
    assert stats["fix_by_other_author_without_link"] == 1


def test_same_author_follow_up_fix_is_a_hotfix():
    pr = pr_record(12, author="alice")
    follow_up = commit("a" * 40, "Fix cache key collision (#20)", MERGED + timedelta(hours=3), files=["src/cache.py"], author="alice")

    result = labels([pr], [follow_up])[12]

    assert result["label_reason"] == "hotfix"
    assert result["label_evidence"][0]["linked_by"] == ["same_author"]


def test_other_author_fix_counts_when_it_cites_the_pr_or_says_regression():
    pr = pr_record(12, author="alice")
    cites = commit("a" * 40, "Fix cache crash (#20)\n\nBroken by #12.", MERGED + timedelta(days=1), files=["src/cache.py"], author="bob")
    regression = commit("b" * 40, "Fix regression in cache warmup (#21)", MERGED + timedelta(days=1), files=["src/cache.py"], author="carol")

    assert labels([pr], [cites])[12]["label_evidence"][0]["linked_by"] == ["cites_pr"]
    assert labels([pr], [regression])[12]["label_evidence"][0]["linked_by"] == ["regression"]


def test_editor_config_directories_are_noise():
    pr = pr_record(12, files=("src/cache.py", ".claude/settings.json", ".vscode/settings.json"))
    fix = commit("a" * 40, "Fix editor settings", MERGED + timedelta(days=1), files=[".claude/settings.json", ".vscode/settings.json"])

    assert labels([pr], [fix])[12]["is_risky"] == 0


def test_hotfix_window_can_be_widened_for_experiments():
    pr = pr_record(12)
    later_fix = commit("a" * 40, "Fix cache crash", MERGED + timedelta(days=10), files=["src/cache.py"])

    assert labels([pr], [later_fix])[12]["is_risky"] == 0
    widened, _ = label_repository([pr], [later_fix], hotfix_window_days=14)
    assert widened[0]["label_reason"] == "hotfix"


def test_signals_can_be_restricted_to_reverts_only():
    pr = pr_record(12)
    commits = [
        commit("a" * 40, "Fix cache crash", MERGED + timedelta(days=1), files=["src/cache.py"]),
        commit("b" * 40, 'Revert "Add cache layer (#12)"', MERGED + timedelta(days=3)),
    ]
    hotfix_only_pr = pr_record(13, title="Other", files=("src/other.py",))
    fix_other = commit("c" * 40, "Fix other crash", MERGED + timedelta(days=1), files=["src/other.py"])

    labeled, _ = label_repository([pr, hotfix_only_pr], commits + [fix_other], signals=("reverted",))
    result = {r["pr"]["number"]: r for r in labeled}

    assert result[12]["label_reason"] == "reverted"
    assert result[13]["is_risky"] == 0


def test_backport_of_this_pr_is_not_its_hotfix():
    pr = pr_record(12, title="Fix file uploads failing on redirects")
    backport = commit("a" * 40, "[PR #12/16703bb9 backport][3.13] Fix file uploads failing on redirects (#20)", MERGED + timedelta(hours=1), files=["src/cache.py"])

    labeled, stats = label_repository([pr], [backport])

    assert labeled[0]["is_risky"] == 0
    assert stats["backport_or_copy_of_a_change"] == 1


def test_cherry_pick_and_repeated_title_are_copies_not_fixes():
    pr = pr_record(12, title="Fix sendfile over-reading")
    commits = [
        commit("a" * 40, "Fix cache crash (#20)\n\n(cherry picked from commit abc1234)", MERGED + timedelta(days=1), files=["src/cache.py"]),
        commit("b" * 40, "Fix sendfile over-reading (#12) (#21)", MERGED + timedelta(days=1), files=["src/cache.py"]),
        commit("c" * 40, "[Backport maintenance/4.0.x] Fix other crash (#22)", MERGED + timedelta(days=1), files=["src/cache.py"]),
    ]

    labeled, stats = label_repository([pr], commits)

    assert labeled[0]["is_risky"] == 0
    assert stats["backport_or_copy_of_a_change"] == 3


def test_genuine_follow_up_fix_still_counts_after_copy_filter():
    pr = pr_record(12, title="Refactor WebSocket reader")
    follow_up = commit("a" * 40, "Fix WebSocket reader with fragmented messages (#20)", MERGED + timedelta(days=1), files=["src/cache.py"])

    assert labels([pr], [follow_up])[12]["label_reason"] == "hotfix"
