"""Audit the labelled dataset before training, and save its feature matrix.

Run from backend/ after ml/label.py:
    ./venv/bin/python -m ml.audit

Writes ml/data/dataset.csv: one row per PR with identifying columns, the label,
and the features in FEATURE_ORDER. Features come from features.build_feature_vector,
the same function the live API uses, so training cannot compute them differently.
CSV rather than parquet: pandas needs pyarrow for parquet, which is not in the
project's dependency list.
"""

import argparse
import json
import random
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from features import FEATURE_ORDER, build_feature_vector

DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_FILENAME = "dataset.csv"
ID_COLUMNS = ["repo", "pr_number", "pr_id", "title", "author", "merged_at", "html_url", "label_reason"]
LABEL_COLUMN = "is_risky"

REPO_DOMINANCE_SHARE = 0.5
AUTHOR_MIN_PRS = 10
TOP_AUTHORS_SHOWN = 10
# A single feature whose ROC-AUC is at least this far from 0.5 visibly separates the classes.
SEPARATING_AUC_MARGIN = 0.05
NEAR_CONSTANT_SHARE = 0.99
EXAMPLES_PER_CLASS = 5


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    records = _read_jsonl(args.data_dir / "labeled_prs.jsonl")
    if not records:
        raise SystemExit("labeled_prs.jsonl is empty or missing. Run ml.label first.")

    dataset = build_dataset(records)
    output = args.data_dir / DATASET_FILENAME
    dataset.to_csv(output, index=False)

    print_audit(dataset, args.seed)
    print(f"\nSaved {len(dataset)} rows ({len(FEATURE_ORDER)} features) to {output}")


def build_dataset(records: list[dict[str, Any]]) -> pd.DataFrame:
    prior_counts = _prior_merged_counts(records)
    rows = []
    for record in records:
        pr = record["pr"]
        features = build_feature_vector(
            pr,
            record.get("files") or [],
            record.get("commits") or [],
            author_pr_count=prior_counts[pr["id"]],
            now=_parse(pr.get("merged_at")),
        )
        rows.append(
            {
                "repo": record["repo"],
                "pr_number": pr["number"],
                "pr_id": pr["id"],
                "title": pr.get("title"),
                "author": (pr.get("user") or {}).get("login"),
                "merged_at": pr.get("merged_at"),
                "html_url": pr.get("html_url"),
                "label_reason": record["label_reason"],
                LABEL_COLUMN: int(record["is_risky"]),
                **{name: features[name] for name in FEATURE_ORDER},
            }
        )
    return pd.DataFrame(rows, columns=ID_COLUMNS + [LABEL_COLUMN] + FEATURE_ORDER)


def _prior_merged_counts(records: list[dict[str, Any]]) -> dict[int, int]:
    """PROVISIONAL, see remaining.txt D0j.

    Counts the author's PRs in the same repository, within the collected sample,
    merged before this PR was opened. The live API counts from its own database
    instead, so the two definitions differ until D0j is decided.
    """
    merged_by_author: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for record in records:
        author = (record["pr"].get("user") or {}).get("login")
        merged = _parse(record["pr"].get("merged_at"))
        if author and merged:
            merged_by_author[(record["repo"], author)].append(merged)
    for times in merged_by_author.values():
        times.sort()

    counts = {}
    for record in records:
        pr = record["pr"]
        author = (pr.get("user") or {}).get("login")
        opened = _parse(pr.get("created_at"))
        times = merged_by_author.get((record["repo"], author), [])
        counts[pr["id"]] = bisect_left(times, opened) if author and opened else 0
    return counts


def print_audit(dataset: pd.DataFrame, seed: int) -> None:
    y = dataset[LABEL_COLUMN]
    total, positives = len(dataset), int(y.sum())
    overall_rate = positives / total if total else 0.0

    _heading("1. Overview")
    print(f"rows {total} | positives {positives} | negatives {total - positives} | positive rate {overall_rate:.1%}")
    print(f"repositories {dataset['repo'].nunique()} | authors {dataset['author'].nunique()}")
    missing = dataset[FEATURE_ORDER].isna().sum()
    print(f"missing feature values: {int(missing.sum())}")

    _heading("2. Per-repository positive rate")
    print(f"{'repository':<28}{'PRs':>6}{'risky':>7}{'rate':>8}{'share of positives':>20}")
    for repo, group in dataset.groupby("repo"):
        risky = int(group[LABEL_COLUMN].sum())
        share = risky / positives if positives else 0.0
        flags = []
        if share > REPO_DOMINANCE_SHARE:
            flags.append("DOMINATES POSITIVES")
        if overall_rate and (risky / len(group) > 2 * overall_rate or risky / len(group) < overall_rate / 2):
            flags.append("rate far from overall")
        print(f"{repo:<28}{len(group):>6}{risky:>7}{risky / len(group):>8.1%}{share:>20.1%}  {', '.join(flags)}")

    _heading("3. Feature means by class (sorted by how well each feature separates the classes alone)")
    print(
        "Means are pulled around by a few very large PRs; medians and ROC-AUC describe the typical PR.\n"
        "ROC-AUC is for that feature alone: 0.5 = no signal, further from 0.5 = stronger."
    )
    print(
        f"{'feature':<22}{'mean risky':>11}{'mean safe':>11}{'ratio':>7}"
        f"{'median risky':>14}{'median safe':>13}{'ROC-AUC':>9}  signal"
    )
    rows = []
    for name in FEATURE_ORDER:
        column = dataset[name]
        risky, safe = column[y == 1], column[y == 0]
        mean_pos = risky.mean() if positives else float("nan")
        mean_neg = safe.mean() if total - positives else float("nan")
        ratio = mean_pos / mean_neg if mean_neg else float("inf") if mean_pos else float("nan")
        auc = roc_auc_score(y, column) if 0 < positives < total and column.nunique() > 1 else float("nan")
        rows.append((name, mean_pos, mean_neg, ratio, risky.median(), safe.median(), auc))
    separating = []
    for name, mean_pos, mean_neg, ratio, median_pos, median_neg, auc in sorted(
        rows, key=lambda r: -abs(r[6] - 0.5) if pd.notna(r[6]) else 1
    ):
        signal = ""
        if pd.notna(auc) and abs(auc - 0.5) >= SEPARATING_AUC_MARGIN:
            signal = "higher when risky" if auc > 0.5 else "lower when risky"
            separating.append(name)
        note = {
            "author_pr_count": "  (provisional D0j; label bias D0o)",
            "review_comment_count": "  (may include post-merge comments)",
        }.get(name, "")
        print(
            f"{name:<22}{mean_pos:>11.2f}{mean_neg:>11.2f}{ratio:>7.2f}"
            f"{median_pos:>14.2f}{median_neg:>13.2f}{auc:>9.3f}  {signal}{note}"
        )

    _heading("4. Constant and near-constant features")
    constant = [n for n in FEATURE_ORDER if dataset[n].nunique() <= 1]
    near = [
        (n, dataset[n].value_counts(normalize=True).iloc[0])
        for n in FEATURE_ORDER
        if n not in constant and dataset[n].value_counts(normalize=True).iloc[0] >= NEAR_CONSTANT_SHARE
    ]
    print(f"constant (dead weight, remove): {', '.join(constant) or 'none'}")
    print(
        "near-constant (one value in >=99% of rows): "
        + (", ".join(f"{n} ({share:.1%})" for n, share in near) or "none")
    )

    _heading("5. Authors (label bias check, remaining.txt D0o)")
    authors = dataset.groupby("author")[LABEL_COLUMN].agg(["count", "sum"])
    busy = authors[authors["count"] >= AUTHOR_MIN_PRS].sort_values("count", ascending=False)
    print(f"{'author':<26}{'PRs':>6}{'risky':>7}{'rate':>8}")
    for author, row in busy.head(TOP_AUTHORS_SHOWN).iterrows():
        print(f"{str(author):<26}{int(row['count']):>6}{int(row['sum']):>7}{row['sum'] / row['count']:>8.1%}")
    top5 = authors.sort_values("sum", ascending=False).head(5)
    top5_share = top5["sum"].sum() / positives if positives else 0.0
    print(f"share of all positives from the 5 authors with most risky PRs: {top5_share:.1%}")

    _heading(f"6. Random examples (seed {seed})")
    rng = random.Random(seed)
    for label, name in ((1, "RISKY"), (0, "SAFE")):
        subset = dataset[y == label]
        for index in rng.sample(list(subset.index), min(EXAMPLES_PER_CLASS, len(subset))):
            row = subset.loc[index]
            print(f"[{name}] {row['repo']}#{row['pr_number']}  reason={row['label_reason']}  {row['title']}")
            print(f"         {row['html_url']}")

    _heading("Gate")
    if len(separating) >= 3:
        print(f"PASS: {len(separating)} features separate the classes: {', '.join(separating)}")
    else:
        print(
            f"FAIL: only {len(separating)} feature(s) separate the classes. The labels may carry "
            "no signal. Go to Step 13b before training."
        )


def _heading(title: str) -> None:
    print(f"\n== {title} ==")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def _parse(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
