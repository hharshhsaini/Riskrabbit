"""Step 16: check that LLM explanations stay faithful to the numbers they were given.

Run from backend/ (calls the configured LLM, about 10 requests):
    ./venv/bin/python -m ml.check_explanations

Runs 10 varied feature vectors through services.explain.generate_explanation and
prints each vector next to its explanation: 7 real PRs spread across the model's
score range, plus 3 hand-made edge cases. Each explanation gets three automatic
checks — numbers that were not in its prompt, alarming wording for its label, and
sentence count or markdown. The checks only point at problems; read every
explanation before signing off.
"""

import logging
import re
import textwrap
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from features import FEATURE_ORDER, build_feature_vector
from services import explain, model

DATASET_PATH = Path(__file__).resolve().parent / "data" / "dataset.csv"
# Score quantiles for the real PRs, so Low, Medium and High are all represented.
SCORE_QUANTILES = (0.02, 0.2, 0.4, 0.6, 0.8, 0.95, 0.995)

NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
MARKDOWN_LINE = re.compile(r"^\s*([-*•#]|\d+[.)])\s|\*\*")
# Wording that overstates risk. "elevated" is only alarming for a Low rating.
ALARMING_ALWAYS = ("dangerous", "danger", "critical", "severe", "alarming", "catastrophic",
                   "will fail", "certain to", "guaranteed", "extremely risky")
ALARMING_BELOW_HIGH = ("high risk", "highly risky", "very risky", "significant risk")
ALARMING_FOR_LOW = ("elevated", "risky")


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="  LOG %(levelname)s: %(message)s")
    cases = real_cases() + edge_cases()
    results = []
    for index, case in enumerate(cases, start=1):
        results.append(run_case(index, len(cases), case))
    print_summary(results)


def real_cases() -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise SystemExit(f"{DATASET_PATH} not found. Run ml.label and ml.audit first.")
    dataset = pd.read_csv(DATASET_PATH)
    dataset["score"] = model.classifier.predict_proba(dataset[FEATURE_ORDER])[:, 1]
    ordered = dataset.sort_values("score").reset_index(drop=True)
    cases = []
    for quantile in SCORE_QUANTILES:
        row = ordered.iloc[min(len(ordered) - 1, int(quantile * len(ordered)))]
        cases.append(
            {
                "name": f"real PR {row['repo']}#{row['pr_number']}",
                "detail": f"{row['title']}  ({row['html_url']}; labelled risky: {'yes' if row['is_risky'] else 'no'})",
                "features": {name: float(row[name]) for name in FEATURE_ORDER},
            }
        )
    return cases


def edge_cases() -> list[dict[str, Any]]:
    wednesday = datetime(2024, 1, 3, 12, 0, tzinfo=timezone.utc)
    saturday_night = datetime(2024, 1, 6, 2, 0, tzinfo=timezone.utc)
    iso = lambda moment: moment.isoformat().replace("+00:00", "Z")  # noqa: E731

    empty = build_feature_vector({}, [], [], now=wednesday)
    typo = build_feature_vector(
        {"changed_files": 1, "additions": 1, "deletions": 1, "commits": 1, "review_comments": 0,
         "created_at": iso(wednesday - timedelta(hours=2)), "merged_at": iso(wednesday)},
        [{"filename": "README.md", "additions": 1, "deletions": 1}],
        [{"commit": {"message": "Fix typo in README"}}],
        author_pr_count=30, now=wednesday,
    )
    files = [{"filename": f"src/core/module_{i}.py", "additions": 70, "deletions": 30} for i in range(117)]
    files += [
        {"filename": ".github/workflows/ci.yml", "additions": 40, "deletions": 20},
        {"filename": "Dockerfile", "additions": 15, "deletions": 5},
        {"filename": "requirements.txt", "additions": 5, "deletions": 5},
    ]
    huge = build_feature_vector(
        {"changed_files": 120, "additions": 8250, "deletions": 3540, "commits": 40, "review_comments": 25,
         "created_at": iso(saturday_night - timedelta(hours=400)), "merged_at": iso(saturday_night)},
        files, [{"commit": {"message": "wip"}}] * 40, author_pr_count=0, now=saturday_night,
    )
    return [
        {"name": "edge case: empty PR", "detail": "every feature is 0", "features": empty},
        {"name": "edge case: one-line README typo fix", "detail": "1 file, 2 lines, weekday", "features": typo},
        {"name": "edge case: huge weekend change to CI and config", "detail": "120 files, 11,790 lines, 40 commits, Saturday 02:00 UTC, first PR by the author", "features": huge},
    ]


def run_case(index: int, total: int, case: dict[str, Any]) -> dict[str, Any]:
    vector = case["features"]
    score, label = model.predict(vector)
    drivers = explain.top_drivers(vector)
    prompt = explain.build_prompt(drivers, score, label)

    started = time.perf_counter()
    text = explain.generate_explanation(vector, score, label)
    elapsed_ms = (time.perf_counter() - started) * 1000
    used_fallback = text == explain.fallback_explanation(drivers, score, label)

    checks = {
        "unsupported_numbers": numbers_not_in(text, prompt),
        "alarming_words": alarming_words(text, label),
        "sentences": count_sentences(text),
        "markdown": has_markdown(text),
    }

    print(f"\n{'=' * 100}\n[{index}/{total}] {case['name']} | score {score:.3f} -> {label}")
    print(f"  {case['detail']}")
    print("  full vector:")
    for line in textwrap.wrap("  ".join(f"{name}={_short(vector[name])}" for name in FEATURE_ORDER), 96):
        print(f"    {line}")
    print("  given to the LLM (top 5 measurements by effect on the score):")
    for row in drivers:
        print(f"    - {explain._describe(row)}")
    source = "FALLBACK (LLM call failed)" if used_fallback else "LLM"
    print(f"  explanation [{source}, {elapsed_ms:.0f} ms]:")
    for line in textwrap.wrap(text, 96):
        print(f"    {line}")
    print(f"  checks: numbers not in prompt: {', '.join(checks['unsupported_numbers']) or 'none'}"
          f" | alarming for {label}: {', '.join(checks['alarming_words']) or 'none'}"
          f" | sentences: {checks['sentences']} | markdown: {'YES' if checks['markdown'] else 'no'}")
    return {"case": case["name"], "score": score, "label": label, "fallback": used_fallback, **checks}


def numbers_not_in(text: str, prompt: str) -> list[str]:
    """Numbers in the explanation that do not appear anywhere in its prompt."""
    allowed = {_as_number(token) for token in NUMBER.findall(prompt)}
    return [token for token in NUMBER.findall(text) if _as_number(token) not in allowed]


def alarming_words(text: str, label: str) -> list[str]:
    lowered = text.casefold()
    phrases = list(ALARMING_ALWAYS)
    if label != "High":
        phrases += ALARMING_BELOW_HIGH
    if label == "Low":
        phrases += ALARMING_FOR_LOW
    found = [p for p in phrases if re.search(rf"\b{re.escape(p)}\b", lowered)]
    # "not dangerous" or "no sign of elevated risk" is fine; only count unnegated uses.
    # "less risky" and "not dangerous" say the opposite of alarming.
    negation = r"\b(not|no|nothing|never|neither|nor|without|isn't|isn’t|less)\b"
    return [p for p in found if not re.search(rf"{negation}[^.]{{0,40}}\b{re.escape(p)}\b", lowered)]


def count_sentences(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part])


def has_markdown(text: str) -> bool:
    return any(MARKDOWN_LINE.search(line) for line in text.splitlines())


def print_summary(results: list[dict[str, Any]]) -> None:
    print(f"\n{'=' * 100}\nSUMMARY")
    print(f"{'case':<50}{'label':<8}{'source':<10}{'numbers not in prompt':<24}{'alarming':<14}{'sentences':<10}")
    for r in results:
        print(
            f"{r['case'][:49]:<50}{r['label']:<8}{'fallback' if r['fallback'] else 'LLM':<10}"
            f"{(', '.join(r['unsupported_numbers']) or 'none')[:23]:<24}{(', '.join(r['alarming_words']) or 'none')[:13]:<14}"
            f"{r['sentences']:<10}"
        )
    flagged = [r for r in results if r["unsupported_numbers"] or r["alarming_words"] or r["markdown"] or not 2 <= r["sentences"] <= 4 or r["fallback"]]
    print(f"\n{len(flagged)} of {len(results)} explanations flagged by the automatic checks.")
    print("The checks only point at problems. Read every explanation above before signing off.")


def _as_number(token: str) -> float:
    return round(float(token.replace(",", "")), 2)


def _short(value: float) -> str:
    return f"{value:g}" if abs(value) < 1000 else f"{value:,.0f}"


if __name__ == "__main__":
    main()
