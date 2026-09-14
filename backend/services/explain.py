"""Plain-language explanations of a risk score, written by an LLM.

The explanation is grounded in what the model actually used: the prompt gets the
score, the label and the five features that moved this PR's score the most, taken
from the model's own per-feature contributions, each marked as raising or lowering
the score. Any LLM failure returns a deterministic summary instead, so a slow or
broken LLM can never fail a prediction.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from openai import OpenAI

import features
from config import settings
from services import model

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent.parent
TOP_FEATURE_COUNT = 5
# Contributions this small are features the model did not use for this PR.
MIN_EFFECT = 1e-6
TEMPERATURE = 0.3
MAX_TOKENS = 200
# The openai client's default timeout is 600 seconds; a prediction must not hang that long.
TIMEOUT_SECONDS = 20
MAX_RETRIES = 1

PROMPT_TEMPLATE = """You are a software engineer explaining a deployment risk assessment to a teammate.

A model scored this pull request {score:.2f} on a 0 to 1 deployment risk scale and labelled it {label}. The score ranks pull requests against past ones; it is not a probability that this change will fail.

These are the {count} measurements that moved this score the most, according to the model itself, listed from the largest effect to the smallest: measurement 1 had the largest effect. "Raised the score" means the measurement pushed the score up and "lowered the score" means it pushed the score down. Averages across all pull requests are given for comparison.
{feature_lines}

Write 2 to 4 sentences of plain prose explaining the rating. Discuss only the first 2 or 3 measurements in the list, in that order, and say whether each raised or lowered the score. Describe them in your own words, but copy their numbers exactly as given. Only state what the list says: do not rank the measurements differently, do not mention measurements that are not listed or claim anything about them, and do not guess how well the change was reviewed or tested, what the code does, who wrote it, why a measurement matters, or what the model has seen before. Do not claim more certainty than the score supports: match your wording to the label, so a Medium rating or a score around 0.55 is "moderately elevated", not "dangerous", and never say a change will fail. Do not use bullet points, numbered lists, headings or markdown."""

# Numeric features: how each reads in a sentence and how its value is formatted.
VALUE_TEXT: dict[str, tuple[str, str]] = {
    "files_changed": ("files changed", "count"),
    "lines_added": ("lines added", "count"),
    "lines_removed": ("lines removed", "count"),
    "total_churn": ("lines changed in total", "count"),
    "unique_dirs_touched": ("directories touched", "count"),
    "commit_count": ("commits", "count"),
    "avg_commit_msg_len": ("average commit message length in characters", "count"),
    "test_file_ratio": ("share of changed files that are tests", "ratio"),
    "review_comment_count": ("review comments", "count"),
    "pr_age_hours": ("hours between opening and merging", "count"),
    "author_pr_count": ("earlier merged pull requests by the same author in this repository", "count"),
}
# Yes/no features, phrased as the PR's actual state so a "no" cannot be read as its opposite.
FLAG_TEXT: dict[str, tuple[str, str]] = {
    "has_tests": ("includes changes to test files", "does not change any test files"),
    "touches_config": ("changes configuration files", "does not change configuration files"),
    "touches_ci": ("changes CI workflow files", "does not change CI workflow files"),
    "is_weekend": ("was merged on a weekend", "was merged on a weekday"),
    "is_off_hours": ("was merged outside 09:00-18:00 UTC", "was merged between 09:00 and 18:00 UTC"),
}


def generate_explanation(feature_values: dict[str, float], score: float, label: str) -> str:
    try:
        drivers = top_drivers(feature_values)
    except Exception as exc:  # the explanation is optional; it must never fail the prediction
        logger.warning("Could not compute feature contributions (%s: %s)", type(exc).__name__, exc)
        return fallback_explanation([], score, label)
    try:
        extra = {"reasoning_effort": settings.LLM_REASONING_EFFORT} if settings.LLM_REASONING_EFFORT else {}
        response = _client().chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[{"role": "user", "content": build_prompt(drivers, score, label)}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            **extra,
        )
        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        if choice.finish_reason != "stop" or not text:
            raise ValueError(
                f"unusable completion (finish_reason={choice.finish_reason!r}, {len(text)} characters)"
            )
        return text
    except Exception as exc:  # any LLM failure falls back; it must never fail the prediction
        logger.warning(
            "LLM explanation failed (%s: %s); using the fallback summary", type(exc).__name__, exc
        )
        return fallback_explanation(drivers, score, label)


def top_drivers(feature_values: dict[str, float], count: int = TOP_FEATURE_COUNT) -> list[dict[str, Any]]:
    """The features that moved this PR's score most, by the model's own contributions."""
    effects = model.contributions(feature_values)
    rows = [
        {"name": name, "value": float(feature_values[name]), "mean": FEATURE_MEANS[name], "effect": effects[name]}
        for name in features.FEATURE_ORDER
        if abs(effects[name]) >= MIN_EFFECT
    ]
    rows.sort(key=lambda row: -abs(row["effect"]))  # stable: ties keep FEATURE_ORDER
    return rows[:count]


def build_prompt(drivers: list[dict[str, Any]], score: float, label: str) -> str:
    return PROMPT_TEMPLATE.format(
        score=score,
        label=label,
        count=len(drivers),
        feature_lines="\n".join(f"{rank}. {_describe(row)}" for rank, row in enumerate(drivers, start=1)),
    )


def fallback_explanation(drivers: list[dict[str, Any]], score: float, label: str) -> str:
    listed = [_describe(row) for row in drivers[:3]]
    if not listed:
        return f"This pull request is rated {label} risk (score {score:.2f})."
    return (
        f"This pull request is rated {label} risk (score {score:.2f}). "
        f"The measurements that moved the score most: {'; '.join(listed)}."
    )


def _describe(row: dict[str, Any]) -> str:
    name, value, mean = row["name"], row["value"], row["mean"]
    direction = "raised the score" if row["effect"] > 0 else "lowered the score"
    if name in FLAG_TEXT:
        yes_text, no_text = FLAG_TEXT[name]
        if value >= 0.5:
            return f"{yes_text}, like {mean:.0%} of pull requests; {direction}"
        return f"{no_text}, like {1 - mean:.0%} of pull requests; {direction}"
    text, kind = VALUE_TEXT[name]
    if kind == "ratio":
        return f"{text}: {value:.0%} (average {mean:.0%}); {direction}"
    return f"{text}: {_number(value)} (average {_number(mean)}); {direction}"


def _number(value: float) -> str:
    return f"{value:,.0f}" if abs(value) >= 10 or float(value).is_integer() else f"{value:,.1f}"


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    # Created on first use, inside generate_explanation's try, so a bad key or
    # missing configuration becomes a fallback rather than an import error.
    return OpenAI(
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        timeout=TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


def _load_feature_means() -> dict[str, float]:
    """Training-set means saved by ml/train.py, read once at import.

    Read from metrics.json rather than copied into this file, so a retrained model
    can never leave stale averages behind.
    """
    model_path = Path(settings.MODEL_PATH)
    if not model_path.is_absolute():
        model_path = BACKEND_DIR / model_path
    metrics_path = model_path.with_name("metrics.json")
    if not metrics_path.exists():
        raise RuntimeError(f"{metrics_path} not found. Train the model with `python -m ml.train`.")
    means = json.loads(metrics_path.read_text()).get("feature_means", {})
    missing = [name for name in features.FEATURE_ORDER if name not in means]
    if missing:
        raise RuntimeError(f"metrics.json has no training means for {missing}. Retrain with `python -m ml.train`.")
    return {name: float(means[name]) for name in features.FEATURE_ORDER}


FEATURE_MEANS = _load_feature_means()
