import logging
from types import SimpleNamespace

import httpx
import openai
import pytest

import config
from features import FEATURE_ORDER
from services import explain

REQUEST = httpx.Request("POST", "https://api.example.com/v1/chat/completions")


def average_pr():
    return dict(explain.FEATURE_MEANS)


def risky_pr():
    values = average_pr()
    values.update(files_changed=50, total_churn=2475, lines_added=1980, is_weekend=1, touches_config=1, has_tests=0)
    return values


def fake_client(content="This PR is moderately elevated risk because it changes 50 files.", finish_reason="stop", calls=None):
    def create(**kwargs):
        if calls is not None:
            calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)])

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def raising_client(exc):
    def create(**kwargs):
        raise exc

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_every_feature_has_readable_text_and_a_training_mean():
    assert set(explain.FEATURE_TEXT) == set(FEATURE_ORDER)
    assert list(explain.FEATURE_MEANS) == FEATURE_ORDER


def test_top_deviations_picks_five_furthest_from_mean_relative_to_mean():
    drivers = explain.top_deviations(risky_pr())

    assert len(drivers) == 5
    assert drivers[0]["name"] == "files_changed"
    assert {"total_churn", "lines_added", "is_weekend"} <= {d["name"] for d in drivers}
    assert all(d["name"] not in {"commit_count", "review_comment_count"} for d in drivers)
    assert [d["deviation"] for d in drivers] == sorted((d["deviation"] for d in drivers), reverse=True)


def test_prompt_contains_score_label_five_features_and_instructions():
    prompt = explain.build_prompt(explain.top_deviations(risky_pr()), 0.55, "Medium")

    assert "0.55" in prompt and "Medium" in prompt
    assert prompt.count("\n- ") == 5
    assert "files changed: 50 (average" in prompt
    assert "engineer explaining" in prompt
    assert "2 to 4 sentences" in prompt
    assert "2 or 3 measurements" in prompt
    assert "Use only the numbers above" in prompt
    assert "do not invent facts" in prompt
    assert '"moderately elevated", not "dangerous"' in prompt
    assert "Do not use bullet points" in prompt
    for name in ("review_comment_count", "commit_count"):
        assert explain.FEATURE_TEXT[name][0] + ":" not in prompt


def test_successful_call_uses_the_prompt_settings(monkeypatch):
    calls = []
    monkeypatch.setattr(explain, "_client", lambda: fake_client(calls=calls))
    monkeypatch.setattr(config.settings, "LLM_REASONING_EFFORT", "low")

    text = explain.generate_explanation(risky_pr(), 0.62, "Medium")

    assert text == "This PR is moderately elevated risk because it changes 50 files."
    (call,) = calls
    assert call["temperature"] == 0.3
    assert call["max_tokens"] == 200
    assert call["model"] == config.settings.LLM_MODEL
    assert call["reasoning_effort"] == "low"
    assert "0.62" in call["messages"][0]["content"]


def test_reasoning_effort_is_omitted_when_not_configured(monkeypatch):
    calls = []
    monkeypatch.setattr(explain, "_client", lambda: fake_client(calls=calls))
    monkeypatch.setattr(config.settings, "LLM_REASONING_EFFORT", "")

    explain.generate_explanation(risky_pr(), 0.4, "Medium")

    assert "reasoning_effort" not in calls[0]


@pytest.mark.parametrize(
    "failure",
    [
        openai.APITimeoutError(request=REQUEST),
        openai.RateLimitError("rate limited", response=httpx.Response(429, request=REQUEST), body=None),
        openai.AuthenticationError("bad key", response=httpx.Response(401, request=REQUEST), body=None),
        openai.APIConnectionError(request=REQUEST),
        RuntimeError("unexpected client bug"),
    ],
    ids=["timeout", "rate_limit", "bad_key", "connection", "unexpected"],
)
def test_any_llm_failure_returns_the_fallback_and_logs_a_warning(monkeypatch, caplog, failure):
    monkeypatch.setattr(explain, "_client", lambda: raising_client(failure))

    with caplog.at_level(logging.WARNING, logger="services.explain"):
        text = explain.generate_explanation(risky_pr(), 0.71, "High")

    assert text == explain.fallback_explanation(explain.top_deviations(risky_pr()), 0.71, "High")
    assert "rated High risk (score 0.71)" in text
    assert "files changed: 50" in text
    assert "LLM explanation failed" in caplog.text
    assert config.settings.OPENAI_API_KEY not in caplog.text


def test_client_construction_failure_also_falls_back(monkeypatch):
    def broken():
        raise openai.OpenAIError("The api_key client option must be set")

    monkeypatch.setattr(explain, "_client", broken)

    assert explain.generate_explanation(risky_pr(), 0.2, "Low").startswith("This pull request is rated Low risk")


@pytest.mark.parametrize(("content", "finish_reason"), [(None, "length"), ("", "stop"), ("   ", "stop"), ("This PR changes 50 fi", "length")])
def test_empty_or_truncated_replies_fall_back(monkeypatch, content, finish_reason):
    monkeypatch.setattr(explain, "_client", lambda: fake_client(content=content, finish_reason=finish_reason))

    assert explain.generate_explanation(risky_pr(), 0.5, "Medium").startswith("This pull request is rated Medium risk")


def test_fallback_is_deterministic_and_describes_flags_in_words():
    drivers = explain.top_deviations(risky_pr())

    first = explain.fallback_explanation(drivers, 0.66, "Medium")

    assert first == explain.fallback_explanation(drivers, 0.66, "Medium")
    assert "merged on a weekend: yes (true for" in explain.build_prompt(drivers, 0.66, "Medium")
