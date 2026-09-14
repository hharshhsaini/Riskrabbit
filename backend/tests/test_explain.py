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
    values.update(files_changed=50, total_churn=2475, lines_added=1980, is_weekend=1, touches_config=0, has_tests=0)
    return values


def controlled_effects(monkeypatch, **effects):
    monkeypatch.setattr(explain.model, "contributions", lambda values: {name: effects.get(name, 0.0) for name in FEATURE_ORDER})


def fake_client(content="Moderately elevated: 50 files changed raised the score.", finish_reason="stop", calls=None):
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
    assert set(explain.VALUE_TEXT) | set(explain.FLAG_TEXT) == set(FEATURE_ORDER)
    assert not set(explain.VALUE_TEXT) & set(explain.FLAG_TEXT)
    assert list(explain.FEATURE_MEANS) == FEATURE_ORDER


def test_top_drivers_rank_by_size_of_effect_and_skip_unused_features(monkeypatch):
    controlled_effects(monkeypatch, files_changed=0.9, has_tests=-1.2, is_weekend=0.3, lines_added=1e-9)

    drivers = explain.top_drivers(risky_pr())

    assert [d["name"] for d in drivers] == ["has_tests", "files_changed", "is_weekend"]
    assert [d["effect"] for d in drivers] == [-1.2, 0.9, 0.3]


def test_top_drivers_on_the_real_model_are_sorted_and_at_most_five():
    drivers = explain.top_drivers(risky_pr())

    assert 1 <= len(drivers) <= 5
    assert [abs(d["effect"]) for d in drivers] == sorted((abs(d["effect"]) for d in drivers), reverse=True)


def test_prompt_marks_direction_and_phrases_flags_as_the_prs_actual_state(monkeypatch):
    controlled_effects(monkeypatch, files_changed=0.9, touches_config=-0.4, is_weekend=0.2)
    means = dict(explain.FEATURE_MEANS, touches_config=0.18, is_weekend=0.26, files_changed=5.7)
    monkeypatch.setattr(explain, "FEATURE_MEANS", means)

    prompt = explain.build_prompt(explain.top_drivers(risky_pr()), 0.55, "Medium")

    assert "1. files changed: 50 (average 5.7); raised the score" in prompt
    assert "2. does not change configuration files, like 82% of pull requests; lowered the score" in prompt
    assert "3. was merged on a weekend, like 26% of pull requests; raised the score" in prompt
    assert "\n4. " not in prompt


def test_prompt_contains_score_label_and_instructions():
    prompt = explain.build_prompt(explain.top_drivers(risky_pr()), 0.55, "Medium")

    assert "0.55" in prompt and "Medium" in prompt
    assert "engineer explaining" in prompt
    assert "moved this score the most, according to the model itself" in prompt
    assert "measurement 1 had the largest effect" in prompt
    assert "Discuss only the first 2 or 3 measurements in the list, in that order" in prompt
    assert "do not rank the measurements differently" in prompt
    assert "do not mention measurements that are not listed" in prompt
    assert "2 to 4 sentences" in prompt
    assert "copy their numbers exactly" in prompt
    assert "do not guess how well the change was reviewed or tested" in prompt
    assert "what the model has seen before" in prompt
    assert '"moderately elevated", not "dangerous"' in prompt
    assert "Do not use bullet points" in prompt


def test_successful_call_uses_the_prompt_settings(monkeypatch):
    calls = []
    monkeypatch.setattr(explain, "_client", lambda: fake_client(calls=calls))
    monkeypatch.setattr(config.settings, "LLM_REASONING_EFFORT", "low")

    text = explain.generate_explanation(risky_pr(), 0.62, "Medium")

    assert text == "Moderately elevated: 50 files changed raised the score."
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

    assert text == explain.fallback_explanation(explain.top_drivers(risky_pr()), 0.71, "High")
    assert "rated High risk (score 0.71)" in text
    assert "moved the score most" in text
    assert "LLM explanation failed" in caplog.text
    assert config.settings.OPENAI_API_KEY not in caplog.text


def test_client_construction_failure_also_falls_back(monkeypatch):
    def broken():
        raise openai.OpenAIError("The api_key client option must be set")

    monkeypatch.setattr(explain, "_client", broken)

    assert explain.generate_explanation(risky_pr(), 0.2, "Low").startswith("This pull request is rated Low risk")


def test_contribution_failure_falls_back_without_calling_the_llm(monkeypatch):
    def broken(values):
        raise ValueError("booster unavailable")

    calls = []
    monkeypatch.setattr(explain.model, "contributions", broken)
    monkeypatch.setattr(explain, "_client", lambda: fake_client(calls=calls))

    assert explain.generate_explanation(risky_pr(), 0.3, "Low") == "This pull request is rated Low risk (score 0.30)."
    assert calls == []


@pytest.mark.parametrize(("content", "finish_reason"), [(None, "length"), ("", "stop"), ("   ", "stop"), ("This PR changes 50 fi", "length")])
def test_empty_or_truncated_replies_fall_back(monkeypatch, content, finish_reason):
    monkeypatch.setattr(explain, "_client", lambda: fake_client(content=content, finish_reason=finish_reason))

    assert explain.generate_explanation(risky_pr(), 0.5, "Medium").startswith("This pull request is rated Medium risk")


def test_fallback_is_deterministic():
    drivers = explain.top_drivers(risky_pr())

    assert explain.fallback_explanation(drivers, 0.66, "Medium") == explain.fallback_explanation(drivers, 0.66, "Medium")
