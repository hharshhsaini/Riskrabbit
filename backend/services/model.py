"""The trained risk model, loaded once when this module is imported.

Importing fails loudly if the model artifacts are missing, or if the model was
trained on a different feature order than features.py computes. Loading happens at
import, never per request: re-reading model.json on every call would add latency to
each prediction for no benefit.
"""

import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier

import features
from config import settings

BACKEND_DIR = Path(__file__).resolve().parent.parent

LOW_RISK_BELOW = 0.34
MEDIUM_RISK_BELOW = 0.67


def predict(feature_dict: dict[str, float]) -> tuple[float, str]:
    score = float(classifier.predict_proba(_row(feature_dict))[0, 1])
    return score, score_to_label(score)


def contributions(feature_dict: dict[str, float]) -> dict[str, float]:
    """How much each feature pushed this PR's score up (+) or down (-).

    XGBoost's built-in tree SHAP values, in log-odds. Together with the model's base
    value they add up exactly to this PR's score before the sigmoid.
    """
    values = classifier.get_booster().predict(xgb.DMatrix(_row(feature_dict)), pred_contribs=True)[0]
    return {name: float(value) for name, value in zip(FEATURE_ORDER, values[:-1])}


def _row(feature_dict: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([features.to_array(feature_dict)], columns=FEATURE_ORDER)


def score_to_label(score: float) -> str:
    if score < LOW_RISK_BELOW:
        return "Low"
    if score < MEDIUM_RISK_BELOW:
        return "Medium"
    return "High"


def _artifact_paths() -> tuple[Path, Path, Path]:
    model_path = Path(settings.MODEL_PATH)
    if not model_path.is_absolute():
        model_path = BACKEND_DIR / model_path
    return model_path, model_path.with_name("feature_order.json"), model_path.with_name("metrics.json")


def _load() -> tuple[XGBClassifier, list[str], str]:
    model_path, order_path, metrics_path = _artifact_paths()
    missing = [str(path) for path in (model_path, order_path, metrics_path) if not path.exists()]
    if missing:
        raise RuntimeError(
            f"Model artifacts not found: {', '.join(missing)}. "
            "Train them from backend/ with `python -m ml.train`."
        )
    feature_order = json.loads(order_path.read_text())
    _check_feature_order(feature_order)
    model = XGBClassifier()
    model.load_model(model_path)
    return model, feature_order, json.loads(metrics_path.read_text())["model_version"]


def _check_feature_order(loaded: list[str]) -> None:
    # An explicit raise, not `assert`: asserts are stripped under `python -O`, and a
    # reordered feature vector produces confident, silently wrong scores.
    expected = features.FEATURE_ORDER
    if loaded == expected:
        return
    unknown_to_model = [name for name in expected if name not in loaded]
    not_computed = [name for name in loaded if name not in expected]
    problems = []
    if unknown_to_model:
        problems.append(f"computed by features.py but unknown to the model: {unknown_to_model}")
    if not_computed:
        problems.append(f"expected by the model but not computed by features.py: {not_computed}")
    if not problems:
        position = next(i for i, (a, b) in enumerate(zip(loaded, expected)) if a != b)
        problems.append(
            f"same names in a different order; first difference at position {position}: "
            f"model has {loaded[position]!r}, features.py has {expected[position]!r}"
        )
    raise RuntimeError(
        "feature_order.json does not match features.FEATURE_ORDER, so predictions would be "
        f"silently wrong. {'; '.join(problems)}. Retrain with `python -m ml.train`."
    )


classifier, FEATURE_ORDER, MODEL_VERSION = _load()
