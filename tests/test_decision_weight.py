"""판정 가중치 검증.

argmax 는 등급이 심하게 불균형할 때 최적이 아니다. 고온·건조 3시간이
그 예였다 — 주의 등급은 실제 300건인데 argmax 가 정상 540건을 주의로
불러, 주의 F1 0.42 하나가 Macro F1 을 0.80 아래로 끌어내렸다.

가중치는 **검증셋에서만** 고르고, 놓침 방지율 하한 아래로 내려가는 후보는
제외한다. F1 을 올리자고 위험을 더 놓치는 것은 재해 경보에서 옳은 교환이
아니기 때문이다.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from ofdf.models.weather_risk import (
    RISK_RECALL_FLOOR,
    decide_levels,
    load_decision_weight,
    tune_decision_weights,
)


def test_no_weight_is_plain_argmax():
    proba = np.array([[0.5, 0.4, 0.1], [0.2, 0.3, 0.5]])
    assert list(decide_levels(proba)) == [0, 2]
    assert list(decide_levels(proba, [1.0, 1.0, 1.0])) == [0, 2]


def test_weight_can_hold_back_an_eager_class():
    """주의를 과하게 부르는 것을 누르는 것이 이 가중치의 목적이다."""
    proba = np.array([[0.45, 0.46, 0.09]])
    assert decide_levels(proba)[0] == 1            # argmax 는 주의
    assert decide_levels(proba, [1.0, 0.5, 1.0])[0] == 0   # 눌러서 정상


def test_tuning_improves_macro_f1_on_imbalanced_middle_class():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    n = 4000
    y = np.zeros(n, dtype=int)
    y[:120] = 1          # 주의는 드물다
    y[120:600] = 2
    rng.shuffle(y)

    # 주의 확률이 전반적으로 부풀려진 모델을 흉내낸다
    proba = np.zeros((n, 3))
    for i, label in enumerate(y):
        base = {0: [0.55, 0.35, 0.10], 1: [0.25, 0.60, 0.15], 2: [0.10, 0.20, 0.70]}[label]
        proba[i] = np.clip(np.array(base) + rng.normal(0, 0.06, 3), 0.01, None)
    proba /= proba.sum(1, keepdims=True)

    weight = tune_decision_weights(proba, y)
    before = f1_score(y, decide_levels(proba), average="macro")
    after = f1_score(y, decide_levels(proba, weight), average="macro")
    assert after > before
    assert weight[1] < 1.0, "과하게 불리는 등급을 눌러야 한다"


def test_tuning_never_trades_away_the_recall_floor():
    """사업계획서 1순위 지표를 깎는 후보는 애초에 고르지 않는다."""
    rng = np.random.default_rng(1)
    n = 3000
    y = (rng.random(n) < 0.05).astype(int) * 2      # 정상 아니면 경계
    proba = np.zeros((n, 3))
    for i, label in enumerate(y):
        base = {0: [0.70, 0.20, 0.10], 2: [0.40, 0.15, 0.45]}[label]
        proba[i] = np.clip(np.array(base) + rng.normal(0, 0.08, 3), 0.01, None)
    proba /= proba.sum(1, keepdims=True)

    weight = tune_decision_weights(proba, y, recall_floor=0.95)
    pred = decide_levels(proba, weight)
    risk = y >= 1
    assert float((pred[risk] >= 1).mean()) >= 0.95


def test_floor_matches_the_business_plan_target():
    assert RISK_RECALL_FLOOR == 0.85


def test_no_risk_samples_falls_back_to_argmax():
    proba = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
    assert tune_decision_weights(proba, np.zeros(2, dtype=int)) == [1.0, 1.0, 1.0]


def test_weights_ship_with_the_model(tmp_path):
    """제어기가 같은 규칙으로 판정하려면 가중치가 모델과 함께 가야 한다."""
    path = tmp_path / "model_x_h3.txt"
    path.write_text("dummy", encoding="utf-8")
    assert load_decision_weight(path) is None       # 없으면 None — argmax 로 돈다

    path.with_suffix(".weights.json").write_text(
        json.dumps({"decision_weight": [1.0, 0.15, 0.5]}), encoding="utf-8")
    assert load_decision_weight(path) == [1.0, 0.15, 0.5]


def test_corrupt_weight_file_does_not_crash_the_controller(tmp_path):
    path = tmp_path / "model_x_h3.txt"
    path.write_text("dummy", encoding="utf-8")
    path.with_suffix(".weights.json").write_text("{깨진", encoding="utf-8")
    assert load_decision_weight(path) is None
