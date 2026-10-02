"""예보 연계와 조치단위 평가 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ofdf.action.recommend import action_distinct_levels, levels_change_action
from ofdf.data.forecast import SimulatedForecast, observed_skill
from ofdf.evaluation import metrics


@pytest.fixture
def observations():
    rng = np.random.default_rng(0)
    index = pd.date_range("2025-10-01", periods=2000, freq="h")
    return pd.DataFrame({
        "rain": rng.choice([0.0, 0.0, 0.0, 0.0, 2.0, 8.0], 2000),
        "t_air": 15 + rng.normal(0, 5, 2000),
        "rh": np.clip(70 + rng.normal(0, 10, 2000), 0, 100),
        "wind_speed": np.abs(rng.normal(1.5, 0.8, 2000)),
    }, index=index)


def test_perfect_forecast_reproduces_future(observations):
    table = SimulatedForecast(observations, skill=1.0, seed=1).build((1,))
    expected = observations["rain"].shift(-1)
    pair = pd.DataFrame({"a": table["fc_rain_h1"], "b": expected}).dropna()
    assert np.allclose(pair["a"], pair["b"])


def test_forecast_quality_rises_with_skill(observations):
    truth = observations["rain"].shift(-1)
    scores = []
    for skill in (0.0, 0.5, 0.9, 1.0):
        table = SimulatedForecast(observations, skill=skill, seed=1).build((1,))
        scores.append(observed_skill(table["fc_rain_h1"], truth)["임계성공지수(CSI)"])
    assert scores == sorted(scores)
    assert scores[-1] == pytest.approx(1.0)


def test_forecast_degrades_with_lead_time(observations):
    truth_1 = observations["rain"].shift(-1)
    truth_6 = observations["rain"].shift(-6)
    table = SimulatedForecast(observations, skill=0.6, seed=1).build((1, 6))
    near = observed_skill(table["fc_rain_h1"], truth_1)["임계성공지수(CSI)"]
    far = observed_skill(table["fc_rain_h6"], truth_6)["임계성공지수(CSI)"]
    assert far < near


def test_forecast_is_reproducible(observations):
    a = SimulatedForecast(observations, skill=0.6, seed=7).build((1,))
    b = SimulatedForecast(observations, skill=0.6, seed=7).build((1,))
    pd.testing.assert_frame_equal(a, b)


def test_forecast_skips_time_gaps():
    """시간축이 끊긴 자리에서 다음 구간 값을 끌어오지 않는다."""
    index = pd.DatetimeIndex(
        list(pd.date_range("2025-10-01", periods=24, freq="h"))
        + list(pd.date_range("2025-11-01", periods=24, freq="h"))
    )
    observations = pd.DataFrame(
        {"rain": [0.0] * 23 + [99.0] + [0.0] * 24, "t_air": 15.0,
         "rh": 70.0, "wind_speed": 1.0},
        index=index,
    )
    table = SimulatedForecast(observations, skill=1.0, seed=1).build((1,))
    assert np.isnan(table.loc["2025-10-01 23:00", "fc_rain_h1"])   # 경계
    assert table.loc["2025-10-01 22:00", "fc_rain_h1"] == pytest.approx(99.0)


def test_observed_skill_counts_hits_and_false_alarms():
    forecast = pd.Series([1.0, 0.0, 1.0, 0.0])
    observed = pd.Series([1.0, 1.0, 0.0, 0.0])
    quality = observed_skill(forecast, observed)
    assert quality["탐지율(POD)"] == pytest.approx(0.5)      # 2건 중 1건 탐지
    assert quality["오경보율(FAR)"] == pytest.approx(0.5)    # 예보 2건 중 1건 헛방


# --------------------------------------------------------------------------
# 조치단위 평가
# --------------------------------------------------------------------------

def test_action_distinct_levels_matches_engine():
    """평가 단위는 조치 엔진에서 직접 끌어온다 — 기준을 두 군데 적지 않는다."""
    distinct = action_distinct_levels()
    # 강우·과습과 저온·서리는 주의와 경계가 같은 조치를 낸다
    assert distinct["rain_wet"] is False
    assert distinct["frost"] is False
    # 고온·건조는 경계에서만 차광, 병해는 경계에서만 방제 일정 조정
    assert distinct["heat_dry"] is True
    assert distinct["disease"] is True


def test_levels_change_action_respects_hour():
    """차광은 10~16시에만 나가므로 시각에 따라 판정이 달라진다."""
    assert levels_change_action("heat_dry", hour=13) is True
    assert levels_change_action("heat_dry", hour=22) is False


def test_operational_score_picks_unit_by_action():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 1, 0, 1, 2])          # 주의/경계만 헷갈린다

    three, unit3 = metrics.operational_score(y_true, y_pred, "heat_dry")
    two, unit2 = metrics.operational_score(y_true, y_pred, "rain_wet")

    assert unit3.startswith("3등급") and unit2.startswith("2등급")
    # 등급을 헷갈린 것은 2등급 평가에서는 전혀 깎이지 않는다
    assert two.macro_f1 == pytest.approx(1.0)
    assert three.macro_f1 < 1.0


def test_binary_score_collapses_caution_and_warning():
    y_true = np.array([0, 1, 2, 2, 0])
    y_pred = np.array([0, 2, 1, 2, 0])
    score = metrics.binary_score(y_true, y_pred)
    assert score.macro_f1 == pytest.approx(1.0)
    assert score.risk_recall == pytest.approx(1.0)
    assert score.n_risk == 3


def test_binary_score_still_penalises_missed_risk():
    y_true = np.array([0, 1, 2, 2])
    y_pred = np.array([0, 0, 0, 2])
    score = metrics.binary_score(y_true, y_pred)
    assert score.risk_recall == pytest.approx(1 / 3)
    assert score.macro_f1 < 0.8
