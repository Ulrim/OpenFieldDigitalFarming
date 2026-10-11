"""위험 라벨 생성 규칙 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ofdf.labels.risk import (
    CAUTION,
    NORMAL,
    WARNING,
    RiskThresholds,
    build_states,
    compose_compound,
    event_ids,
    future_labels,
)


def make_frame(hours: int = 48, **overrides) -> pd.DataFrame:
    index = pd.date_range("2025-10-01", periods=hours, freq="h")
    base = {
        "t_air": 15.0, "rh": 60.0, "rain": 0.0, "solar_w": 0.0,
        "wind_speed": 1.0, "leaf_wetness": 0.0, "canopy_temp": 15.0,
        "soil_index": 50.0, "cloud_cover": 0.5,
    }
    base.update(overrides)
    return pd.DataFrame({k: np.full(hours, v, dtype=float) for k, v in base.items()}, index=index)


def test_heat_dry_needs_both_temperature_and_solar():
    """고온만으로는 주의가 아니고, 일사까지 높아야 한다."""
    hot_only = make_frame(t_air=32.0, solar_w=100.0)
    assert (build_states(hot_only)["heat_dry"] == NORMAL).all()

    hot_and_bright = make_frame(t_air=32.0, solar_w=750.0)
    assert (build_states(hot_and_bright)["heat_dry"] == CAUTION).all()

    very_hot = make_frame(t_air=34.0, solar_w=850.0)
    assert (build_states(very_hot)["heat_dry"] == WARNING).all()


def test_winter_dry_soil_is_not_heat_stress():
    """겨울철 건조한 토양은 고온·건조 위험이 아니다."""
    cold_dry = make_frame(t_air=2.0, soil_index=10.0)
    assert (build_states(cold_dry)["heat_dry"] == NORMAL).all()


def test_disease_needs_wet_duration_humidity_and_temperature():
    """엽면습윤 지속시간·습도·기온이 모두 맞아야 병해 유리환경이다."""
    wet = make_frame(leaf_wetness=1.0, rh=92.0, t_air=15.0)
    states = build_states(wet)["disease"]
    # 6시간 연속 습윤이 쌓이기 전에는 정상
    assert states.iloc[:5].eq(NORMAL).all()
    assert states.iloc[6:].eq(CAUTION).all()

    too_cold = make_frame(leaf_wetness=1.0, rh=92.0, t_air=3.0)
    assert (build_states(too_cold)["disease"] == NORMAL).all()


def test_frost_only_at_night():
    """저온·서리는 야간에만 판정한다."""
    cold = make_frame(t_air=1.0, canopy_temp=-1.0, wind_speed=0.5, cloud_cover=0.1)
    states = build_states(cold)["frost"]
    hours = states.index.hour
    assert states[(hours >= 18) | (hours < 8)].ge(CAUTION).all()
    assert states[(hours >= 8) & (hours < 18)].eq(NORMAL).all()


def test_rain_caution_mode_changes_label_density():
    """주의 결합 방식에 따라 라벨 수가 달라진다(AND 가 더 엄격)."""
    frame = make_frame(hours=24, rain=8.0, soil_index=95.0)
    strict = build_states(frame, RiskThresholds(rain_caution_mode="and"))["rain_wet"]
    loose = build_states(frame, RiskThresholds(rain_caution_mode="or"))["rain_wet"]
    assert (loose >= strict).all()


def test_compound_requires_two_risks():
    """복합위험은 구성 위험이 둘 이상 겹쳐야 한다."""
    assert compose_compound([2], [0], [0]).tolist() == [NORMAL]
    assert compose_compound([1], [1], [0]).tolist() == [CAUTION]
    assert compose_compound([2], [1], [0]).tolist() == [WARNING]
    assert compose_compound([1], [0], [2]).tolist() == [WARNING]


def test_future_labels_look_forward_only():
    """예측 라벨은 미래 구간의 최대 등급이고, 현재값을 포함하지 않는다."""
    states = pd.DataFrame(
        {"x": [0, 0, 2, 0, 0]},
        index=pd.date_range("2025-10-01", periods=5, freq="h"),
    )
    labels = future_labels(states, horizons=(1, 2))
    assert labels["x_h1"].tolist()[:4] == [0, 2, 0, 0]
    assert labels["x_h2"].tolist()[:3] == [2, 2, 0]


def test_event_ids_separate_by_gap():
    """해소 후 기준시간이 지나야 다음 발생을 별개 이벤트로 센다."""
    state = pd.Series([1, 1, 0, 1, 0, 0, 0, 0, 1])
    ids = event_ids(state, gap_hours=2)
    assert ids.tolist() == [0, 0, -1, 0, -1, -1, -1, -1, 1]


@pytest.mark.parametrize("mode", ["and", "or"])
def test_states_are_within_valid_range(mode):
    frame = make_frame(hours=72, rain=3.0, rh=95.0, leaf_wetness=1.0, t_air=18.0)
    states = build_states(frame, RiskThresholds(rain_caution_mode=mode))
    assert states.min().min() >= NORMAL
    assert states.max().max() <= WARNING


# --------------------------------------------------------------------------
# 라벨 조건과 피처의 대응
# --------------------------------------------------------------------------

def test_duration_features_cover_the_heat_dry_label_conditions():
    """고온·건조 라벨은 지속시간으로 주의와 경계를 가른다.

    건조 분기는 같은 조건이 1시간이면 주의, 2시간이면 경계다. 즉 주의는
    사실상 '건조 구간의 첫 1시간'이다. 그런데 피처에 그 지속시간이 없으면
    모델이 '이미 마른 지 오래'인지 '방금 마르기 시작'인지 구분할 수 없고,
    정상을 주의로 부르게 된다(실제로 주의 정밀도가 0.31 이었다).

    run_dry 는 '비가 안 온 시간'이라 이 역할을 못 한다. 토양 건조는 비와
    따로 움직인다.
    """
    import pandas as pd
    from ofdf.features.weather import duration_features
    from ofdf.labels.risk import RiskThresholds

    th = RiskThresholds()
    index = pd.date_range("2025-09-01", periods=6, freq="h")
    # 비는 내내 오지 않지만, 토양이 마르는 것은 3시각부터다
    df = pd.DataFrame(
        {
            "t_air": 25.0, "rain": 0.0, "rh": 60.0, "solar_w": 300.0,
            "soil_index": [60.0, 60.0, 60.0, 20.0, 20.0, 20.0],
            "leaf_wetness": 0.0, "canopy_temp": 20.0,
        },
        index=index,
    )
    got = duration_features(df)

    # 무강수는 처음부터 이어지지만 토양 건조는 아니다 — 둘은 다른 값이다
    assert list(got["run_dry"]) == [1, 2, 3, 4, 5, 6]
    assert list(got["run_dry_warm"]) == [0, 0, 0, 1, 2, 3]
    assert list(got["run_soil_low"]) == [0, 0, 0, 1, 2, 3]

    # 라벨이 주의/경계를 가르는 지점(1시간 / 2시간)이 피처에 그대로 보인다
    assert got["run_dry_warm"].iloc[3] == th.dry_caution_hours
    assert got["run_dry_warm"].iloc[4] == th.dry_warning_hours


def test_hot_duration_feature_requires_solar_like_the_label():
    """라벨의 고온 분기는 기온과 일사를 함께 본다. 기온만 보면 어긋난다."""
    import pandas as pd
    from ofdf.features.weather import duration_features

    index = pd.date_range("2025-09-01", periods=4, freq="h")
    df = pd.DataFrame(
        {
            "t_air": 31.0,                       # 내내 30℃ 이상
            "solar_w": [100.0, 100.0, 750.0, 750.0],   # 일사는 뒤늦게 오른다
            "rain": 0.0, "rh": 60.0, "soil_index": 60.0,
            "leaf_wetness": 0.0, "canopy_temp": 25.0,
        },
        index=index,
    )
    got = duration_features(df)

    assert list(got["run_hot30"]) == [1, 2, 3, 4]        # 기온만 본 값
    assert list(got["run_hot_solar"]) == [0, 0, 1, 2]    # 라벨 조건을 본 값


def test_duration_thresholds_follow_field_calibration():
    """임계값을 하드코딩하지 않는다. 현장 보정이 바뀌면 피처도 따라가야 한다."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src" / "ofdf" / "features" / "weather.py"
    text = src.read_text(encoding="utf-8")
    assert "th.soil_low" in text
    assert "th.dry_min_temp" in text
    assert "th.heat_caution_temp" in text
    assert "th.heat_caution_solar" in text
