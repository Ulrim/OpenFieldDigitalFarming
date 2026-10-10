"""데이터 파싱·품질관리·파생변수 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ofdf.data import quality
from ofdf.data.aaos import decumulate
from ofdf.features.derived import canopy_temperature, dew_point, leaf_wetness, soil_water_index


def test_decumulate_restores_hourly_values():
    """자정 기준 누적값을 시간값으로 되돌린다."""
    ts = pd.date_range("2020-07-13", periods=6, freq="h")
    df = pd.DataFrame({"station": "A", "ts": ts, "rain": [1.5, 9.5, 21.5, 29.5, 38.5, 51.0]})
    out = decumulate(df, ["rain"])
    assert out["rain"].tolist() == [1.5, 8.0, 12.0, 8.0, 9.0, 12.5]
    assert out["rain"].sum() == 51.0           # 하루 총량 보존
    assert out["rain_cum"].tolist() == [1.5, 9.5, 21.5, 29.5, 38.5, 51.0]


def test_decumulate_resets_each_day():
    ts = list(pd.date_range("2020-07-13", periods=2, freq="h")) + list(
        pd.date_range("2020-07-14", periods=2, freq="h")
    )
    df = pd.DataFrame({"station": "A", "ts": ts, "rain": [5.0, 9.0, 2.0, 6.0]})
    out = decumulate(df, ["rain"])
    assert out["rain"].tolist() == [5.0, 4.0, 2.0, 4.0]


def test_decumulate_clips_negative_steps():
    """관측기 리셋으로 누적값이 줄어도 음수 강수가 나오지 않는다."""
    ts = pd.date_range("2020-07-13", periods=3, freq="h")
    df = pd.DataFrame({"station": "A", "ts": ts, "rain": [10.0, 4.0, 6.0]})
    assert (decumulate(df, ["rain"])["rain"] >= 0).all()


def test_quality_flags_out_of_range_and_spikes():
    df = pd.DataFrame(
        {
            "station": "A",
            "ts": pd.date_range("2024-10-30", periods=4, freq="h"),
            "t_air": [12.0, -44.6, 13.0, 40.0],
            "rh": [60.0, 61.0, 62.0, 63.0],
        }
    )
    flagged, report = quality.run(df)
    assert flagged["qc_range_t_air"].tolist() == [False, True, False, False]
    assert flagged["qc_spike_t_air"].any()
    assert report.flagged["qc_range_t_air"] == 1

    masked = quality.apply_mask(flagged)
    assert np.isnan(masked["t_air"].iloc[1])
    assert masked["t_air"].iloc[0] == 12.0      # 정상값은 건드리지 않는다


def test_quality_counts_duplicates_per_station():
    """지점이 여럿이면 같은 시각이 여러 번 나오는 것은 중복이 아니다."""
    ts = pd.date_range("2025-10-01", periods=3, freq="h")
    df = pd.DataFrame({"station": ["A"] * 3 + ["B"] * 3, "ts": list(ts) * 2, "t_air": 10.0})
    assert quality.check_index(df) == (0, 0)


def test_dew_point_below_air_temperature():
    t = pd.Series([20.0, 20.0, 20.0])
    rh = pd.Series([100.0, 70.0, 40.0])
    td = dew_point(t, rh)
    assert td.iloc[0] == max(td)                 # 포화 시 이슬점 = 기온
    assert (td <= t + 1e-6).all()
    assert td.is_monotonic_decreasing


def test_leaf_wetness_triggers_on_rain_and_humidity():
    t = pd.Series([15.0] * 3)
    rh = pd.Series([50.0, 95.0, 50.0])
    rain = pd.Series([0.0, 0.0, 2.0])
    solar = pd.Series([500.0, 0.0, 500.0])
    assert leaf_wetness(t, rh, rain, solar).tolist() == [0.0, 1.0, 1.0]


def test_canopy_temperature_drops_only_at_night_and_under_clear_sky():
    t = pd.Series([5.0, 5.0, 5.0])
    rh = pd.Series([60.0, 60.0, 60.0])
    wind = pd.Series([0.2, 0.2, 0.2])
    solar = pd.Series([0.0, 0.0, 600.0])          # 야간, 야간, 주간
    cloud = pd.Series([0.0, 1.0, 0.0])            # 맑음, 흐림, 맑음
    canopy = canopy_temperature(t, rh, wind, solar, cloud)
    assert canopy.iloc[0] < canopy.iloc[1]        # 맑은 밤이 더 식는다
    assert canopy.iloc[2] == 5.0                  # 주간에는 기온과 같다
    assert canopy.iloc[0] >= 5.0 - 4.0            # 최대 강하폭을 넘지 않는다


def test_soil_water_index_drains_and_never_sticks_at_saturation():
    """큰 비 뒤에는 배수로 빠져 포화에 고착되지 않는다."""
    rain = pd.Series([100.0] + [0.0] * 47)
    et0 = pd.Series([0.05] * 48)
    index = soil_water_index(rain, et0)
    assert index.iloc[0] <= 100.0
    assert index.iloc[-1] < index.iloc[0]
    assert (index.between(0, 100)).all()


def test_soil_water_index_slows_drying_when_dry():
    """수분스트레스 계수 때문에 마를수록 증발산이 줄어 0으로 붙지 않는다."""
    rain = pd.Series([0.0] * 200)
    et0 = pd.Series([0.2] * 200)
    index = soil_water_index(rain, et0)
    assert index.iloc[-1] > 0.0
    assert index.is_monotonic_decreasing


def test_daily_columns_are_lagged_by_one_day():
    """일자료는 하루 전 값이어야 한다 — 미래 정보 누설 방지.

    그날 값을 그날 전체에 깔면 01시에 그날 밤의 실현 최저기온을 보게 되고,
    그것은 서리 정답을 미리 보는 것에 가깝다. 제어기가 운용 시점에 실제로
    가진 것은 어제까지 실현된 일자료뿐이다.
    """
    import pandas as pd
    from ofdf.data import agera5

    index = pd.date_range("2025-10-02", periods=48, freq="h")
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2025-10-01", "2025-10-02", "2025-10-03"]),
        "t_min_night": [1.0, 2.0, 3.0],
    })

    prev_day = index.normalize() - pd.Timedelta(days=1)
    mapped = pd.Series(prev_day, index=index).map(daily.set_index("date")["t_min_night"])

    # 10/2 의 시각들은 10/1 값을, 10/3 의 시각들은 10/2 값을 본다
    assert mapped.loc["2025-10-02 00:00"] == 1.0
    assert mapped.loc["2025-10-02 23:00"] == 1.0
    assert mapped.loc["2025-10-03 00:00"] == 2.0
    assert agera5.RENAME["2m_temperature_night_time_minimum"] == "t_min_night"


def test_assembly_source_uses_the_lagged_day():
    """조립 코드가 실제로 하루 전 날짜로 매핑하는지 본다."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "src" / "ofdf" / "data" / "dataset.py"
    text = src.read_text(encoding="utf-8")
    assert "prev_day = one.index.normalize() - pd.Timedelta(days=1)" in text
    assert "pd.Series(prev_day, index=one.index).map(" in text
