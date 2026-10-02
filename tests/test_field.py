"""현장 센서 수집·보정 검증."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ofdf.data import field, quality
from ofdf.features import calibration as calib


@pytest.fixture
def long_records(tmp_path):
    """적용구간 2점 + 마스트 1점으로 이루어진 1분 수집 기록."""
    ts = pd.date_range("2025-10-01", periods=240, freq="1min")
    rows = []
    for point, offset in enumerate([0.0, 0.4], start=1):
        rows.append(pd.DataFrame({
            "ts": ts, "zone": field.ZONE_TREATMENT,
            "sensor_id": f"TR-canopy-{point:02d}", "variable": "canopy_temp",
            "value": 10.0 + offset, "quality": "ok",
        }))
    rows.append(pd.DataFrame({
        "ts": ts, "zone": field.ZONE_MAST, "sensor_id": "MA-solar-01",
        "variable": "solar", "value": -12.0, "quality": "ok",   # 야간 음수 오프셋
    }))
    rows.append(pd.DataFrame({
        "ts": ts, "zone": field.ZONE_MAST, "sensor_id": "MA-rain-01",
        "variable": "rain", "value": 0.1, "quality": "ok",
    }))
    path = tmp_path / "field.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    return path


def test_reads_long_format(long_records):
    df = field.read_field(long_records)
    assert {"ts", "zone", "variable", "value"} <= set(df.columns)
    assert df["zone"].nunique() == 2


def test_bad_quality_becomes_missing(tmp_path):
    ts = pd.date_range("2025-10-01", periods=3, freq="1min")
    path = tmp_path / "f.csv"
    pd.DataFrame({
        "ts": ts, "zone": "mast", "sensor_id": "A", "variable": "t_air",
        "value": [10.0, 11.0, 12.0], "quality": ["ok", "bad", "ok"],
    }).to_csv(path, index=False)
    assert field.read_field(path)["value"].isna().sum() == 1


def test_multiple_points_use_median_and_report_spread(long_records):
    """측정점이 여럿이면 중앙값으로 합치고 벌어짐을 남긴다."""
    reconciled = field.reconcile_points(field.read_field(long_records))
    canopy = reconciled[reconciled["variable"] == "canopy_temp"]
    assert (canopy["n_points"] == 2).all()
    assert np.allclose(canopy["value"], 10.2)
    assert np.allclose(canopy["spread"], 0.4)
    assert not canopy["spread_alarm"].any()      # 0.4도 차이는 정상


def test_spread_alarm_fires_on_disagreeing_points(tmp_path):
    ts = pd.date_range("2025-10-01", periods=5, freq="1min")
    rows = [
        pd.DataFrame({"ts": ts, "zone": "treatment", "sensor_id": f"P{i}",
                      "variable": "canopy_temp", "value": v, "quality": "ok"})
        for i, v in enumerate([10.0, 18.0])
    ]
    path = tmp_path / "f.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    assert field.reconcile_points(field.read_field(path))["spread_alarm"].all()


def test_small_negative_solar_is_clamped_not_dropped(long_records):
    """일사계 야간 음수 오프셋은 결측이 아니라 0 으로 자른다."""
    reconciled = field.reconcile_points(field.read_field(long_records))
    solar = reconciled[reconciled["variable"] == "solar"]
    assert solar["value"].notna().all()
    assert (solar["value"] == 0.0).all()


def test_large_out_of_range_still_dropped(tmp_path):
    ts = pd.date_range("2025-10-01", periods=3, freq="1min")
    path = tmp_path / "f.csv"
    pd.DataFrame({
        "ts": ts, "zone": "mast", "sensor_id": "A", "variable": "solar",
        "value": [-999.0, 500.0, 600.0], "quality": "ok",
    }).to_csv(path, index=False)
    reconciled = field.reconcile_points(field.read_field(path))
    assert reconciled["value"].isna().sum() == 1


def test_resample_uses_variable_specific_rules(long_records):
    """강수는 적산, 나머지는 평균으로 집계한다."""
    wide = field.to_wide(field.reconcile_points(field.read_field(long_records)))
    hourly = field.resample(wide, "h")
    assert np.isclose(hourly["mast.rain"].iloc[0], 6.0)            # 0.1 x 60분
    assert np.isclose(hourly["treatment.canopy_temp"].iloc[0], 10.2)


def test_zone_frame_falls_back_to_mast(long_records):
    """구간에 없는 항목은 기상 마스트 값으로 채운다."""
    hourly = field.resample(field.to_wide(field.reconcile_points(field.read_field(long_records))), "h")
    zone = field.zone_frame(hourly, field.ZONE_TREATMENT)
    assert "canopy_temp" in zone.columns      # 구간 자체 측정
    assert "rain" in zone.columns             # 마스트에서 보충


# --------------------------------------------------------------------------
# 보정
# --------------------------------------------------------------------------

@pytest.fixture
def paired_frames():
    index = pd.date_range("2025-10-01", periods=480, freq="h")
    rng = np.random.default_rng(0)
    station = pd.DataFrame({
        "t_air": 15 + 8 * np.sin(np.arange(480) / 24 * 2 * np.pi) + rng.normal(0, 0.3, 480),
        "rh": np.clip(70 + rng.normal(0, 8, 480), 0, 100),
    }, index=index)
    field_frame = pd.DataFrame({
        "t_air": station["t_air"] - 1.5,        # 계기 편차 -1.5도
        "rh": np.clip(station["rh"] + 4.0, 0, 100),
    }, index=index)
    return field_frame, station


def test_fit_recovers_injected_bias(paired_frames):
    field_frame, station = paired_frames
    calibration = calib.fit(field_frame, station, strength=1.0)
    assert calibration.biases["t_air"].reliable
    assert calibration.biases["t_air"].offset == pytest.approx(1.5, abs=0.1)
    assert calibration.biases["rh"].offset == pytest.approx(-4.0, abs=0.3)


def test_calibration_reduces_disagreement(paired_frames):
    field_frame, station = paired_frames
    calibration = calib.fit(field_frame, station, strength=1.0)
    report = calib.agreement_report(field_frame, station, calibration)
    assert (report["보정후 MAE"] < report["보정전 MAE"]).all()


def test_strength_zero_leaves_values_untouched(paired_frames):
    field_frame, station = paired_frames
    calibration = calib.fit(field_frame, station, strength=0.0)
    assert calibration.apply(field_frame)["t_air"].equals(field_frame["t_air"])


def test_short_overlap_is_not_trusted(paired_frames):
    """중복이 짧으면 편차를 믿지 않고 보정하지 않는다."""
    field_frame, station = paired_frames
    short = field_frame.iloc[:24]
    calibration = calib.fit(short, station, strength=1.0)
    assert not calibration.biases["t_air"].reliable
    assert calibration.apply(short)["t_air"].equals(short["t_air"])


def test_quantile_map_matches_reference_scale():
    """척도가 다른 측정값을 기준 분포 위로 옮긴다."""
    reference = pd.Series(np.linspace(0, 100, 500))      # 물수지 지수
    measured = pd.Series(np.linspace(25, 38, 300))       # 체적수분율 %
    mapped = calib.quantile_map(measured, reference)
    assert mapped.min() >= reference.min() - 1
    assert mapped.max() <= reference.max() + 1
    assert mapped.std() > measured.std() * 3             # 척도가 늘어난다
    assert mapped.is_monotonic_increasing                # 순서는 보존


def test_match_threshold_keeps_exceedance_rate():
    """임계값을 옮겨도 초과 비율이 보존된다."""
    reference = pd.Series(np.linspace(0, 100, 1000))
    measured = pd.Series(np.linspace(25, 38, 1000))
    moved = calib.match_threshold_by_quantile(reference, 30.0, measured)
    assert (reference <= 30).mean() == pytest.approx((measured <= moved).mean(), abs=0.02)


def test_replace_proxies_prefers_measured():
    index = pd.date_range("2025-10-01", periods=5, freq="h")
    model = pd.DataFrame({"canopy_temp": [1.0] * 5, "leaf_wetness": [0.0] * 5}, index=index)
    measured = pd.DataFrame({"canopy_temp": [-2.0] * 5, "leaf_wetness": [0.9] * 5}, index=index)
    out, replaced = calib.replace_proxies(model, measured)
    assert set(replaced) == {"canopy_temp", "leaf_wetness"}
    assert (out["canopy_temp"] == -2.0).all()


def test_station_health_detects_humidity_ceiling():
    """습도 최댓값이 낮게 묶이면 상한 포화로 본다."""
    ts = pd.date_range("2025-09-01", periods=720, freq="h")
    rng = np.random.default_rng(1)
    healthy = pd.DataFrame({
        "station": "정상지점", "ts": ts,
        "rh": np.clip(rng.uniform(40, 100, 720), 0, 100),
        "rain": 0.0, "t_air": 15.0, "solar_mj": 0.0,
    })
    broken = healthy.copy()
    broken["station"] = "포화지점"
    broken["rh"] = broken["rh"].clip(upper=89.0)

    report = quality.station_health(pd.concat([healthy, broken], ignore_index=True))
    by_station = report.set_index("station")
    assert by_station.loc["정상지점", "rh_ceiling_ok"]
    assert not by_station.loc["포화지점", "rh_ceiling_ok"]
    assert "습도 상한 포화" in by_station.loc["포화지점", "verdict"]
