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


# ── 유량·상태의 시간 집계 ──────────────────────────────────────────

def test_유량은_더하지_않는다():
    """유량은 비율이다.

    1분마다 잰 43 L/min 다섯 개를 더하면 216 L/min 이 된다. 단위는 L/min 인데
    값은 아닌 상태가 되고, 그 숫자가 화면에 그대로 나간다. 구간에 흘러간
    양을 묻는 자리는 적산 수량계(irrigation_volume)가 답한다.
    """
    assert field.TIME_AGGREGATION["irrigation_flow"] == "mean"
    assert field.TIME_AGGREGATION["spray_flow"] == "mean"
    assert field.TIME_AGGREGATION["irrigation_volume"] == "sum"

    stamps = pd.date_range("2026-10-14 02:30", periods=5, freq="1min")
    wide = pd.DataFrame({"facility.irrigation_flow": [43.0] * 5}, index=stamps)
    assert field.resample(wide, "5min").iloc[0, 0] == pytest.approx(43.0)


def test_상태는_평균이_아니라_최댓값으로_모은다():
    """5분 동안 한 번이라도 열렸으면 '열렸다'가 그 구간의 사실이다.

    제어응답 확인은 그것을 묻는다. 평균을 내면 0.4 가 나오는데, 밸브가
    0.4 만큼 열린 적은 없다.
    """
    for variable in ("valve_state", "circuit_state", "aux_circuit_state"):
        assert field.TIME_AGGREGATION[variable] == "max"

    stamps = pd.date_range("2026-10-14 02:30", periods=5, freq="1min")
    wide = pd.DataFrame({"facility.valve_state": [0.0, 0.0, 1.0, 1.0, 0.0]},
                        index=stamps)
    assert field.resample(wide, "5min").iloc[0, 0] == 1.0


def test_서로_다른_회로는_같은_변수를_쓰지_않는다():
    """관수펌프와 보조회로를 한 변수에 담으면 중앙값이 0.5 가 된다.

    측정점 통합(:func:`reconcile_points`)은 '같은 것을 여러 점에서 재는'
    경우를 위한 장치다. 서로 다른 회로는 그 경우가 아니다.
    """
    raw = pd.DataFrame({
        "ts": pd.to_datetime(["2026-10-14 02:30"] * 2),
        "zone": ["facility"] * 2,
        "sensor_id": ["PUMP-01", "AUX-01"],
        "variable": ["circuit_state", "aux_circuit_state"],
        "value": [1.0, 0.0],
    })
    out = field.reconcile_points(raw)
    values = dict(zip(out["variable"], out["value"]))
    assert values["circuit_state"] == 1.0
    assert values["aux_circuit_state"] == 0.0
    assert all(out["n_points"] == 1)       # 서로를 합치지 않았다


def test_밸브가_여럿이면_장치별로_남는다():
    """밸브 세 개는 세 라인이다. 하나가 열리고 둘이 닫혀 있을 때 중앙값 0 은
    '닫혔다'는 거짓이 된다. 장치별 컬럼으로 남아 서로를 덮지 않아야 한다."""
    stamp = pd.Timestamp("2026-10-14 02:30")
    raw = pd.DataFrame({
        "ts": [stamp] * 3,
        "zone": ["facility"] * 3,
        "sensor_id": ["VALVE-01", "VALVE-02", "VALVE-03"],
        "variable": ["valve_state"] * 3,
        "value": [1.0, 0.0, 0.0],
    })
    wide = field.to_wide(field.reconcile_points(raw))
    assert set(wide.columns) == {
        "facility.VALVE-01.valve_state", "facility.VALVE-02.valve_state",
        "facility.VALVE-03.valve_state",
    }
    assert wide.loc[stamp, "facility.VALVE-01.valve_state"] == 1.0
    assert wide.loc[stamp, "facility.VALVE-02.valve_state"] == 0.0


def test_같은_것을_재는_여러_점은_여전히_하나로_합친다():
    """초관부 온도 2점은 같은 구간의 같은 양이다. 중앙값이 맞다."""
    stamp = pd.Timestamp("2026-10-14 02:30")
    raw = pd.DataFrame({
        "ts": [stamp] * 3,
        "zone": ["treatment"] * 3,
        "sensor_id": ["CN-T1", "CN-T2", "CN-T3"],
        "variable": ["canopy_temp"] * 3,
        "value": [3.0, 3.4, 9.9],                  # 한 점이 그늘에서 벗어났다
    })
    out = field.reconcile_points(raw)
    assert len(out) == 1
    assert out.iloc[0]["value"] == 3.4             # 중앙값. 9.9 에 끌려가지 않는다
    assert out.iloc[0]["n_points"] == 3
    wide = field.to_wide(out)
    assert list(wide.columns) == ["treatment.canopy_temp"]


def test_장치별_컬럼은_모델_입력으로_새지_않는다():
    stamp = pd.Timestamp("2026-10-14 02:30")
    wide = pd.DataFrame({
        "treatment.canopy_temp": [3.0],
        "treatment.VALVE-09.valve_state": [1.0],   # 구간 안에 밸브를 두어도
        "mast.t_air": [5.0],
    }, index=[stamp])
    frame = field.zone_frame(wide, "treatment")
    assert "canopy_temp" in frame.columns
    assert "t_air" in frame.columns
    assert not any("VALVE" in c for c in frame.columns)


def test_시간_집계도_장치별_컬럼을_변수_규칙으로_모은다():
    """컬럼 이름이 세 마디라도 변수는 마지막 마디다. 적산은 더하고 상태는 최댓값."""
    stamps = pd.date_range("2026-10-14 02:30", periods=5, freq="1min")
    wide = pd.DataFrame({
        "facility.VALVE-01.irrigation_volume": [0.7] * 5,
        "facility.VALVE-01.valve_state": [0, 0, 1, 1, 0],
    }, index=stamps)
    agg = field.resample(wide, "5min")
    assert agg.iloc[0]["facility.VALVE-01.irrigation_volume"] == pytest.approx(3.5)
    assert agg.iloc[0]["facility.VALVE-01.valve_state"] == 1.0
