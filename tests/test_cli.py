"""엣지 실행 루프 검증."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ofdf import cli
from ofdf.action.journal import Journal
from ofdf.action.recommend import SensorState
from ofdf.features.derived import add_derived


@pytest.fixture
def field_dir(tmp_path):
    """현장 수집 디렉터리를 흉내 낸다 — 서리가 오는 밤."""
    ts = pd.date_range("2025-11-20", periods=60 * 60, freq="1min")
    rows = []
    hours = ts.hour.to_numpy()
    night = (hours >= 18) | (hours < 8)
    base_t = np.where(night, 3.0, 12.0)

    for zone, variables in [
        ("mast", {"t_air": base_t, "rh": 95.0, "solar": np.where(night, 0.0, 400.0),
                  "rain": 0.0, "wind_speed": 0.5, "wind_gust_3s": 1.0}),
        ("treatment", {"t_air": base_t - 0.5, "rh": 97.0,
                       "canopy_temp": base_t - np.where(night, 2.5, 0.0),
                       "leaf_wetness": 0.9, "soil_moisture": 30.0}),
    ]:
        for variable, value in variables.items():
            rows.append(pd.DataFrame({
                "ts": ts, "zone": zone, "sensor_id": f"{zone}-{variable}",
                "variable": variable, "value": value, "quality": "ok",
            }))
    pd.concat(rows, ignore_index=True).to_csv(tmp_path / "field.csv", index=False)
    return tmp_path


@pytest.fixture
def models(tmp_path):
    """작은 모델 4종을 만들어 둔다."""
    import lightgbm as lgb

    rng = np.random.default_rng(0)
    out = tmp_path / "models"
    out.mkdir()
    index = pd.date_range("2025-10-01", periods=900, freq="h")
    frame = pd.DataFrame({
        "t_air": rng.normal(10, 6, 900), "rh": rng.uniform(40, 100, 900),
        "rain": 0.0, "solar_w": 0.0, "wind_speed": 1.0,
        "canopy_temp": rng.normal(8, 6, 900), "soil_index": 50.0,
        "leaf_wetness": 1.0, "vpd": 0.3, "dew_point": 5.0,
        "cloud_cover": 0.3, "et0": 0.1,
    }, index=index)
    from ofdf.features.weather import build_features
    features = build_features(add_derived(frame))

    for risk in cli.BASE_RISKS:
        y = (features["t_air"] < 5).astype(int) if risk == "frost" else np.zeros(len(features), int)
        y[:3] = [0, 1, 2]                       # 세 등급이 모두 나오게 한다
        booster = lgb.train(
            {"objective": "multiclass", "num_class": 3, "verbose": -1, "num_leaves": 4},
            lgb.Dataset(features, label=y), num_boost_round=5,
        )
        booster.save_model(str(out / f"model_{risk}_h3.txt"))
    return out


def test_once_writes_journal_and_dashboard(field_dir, models, tmp_path):
    journal = tmp_path / "journal.jsonl"
    dashboard = tmp_path / "data.json"
    code = cli.main([
        "once", "--field", str(field_dir), "--models", str(models),
        "--journal", str(journal), "--dashboard", str(dashboard),
    ])
    assert code == 0
    assert journal.exists() and dashboard.exists()

    record = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
    assert record["station"] and record["model_version"].startswith("ofdf-weather-")
    assert "recommended" in record

    payload = json.loads(dashboard.read_text(encoding="utf-8"))
    assert {"station", "now", "mode", "cards", "actions"} <= payload.keys()
    assert len(payload["cards"]) == len(cli.BASE_RISKS) + 1      # 복합위험 포함


def test_missing_models_fail_loudly(field_dir, tmp_path):
    assert cli.main([
        "once", "--field", str(field_dir), "--models", str(tmp_path / "없음"),
        "--journal", str(tmp_path / "j.jsonl"), "--dashboard", "",
    ]) == 1


def test_missing_field_data_does_not_crash(models, tmp_path):
    """수집 파일이 없어도 예외로 죽지 않는다 — 제어기는 계속 돌아야 한다."""
    assert cli.main([
        "once", "--field", str(tmp_path / "없음"), "--models", str(models),
        "--journal", str(tmp_path / "j.jsonl"), "--dashboard", "",
    ]) == 1


def test_sensors_without_gust_do_not_fake_wind():
    """3초 순간최대풍속이 없으면 강풍을 판정하지 않는다.

    평균 풍속으로 순간최대 기준을 대신하면 강풍 안전규칙이 틀린 값으로 돈다.
    """
    index = pd.date_range("2025-11-20 22:00", periods=2, freq="h")
    frame = pd.DataFrame({"wind_speed": [3.0, 3.0], "rain": [0.0, 0.0]}, index=index)
    state = cli.sensors_from(frame, pd.Timestamp("2025-11-20 23:00"))
    assert state.gust_3s is None


def test_sensors_use_measured_gust():
    index = pd.date_range("2025-11-20 22:00", periods=2, freq="h")
    frame = pd.DataFrame({"wind_gust_3s": [2.0, 9.4], "rain": [0.0, 0.0]}, index=index)
    state = cli.sensors_from(frame, pd.Timestamp("2025-11-20 23:00"))
    assert state.gust_3s == pytest.approx(9.4)


def test_dashboard_write_is_atomic(tmp_path):
    """화면이 읽는 중 깨진 파일을 보지 않도록 임시파일을 거친다."""
    path = tmp_path / "data.json"
    cli.write_dashboard(path, {"a": 1})
    cli.write_dashboard(path, {"a": 2})
    assert json.loads(path.read_text(encoding="utf-8"))["a"] == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_add_derived_accepts_both_solar_units():
    """관측소는 MJ/m², 현장 일사계는 W/m² 를 준다. 둘 다 받아야 한다."""
    index = pd.date_range("2025-10-01", periods=3, freq="h")
    base = {"t_air": 15.0, "rh": 70.0, "rain": 0.0, "wind_speed": 1.0, "et0": 0.1}

    station = add_derived(pd.DataFrame({**base, "solar_mj": 1.0}, index=index))
    field = add_derived(pd.DataFrame({**base, "solar": 277.8}, index=index))
    # WS90 은 일사계가 아니라 조도계라 lux 로 들어온다.
    lux = add_derived(pd.DataFrame({**base, "illuminance": 35197.0}, index=index))

    assert station["solar_w"].iloc[0] == pytest.approx(277.8, rel=1e-3)
    assert field["solar_w"].iloc[0] == pytest.approx(277.8, rel=1e-3)
    assert lux["solar_w"].iloc[0] == pytest.approx(277.8, rel=1e-3)


def test_measured_irradiance_wins_over_lux_conversion():
    """lux 환산은 근사다. 일사계 실측이 있으면 그쪽을 써야 한다."""
    index = pd.date_range("2025-10-01", periods=2, freq="h")
    frame = pd.DataFrame(
        {
            "t_air": 15.0, "rh": 70.0, "rain": 0.0, "wind_speed": 1.0, "et0": 0.1,
            "solar": 500.0,
            "illuminance": 35197.0,   # 환산하면 277.8 W/m^2 — 쓰이면 안 된다
        },
        index=index,
    )
    assert add_derived(frame)["solar_w"].iloc[0] == pytest.approx(500.0)


def test_measured_canopy_temp_is_not_overwritten_by_proxy():
    """초관부 온도센서를 달았으면 추정값이 아니라 실측으로 판단해야 한다."""
    index = pd.date_range("2025-10-01", periods=3, freq="h")
    frame = pd.DataFrame(
        {
            "t_air": 2.0, "rh": 90.0, "rain": 0.0,
            "wind_speed": 0.5, "solar": 0.0, "et0": 0.1,
            # 복사냉각으로 2m 기온보다 훨씬 낮게 측정된 값
            "canopy_temp": [-2.5, -3.0, np.nan],
        },
        index=index,
    )
    out = add_derived(frame)

    assert out["canopy_temp"].iloc[0] == pytest.approx(-2.5)
    assert out["canopy_temp"].iloc[1] == pytest.approx(-3.0)
    assert list(out["canopy_temp_source"]) == ["measured", "measured", "proxy"]
    # 결측 시각은 대리지표로 메워서 판단이 비지 않아야 한다.
    assert not pd.isna(out["canopy_temp"].iloc[2])


def test_proxy_is_used_when_sensor_column_is_absent_or_empty():
    """센서가 없거나 통째로 결측이면 종전처럼 대리지표로 돌아간다."""
    index = pd.date_range("2025-10-01", periods=2, freq="h")
    base = {
        "t_air": 2.0, "rh": 90.0, "rain": 0.0,
        "wind_speed": 0.5, "solar": 0.0, "et0": 0.1,
    }
    absent = add_derived(pd.DataFrame(base, index=index))
    empty = add_derived(pd.DataFrame({**base, "canopy_temp": np.nan}, index=index))

    assert set(absent["canopy_temp_source"]) == {"proxy"}
    assert set(empty["canopy_temp_source"]) == {"proxy"}
    assert absent["canopy_temp"].notna().all()
    assert absent["canopy_temp"].equals(empty["canopy_temp"])


def test_source_columns_do_not_leak_into_model_features():
    """출처 컬럼은 기록용이다. 모델 입력으로 새면 안 된다."""
    from ofdf.features.weather import build_features

    index = pd.date_range("2025-10-01", periods=4, freq="h")
    frame = pd.DataFrame(
        {
            "t_air": 2.0, "rh": 90.0, "rain": 0.0,
            "wind_speed": 0.5, "solar": 0.0, "et0": 0.1,
            "canopy_temp": -2.0, "leaf_wetness": 1.0,
        },
        index=index,
    )
    features = build_features(add_derived(frame))
    assert not [c for c in features.columns if c.endswith("_source")]
