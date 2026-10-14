"""에코윗 게이트웨이(와이파이) 수신 검증.

WS90 자체에는 와이파이가 없다. 게이트웨이가 433MHz 를 받아 와이파이로
HTTP POST 한다. 경로가 바뀌어도 센서는 같은 WS90 이다.
"""

from __future__ import annotations

import pytest

from ofdf.data import ecowitt
from ofdf.data.ecowitt import DEFAULT_PATH, report_to_rows, reports_to_frame
from ofdf.data.rtl433 import VAR_GUST_WINDOW, RainAccumulator


def body(**over) -> str:
    fields = {
        "PASSKEY": "A1B2C3", "stationtype": "GW2000A_V3.1.7",
        "dateutc": "2026-11-01 05:00:00", "tempf": "36.3", "humidity": "92",
        "windspeedmph": "1.57", "windgustmph": "4.25", "winddir": "310",
        "solarradiation": "0.00", "uv": "0", "dailyrainin": "0.0",
        "baromrelin": "30.07", "ws90cap_volt": "4.1", "wh90batt": "3.10",
        "freq": "433M", "model": "WS90", "interval": "60",
    }
    fields.update({k: str(v) for k, v in over.items()})
    return "&".join(f"{k}={v}".replace(" ", "+") for k, v in fields.items())


def got(**over) -> dict[str, float]:
    return {r["variable"]: r["value"] for r in report_to_rows(body(**over))}


def test_imperial_units_are_converted():
    """게이트웨이는 웹UI 표시 설정과 무관하게 야드파운드로 보낸다.

    그대로 넣으면 기온이 36도로 잡힌다. 실제로는 2.4도다.
    """
    v = got()
    assert v["t_air"] == pytest.approx(2.389, abs=1e-3)      # 36.3°F
    assert v["wind_speed"] == pytest.approx(0.702, abs=1e-3)  # 1.57 mph
    assert v["pressure_hpa"] == pytest.approx(1018.287, abs=1e-2)  # 30.07 inHg


def test_gust_is_the_window_maximum_not_the_three_second_gust():
    """경로가 바뀌었다고 통계량이 바뀌지는 않는다. 같은 WS90 이다."""
    v = got()
    assert v[VAR_GUST_WINDOW] == pytest.approx(1.900, abs=1e-3)
    assert "wind_gust_3s" not in v


def test_daily_cumulative_rain_becomes_interval_rain():
    """dailyrainin 은 그날 누적이고 자정에 0 으로 돌아간다."""
    rain = RainAccumulator()
    totals = [0.0, 0.04, 0.08, 0.08, 0.0]        # 마지막은 자정 초기화
    out = [report_to_rows(body(dailyrainin=t), rain=rain) for t in totals]
    values = [{r["variable"]: r["value"] for r in rows}["rain"] for rows in out]
    assert values[0] == 0.0
    assert values[1] == pytest.approx(1.016, abs=1e-3)   # 0.04 inch
    assert values[2] == pytest.approx(1.016, abs=1e-3)
    assert values[3] == 0.0                              # 안 늘었다
    assert values[4] == 0.0                              # 줄었다 — 음수가 아니다


def test_gateway_gives_irradiance_directly_unlike_rtl433():
    """게이트웨이는 W/m^2 를 계산해 준다. rtl_433 은 조도(lux)만 준다."""
    v = got(solarradiation="412.5")
    assert v["solar"] == 412.5
    assert "illuminance" not in v


def test_station_metadata_is_not_treated_as_a_measurement():
    v = got()
    for key in ("PASSKEY", "stationtype", "interval", "freq", "runtime"):
        assert key not in v


def test_unmapped_field_is_reported():
    unknown: list[str] = []
    report_to_rows(body(brandnewfield="7"), unknown=unknown)
    assert any("brandnewfield" in u for u in unknown)


def test_observation_time_comes_from_the_gateway():
    """파이 시계가 틀어져도 관측 시각이 유지돼야 한다."""
    rows = report_to_rows(body(dateutc="2026-11-01 05:00:00"))
    assert rows[0]["ts"].startswith("2026-11-01T05:00:00")


def test_rows_match_the_field_collector_format():
    from ofdf.data.field import LONG_COLUMNS

    rows = report_to_rows(body())
    assert rows and LONG_COLUMNS <= set(rows[0])
    assert all(r["zone"] == "mast" for r in rows)


def test_many_reports_keep_counting_rain_in_order():
    frame = reports_to_frame([body(dailyrainin=t) for t in (0.0, 0.04, 0.08)])
    rain = frame[frame["variable"] == "rain"]["value"].tolist()
    assert rain == [0.0, pytest.approx(1.016, abs=1e-3), pytest.approx(1.016, abs=1e-3)]


def test_default_path_matches_what_the_gateway_ui_expects():
    assert DEFAULT_PATH == "/data/report/"


# ── 압전 강수와 보조 채널 ──────────────────────────────────────────
#
# WS90 은 티핑버킷이 아니라 압전식이라 강수 필드 이름이 다르다. 티핑버킷
# 이름만 받아 두면 와이파이 경로에서 강수가 통째로 비는데, **비가 와도
# 화면에 0 으로 나오므로 아무도 눈치채지 못한다.**

def _values(rows):
    return {r["variable"]: r["value"] for r in rows}


def test_압전_강수를_받는다():
    rows = ecowitt.report_to_rows({"drain_piezo": "0.08", "rrain_piezo": "0.004"})
    values = _values(rows)
    assert values["rain"] == 0.0                      # 첫 보고는 기준점
    assert values["rain_rate"] == pytest.approx(0.102, abs=0.001)


def test_압전_강수가_구간값이_된다():
    rain = ecowitt.RainAccumulator()
    ecowitt.report_to_rows({"drain_piezo": "0.08"}, rain=rain)
    rows = ecowitt.report_to_rows({"drain_piezo": "0.12"}, rain=rain)
    # 0.04 inch = 1.016 mm
    assert _values(rows)["rain"] == pytest.approx(1.016, abs=0.001)


def test_강우감지를_받는다():
    """사업계획서가 요구하는 '강우 감지'다. 압전식이라 티핑버킷보다 빠르다."""
    assert _values(ecowitt.report_to_rows({"srain_piezo": "1"}))["rain_detect"] == 1.0
    assert _values(ecowitt.report_to_rows({"srain_piezo": "0"}))["rain_detect"] == 0.0


def test_압전과_티핑버킷이_같이_오면_한_번만_센다():
    """같은 비를 두 번 더하면 5분 강수가 두 배가 된다."""
    rain = ecowitt.RainAccumulator()
    ecowitt.report_to_rows({"drain_piezo": "0.08", "dailyrainin": "0.09"}, rain=rain)
    rows = ecowitt.report_to_rows(
        {"drain_piezo": "0.12", "dailyrainin": "0.13"}, rain=rain)
    values = _values(rows)
    assert sum(1 for r in rows if r["variable"] == "rain") == 1
    assert values["rain"] == pytest.approx(1.016, abs=0.001)
    # 쓰지 않은 쪽도 버리지 않는다 — 두 계측기 차이는 현장에서 따질 거리다
    assert "rain_alt_total" in values


def test_보조_채널의_토양수분을_받는다():
    rows = ecowitt.report_to_rows({"soilmoisture1": "34", "soilbatt1": "1.5"})
    row = next(r for r in rows if r["variable"] == "soil_moisture")
    assert row["value"] == 34.0
    assert row["sensor_id"] == "ECO-SOIL1"


def test_채널_번호는_센서_계열마다_따로_센다():
    """soilmoisture1 과 leafwetness_ch1 과 temp1f 은 모두 '1번'이지만 다른 장치다.

    번호만 보고 묶으면 셋이 한 센서가 되어 배터리 값이 서로를 덮어쓴다.
    """
    rows = ecowitt.report_to_rows(
        {"soilmoisture1": "34", "soilbatt1": "1.5",
         "leafwetness_ch1": "12", "leaf_batt1": "1.2"})
    by_sensor = {}
    for row in rows:
        by_sensor.setdefault(row["sensor_id"], {})[row["variable"]] = row["value"]
    assert by_sensor["ECO-SOIL1"]["battery_v"] == 1.5
    assert by_sensor["ECO-LEAF1"]["battery_v"] == 1.2


def test_엽면습윤은_비율로_바꾼다():
    """게이트웨이는 0~100% 로 주고 우리 변수는 0~1 비율이다."""
    rows = ecowitt.report_to_rows({"leafwetness_ch1": "12"})
    assert _values(rows)["leaf_wetness"] == pytest.approx(0.12)


def test_지온_프로브를_화씨에서_바꾼다():
    rows = ecowitt.report_to_rows({"tf_ch2": "55.4"})
    row = next(r for r in rows if r["variable"] == "soil_temp")
    assert row["value"] == pytest.approx(13.0, abs=0.01)
    assert row["sensor_id"] == "ECO-TF2"


def test_범용_온습도계는_설정_없이_받지_않는다():
    """temp1f 는 2m 마스트면 기온이고 지상 10cm 면 초관부 온도다."""
    unknown = []
    rows = ecowitt.report_to_rows({"temp1f": "50.1", "humidity1": "71"},
                                  unknown=unknown)
    assert _values(rows) == {}
    assert len(unknown) == 2
    assert all("ecowitt.channels.th1" in note for note in unknown)


def test_설정에_적으면_범용_온습도계도_받는다(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(
        "ecowitt:\n"
        "  gateway: {zone: mast, sensor_id: WS90-01}\n"
        "  channels:\n"
        "    th1:\n"
        "      zone: control\n"
        "      sensor_id: CN-02\n"
        "      note: 비교구간 지상 10cm\n"
        "      map: {t_air: canopy_temp}\n",
        encoding="utf-8")
    gateway, channels = ecowitt.load_channels(path)
    assert gateway.sensor_id == "WS90-01"
    assert channels["th1"].zone == "control"

    unknown = []
    rows = ecowitt.report_to_rows(
        {"temp1f": "50.1", "humidity1": "71"},
        gateway=gateway, channels=channels, unknown=unknown)
    by = {r["variable"]: r for r in rows}
    assert by["canopy_temp"]["value"] == pytest.approx(10.06, abs=0.01)
    assert by["canopy_temp"]["zone"] == "control"
    assert by["canopy_temp"]["sensor_id"] == "CN-02"
    assert by["rh"]["sensor_id"] == "CN-02"
    assert unknown == []


def test_게이트웨이_설정이_구간과_이름을_정한다(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(
        "ecowitt:\n  gateway: {zone: treatment, sensor_id: GW-01}\n",
        encoding="utf-8")
    gateway, channels = ecowitt.load_channels(path)
    rows = ecowitt.report_to_rows({"tempf": "68.0"}, gateway=gateway,
                                  channels=channels)
    assert rows[0]["zone"] == "treatment"
    assert rows[0]["sensor_id"] == "GW-01"


def test_설정에_없는_채널도_받되_기본_이름을_쓴다():
    """센서를 꽂았는데 설정을 안 적었다고 자료를 버리면 안 된다."""
    rows = ecowitt.report_to_rows({"soilmoisture5": "22"})
    assert rows[0]["sensor_id"] == "ECO-SOIL5"


def test_채널_설정이_빈_값이면_버린다(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(
        "ecowitt:\n  channels:\n    soil1:\n      map: {soil_moisture: ''}\n",
        encoding="utf-8")
    _, channels = ecowitt.load_channels(path)
    rows = ecowitt.report_to_rows({"soilmoisture1": "34"}, channels=channels)
    assert rows == []


def test_실내_온습도는_진단으로_남긴다():
    """게이트웨이 함체 온도다. 작물 판단에 쓰면 안 되지만 버릴 것도 아니다."""
    rows = ecowitt.report_to_rows({"tempinf": "68.2", "humidityin": "41"})
    values = _values(rows)
    assert values["enclosure_temp"] == pytest.approx(20.11, abs=0.01)
    assert values["enclosure_rh"] == 41.0
    assert "t_air" not in values


def test_모르는_필드는_값까지_적어_알린다():
    unknown = []
    ecowitt.report_to_rows({"무엇인가": "7"}, unknown=unknown)
    assert any("무엇인가" in note for note in unknown)
