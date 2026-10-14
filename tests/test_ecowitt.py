"""에코윗 게이트웨이(와이파이) 수신 검증.

WS90 자체에는 와이파이가 없다. 게이트웨이가 433MHz 를 받아 와이파이로
HTTP POST 한다. 경로가 바뀌어도 센서는 같은 WS90 이다.
"""

from __future__ import annotations

import pytest

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
