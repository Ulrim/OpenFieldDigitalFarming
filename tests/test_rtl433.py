"""rtl_433(433MHz) 수집 검증 — Ecowitt WS90.

WS90 은 Zigbee 가 아니라 433MHz 전용 RF 다. RTL-SDR 동글로 rtl_433 이
받는다. 여기서 바로잡아야 할 것이 둘이다 — 돌풍의 통계량과 누적 강수.
"""

from __future__ import annotations

import json

import pytest

from ofdf.data.rtl433 import (
    VAR_GUST_WINDOW,
    RainAccumulator,
    message_to_rows,
    read_capture,
)


def ws90(**extra) -> dict:
    base = {"time": "2026-11-01 05:00:00", "model": "Fineoffset-WS90", "id": 9231,
            "temperature_C": 2.4, "humidity": 92, "wind_avg_m_s": 0.7,
            "wind_max_m_s": 1.9, "rain_mm": 13.2, "light_lux": 0,
            "battery_mV": 3100, "supercap_V": 4.1, "mic": "CRC"}
    base.update(extra)
    return base


def by_variable(rows) -> dict[str, float]:
    return {r["variable"]: r["value"] for r in rows}


def test_window_gust_is_not_recorded_as_the_three_second_gust():
    """사업계획서의 강풍 기준은 3초 순간최대다.

    WS90 의 wind_max_m_s 는 2초 간격으로 재어 최근 28초 중 최댓값이다.
    다른 통계량이므로 같은 이름으로 넣으면 안 된다 — 넣으면 농가 화면이
    '3초 순간최대'라고 적는데 실제로는 28초 창 값인 상태가 된다.
    """
    got = by_variable(message_to_rows(ws90()))
    assert got[VAR_GUST_WINDOW] == 1.9
    assert "wind_gust_3s" not in got
    assert VAR_GUST_WINDOW != "wind_gust_3s"


def test_average_wind_is_separate_from_the_gust():
    got = by_variable(message_to_rows(ws90()))
    assert got["wind_speed"] == 0.7 and got[VAR_GUST_WINDOW] == 1.9


def test_cumulative_rain_becomes_interval_rain():
    """rain_mm 은 초기화 이후 누적이다. 그대로 넣으면 시간당 수백 mm 가 된다."""
    rain = RainAccumulator()
    totals = [13.2, 13.2, 13.6, 14.0, 14.0]
    got = [by_variable(message_to_rows(ws90(rain_mm=t), rain=rain))["rain"]
           for t in totals]
    assert got == [0.0, 0.0, 0.4, 0.4, 0.0]


def test_rain_counter_reset_does_not_produce_negative_rain():
    """장치 교체·초기화로 값이 줄면 그 구간은 0 이다. 음수 강수는 물수지를 깬다."""
    rain = RainAccumulator()
    message_to_rows(ws90(rain_mm=120.0), rain=rain)
    got = by_variable(message_to_rows(ws90(rain_mm=0.2), rain=rain))
    assert got["rain"] == 0.0


def test_each_device_counts_its_own_rain():
    rain = RainAccumulator()
    message_to_rows(ws90(id=1, rain_mm=10.0), rain=rain)
    message_to_rows(ws90(id=2, rain_mm=50.0), rain=rain)
    a = by_variable(message_to_rows(ws90(id=1, rain_mm=10.5), rain=rain))
    assert a["rain"] == pytest.approx(0.5)


def test_power_state_is_kept_as_a_diagnostic():
    """태양광 슈퍼캡이 비어 가는 것이 송신이 끊기기 전의 신호다."""
    got = by_variable(message_to_rows(ws90()))
    assert got["supercap_v"] == 4.1 and got["battery_mv"] == 3100.0


def test_light_goes_to_illuminance_so_the_lux_conversion_applies():
    """WS90 은 일사계가 아니라 조도계다. 환산은 add_derived 가 한다."""
    got = by_variable(message_to_rows(ws90(light_lux=35197.0)))
    assert got["illuminance"] == 35197.0
    assert "solar" not in got


def test_rows_match_the_field_collector_format():
    from ofdf.data.field import LONG_COLUMNS

    rows = message_to_rows(ws90())
    assert rows and LONG_COLUMNS <= set(rows[0])
    assert all(r["zone"] == "mast" for r in rows)


def test_capture_replay_skips_broken_lines(tmp_path):
    path = tmp_path / "rtl.jsonl"
    path.write_text("\n".join([
        json.dumps(ws90(rain_mm=1.0)), "{깨진", json.dumps(ws90(rain_mm=1.5)),
    ]), encoding="utf-8")
    frame = read_capture(path)
    assert set(frame["variable"]) >= {"t_air", "rh", VAR_GUST_WINDOW, "rain"}
    assert float(frame[frame["variable"] == "rain"]["value"].iloc[-1]) == pytest.approx(0.5)


def test_unmapped_field_is_reported_not_silently_dropped():
    unknown: list[str] = []
    message_to_rows(ws90(brand_new_field=42.0), unknown=unknown)
    assert any("brand_new_field" in u for u in unknown)
