"""rtl_433(433MHz) 수집 검증 — Ecowitt WS90.

WS90 은 Zigbee 가 아니라 433MHz 전용 RF 다. RTL-SDR 동글로 rtl_433 이
받는다. 여기서 바로잡아야 할 것이 둘이다 — 돌풍의 통계량과 누적 강수.
"""

from __future__ import annotations

import json

import pytest

from ofdf.data import rtl433
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


# ── 설정 기반 장치표와 모델별 필드 ────────────────────────────────
#
# 433MHz 는 페어링이 없다. 전파만 받으면 들어오므로 이웃집 온도계도 들어온다.
# 그래서 받을 모델을 설정에서 정하고, 어디에 달렸는지도 거기 적는다.

def test_설정에서_구간과_이름을_읽는다(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(
        "rf:\n"
        "  Fineoffset-WS90:\n"
        "    zone: mast\n"
        "    sensor_id: WS90-01\n"
        "    note: 폴대 상단 2m\n",
        encoding="utf-8")
    devices = rtl433.load_rf_devices(path)
    device = devices["Fineoffset-WS90"]
    assert device.sensor_id == "WS90-01"
    assert device.note == "폴대 상단 2m"

    rows = rtl433.message_to_rows(
        {"model": "Fineoffset-WS90", "id": 1, "temperature_C": 2.4},
        device=device)
    assert rows[0]["zone"] == "mast"
    assert rows[0]["sensor_id"] == "WS90-01"


def test_같은_모델이_둘이면_id_로_가른다(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(
        "rf:\n"
        "  Fineoffset-WH51/111:\n"
        "    zone: treatment\n"
        "    sensor_id: SOIL-T3\n"
        "  Fineoffset-WH51/222:\n"
        "    zone: control\n"
        "    sensor_id: SOIL-C3\n",
        encoding="utf-8")
    devices = rtl433.load_rf_devices(path)
    first = rtl433.match_device({"model": "Fineoffset-WH51", "id": 111}, devices)
    second = rtl433.match_device({"model": "Fineoffset-WH51", "id": 222}, devices)
    assert first.sensor_id == "SOIL-T3"
    assert second.sensor_id == "SOIL-C3"
    assert rtl433.match_device({"model": "Acme-Other", "id": 9}, devices) is None


def test_설정의_매핑이_기본값을_덮는다(tmp_path):
    """같은 temperature_C 라도 흙에 꽂으면 지온이다."""
    path = tmp_path / "d.yaml"
    path.write_text(
        "rf:\n"
        "  Fineoffset-WH31:\n"
        "    zone: control\n"
        "    sensor_id: CN-02\n"
        "    map: {temperature_C: canopy_temp}\n",
        encoding="utf-8")
    devices = rtl433.load_rf_devices(path)
    rows = rtl433.message_to_rows(
        {"model": "Fineoffset-WH31", "id": 1, "temperature_C": 9.0},
        device=devices["Fineoffset-WH31"])
    assert rows[0]["variable"] == "canopy_temp"


def test_모델별_필드가_WS90_의_실제_구성과_맞다():
    """WS90 은 엽면습윤도 토양수분도 재지 않는다.

    환산표에 그 이름이 있는 것은 WH35·WH51 을 사면 같은 동글로 들어오기
    때문이다. 모델별 필드와 환산표를 섞으면 없는 센서를 있다고 적게 된다.
    """
    ws90 = rtl433.MODEL_FIELDS["Fineoffset-WS90"]
    assert "temperature_C" in ws90
    assert "wind_max_m_s" in ws90
    assert "rain_mm" in ws90
    assert "leaf_wetness" not in ws90
    assert "moisture" not in ws90
    assert "soil_moisture" not in ws90


def test_토양수분계를_사면_코드를_고치지_않는다():
    """WH51 은 moisture 로 보낸다. 환산표에 이미 있다."""
    rows = rtl433.message_to_rows(
        {"model": "Fineoffset-WH51", "id": 7, "moisture": 31, "battery_mV": 1500},
        sensor_id="SOIL-T3")
    values = {r["variable"]: r["value"] for r in rows}
    assert values["soil_moisture"] == 31.0
    assert values["battery_mv"] == 1500.0


def test_수신세기를_남긴다():
    """전파가 약해지는 것은 서서히 진행한다. 끊긴 뒤가 아니라 그전에 보여야 한다."""
    rows = rtl433.message_to_rows(
        {"model": "Fineoffset-WS90", "id": 1, "rssi": -7.5, "snr": 12.0})
    values = {r["variable"]: r["value"] for r in rows}
    assert values["rssi"] == -7.5
    assert values["snr"] == 12.0


def test_참거짓_배터리도_받는다():
    rows = rtl433.message_to_rows(
        {"model": "Fineoffset-WS90", "id": 1, "battery_ok": True})
    assert rows[0]["variable"] == "battery_ok"
    assert rows[0]["value"] == 1.0


def test_복조_부산물은_경고를_띄우지_않는다():
    unknown = []
    rtl433.message_to_rows(
        {"model": "Fineoffset-WS90", "id": 1, "mic": "CRC", "data": "ab12",
         "flags": 0, "channel": 1, "temperature_C": 2.0},
        unknown=unknown)
    assert unknown == []
