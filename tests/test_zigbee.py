"""Zigbee2MQTT 수집 검증.

현장 센서는 라즈베리파이에 꽂은 코디네이터로 들어오고 Zigbee2MQTT 가
장치마다 JSON 을 내보낸다. 그것을 수집 표준 형식으로 바꾸는 경로다.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from ofdf.data.zigbee import (
    AMBIGUOUS_FIELDS,
    Device,
    device_status,
    load_devices,
    message_to_rows,
    read_capture,
    rows_to_frame,
)

NOW = dt.datetime(2026, 10, 14, 5, 0)


def canopy() -> dict[str, Device]:
    return {"c1": Device(name="c1", zone="treatment", sensor_id="CN-01",
                         map={"temperature": "canopy_temp", "humidity": "rh"},
                         note="지상 10cm")}


def test_same_field_means_different_variables_by_mounting():
    """같은 temperature 라도 어디에 달렸느냐에 따라 다른 변수다.

    2m 마스트면 기온, 지상 10cm 면 초관부 온도, 흙에 꽂으면 지온이다.
    서리 판단은 이 셋을 구분하지 못하면 성립하지 않는다.
    """
    payload = {"temperature": 2.4}
    for mapping, expected in (
        ({"temperature": "t_air"}, "t_air"),
        ({"temperature": "canopy_temp"}, "canopy_temp"),
        ({"temperature": "soil_temp"}, "soil_temp"),
    ):
        dev = {"x": Device(name="x", sensor_id="X", map=mapping)}
        rows = message_to_rows("zigbee2mqtt/x", payload, dev, now=NOW)
        assert [r["variable"] for r in rows] == [expected]


def test_ambiguous_field_is_dropped_with_a_reason_not_guessed():
    """설정이 말해 주지 않으면 추측하지 않는다.

    기본값을 두면 지상 10cm 센서가 조용히 기온으로 들어가고, 아무도
    모르는 채로 서리 판단이 어긋난다.
    """
    dev = {"x": Device(name="x", sensor_id="X")}      # map 없음
    unknown: list[str] = []
    rows = message_to_rows(
        "zigbee2mqtt/x", {"temperature": 2.4, "soil_moisture": 30},
        dev, now=NOW, unknown=unknown,
    )
    assert [r["variable"] for r in rows] == ["soil_moisture"]
    assert any("temperature" in u for u in unknown)
    assert "temperature" in AMBIGUOUS_FIELDS


def test_battery_and_link_are_kept_as_diagnostics():
    """자료가 끊기기 전에 알려 주는 유일한 신호다."""
    rows = message_to_rows(
        "zigbee2mqtt/c1",
        {"temperature": 2.4, "battery": 87, "linkquality": 120},
        canopy(), now=NOW,
    )
    got = {r["variable"]: r["value"] for r in rows}
    assert got["battery_pct"] == 87.0
    assert got["linkquality"] == 120.0
    assert got["canopy_temp"] == 2.4


def test_booleans_become_numbers():
    dev = {"leak": Device(name="leak", zone="facility", sensor_id="LK")}
    rows = message_to_rows("zigbee2mqtt/leak", {"water_leak": True}, dev, now=NOW)
    assert rows[0]["variable"] == "enclosure_flood" and rows[0]["value"] == 1.0


def test_out_of_range_value_is_flagged_not_dropped():
    """버리면 왜 비었는지 알 수 없다. 표시하고 남긴다."""
    rows = message_to_rows("zigbee2mqtt/c1", {"temperature": 999.0},
                           canopy(), now=NOW)
    assert rows[0]["quality"] == "range"
    assert rows[0]["value"] == 999.0


def test_unknown_device_is_reported():
    unknown: list[str] = []
    rows = message_to_rows("zigbee2mqtt/nobody", {"temperature": 1.0},
                           canopy(), now=NOW, unknown=unknown)
    assert rows == []
    assert any("nobody" in u for u in unknown)


def test_bridge_topics_are_not_sensor_data():
    assert message_to_rows("zigbee2mqtt/bridge/state", {"state": "online"},
                           canopy(), now=NOW) == []


def test_string_states_are_skipped():
    """ON/OFF 같은 문자열은 수치 변수가 아니다."""
    dev = {"sw": Device(name="sw", zone="facility", sensor_id="SW",
                        map={"current": "pump_current"})}
    rows = message_to_rows("zigbee2mqtt/sw", {"state": "ON", "current": 4.2},
                           dev, now=NOW)
    assert [r["variable"] for r in rows] == ["pump_current"]


def test_capture_replay_survives_broken_lines(tmp_path):
    """한 줄이 깨져도 나머지를 읽는다 — JSON Lines 를 쓴 이유다."""
    path = tmp_path / "capture.jsonl"
    path.write_text("\n".join([
        json.dumps({"ts": NOW.isoformat(), "topic": "zigbee2mqtt/c1",
                    "payload": {"temperature": 2.4}}),
        "{깨진 줄",
        json.dumps({"ts": NOW.isoformat(), "topic": "zigbee2mqtt/c1",
                    "payload": {"temperature": 2.1}}),
    ]), encoding="utf-8")
    frame = read_capture(path, canopy())
    assert len(frame) == 2
    assert list(frame["variable"]) == ["canopy_temp", "canopy_temp"]


def test_collected_rows_match_the_field_collector_format():
    """ofdf.data.field 가 그대로 읽을 수 있어야 한다."""
    from ofdf.data.field import LONG_COLUMNS

    frame = rows_to_frame(message_to_rows(
        "zigbee2mqtt/c1", {"temperature": 2.4}, canopy(), now=NOW))
    assert LONG_COLUMNS <= set(frame.columns)


def test_device_status_surfaces_a_quietly_dead_sensor():
    """마지막 수신 시각이 있어야 조용히 죽은 센서를 찾는다."""
    dev = canopy()
    rows = []
    for minute in (0, 5, 10):
        rows += message_to_rows(
            "zigbee2mqtt/c1", {"temperature": 2.0, "battery": 40, "linkquality": 70},
            dev, now=NOW + dt.timedelta(minutes=minute))
    status = device_status(rows_to_frame(rows), dev)
    row = status.iloc[0]
    assert row["sensor_id"] == "CN-01"
    assert row["battery_pct"] == 40.0
    assert row["note"] == "지상 10cm"
    assert row["last_seen"] == NOW + dt.timedelta(minutes=10)


def test_shipped_device_config_matches_what_was_bought():
    """저장소의 설정이 실제 구매분과 맞아야 한다.

    뜻이 갈리는 temperature 는 장치마다 지정돼 있어야 한다. 하나라도
    비어 있으면 그 값은 버려지고 판단이 조용히 어긋난다.
    """
    from pathlib import Path

    devices = load_devices(Path(__file__).resolve().parents[1] / "configs" / "devices.yaml")
    assert devices, "장치가 비어 있다"

    # temperature 를 내보내는 장치는 모두 어느 변수인지 적혀 있어야 한다
    soil = [d for d in devices.values() if d.sensor_id.startswith("SOIL")]
    assert soil, "토양 센서가 없다"
    assert all(d.map.get("temperature") == "soil_temp" for d in soil), \
        "토양 센서의 temperature 는 지온이다"

    canopy = [d for d in devices.values()
              if d.map.get("temperature") == "canopy_temp"]
    assert canopy, "초관부 온도로 지정된 장치가 없다"
    assert {d.zone for d in canopy} == {"treatment"}, (
        "온습도 센서 구매분이 1대라 적용구간에만 있다. "
        "비교구간에도 달게 되면 이 기대값을 함께 고친다"
    )


def test_metering_breaker_voltage_is_line_voltage_not_battery():
    """배터리 장치의 voltage 는 mV, 차단기의 voltage 는 선간전압(V)이다.

    기본 매핑은 배터리 쪽이므로 차단기는 설정에서 덮어야 한다. 안 덮으면
    220V 가 배터리 전압 220mV 로 기록된다.
    """
    from pathlib import Path

    devices = load_devices(Path(__file__).resolve().parents[1] / "configs" / "devices.yaml")
    breakers = [d for d in devices.values() if d.map.get("current")]
    assert breakers, "전력계측 차단기가 없다"
    assert all(d.map.get("voltage") == "line_voltage" for d in breakers)


def test_valve_body_temperature_is_explicitly_discarded():
    """밸브 몸체 온도는 작물 판단에 쓰지 않는다. 빈 값으로 버린다고 적는다."""
    from pathlib import Path

    devices = load_devices(Path(__file__).resolve().parents[1] / "configs" / "devices.yaml")
    valve = next(d for d in devices.values() if d.sensor_id == "VALVE-01")
    assert valve.map.get("temperature") == ""
    assert valve.variable_for("temperature") is None
    assert valve.variable_for("flow") == "irrigation_flow"
