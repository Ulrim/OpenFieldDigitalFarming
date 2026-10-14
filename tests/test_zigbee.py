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


def test_shipped_device_config_is_loadable_and_maps_canopy():
    """저장소에 든 설정 예시가 실제로 읽히고, 초관부가 지정돼 있어야 한다."""
    from pathlib import Path

    devices = load_devices(Path(__file__).resolve().parents[1] / "configs" / "devices.yaml")
    assert devices, "장치가 비어 있다"
    canopy_devices = [d for d in devices.values()
                      if d.map.get("temperature") == "canopy_temp"]
    assert canopy_devices, "초관부 온도로 지정된 장치가 없다"
    # 적용구간과 비교구간에 대칭으로 있어야 한다
    assert {d.zone for d in canopy_devices} >= {"treatment", "control"}
