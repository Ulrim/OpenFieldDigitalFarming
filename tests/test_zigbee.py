"""Zigbee2MQTT 수집 검증.

현장 센서는 라즈베리파이에 꽂은 코디네이터로 들어오고 Zigbee2MQTT 가
장치마다 JSON 을 내보낸다. 그것을 수집 표준 형식으로 바꾸는 경로다.
"""

from __future__ import annotations

import datetime as dt
import json

import pytest

from ofdf.data import zigbee
from ofdf.data.counter import CumulativeCounter
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

    # 사업계획서의 설계는 구간마다 초관부 온습도 2점·토양 2점이고, 비교구간에
    # 같은 구성을 대칭으로 둔다. 센서는 여러 대 붙는다 — 1점이면 그 센서가
    # 죽는 순간 그 구간의 판단이 빈다.
    for zone in ("treatment", "control"):
        assert sum(1 for d in canopy if d.zone == zone) >= 2, \
            f"{zone} 구간 초관부 온습도가 2점 미만이다"
        assert sum(1 for d in soil if d.zone == zone) >= 2, \
            f"{zone} 구간 토양 센서가 2점 미만이다"


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


# ── 상태값과 적산값 ────────────────────────────────────────────────
#
# 이 둘이 빠져 있던 것이 가장 큰 구멍이었다. 밸브와 차단기는 자기 상태를
# 문자열로 보내는데 '숫자가 아니라서' 버려지고 있었다. 그러면 명령은 보냈는데
# 실행됐는지 모르는 상태가 된다 — 공인시험 T-07 이 묻는 것이 바로 그것이다.

FACILITY = {
    "valve-treat-01": zigbee.Device(
        name="valve-treat-01", zone="facility", sensor_id="VALVE-01",
        map={"state": "valve_state", "flow": "irrigation_flow",
             "water_consumed": "irrigation_volume", "temperature": ""},
    ),
    "breaker-pump-01": zigbee.Device(
        name="breaker-pump-01", zone="facility", sensor_id="PUMP-01",
        map={"state": "circuit_state", "current": "pump_current",
             "power": "pump_power", "voltage": "line_voltage",
             "energy": "pump_energy", "temperature": ""},
    ),
}


def _values(rows):
    return {r["variable"]: r["value"] for r in rows}


def test_밸브_열림_상태를_받는다():
    rows = zigbee.message_to_rows(
        "zigbee2mqtt/valve-treat-01", {"state": "ON", "flow": 14.2}, FACILITY)
    assert _values(rows)["valve_state"] == 1.0
    assert _values(rows)["irrigation_flow"] == 14.2


def test_밸브_닫힘_상태를_받는다():
    rows = zigbee.message_to_rows(
        "zigbee2mqtt/valve-treat-01", {"state": "OFF"}, FACILITY)
    assert _values(rows)["valve_state"] == 0.0


@pytest.mark.parametrize("word,expected", [
    ("ON", 1.0), ("on", 1.0), ("OPEN", 1.0), ("opened", 1.0),
    ("OFF", 0.0), ("close", 0.0), ("CLOSED", 0.0),
    ("detected", 1.0), ("clear", 0.0), ("wet", 1.0), ("dry", 0.0),
])
def test_상태말을_대소문자_가리지_않고_읽는다(word, expected):
    assert zigbee.to_number(word) == expected


def test_모르는_상태말은_추측하지_않고_알린다():
    """밸브 상태를 거꾸로 읽으면 관수했다고 기록하면서 작물은 마른다."""
    unknown = []
    rows = zigbee.message_to_rows(
        "zigbee2mqtt/valve-treat-01", {"state": "부분개방"}, FACILITY,
        unknown=unknown)
    assert "valve_state" not in _values(rows)
    assert any("모르는 상태값" in note for note in unknown)


def test_참거짓도_받는다():
    assert zigbee.to_number(True) == 1.0
    assert zigbee.to_number(False) == 0.0


def test_숫자를_문자열로_보내는_장치도_받는다():
    assert zigbee.to_number("23.5") == 23.5


def test_빈_문자열과_사전은_버린다():
    assert zigbee.to_number("") is None
    assert zigbee.to_number({"a": 1}) is None
    assert zigbee.to_number(None) is None


def test_적산_전력량은_구간_차분이_된다():
    """적산값을 그대로 넣으면 5분 전력량이 설치 이후 총량으로 잡힌다."""
    counter = CumulativeCounter()
    first = zigbee.message_to_rows(
        "zigbee2mqtt/breaker-pump-01", {"energy": 51.20}, FACILITY,
        cumulative=counter)
    second = zigbee.message_to_rows(
        "zigbee2mqtt/breaker-pump-01", {"energy": 51.27}, FACILITY,
        cumulative=counter)
    assert _values(first)["pump_energy"] == 0.0        # 처음은 기준점
    assert _values(second)["pump_energy"] == pytest.approx(0.07)


def test_적산값이_줄면_그_구간은_0_이다():
    """센서 교체·초기화. 음수 관수량은 물수지를 망가뜨린다."""
    counter = CumulativeCounter()
    for total in (120.0, 141.5):
        zigbee.message_to_rows("zigbee2mqtt/valve-treat-01",
                               {"water_consumed": total}, FACILITY,
                               cumulative=counter)
    rows = zigbee.message_to_rows("zigbee2mqtt/valve-treat-01",
                                  {"water_consumed": 0.0}, FACILITY,
                                  cumulative=counter)
    assert _values(rows)["irrigation_volume"] == 0.0


def test_장치가_둘이면_적산을_따로_센다():
    counter = CumulativeCounter()
    zigbee.message_to_rows("zigbee2mqtt/breaker-pump-01", {"energy": 100.0},
                           FACILITY, cumulative=counter)
    rows = zigbee.message_to_rows("zigbee2mqtt/valve-treat-01",
                                  {"water_consumed": 5.0}, FACILITY,
                                  cumulative=counter)
    # 차단기의 100 이 밸브의 기준점이 되어 음수가 나오면 안 된다
    assert _values(rows)["irrigation_volume"] == 0.0


def test_설정이_버리라고_한_필드는_다시_묻지_않는다():
    """밸브 몸체 온도는 작물 판단에 쓰지 않는다고 설정이 이미 말했다."""
    unknown = []
    rows = zigbee.message_to_rows(
        "zigbee2mqtt/valve-treat-01", {"temperature": 21.0}, FACILITY,
        unknown=unknown)
    assert rows == []
    assert unknown == []


def test_차단기_설정값은_경고를_띄우지_않는다():
    """설정 임계치까지 '매핑 없음'으로 찍으면 정작 봐야 할 경고가 묻힌다."""
    unknown = []
    zigbee.message_to_rows(
        "zigbee2mqtt/breaker-pump-01",
        {"over_current_threshold": 63, "power_on_behavior": "previous",
         "current": 4.1},
        FACILITY, unknown=unknown)
    assert unknown == []


def test_설비_진단값을_남긴다():
    """차단기 과열은 화재 전조다. 작물 변수와 별도로 남긴다."""
    rows = zigbee.message_to_rows(
        "zigbee2mqtt/breaker-pump-01",
        {"device_temperature": 38, "power_factor": 98, "ac_frequency": 60},
        FACILITY)
    values = _values(rows)
    assert values["device_temp"] == 38.0
    assert values["power_factor"] == 98.0
    assert values["ac_frequency"] == 60.0


def test_모르는_필드는_값까지_적어_알린다():
    unknown = []
    zigbee.message_to_rows("zigbee2mqtt/valve-treat-01", {"무엇인가": 7},
                           FACILITY, unknown=unknown)
    assert any("무엇인가" in note and "7" in note for note in unknown)


def test_화면용_상태가_말로_나온다():
    """0.5 는 '밸브가 지금 열려 있나'의 답이 아니다. 마지막 값을 말로 쓴다."""
    frame = zigbee.rows_to_frame([
        {"ts": "2026-10-14T05:00:00", "zone": "facility", "sensor_id": "VALVE-01",
         "variable": "valve_state", "value": 0.0, "unit": "bool", "quality": "ok"},
        {"ts": "2026-10-14T05:05:00", "zone": "facility", "sensor_id": "VALVE-01",
         "variable": "valve_state", "value": 1.0, "unit": "bool", "quality": "ok"},
    ])
    status = zigbee.device_status(frame, FACILITY)
    states = status.iloc[0]["states"]
    assert {"name": "관수 밸브", "on": True, "text": "열림"} in states


def test_Zigbee가_아닌_장치도_센서_상태에_나온다():
    """433MHz 장치가 화면에서 빠지면 기상 마스트가 죽어도 아무도 모른다."""
    frame = zigbee.rows_to_frame([
        {"ts": "2026-10-14T05:00:00", "zone": "mast", "sensor_id": "WS90-01",
         "variable": "t_air", "value": 5.0, "unit": "degC", "quality": "ok"},
    ])
    status = zigbee.device_status(
        frame, {}, extra={"WS90-01": ("mast", "폴대 상단 2m")})
    assert status.iloc[0]["note"] == "폴대 상단 2m"
    assert status.iloc[0]["zone"] == "mast"


# ── 센서가 여러 대일 때 ─────────────────────────────────────────

KINDS_YAML = """
kinds:
  soil:
    note: 토양
    map: {temperature: soil_temp}
    sends: [soil_moisture, battery, linkquality]
  valve:
    map: {state: valve_state, flow: irrigation_flow, temperature: ""}
devices:
  soil-treat-01: {kind: soil, zone: treatment, sensor_id: SOIL-T1, note: 깊이 10cm}
  soil-treat-02: {kind: soil, zone: treatment, sensor_id: SOIL-T2}
  soil-ctrl-01:  {kind: soil, zone: control,   sensor_id: SOIL-C1}
  valve-a:       {kind: valve, zone: facility, sensor_id: VALVE-01}
  valve-b:       {kind: valve, zone: facility, sensor_id: VALVE-02,
                  map: {flow: spray_flow}}
  lone-01:       {zone: facility, sensor_id: LONE-01, map: {water_leak: enclosure_flood}}
"""


def test_종류에_적은_매핑을_장치가_물려받는다(tmp_path):
    """토양센서 넉 대에 같은 줄을 네 번 적지 않는다."""
    path = tmp_path / "d.yaml"
    path.write_text(KINDS_YAML, encoding="utf-8")
    devices = load_devices(path)
    for name in ("soil-treat-01", "soil-treat-02", "soil-ctrl-01"):
        assert devices[name].map == {"temperature": "soil_temp"}
        assert devices[name].sends == ("soil_moisture", "battery", "linkquality")
        assert devices[name].kind == "soil"
    assert devices["soil-treat-02"].note == "토양"          # 종류의 메모
    assert devices["soil-treat-01"].note == "깊이 10cm"      # 장치가 덮는다


def test_장치의_매핑이_종류_위에_덮인다(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(KINDS_YAML, encoding="utf-8")
    devices = load_devices(path)
    assert devices["valve-a"].map["flow"] == "irrigation_flow"
    assert devices["valve-b"].map["flow"] == "spray_flow"     # 덮였다
    assert devices["valve-b"].map["state"] == "valve_state"   # 나머지는 그대로


def test_종류_없이_직접_적어도_된다(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(KINDS_YAML, encoding="utf-8")
    devices = load_devices(path)
    assert devices["lone-01"].kind == ""
    assert devices["lone-01"].map == {"water_leak": "enclosure_flood"}


def test_없는_종류를_적으면_바로_알린다(tmp_path):
    """오타로 매핑이 통째로 빠지면 값이 조용히 버려진다. 읽을 때 막는다."""
    path = tmp_path / "d.yaml"
    path.write_text("devices:\n  x: {kind: soli, zone: treatment}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="soli"):
        load_devices(path)


def test_페어링_목록에서_설정에_없는_장치를_찾는다():
    payload = json.dumps([
        {"friendly_name": "Coordinator", "type": "Coordinator"},
        {"friendly_name": "soil-treat-01", "type": "EndDevice",
         "ieee_address": "0xa1",
         "definition": {"model": "TS0601_soil", "vendor": "Tuya",
                        "description": "Soil sensor"}},
        {"friendly_name": "0x00124b00aabbccdd", "type": "EndDevice",
         "ieee_address": "0x00124b00aabbccdd",
         "definition": {"model": "TS0201", "vendor": "Tuya",
                        "description": "Temperature & humidity sensor"}},
    ])
    paired = zigbee.parse_bridge_devices(payload)
    assert [d["friendly_name"] for d in paired] == ["soil-treat-01", "0x00124b00aabbccdd"]
    assert paired[1]["model"] == "TS0201"

    devices = {"soil-treat-01": Device(name="soil-treat-01")}
    missing = zigbee.unconfigured(paired, devices)
    assert [d["friendly_name"] for d in missing] == ["0x00124b00aabbccdd"]


def test_깨진_페어링_목록은_빈_목록이다():
    assert zigbee.parse_bridge_devices("{}") == []
    assert zigbee.parse_bridge_devices([1, "x", None]) == []
