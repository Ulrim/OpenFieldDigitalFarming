"""Zigbee2MQTT 출력을 현장 수집 표준 형식으로 바꾼다.

실증포장의 Zigbee 센서는 라즈베리파이에 꽂은 코디네이터(SONOFF Dongle)로
들어오고, Zigbee2MQTT 가 장치마다 JSON 한 덩어리를 MQTT 로 내보낸다::

    zigbee2mqtt/soil-treat-01  {"soil_moisture":32,"temperature":14.2,"battery":87,...}

:mod:`ofdf.data.field` 가 읽는 형식은 세로(long)다::

    ts,zone,sensor_id,variable,value,unit,quality

이 모듈이 그 사이를 잇는다. **장치 이름과 변수 이름은 설정이 정한다.**
코드에 박으면 센서를 하나 옮길 때마다 코드를 고쳐야 하기 때문이다.

왜 장치마다 매핑이 필요한가
---------------------------
같은 ``temperature`` 라도 어디에 달렸느냐에 따라 다른 변수다. 2m 마스트에
달면 기온(``t_air``)이고, 지상 10cm 에 달면 초관부 온도(``canopy_temp``)이며,
흙에 꽂으면 지온(``soil_temp``)이다. 서리 판단은 이 셋을 구분하지 못하면
성립하지 않는다. 그래서 장치별로 어느 변수인지 설정에 적는다.

통신 상태도 함께 남긴다
-----------------------
``battery`` 와 ``linkquality`` 는 작물 위험과 무관하지만, **자료가 끊기기 전에
알려 주는 유일한 신호**다. 배터리가 떨어지거나 전파가 약해지는 것은 서서히
진행되므로, 값이 사라진 뒤가 아니라 사라지기 전에 보여야 한다.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from ofdf.data.field import FIELD_VARIABLES, ZONE_TREATMENT

#: Zigbee2MQTT 필드 이름 -> 기본 변수 이름.
#:
#: 설치 위치로 뜻이 갈리는 것(``temperature``·``humidity``)은 일부러 빼
#: 두었다. 기본값을 두면 지상 10cm 센서가 조용히 기온으로 들어간다.
#: 그런 센서는 설정에 어느 변수인지 적어야 한다.
Z2M_DEFAULT_MAP: dict[str, str] = {
    "soil_moisture": "soil_moisture",
    "soil_temperature": "soil_temp",
    "illuminance_lux": "illuminance",
    "illuminance": "illuminance",
    "water_leak": "enclosure_flood",
    "rain": "rain",
    "leaf_wetness": "leaf_wetness",
    "current": "pump_current",
    "power": "pump_power",
    "voltage_rms": "line_voltage",
    "flow": "irrigation_flow",
    "water_flow": "irrigation_flow",
}

#: 뜻이 설치 위치로 갈리는 필드. 설정에 매핑이 없으면 버리고 경고한다.
AMBIGUOUS_FIELDS = frozenset({"temperature", "humidity"})

#: 작물 위험 판단에는 안 쓰지만 통신·전원 상태로 남기는 필드.
DIAGNOSTIC_FIELDS: dict[str, str] = {
    "battery": "battery_pct",
    #: 배터리 장치의 voltage 는 배터리 전압(mV)이다. 전력계측 차단기의
    #: voltage 는 선간전압(V)이라 뜻이 다르므로, 그런 장치는 설정에서
    #: ``voltage: line_voltage`` 로 지정한다.
    "voltage": "battery_mv",
    "linkquality": "linkquality",
}

#: 진단 변수의 단위
DIAGNOSTIC_UNITS = {"battery_pct": "%", "battery_mv": "mV", "linkquality": "lqi"}


@dataclass
class Device:
    """센서 한 대의 설치 정보."""

    name: str                                   #: Zigbee2MQTT friendly_name
    zone: str = ZONE_TREATMENT
    sensor_id: str = ""
    #: Z2M 필드 -> 우리 변수. 기본 매핑을 덮는다.
    map: dict[str, str] = dc_field(default_factory=dict)
    #: 설치 높이·깊이 등 사람이 읽을 메모. 화면에 그대로 나간다.
    note: str = ""

    def __post_init__(self) -> None:
        if not self.sensor_id:
            self.sensor_id = self.name

    def variable_for(self, field_name: str) -> str | None:
        """Z2M 필드 하나가 어느 변수인지 정한다. 모르면 ``None``."""
        if field_name in self.map:
            return self.map[field_name] or None
        if field_name in AMBIGUOUS_FIELDS:
            return None          # 설정이 말해 주지 않으면 추측하지 않는다
        return Z2M_DEFAULT_MAP.get(field_name)


def load_devices(path: str | Path) -> dict[str, Device]:
    """장치 설정 YAML 을 읽는다.

    ``devices:`` 아래에 friendly_name 을 키로 둔다::

        devices:
          canopy-treat-01:
            zone: treatment
            sensor_id: CN-01
            note: 지상 10cm, 라디에이션 실드
            map:
              temperature: canopy_temp
              humidity: rh
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: dict[str, Device] = {}
    for name, cfg in (raw.get("devices") or {}).items():
        cfg = cfg or {}
        out[name] = Device(
            name=name,
            zone=cfg.get("zone", ZONE_TREATMENT),
            sensor_id=cfg.get("sensor_id", ""),
            map=dict(cfg.get("map") or {}),
            note=cfg.get("note", ""),
        )
    return out


def _quality(variable: str, value: float) -> str:
    """물리 유효범위를 벗어난 값을 표시한다. 버리지는 않는다."""
    spec = FIELD_VARIABLES.get(variable)
    if spec is None:
        return "ok"
    _, low, high = spec
    return "ok" if low <= value <= high else "range"


def message_to_rows(
    topic: str,
    payload: str | dict,
    devices: dict[str, Device],
    *,
    now: dt.datetime | None = None,
    base_topic: str = "zigbee2mqtt",
    unknown: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Zigbee2MQTT 메시지 하나를 세로 형식 행들로 바꾼다.

    모르는 장치, 모르는 필드는 **조용히 버리지 않고** ``unknown`` 에 적는다.
    센서를 달았는데 화면에 안 나오는 이유를 찾을 수 있어야 한다.
    """
    name = topic.split("/", 1)[1] if topic.startswith(f"{base_topic}/") else topic
    # 브리지 자체가 내보내는 토픽은 센서 자료가 아니다
    if name.startswith("bridge"):
        return []

    data = json.loads(payload) if isinstance(payload, str) else dict(payload)
    device = devices.get(name)
    if device is None:
        if unknown is not None:
            unknown.append(f"모르는 장치: {name}")
        return []

    stamp = (now or dt.datetime.now()).isoformat()
    rows: list[dict[str, Any]] = []

    for key, value in data.items():
        if isinstance(value, bool):
            value = 1.0 if value else 0.0
        if not isinstance(value, (int, float)):
            continue            # 문자열 상태값(ON/OFF 등)은 따로 다룬다

        variable = device.variable_for(key)
        if variable is not None:
            unit = FIELD_VARIABLES.get(variable, ("", 0, 0))[0]
            rows.append({
                "ts": stamp, "zone": device.zone, "sensor_id": device.sensor_id,
                "variable": variable, "value": float(value), "unit": unit,
                "quality": _quality(variable, float(value)),
            })
            continue

        diag = DIAGNOSTIC_FIELDS.get(key)
        if diag is not None:
            rows.append({
                "ts": stamp, "zone": device.zone, "sensor_id": device.sensor_id,
                "variable": diag, "value": float(value),
                "unit": DIAGNOSTIC_UNITS.get(diag, ""), "quality": "ok",
            })
            continue

        if unknown is not None and key in AMBIGUOUS_FIELDS:
            unknown.append(
                f"{name}.{key}: 설치 위치로 뜻이 갈리는 필드다. "
                f"설정의 map 에 어느 변수인지 적어야 한다"
            )
    return rows


def rows_to_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """행 묶음을 수집기 표준 형식 데이터프레임으로 만든다."""
    columns = ["ts", "zone", "sensor_id", "variable", "value", "unit", "quality"]
    frame = pd.DataFrame(list(rows), columns=columns)
    if not frame.empty:
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
    return frame


def read_capture(
    path: str | Path, devices: dict[str, Device], *, base_topic: str = "zigbee2mqtt"
) -> pd.DataFrame:
    """저장해 둔 Z2M 메시지 기록(JSON Lines)을 읽는다.

    한 줄이 ``{"topic": ..., "payload": {...}, "ts": ...}`` 이다. 브리지가
    받은 것을 그대로 쌓아 두면, 현장에 나가지 않고도 같은 자료로 파이프라인을
    다시 돌릴 수 있다. 공인시험의 재현 절차가 이것을 쓴다.
    """
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue            # 깨진 줄은 건너뛰고 나머지를 읽는다
        stamp = record.get("ts")
        when = None
        if stamp:
            try:
                when = dt.datetime.fromisoformat(stamp)
            except ValueError:
                when = None
        rows += message_to_rows(
            record.get("topic", ""), record.get("payload", {}), devices,
            now=when, base_topic=base_topic,
        )
    return rows_to_frame(rows)


def device_status(frame: pd.DataFrame, devices: dict[str, Device]) -> pd.DataFrame:
    """장치별 최근 상태 — 화면의 '센서 상태' 칸이 읽는다.

    배터리와 전파세기는 자료가 끊기기 전에 알려 주는 유일한 신호다.
    마지막 수신 시각이 함께 있어야 '조용히 죽은 센서'를 찾을 수 있다.
    """
    if frame.empty:
        return pd.DataFrame(columns=["sensor_id", "zone", "note", "last_seen",
                                     "battery_pct", "battery_mv", "supercap_v",
                                     "linkquality", "variables"])
    by_id = {d.sensor_id: d for d in devices.values()}
    #: 433MHz 로 들어오는 장치는 Zigbee 장치 목록에 없다. 화면에서 빠지면
    #: 기상 마스트가 죽어도 아무도 모른다.
    rf_notes = {"WS90-01": ("mast", "Ecowitt WS90 · 433MHz · 폴대 상단 2m")}
    out = []
    for sensor_id, part in frame.groupby("sensor_id"):
        device = by_id.get(str(sensor_id))
        rf = rf_notes.get(str(sensor_id))
        latest = part.sort_values("ts").groupby("variable")["value"].last()
        measured = sorted(
            v for v in latest.index
            if v not in DIAGNOSTIC_UNITS and v in FIELD_VARIABLES
        )
        out.append({
            "sensor_id": sensor_id,
            "zone": device.zone if device else (rf[0] if rf else ""),
            "note": device.note if device else (rf[1] if rf else ""),
            "last_seen": part["ts"].max(),
            "battery_pct": float(latest.get("battery_pct", float("nan"))),
            "battery_mv": float(latest.get("battery_mv", float("nan"))),
            "supercap_v": float(latest.get("supercap_v", float("nan"))),
            "linkquality": float(latest.get("linkquality", float("nan"))),
            "variables": measured,
        })
    return pd.DataFrame(out).sort_values(["zone", "sensor_id"]).reset_index(drop=True)
