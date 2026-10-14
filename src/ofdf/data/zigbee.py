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

숫자가 아닌 값도 버리지 않는다
------------------------------
Tuya 밸브와 차단기는 켜짐·꺼짐을 **문자열**로 보낸다::

    zigbee2mqtt/valve-treat-01  {"state":"ON","flow":12.4,"water_consumed":83.2}

이것이 공인시험 T-07(제어응답)이 보는 값이다. '밸브를 열어라'는 지시를
내렸을 때 정말 열렸는지는 장치가 되읽어 주는 이 값으로만 확인된다.
숫자가 아니라고 버리면 명령은 보냈는데 실행 여부를 모르는 상태가 된다.

적산값은 구간값으로 바꾼다
--------------------------
``energy``(kWh)와 ``water_consumed``(L)는 설치 이후 **누적 총량**이다. 그대로
넣으면 5분 관수량이 수백 L 로 잡힌다. 직전 값과의 차이를 쓰고, 값이 줄면
(초기화·교체) 그 구간은 0 으로 둔다.

적산값은 유량·전류보다 한 수 위다. 1분마다 재는 유량계는 30초짜리 관수를
통째로 놓치지만 적산 수량계는 놓치지 않는다.

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

from ofdf.data.counter import CumulativeCounter
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
    "energy": "pump_energy",
    "water_consumed": "irrigation_volume",
    "volume": "irrigation_volume",
    "ec": "soil_ec",
    "soil_ec": "soil_ec",
}

#: 뜻이 설치 위치·장치 종류로 갈리는 필드. 설정에 매핑이 없으면 버리고
#: 경고한다.
#:
#: ``state`` 가 여기 있는 이유는 ``temperature`` 와 같다. 밸브의 ``state`` 는
#: 열림·닫힘이고 차단기의 ``state`` 는 통전·차단이다. 기본값을 하나 두면
#: 차단기가 조용히 밸브로 들어간다.
AMBIGUOUS_FIELDS = frozenset({"temperature", "humidity", "state"})

#: 설치 이후 누적으로 올라오는 **필드 이름**. 직전 값과의 차이를 쓴다.
#:
#: 변수 이름이 아니라 필드 이름으로 판단한다. 설정에서 변수를 바꿔 달아도
#: (``energy: pump_energy`` -> 다른 이름) 적산이라는 사실은 그대로이기 때문이다.
CUMULATIVE_FIELDS = frozenset({
    "energy", "energy_consumed", "water_consumed", "volume",
})

#: 상태 문자열 -> 수치. 대소문자는 가린다.
#:
#: Zigbee2MQTT 는 장치 종류에 따라 ``"ON"``, ``"OPEN"``, ``"detected"`` 등으로
#: 보낸다. 모르는 말이 오면 **추측하지 않고** 경고한다 — 밸브 상태를 거꾸로
#: 읽으면 관수했다고 기록하면서 작물은 마른다.
STATE_WORDS: dict[str, float] = {
    "on": 1.0, "off": 0.0,
    "open": 1.0, "opened": 1.0, "close": 0.0, "closed": 0.0,
    "true": 1.0, "false": 0.0,
    "detected": 1.0, "not_detected": 0.0, "clear": 0.0,
    "alarm": 1.0, "normal": 0.0,
    "wet": 1.0, "dry": 0.0,
    "lock": 1.0, "unlock": 0.0, "locked": 1.0, "unlocked": 0.0,
    "online": 1.0, "offline": 0.0,
}

#: 작물 위험 판단에는 안 쓰지만 통신·전원·설비 상태로 남기는 필드.
DIAGNOSTIC_FIELDS: dict[str, str] = {
    "battery": "battery_pct",
    #: 배터리 장치의 voltage 는 배터리 전압(mV)이다. 전력계측 차단기의
    #: voltage 는 선간전압(V)이라 뜻이 다르므로, 그런 장치는 설정에서
    #: ``voltage: line_voltage`` 로 지정한다.
    "voltage": "battery_mv",
    "linkquality": "linkquality",
    "battery_low": "battery_low",
    #: 차단기 본체 온도. 과열은 화재 전조라 작물 변수와 별도로 남긴다.
    "device_temperature": "device_temp",
    "power_factor": "power_factor",
    "ac_frequency": "ac_frequency",
    "frequency": "ac_frequency",
    "tamper": "tamper",
    "child_lock": "child_lock",
}

#: 진단 변수의 단위
DIAGNOSTIC_UNITS = {
    "battery_pct": "%", "battery_mv": "mV", "linkquality": "lqi",
    "battery_low": "bool", "device_temp": "degC", "power_factor": "",
    "ac_frequency": "Hz", "tamper": "bool", "child_lock": "bool",
}

#: 설정값·임계치처럼 측정이 아닌 필드. 경고를 띄우지 않고 지나간다.
#:
#: Tuya 차단기는 자기 설정(과전류 임계치 등)을 측정값과 같은 메시지에 담아
#: 보낸다. 이것까지 '매핑 없음'으로 찍으면 정작 봐야 할 경고가 묻힌다.
IGNORED_FIELDS = frozenset({
    "last_seen", "update", "update_available", "elapsed", "action",
    "over_current_threshold", "over_voltage_threshold", "under_voltage_threshold",
    "over_current_breaker", "over_voltage_breaker", "under_voltage_breaker",
    "power_on_behavior", "indicator_mode", "countdown", "cycle_timer",
    "random_timer", "schedule", "calibration", "preset", "clear_device_data",
    "test", "factory_reset", "sensitivity",
})


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
    #: 이 장치가 보내는 필드 가운데 기본 매핑으로 충분한 것들.
    #:
    #: ``map`` 은 '뜻을 바로잡아야 하는 필드'만 적는 자리다. 토양센서의
    #: ``soil_moisture`` 처럼 기본 매핑이 맞는 필드는 적을 것이 없어서
    #: 비어 있는데, 그러면 수집 범위 점검이 '이 장치가 수분을 보낸다'는 것을
    #: 알 길이 없다. 그래서 보내는 필드 목록을 따로 받는다.
    #:
    #: 수집 자체에는 영향이 없다 — 적지 않아도 기본 매핑으로 들어온다.
    sends: tuple[str, ...] = ()

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
            sends: [battery, linkquality]
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
            sends=tuple(cfg.get("sends") or ()),
        )
    return out


def to_number(raw: Any) -> float | None:
    """Z2M 값 하나를 수치로 읽는다. 못 읽으면 ``None``.

    참·거짓, 숫자, 그리고 **상태말**을 받는다. 상태말을 받는 것이 요점이다 —
    밸브와 차단기는 자기 상태를 ``"ON"`` 같은 문자열로 보내고, 그것이
    제어응답 확인에 쓰는 유일한 되읽기 값이다.

    모르는 말은 추측하지 않고 ``None`` 을 돌려준다. 호출부가 경고를 남긴다.
    """
    if isinstance(raw, bool):
        return 1.0 if raw else 0.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        word = STATE_WORDS.get(text.lower())
        if word is not None:
            return word
        try:
            return float(text)          # "23.5" 처럼 숫자를 문자열로 보내는 장치
        except ValueError:
            return None
    return None


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
    cumulative: CumulativeCounter | None = None,
) -> list[dict[str, Any]]:
    """Zigbee2MQTT 메시지 하나를 세로 형식 행들로 바꾼다.

    모르는 장치, 모르는 필드, 모르는 상태말은 **조용히 버리지 않고**
    ``unknown`` 에 적는다. 센서를 달았는데 화면에 안 나오는 이유를 찾을 수
    있어야 한다.

    ``cumulative`` 를 넘기면 적산 필드(``energy``·``water_consumed``)의
    구간 차분을 메시지 사이에 이어서 센다. 넘기지 않으면 메시지마다 처음
    보는 값이 되어 첫 구간이 0 이 된다.
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
    counter = cumulative if cumulative is not None else CumulativeCounter()

    def emit(variable: str, value: float, unit: str, quality: str) -> None:
        rows.append({
            "ts": stamp, "zone": device.zone, "sensor_id": device.sensor_id,
            "variable": variable, "value": float(value), "unit": unit,
            "quality": quality,
        })

    for key, raw in data.items():
        if key in IGNORED_FIELDS:
            continue

        number = to_number(raw)
        if number is None:
            # 숫자로도 상태말로도 읽히지 않는다. 사전·목록(예: Z2M 의
            # ``update`` 블록)이거나 우리가 모르는 상태말이다.
            if unknown is not None and isinstance(raw, str):
                unknown.append(
                    f"{name}.{key}: 모르는 상태값 '{raw}'. "
                    f"ofdf.data.zigbee.STATE_WORDS 에 뜻을 적어야 한다"
                )
            continue

        variable = device.variable_for(key)
        if variable is not None:
            if key in CUMULATIVE_FIELDS:
                number = counter.step(f"{device.sensor_id}:{key}", number)
            unit = FIELD_VARIABLES.get(variable, ("", 0, 0))[0]
            emit(variable, round(number, 4), unit, _quality(variable, number))
            continue

        diag = DIAGNOSTIC_FIELDS.get(key)
        if diag is not None:
            emit(diag, number, DIAGNOSTIC_UNITS.get(diag, ""), "ok")
            continue

        if unknown is None:
            continue
        if key in device.map:
            # 설정이 빈 값으로 '쓰지 않는다'고 이미 말했다(예: 밸브 몸체
            # 온도). 답이 있는 물음을 다시 묻지 않는다.
            continue
        if key in AMBIGUOUS_FIELDS:
            unknown.append(
                f"{name}.{key}: 장치 종류·설치 위치로 뜻이 갈리는 필드다. "
                f"설정의 map 에 어느 변수인지 적어야 한다"
            )
        else:
            unknown.append(f"{name}.{key}: 매핑 없음(값 {raw!r})")
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
    counter = CumulativeCounter()       # 적산값은 파일 전체를 이어서 센다
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
            now=when, base_topic=base_topic, cumulative=counter,
        )
    return rows_to_frame(rows)


#: 화면에 '상태'로 보여 줄 변수와 그 뜻. 값이 1 일 때·0 일 때의 말이다.
#:
#: 평균이 아니라 마지막 값을 쓴다. 밸브가 지금 열려 있는지를 묻는 자리이고,
#: 0.5 는 답이 아니다.
STATE_LABELS: dict[str, tuple[str, str, str]] = {
    "valve_state": ("관수 밸브", "열림", "닫힘"),
    "circuit_state": ("관수 회로", "통전", "끊김"),
    "aux_circuit_state": ("보조 회로", "통전", "끊김"),
    "enclosure_flood": ("침수", "감지", "정상"),
    "emergency_stop": ("비상정지", "눌림", "정상"),
    "rain_detect": ("강우", "내림", "없음"),
    "tamper": ("뜯김", "감지", "정상"),
    "battery_low": ("배터리", "부족", "정상"),
}

STATUS_COLUMNS = ["sensor_id", "zone", "note", "last_seen", "battery_pct",
                  "battery_mv", "battery_v", "supercap_v", "linkquality",
                  "rssi", "variables", "states"]


def device_status(
    frame: pd.DataFrame,
    devices: dict[str, Device],
    *,
    extra: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """장치별 최근 상태 — 화면의 '센서 상태' 칸이 읽는다.

    배터리와 전파세기는 자료가 끊기기 전에 알려 주는 유일한 신호다.
    마지막 수신 시각이 함께 있어야 '조용히 죽은 센서'를 찾을 수 있다.

    ``extra`` 는 Zigbee 가 아닌 장치의 ``{센서이름: (구간, 메모)}`` 다.
    433MHz·에코윗 장치는 Zigbee 목록에 없으므로, 넘기지 않으면 화면에서
    빠진다 — 기상 마스트가 죽어도 아무도 모르는 상태가 된다.
    """
    if frame.empty:
        return pd.DataFrame(columns=STATUS_COLUMNS)
    by_id = {d.sensor_id: d for d in devices.values()}
    extra = extra or {}
    out = []
    for sensor_id, part in frame.groupby("sensor_id"):
        device = by_id.get(str(sensor_id))
        other = extra.get(str(sensor_id))
        latest = part.sort_values("ts").groupby("variable")["value"].last()
        measured = sorted(
            v for v in latest.index
            if v not in DIAGNOSTIC_UNITS and v in FIELD_VARIABLES
            and v not in STATE_LABELS
        )
        states = []
        for variable, (name, on_word, off_word) in STATE_LABELS.items():
            value = latest.get(variable)
            if value is None or pd.isna(value):
                continue
            states.append({"name": name, "on": bool(value >= 0.5),
                           "text": on_word if value >= 0.5 else off_word})

        def pick(key: str) -> float:
            return float(latest.get(key, float("nan")))

        out.append({
            "sensor_id": sensor_id,
            "zone": device.zone if device else (other[0] if other else ""),
            "note": device.note if device else (other[1] if other else ""),
            "last_seen": part["ts"].max(),
            "battery_pct": pick("battery_pct"),
            "battery_mv": pick("battery_mv"),
            "battery_v": pick("battery_v"),
            "supercap_v": pick("supercap_v"),
            "linkquality": pick("linkquality"),
            "rssi": pick("rssi"),
            "variables": measured,
            "states": states,
        })
    return pd.DataFrame(out).sort_values(["zone", "sensor_id"]).reset_index(drop=True)
