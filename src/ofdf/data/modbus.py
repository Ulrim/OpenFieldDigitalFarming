"""RS-485(MODBUS-RTU) 수질 센서를 현장 수집 표준 형식으로 바꾼다.

라즈베리파이 하드웨어는 Shrimp365 수질 모니터링 장비와 같다. 거기 붙던
USB-RS485 변환기와 Nengshi 디지털 센서(pH·DO·EC)가 그대로 꽂히므로
**프로토콜·레지스터·슬레이브 ID 를 그 코드와 같게** 둔다. 두 사업의 장비를
오가며 설정을 바꿀 일이 없어야 한다.

    [pH 센서 ID1]──┐
    [DO 센서 ID3]──┼── A/B 두 선 ──[USB-RS485]──USB──[라즈베리파이]
    [EC 센서 ID4]──┘

통신: 9600bps 8N1, 기능코드 04 로 입력 레지스터 0x0000 부터 16개.
레지스터는 '값'과 '소수점+단위'가 짝이다 — 상위 바이트가 소수 자릿수,
하위 바이트가 단위 코드. 실제 값 = 원시값 / 10^소수자릿수.

**수질 센서가 있을 때만 돈다.** 노지 대파 실증에서 수질은 관수원수·양액의
보조 지표다. 변환기가 안 꽂혀 있으면 브리지는 조용히 끝나고 나머지 수집은
영향이 없다 — 없는 장비를 매 주기 읽으려다 오류를 쌓으면 안 된다.

이 모듈은 **직렬 포트를 열지 않는다.** 프레임을 만들고 응답을 해석하는 순수
함수와, 읽기 함수 하나를 받아 도는 폴러만 있다. 그래서 변환기 없이 시험이
돌고, 저장된 레지스터 기록으로 다시 돌릴 수 있다.
"""

from __future__ import annotations

import datetime as dt
import math
import struct
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from ofdf.data.field import FIELD_VARIABLES, ZONE_FACILITY

#: 기본 직렬 설정. Shrimp365 와 같다.
DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 9600
REGISTER_COUNT = 16


# ── MODBUS-RTU 프레임 ──────────────────────────────────────────────

def crc16(payload: bytes) -> bytes:
    """MODBUS CRC16. 전송 순서(하위 바이트 먼저)로 돌려준다."""
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return struct.pack("<H", crc)


def build_read_request(slave_id: int, start: int = 0, count: int = REGISTER_COUNT) -> bytes:
    """기능코드 04 요청 프레임."""
    body = struct.pack(">BBHH", slave_id, 0x04, start, count)
    return body + crc16(body)


class ModbusError(RuntimeError):
    pass


def parse_read_response(response: bytes, slave_id: int) -> list[int]:
    """기능코드 04 응답을 부호 있는 16비트 정수 목록으로.

    실패 사유를 사람이 고칠 수 있는 말로 적는다 — 응답 없음은 배선·전원·ID,
    CRC 불일치는 노이즈·종단저항이다. 고쳐야 할 곳이 다르다.
    """
    if len(response) < 5:
        raise ModbusError(f"ID {slave_id}: 응답 없음 (배선·전원·슬레이브 ID 확인)")
    if response[0] != slave_id:
        raise ModbusError(f"ID {slave_id}: 다른 기기가 응답함({response[0]})")
    if response[1] & 0x80:
        raise ModbusError(f"ID {slave_id}: 예외 응답 코드 {response[2]}")
    if crc16(response[:-2]) != response[-2:]:
        raise ModbusError(f"ID {slave_id}: CRC 불일치 (노이즈·종단저항 확인)")
    byte_count = response[2]
    data = response[3:3 + byte_count]
    return [struct.unpack(">h", data[i:i + 2])[0] for i in range(0, len(data) - 1, 2)]


# ── 값 해석 (Nengshi 프로토콜) ─────────────────────────────────────

#: 프로토콜 문서의 단위 코드표. Shrimp365 와 같다.
UNITS = {
    0x00: "mV", 0x01: "nA", 0x02: "uA", 0x03: "mA", 0x04: "ohm", 0x05: "kohm",
    0x06: "Mohm", 0x07: "uS", 0x08: "mS", 0x09: "S", 0x0A: "pH", 0x0B: "C",
    0x0C: "F", 0x0D: "ug/L", 0x0E: "mg/L", 0x0F: "g/L", 0x10: "ppb",
    0x11: "ppm", 0x12: "ppt", 0x13: "%", 0x14: "mbar", 0x15: "bar",
    0x16: "mmHg", 0x17: "ppt",
}


def decode(regs: list[int], value_idx: int) -> tuple[float, str] | None:
    """값 레지스터와 그 다음 '소수점+단위' 레지스터를 함께 읽어 실수로."""
    if value_idx + 1 >= len(regs):
        return None
    raw = regs[value_idx]
    meta = regs[value_idx + 1] & 0xFFFF
    decimals = (meta >> 8) & 0xFF
    if decimals > 4:                  # 준비 안 된 센서의 쓰레기값
        return None
    return raw / (10 ** decimals), UNITS.get(meta & 0xFF, "")


@dataclass
class SensorSpec:
    """센서 한 대 — 어느 ID 에서 어느 레지스터를 어느 변수로."""

    key: str
    slave_id: int
    #: {우리 변수: 값 레지스터}
    fields: dict[str, int] = dc_field(default_factory=dict)
    label: str = ""
    #: 설정에 없을 때 켤지. 노지 실증은 수질 센서가 선택이라 전부 꺼 둔다.
    default_enabled: bool = False


#: 레지스터 배치는 Shrimp365 와 같다. 변수 이름만 이 저장소 규칙(water_*)이다.
#: 수온은 세 센서 모두 0x0008 에 있어 :data:`TEMPERATURE_PRIORITY` 대로 하나만 쓴다.
SENSOR_SPECS: dict[str, SensorSpec] = {
    "ph": SensorSpec("ph", 1, {"water_ph": 0x0000, "water_orp": 0x0004,
                               "water_temp": 0x0008}, "pH / ORP"),
    "do": SensorSpec("do", 3, {"water_do": 0x0000, "water_do_sat": 0x0002,
                               "water_temp": 0x0008}, "용존산소"),
    #: 염도 레지스터(0x0006)는 믿을 수 없어 읽지 않는다. 전도도에서 환산한다.
    "ec": SensorSpec("ec", 4, {"water_ec": 0x0000, "water_tds": 0x0002,
                               "water_temp": 0x0008}, "전도도"),
    "flow": SensorSpec("flow", 5, {"irrigation_flow": 0x0000}, "유량"),
    "dp": SensorSpec("dp", 6, {"diff_pressure": 0x0000}, "차압"),
}

#: 수온을 어느 센서 것으로 쓸지. DO 변환기가 자기 수온으로 포화도를 보정하므로
#: 그 수온을 기록에 남겨야 나중에 'DO 가 왜 이런가'를 따질 수 있다.
TEMPERATURE_PRIORITY = ("do", "ph", "ec")

#: 첫 레지스터 단위로 어느 변환기인지 가른다 — 선을 훑어 ID 를 찾을 때 쓴다.
UNIT_TO_KEY = {"pH": "ph", "mg/L": "do", "ug/L": "do", "uS": "ec", "mS": "ec", "S": "ec"}

#: 전극이 값을 못 낼 때 변환기가 보내는 표시값
PROBE_FAULT_RAW = {0x7FFF, -1, 0xFFFF}


def normalize(variable: str, value: float, unit: str) -> float | None:
    """단위를 :data:`FIELD_VARIABLES` 가 정한 것으로 맞춘다."""
    if variable == "water_temp":
        return round((value - 32) * 5 / 9, 2) if unit == "F" else round(value, 2)
    if variable == "water_ec":
        # mS/cm 로 통일한다. 농업 관례(dS/m)와 수치가 같다.
        if unit == "uS":
            return round(value / 1000.0, 3)
        if unit == "S":
            return round(value * 1000.0, 3)
        return round(value, 3)
    if variable in ("water_do", "water_tds"):
        return round(value / 1000.0, 3) if unit == "ug/L" else round(value, 2)
    if variable == "diff_pressure":
        if unit == "bar":
            return round(value * 100.0, 2)
        if unit == "mbar":
            return round(value * 0.1, 2)
        if unit == "mmHg":
            return round(value * 0.133322, 2)
        return round(value, 2)
    return round(value, 3)


def salinity_from_ec(ec_ms_cm: float, factor: float = 0.55) -> float:
    """전도도(mS/cm)에서 염도(ppt). 관수원수 염해 판단용 — 낮은 농도 구간의 단순 환산."""
    return round(ec_ms_cm * 1000.0 * factor / 1000.0, 3)


def do_saturation_mgl(celsius: float) -> float:
    """수온에서의 100% 포화 용존산소(mg/L). 못 구하면 0."""
    if celsius < -5 or celsius > 60:
        return 0.0
    t = celsius + 273.15
    try:
        return math.exp(-139.34411 + 1.575701e5 / t - 6.642308e7 / t ** 2
                        + 1.243800e10 / t ** 3 - 8.621949e11 / t ** 4)
    except (OverflowError, ValueError):
        return 0.0


def _quality(variable: str, value: float) -> str:
    spec = FIELD_VARIABLES.get(variable)
    if spec is None:
        return "ok"
    return "ok" if spec[1] <= value <= spec[2] else "range"


# ── 설정 ──────────────────────────────────────────────────────────

@dataclass
class Rs485Config:
    enabled: bool = False
    port: str = DEFAULT_PORT
    baudrate: int = DEFAULT_BAUD
    zone: str = ZONE_FACILITY
    sensor_id: str = "WQ-01"
    note: str = "관수원수 수질 (RS-485)"
    #: {센서 key: 슬레이브 ID}. 켜 둔 것만.
    sensors: dict[str, int] = dc_field(default_factory=dict)
    salinity_factor: float = 0.55


def load_rs485(path: str | Path) -> Rs485Config:
    """설치 정보 YAML 의 ``rs485:`` 칸을 읽는다.

    다른 경로와 같은 파일에 둔다::

        rs485:
          enabled: true
          port: /dev/ttyUSB0
          sensor_id: WQ-01
          sensors:
            ph: 1          # 슬레이브 ID. 끄려면 줄을 지우거나 0
            do: 3
            ec: 4
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section = raw.get("rs485") or {}
    sensors: dict[str, int] = {}
    for key, slave in (section.get("sensors") or {}).items():
        if key in SENSOR_SPECS and slave:
            sensors[str(key)] = int(slave)
    if "sensors" not in section:
        sensors = {k: s.slave_id for k, s in SENSOR_SPECS.items() if s.default_enabled}
    return Rs485Config(
        enabled=bool(section.get("enabled", False)),
        port=str(section.get("port", DEFAULT_PORT)),
        baudrate=int(section.get("baudrate", DEFAULT_BAUD)),
        zone=str(section.get("zone", ZONE_FACILITY)),
        sensor_id=str(section.get("sensor_id", "WQ-01")),
        note=str(section.get("note", "관수원수 수질 (RS-485)")),
        sensors=sensors,
        salinity_factor=float(section.get("salinity_factor", 0.55)),
    )


# ── 레지스터 → 세로 형식 ───────────────────────────────────────────

def registers_to_rows(
    readings: dict[str, list[int] | None],
    config: Rs485Config,
    *,
    now: dt.datetime | None = None,
    unknown: list[str] | None = None,
) -> list[dict[str, Any]]:
    """센서별 레지스터 묶음을 세로 형식 행들로 바꾼다.

    ``readings`` 는 ``{센서 key: 레지스터 16개 또는 None(응답 없음)}``.
    수온은 우선순위대로 **하나만** 낸다 — 같은 값을 셋 다 보내면 중앙값이
    엉뚱한 센서를 고른다.
    """
    stamp = (now or dt.datetime.now()).isoformat()
    rows: list[dict[str, Any]] = []
    temps: dict[str, float] = {}
    ec_value: float | None = None

    def add(variable: str, value: float, quality: str = "ok") -> None:
        unit = FIELD_VARIABLES.get(variable, ("", 0, 0))[0]
        rows.append({"ts": stamp, "zone": config.zone, "sensor_id": config.sensor_id,
                     "variable": variable, "value": float(value), "unit": unit,
                     "quality": quality})

    for key, regs in readings.items():
        spec = SENSOR_SPECS.get(key)
        if spec is None or regs is None:
            continue
        for variable, idx in spec.fields.items():
            decoded = decode(regs, idx)
            if decoded is None:
                continue
            raw = regs[idx] if idx < len(regs) else None
            if raw in PROBE_FAULT_RAW:
                # 변환기는 살아 있고 전극이 값을 못 낸다. 버리지 않고 표시한다.
                if unknown is not None:
                    unknown.append(f"{spec.label}: {variable} 전극 값 없음 (전극·케이블 확인)")
                continue
            value = normalize(variable, decoded[0], decoded[1])
            if value is None:
                continue
            if variable == "water_temp":
                temps[key] = value
                continue
            add(variable, value, _quality(variable, value))
            if variable == "water_ec":
                ec_value = value

    for key in TEMPERATURE_PRIORITY:
        if key in temps:
            add("water_temp", temps[key], _quality("water_temp", temps[key]))
            break

    if ec_value is not None:
        add("water_salinity", salinity_from_ec(ec_value, config.salinity_factor))

    # 수온이 있으면 DO 가 물에서 나올 수 있는 값인지 본다. 전극이 공중이면
    # 포화도 200% 를 넘는다 — 값은 범위 안이라 범위검사로는 못 잡는다.
    water_t = temps.get(TEMPERATURE_PRIORITY[0]) or next(iter(temps.values()), None)
    if water_t is not None:
        sat = do_saturation_mgl(water_t)
        for row in rows:
            if row["variable"] == "water_do" and sat > 0 and row["value"] / sat > 1.5:
                row["quality"] = "range"
                if unknown is not None:
                    unknown.append(f"용존산소 {row['value']}mg/L 는 {water_t}℃ 물에서 "
                                   f"나올 수 없다({row['value']/sat*100:.0f}%). 전극이 공중인지 확인")
    return rows


def classify(regs: list[int]) -> str | None:
    """레지스터를 보고 어느 변환기인지. 모르면 None."""
    decoded = decode(regs, 0)
    return UNIT_TO_KEY.get(decoded[1]) if decoded else None


def scan(read: Callable[[int], list[int] | None], first: int = 1, last: int = 32) -> dict:
    """선을 훑어 어느 ID 에 무엇이 있는지. 센서에 쓰기를 하지 않는다.

    ``read(slave_id)`` 는 레지스터 목록을 돌려주거나 응답이 없으면 None.
    """
    assign: dict[str, int] = {}
    conflicts: dict[str, list[int]] = {}
    others: list[dict] = []
    for slave_id in range(first, last + 1):
        regs = read(slave_id)
        if not regs:
            continue
        key = classify(regs)
        decoded = decode(regs, 0)
        note = f"{decoded[0]:g} {decoded[1]}".strip() if decoded else "값 해석 실패"
        if key is None:
            others.append({"id": slave_id, "note": note})
        elif key in assign:
            conflicts.setdefault(key, [assign[key]]).append(slave_id)
        else:
            assign[key] = slave_id
    return {"assign": assign, "conflicts": conflicts, "others": others,
            "missing": [k for k in ("ph", "do", "ec") if k not in assign]}


def rows_to_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    columns = ["ts", "zone", "sensor_id", "variable", "value", "unit", "quality"]
    frame = pd.DataFrame(list(rows), columns=columns)
    if not frame.empty:
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
    return frame
