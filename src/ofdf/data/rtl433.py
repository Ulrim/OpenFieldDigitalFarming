"""rtl_433 출력을 현장 수집 표준 형식으로 바꾼다.

Ecowitt WS90 은 433MHz 전용 RF 라 Zigbee 로 들어오지 않는다. 라즈베리파이에
꽂은 RTL-SDR 동글로 ``rtl_433`` 이 받아 장치마다 JSON 한 줄을 내보낸다::

    {"time":"2026-11-01 05:00:00","model":"Fineoffset-WS90","id":12345,
     "temperature_C":2.4,"humidity":92,"wind_avg_m_s":0.7,"wind_max_m_s":1.9,
     "rain_mm":13.2,"light_lux":0,"uvi":0,"battery_mV":3100,"supercap_V":4.1}

두 가지를 바로잡아야 한다.

돌풍은 3초 순간최대가 아니다
----------------------------
사업계획서의 강풍 안전규칙은 **3초 순간최대풍속**을 기준으로 한다. 그런데
WS90 이 내보내는 ``wind_max_m_s`` 는 2초 간격으로 재어 **최근 28초 중
최댓값**이다. 다른 통계량이므로 같은 이름으로 넣으면 안 된다. 농가 화면이
'3초 순간최대풍속'이라고 적는데 실제로는 28초 창 값인 상태가 된다.

그래서 :data:`VAR_GUST_WINDOW` 라는 별도 변수로 받는다. 제어기는 참 3초
계측값이 있으면 그것을 먼저 쓰고, 없으면 이 값으로 물러서되 **어느 쪽을
썼는지 조치 사유에 적는다.**

강수는 누적값이다
-----------------
Fine Offset 계열의 ``rain_mm`` 은 초기화 이후 누적 총량이다. 그대로 넣으면
시간당 강수가 수백 mm 로 잡힌다. 직전 값과의 차이를 쓰고, 값이 줄어들면
(초기화·교체) 그 구간은 0 으로 둔다.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ofdf.data.counter import CumulativeCounter
from ofdf.data.field import ZONE_MAST

#: WS90 의 28초 창 최댓값. 3초 순간최대(``wind_gust_3s``)와 다른 통계량이다.
VAR_GUST_WINDOW = "wind_gust_window"

#: rtl_433 필드 -> 우리 변수.
#:
#: ``wind_max_m_s`` 를 ``wind_gust_3s`` 로 보내지 않는 것이 요점이다.
RTL433_MAP: dict[str, str] = {
    "temperature_C": "t_air",
    "humidity": "rh",
    "wind_avg_m_s": "wind_speed",
    "wind_max_m_s": VAR_GUST_WINDOW,
    "light_lux": "illuminance",
    "soil_moisture": "soil_moisture",
    #: Fine Offset 토양수분(WH51)은 ``moisture`` 로 보낸다. 같은 433MHz 대역
    #: 이라 WS90 을 받는 동글이 그대로 받는다 — 한 대 더 사면 코드는 그대로다.
    "moisture": "soil_moisture",
    "rain_rate_mm_h": "rain_rate",
    #: 엽면습윤(WH35). 사 놓으면 바로 들어오도록 자리를 비워 둔다.
    "leaf_wetness": "leaf_wetness",
}

#: 누적으로 올라오는 필드 -> 우리 변수. 직전 값과의 차이를 쓴다.
CUMULATIVE_MAP: dict[str, str] = {"rain_mm": "rain", "rain_total_mm": "rain"}

#: 작물 위험에는 안 쓰지만 전원·수신 상태로 남기는 필드.
RTL433_DIAGNOSTICS: dict[str, tuple[str, str]] = {
    "battery_mV": ("battery_mv", "mV"),
    "battery_V": ("battery_v", "V"),
    "battery_ok": ("battery_ok", "bool"),
    "supercap_V": ("supercap_v", "V"),
    "uvi": ("uv_index", ""),
    "uv": ("uv_index", ""),
    "pressure_hPa": ("pressure_hpa", "hPa"),
    #: 수신 세기. Zigbee 의 linkquality 와 같은 자리다 — 전파가 약해지는 것은
    #: 서서히 진행하므로, 끊긴 뒤가 아니라 끊기기 전에 보여야 한다.
    "rssi": ("rssi", "dB"),
    "snr": ("snr", "dB"),
}

#: 모델마다 **실제로 보내는** 필드. 수집 범위 점검(ofdf.data.coverage)이 쓴다.
#:
#: :data:`RTL433_MAP` 은 '이 이름이 오면 이 변수'라는 환산표일 뿐, 어느 장치가
#: 그 이름을 보내는지는 말해 주지 않는다. WS90 은 엽면습윤도 토양수분도 재지
#: 않는데, 환산표에 그 이름이 있다는 이유로 '받고 있다'고 보고하면 없는 센서를
#: 있다고 적는 셈이 된다. 그래서 모델별 필드를 따로 둔다.
#:
#: 모르는 모델은 여기에 없다. 그 경우 점검은 '받는다'고 단정하지 않는다.
MODEL_FIELDS: dict[str, frozenset[str]] = {
    "Fineoffset-WS90": frozenset({
        "temperature_C", "humidity", "wind_dir_deg", "wind_avg_m_s",
        "wind_max_m_s", "uvi", "light_lux", "rain_mm", "supercap_V",
        "battery_mV", "battery_ok", "firmware",
    }),
    #: 토양수분 (WH51) — 수분만 잰다. 지온도 EC 도 없다.
    "Fineoffset-WH51": frozenset({"moisture", "battery_mV", "battery_ok"}),
    #: 범용 온습도 (WH31) — 설치 위치로 뜻이 갈린다.
    "Fineoffset-WH31": frozenset({"temperature_C", "humidity", "battery_ok"}),
    #: 엽면습윤 (WH35)
    "Fineoffset-WH35": frozenset({"leaf_wetness", "battery_mV", "battery_ok"}),
    #: 티핑버킷 강우 (WH40)
    "Fineoffset-WH40": frozenset({"rain_mm", "battery_mV", "battery_ok"}),
}

#: 우리가 안 쓰는 복조 부산물. 경고를 띄우지 않는다.
IGNORED_FIELDS = frozenset({
    "id", "mic", "flags", "unknown", "firmware", "rain_start", "data",
    "channel", "subtype", "protocol", "freq", "noise", "duration", "modulation",
    "wind_dir_deg",                      # 아래에서 따로 넣는다
    "battery",                           # 0/1 이 아니라 등급이라 뜻이 갈린다
})

#: 변수 단위 — FIELD_VARIABLES 에 없는 것만 여기서 채운다.
EXTRA_UNITS = {
    VAR_GUST_WINDOW: "m/s",
    "wind_speed_avg10m": "m/s", "wind_dir_avg10m": "deg", "gust_dir": "deg",
    "gust_daily_max": "m/s", "soil_raw_ad": "", "pm25": "ug/m3",
    "pm25_24h": "ug/m3", "rain_alt_total": "mm",
}


def _unit(variable: str) -> str:
    from ofdf.data.field import FIELD_VARIABLES

    spec = FIELD_VARIABLES.get(variable)
    return spec[0] if spec else EXTRA_UNITS.get(variable, "")


class RainAccumulator(CumulativeCounter):
    """누적 강수에서 구간 강수를 만든다.

    셈은 :class:`ofdf.data.counter.CumulativeCounter` 와 같다 — 전력량계·
    수량계도 같은 문제를 겪기 때문에 한 곳에 두었다. 이름은 강수 쪽
    호출부가 많아 그대로 남긴다.
    """


def message_to_rows(
    record: dict[str, Any],
    *,
    zone: str = ZONE_MAST,
    sensor_id: str | None = None,
    rain: RainAccumulator | None = None,
    now: dt.datetime | None = None,
    unknown: list[str] | None = None,
    device: RfDevice | None = None,
) -> list[dict[str, Any]]:
    """rtl_433 JSON 한 줄을 세로 형식 행들로 바꾼다.

    ``device`` 를 넘기면 구간·센서이름·변수매핑을 설정에서 가져온다. 안
    넘기면 인자로 받은 ``zone``·``sensor_id`` 와 기본 매핑을 쓴다.
    """
    model = str(record.get("model", ""))
    rain_key = f"{model}-{record.get('id', '')}"
    if device is not None:
        zone = device.zone
        sensor = device.sensor_id
    else:
        sensor = sensor_id or (model.split("-")[-1] or "RF") + "-01"

    stamp = record.get("time")
    when = now
    if when is None and isinstance(stamp, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                when = dt.datetime.strptime(stamp[:19], fmt)
                break
            except ValueError:
                continue
    when = when or dt.datetime.now()
    iso = when.isoformat()

    rows: list[dict[str, Any]] = []

    def add(variable: str, value: float, unit: str | None = None) -> None:
        rows.append({
            "ts": iso, "zone": zone, "sensor_id": sensor, "variable": variable,
            "value": float(value), "unit": unit if unit is not None else _unit(variable),
            "quality": "ok",
        })

    for key, value in record.items():
        if isinstance(value, bool):
            value = 1.0 if value else 0.0      # battery_ok 가 참·거짓으로 올 때
        if not isinstance(value, (int, float)):
            continue

        variable = (device.variable_for(key) if device is not None
                    else RTL433_MAP.get(key))
        if variable:
            add(variable, value)
            continue

        if key in CUMULATIVE_MAP:
            counter = rain or RainAccumulator()
            add(CUMULATIVE_MAP[key], round(counter.step(rain_key, float(value)), 3))
            continue

        diag = RTL433_DIAGNOSTICS.get(key)
        if diag:
            add(diag[0], value, diag[1])
            continue

        if unknown is not None and key not in IGNORED_FIELDS:
            unknown.append(f"{model}.{key}: 매핑 없음(값 {value!r})")

    # 풍향은 각도라 그대로 둔다(평균을 내면 안 되는 값이라 집계에서 제외된다)
    if isinstance(record.get("wind_dir_deg"), (int, float)):
        add("wind_dir", float(record["wind_dir_deg"]), "deg")
    return rows


@dataclass
class RfDevice:
    """433MHz 로 들어오는 센서 한 대의 설치 정보.

    Zigbee 장치와 달리 페어링이 없다. 전파만 받으면 들어오므로 **이웃집
    온도계도 들어온다.** 그래서 모델(또는 모델/id)로 받을 것을 정하고,
    어디에 달렸는지를 여기 적는다.
    """

    model: str                                  #: rtl_433 의 model 값
    zone: str = ZONE_MAST
    sensor_id: str = ""
    note: str = ""
    device_id: str = ""                         #: 같은 모델이 둘일 때 구분
    map: dict[str, str] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sensor_id:
            self.sensor_id = self.model

    def variable_for(self, field_name: str) -> str | None:
        if field_name in self.map:
            return self.map[field_name] or None
        return RTL433_MAP.get(field_name)


def load_rf_devices(path: str | Path) -> dict[str, RfDevice]:
    """설치 정보 YAML 의 ``rf:`` 칸을 읽는다.

    Zigbee 장치와 **같은 파일**에 둔다. 포장에 무엇이 달려 있는지를 두 곳에
    나눠 적으면 한쪽만 고치는 날이 온다::

        rf:
          Fineoffset-WS90:
            zone: mast
            sensor_id: WS90-01
            note: 폴대 상단 2m · 개활지
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out: dict[str, RfDevice] = {}
    for key, cfg in (raw.get("rf") or {}).items():
        cfg = cfg or {}
        model, _, device_id = str(key).partition("/")
        out[str(key)] = RfDevice(
            model=model,
            zone=cfg.get("zone", ZONE_MAST),
            sensor_id=cfg.get("sensor_id", ""),
            note=cfg.get("note", ""),
            device_id=cfg.get("id", device_id),
            map=dict(cfg.get("map") or {}),
        )
    return out


def match_device(
    record: dict[str, Any], devices: dict[str, RfDevice]
) -> RfDevice | None:
    """수신한 줄이 설정의 어느 장치인지 찾는다. 모델/id 가 먼저다."""
    model = str(record.get("model", ""))
    ident = str(record.get("id", ""))
    return devices.get(f"{model}/{ident}") or devices.get(model)


def read_capture(
    path: str | Path, *, zone: str = ZONE_MAST, sensor_id: str | None = None
) -> pd.DataFrame:
    """``rtl_433 -F json`` 출력을 모아 둔 파일을 읽는다.

    현장에 나가지 않고 같은 자료로 파이프라인을 다시 돌릴 수 있어야 공인시험
    재현 절차가 성립한다. 깨진 줄은 건너뛰고 나머지를 읽는다.
    """
    rain = RainAccumulator()
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        rows += message_to_rows(record, zone=zone, sensor_id=sensor_id, rain=rain)
    return rows_to_frame(rows)


def rows_to_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    columns = ["ts", "zone", "sensor_id", "variable", "value", "unit", "quality"]
    frame = pd.DataFrame(list(rows), columns=columns)
    if not frame.empty:
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
    return frame
