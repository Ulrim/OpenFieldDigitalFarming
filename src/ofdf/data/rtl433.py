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
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

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
}

#: 누적으로 올라오는 필드 -> 우리 변수. 직전 값과의 차이를 쓴다.
CUMULATIVE_MAP: dict[str, str] = {"rain_mm": "rain"}

#: 작물 위험에는 안 쓰지만 전원·수신 상태로 남기는 필드.
RTL433_DIAGNOSTICS: dict[str, tuple[str, str]] = {
    "battery_mV": ("battery_mv", "mV"),
    "supercap_V": ("supercap_v", "V"),
    "uvi": ("uv_index", ""),
    "pressure_hPa": ("pressure_hpa", "hPa"),
}

#: 변수 단위 — FIELD_VARIABLES 에 없는 것만 여기서 채운다.
EXTRA_UNITS = {VAR_GUST_WINDOW: "m/s"}


def _unit(variable: str) -> str:
    from ofdf.data.field import FIELD_VARIABLES

    spec = FIELD_VARIABLES.get(variable)
    return spec[0] if spec else EXTRA_UNITS.get(variable, "")


class RainAccumulator:
    """누적 강수에서 구간 강수를 만든다.

    장치마다 따로 센다. 값이 줄어들면 초기화·교체로 보고 그 구간은 0 이다 —
    음수 강수를 내보내면 물수지가 망가진다.
    """

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def step(self, device: str, total: float) -> float:
        previous = self._last.get(device)
        self._last[device] = total
        if previous is None or total < previous:
            return 0.0
        return total - previous


def message_to_rows(
    record: dict[str, Any],
    *,
    zone: str = ZONE_MAST,
    sensor_id: str | None = None,
    rain: RainAccumulator | None = None,
    now: dt.datetime | None = None,
    unknown: list[str] | None = None,
) -> list[dict[str, Any]]:
    """rtl_433 JSON 한 줄을 세로 형식 행들로 바꾼다."""
    model = str(record.get("model", ""))
    device = f"{model}-{record.get('id', '')}"
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
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue

        variable = RTL433_MAP.get(key)
        if variable:
            add(variable, value)
            continue

        if key in CUMULATIVE_MAP:
            counter = rain or RainAccumulator()
            add(CUMULATIVE_MAP[key], round(counter.step(device, float(value)), 3))
            continue

        diag = RTL433_DIAGNOSTICS.get(key)
        if diag:
            add(diag[0], value, diag[1])
            continue

        if unknown is not None and key not in {"id", "mic", "flags", "unknown",
                                               "firmware", "rain_start",
                                               "wind_dir_deg", "battery_ok"}:
            unknown.append(f"{model}.{key}: 매핑 없음")

    # 풍향은 각도라 그대로 둔다(평균을 내면 안 되는 값이라 집계에서 제외된다)
    if isinstance(record.get("wind_dir_deg"), (int, float)):
        add("wind_dir", float(record["wind_dir_deg"]), "deg")
    return rows


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
