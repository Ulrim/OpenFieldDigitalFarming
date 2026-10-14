"""에코윗 게이트웨이(와이파이) 전송을 현장 수집 표준 형식으로 바꾼다.

WS90 자체에는 와이파이가 없다. 433MHz 송신만 한다. 와이파이를 가진 것은
**게이트웨이**(GW1200·GW2000 등)이고, 그것이 433MHz 를 받아 네트워크로
넘긴다. 그래서 라즈베리파이가 WS90 을 받는 길은 둘이다.

    WS90 --433MHz--> RTL-SDR 동글 --> rtl_433        (ofdf.data.rtl433)
    WS90 --433MHz--> 에코윗 게이트웨이 --WiFi--> 이 모듈

게이트웨이 설정에서 'Customized' 업로드를 켜고 프로토콜을 **Ecowitt** 로,
주소를 라즈베리파이로 지정하면 주기마다 HTTP POST 가 온다. 바깥 인터넷을
거치지 않으므로 통신이 끊겨도 돈다.

두 가지를 바로잡아야 한다.

단위가 야드파운드다
-------------------
게이트웨이는 웹UI 표시 설정과 **무관하게** 화씨·마일·인치로 보낸다.
``tempf``, ``windspeedmph``, ``dailyrainin`` 이 그렇다. 그대로 넣으면 기온이
40도 넘게 잡힌다.

강수가 누적이다
---------------
``dailyrainin`` 은 그날 누적이고 자정에 0 으로 돌아간다. 직전 값과의
차이를 쓰고, 값이 줄면(자정·초기화) 그 구간은 0 으로 둔다.

돌풍은 여전히 3초 순간최대가 아니다
-----------------------------------
게이트웨이를 거쳐도 센서는 같은 WS90 이다. ``windgustmph`` 는 2초 간격으로
재어 28초 창의 최댓값이므로 :data:`ofdf.data.rtl433.VAR_GUST_WINDOW` 로
받는다. 경로가 바뀌었다고 통계량이 바뀌지는 않는다.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Iterable
from urllib.parse import parse_qs

import pandas as pd

from ofdf.data.field import ZONE_MAST
from ofdf.data.rtl433 import VAR_GUST_WINDOW, RainAccumulator, rows_to_frame

#: 게이트웨이가 POST 하는 기본 경로. 설정 화면의 Path 와 같아야 한다.
DEFAULT_PATH = "/data/report/"

#: 단위 환산 — 게이트웨이는 야드파운드로 보낸다.
def _f_to_c(value: float) -> float:
    return (value - 32.0) * 5.0 / 9.0


def _mph_to_ms(value: float) -> float:
    return value * 0.44704


def _inch_to_mm(value: float) -> float:
    return value * 25.4


def _inhg_to_hpa(value: float) -> float:
    return value * 33.8639

#: 에코윗 필드 -> (우리 변수, 환산함수).
#:
#: ``solarradiation`` 은 게이트웨이가 이미 W/m^2 로 계산해 준다. rtl_433
#: 경로는 조도(lux)만 주므로 이쪽이 한 단계 정확하다.
ECOWITT_MAP: dict[str, tuple[str, Any]] = {
    "tempf": ("t_air", _f_to_c),
    "humidity": ("rh", None),
    "windspeedmph": ("wind_speed", _mph_to_ms),
    "windgustmph": (VAR_GUST_WINDOW, _mph_to_ms),
    "winddir": ("wind_dir", None),
    "solarradiation": ("solar", None),
    "uv": ("uv_index", None),
    "soilmoisture1": ("soil_moisture", None),
    "tf_ch1": ("soil_temp", _f_to_c),
}

#: 누적으로 오는 필드. 직전 값과의 차이를 쓴다.
CUMULATIVE_MAP: dict[str, tuple[str, Any]] = {
    "dailyrainin": ("rain", _inch_to_mm),
}

#: 작물 위험에는 안 쓰지만 전원·상태로 남기는 필드.
DIAGNOSTICS: dict[str, tuple[str, str, Any]] = {
    "baromrelin": ("pressure_hpa", "hPa", _inhg_to_hpa),
    "ws90cap_volt": ("supercap_v", "V", None),
    "wh90batt": ("battery_v", "V", None),
    "wh90_ver": ("firmware", "", None),
}

#: 수치가 아니거나 우리가 안 쓰는 메타데이터. 경고를 띄우지 않는다.
IGNORED = frozenset({
    "PASSKEY", "stationtype", "runtime", "dateutc", "freq", "model",
    "interval", "heap", "ws90_ver",
})


def _unit(variable: str) -> str:
    from ofdf.data.field import FIELD_VARIABLES
    from ofdf.data.rtl433 import EXTRA_UNITS

    spec = FIELD_VARIABLES.get(variable)
    return spec[0] if spec else EXTRA_UNITS.get(variable, "")


def report_to_rows(
    body: str | dict[str, Any],
    *,
    zone: str = ZONE_MAST,
    sensor_id: str = "WS90-01",
    rain: RainAccumulator | None = None,
    now: dt.datetime | None = None,
    unknown: list[str] | None = None,
) -> list[dict[str, Any]]:
    """POST 본문 하나를 세로 형식 행들로 바꾼다.

    ``body`` 는 ``application/x-www-form-urlencoded`` 문자열이거나 이미
    풀어 놓은 사전이다.
    """
    if isinstance(body, str):
        data: dict[str, Any] = {k: v[0] for k, v in parse_qs(body).items() if v}
    else:
        data = dict(body)

    # 게이트웨이가 보내는 dateutc 가 있으면 그것을 쓴다. 파이 시계가 틀어져도
    # 관측 시각이 유지된다.
    when = now
    if when is None:
        stamp = str(data.get("dateutc", ""))
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
            try:
                when = dt.datetime.strptime(stamp[:19], fmt)
                break
            except ValueError:
                continue
    when = when or dt.datetime.now()
    iso = when.isoformat()
    device = str(data.get("PASSKEY", sensor_id))

    rows: list[dict[str, Any]] = []

    def add(variable: str, value: float, unit: str | None = None) -> None:
        rows.append({
            "ts": iso, "zone": zone, "sensor_id": sensor_id, "variable": variable,
            "value": float(value), "unit": unit if unit is not None else _unit(variable),
            "quality": "ok",
        })

    for key, raw in data.items():
        if key in IGNORED:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue

        mapped = ECOWITT_MAP.get(key)
        if mapped:
            variable, convert = mapped
            add(variable, round(convert(value) if convert else value, 3))
            continue

        cumulative = CUMULATIVE_MAP.get(key)
        if cumulative:
            variable, convert = cumulative
            total = convert(value) if convert else value
            counter = rain or RainAccumulator()
            add(variable, round(counter.step(f"{device}:{key}", total), 3))
            continue

        diag = DIAGNOSTICS.get(key)
        if diag:
            variable, unit, convert = diag
            add(variable, round(convert(value) if convert else value, 3), unit)
            continue

        if unknown is not None:
            unknown.append(f"에코윗 {key}: 매핑 없음")
    return rows


def reports_to_frame(bodies: Iterable[str | dict], **kwargs) -> pd.DataFrame:
    """여러 건을 한 번에 바꾼다. 강수 누적은 이어서 센다."""
    rain = kwargs.pop("rain", None) or RainAccumulator()
    rows: list[dict[str, Any]] = []
    for body in bodies:
        rows += report_to_rows(body, rain=rain, **kwargs)
    return rows_to_frame(rows)
