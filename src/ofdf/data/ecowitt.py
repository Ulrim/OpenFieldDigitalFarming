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

WS90 의 강수는 ``*_piezo`` 로 온다
----------------------------------
WS90 은 티핑버킷이 아니라 **압전식**이다. 그래서 게이트웨이가 보내는
강수 필드 이름이 다르다 — ``dailyrainin`` 이 아니라 ``drain_piezo``,
강우강도는 ``rrain_piezo`` 다. 티핑버킷 이름만 받아 두면 와이파이 경로에서
강수가 통째로 비는데, 비가 와도 화면에 0 으로 나오므로 아무도 눈치채지
못한다.

``srain_piezo`` 는 지금 비가 오는지를 0/1 로 준다. 사업계획서가 요구하는
'강우 감지'가 이것이고, 압전식이라 티핑버킷보다 강우 시작을 빨리 잡는다.

보조 채널도 받는다
------------------
게이트웨이는 WS90 하나만 받는 것이 아니다. 토양수분(WH51)·지온(WN34)·
온습도(WH31)·엽면습윤(WH35) 을 채널 1~8 에 붙일 수 있고, 각각
``soilmoisture3``, ``tf_ch2``, ``temp1f``, ``leafwetness_ch1`` 처럼 번호가
붙은 이름으로 온다. 센서를 한 대 더 사는 날 코드를 고치지 않도록 번호를
일반화해 받고, 어느 구간에 달렸는지는 설정에서 읽는다.

돌풍은 여전히 3초 순간최대가 아니다
-----------------------------------
게이트웨이를 거쳐도 센서는 같은 WS90 이다. ``windgustmph`` 는 2초 간격으로
재어 28초 창의 최댓값이므로 :data:`ofdf.data.rtl433.VAR_GUST_WINDOW` 로
받는다. 경로가 바뀌었다고 통계량이 바뀌지는 않는다.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field as dc_field
from pathlib import Path
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


def _pct_to_ratio(value: float) -> float:
    #: 엽면습윤(WH35)은 0~100% 로 오는데 우리 변수는 0~1 비율이다.
    return value / 100.0

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
    #: 압전식(WS90) 강우강도·강우감지. 티핑버킷과 이름이 다르다.
    "rrain_piezo": ("rain_rate", _inch_to_mm),
    "rainratein": ("rain_rate", _inch_to_mm),
    "srain_piezo": ("rain_detect", None),
    #: 10분 평균 풍속·풍향. 순간값과 다른 통계량이라 따로 받는다.
    "windspdmph_avg10m": ("wind_speed_avg10m", _mph_to_ms),
}

#: 누적으로 오는 필드. 직전 값과의 차이를 쓴다.
#:
#: 일·누적 둘 다 오면 **일 누적**을 쓴다. 자정 초기화 한 번만 0 으로
#: 떨어지고 나머지는 단조증가라 차분이 가장 안정적이다.
CUMULATIVE_MAP: dict[str, tuple[str, Any]] = {
    "drain_piezo": ("rain", _inch_to_mm),     # WS90(압전) — 이쪽이 먼저다
    "dailyrainin": ("rain", _inch_to_mm),     # 티핑버킷을 따로 달았을 때
}

#: 강수 적산 필드 중 하나만 쓴다. 왼쪽이 먼저다.
#:
#: 같은 비를 두 번 더하면 5분 강수가 두 배가 된다. 압전과 티핑버킷이 같이
#: 달려 있으면 둘 다 ``rain`` 으로 들어오기 때문에, 들어온 것 중 앞선
#: 하나만 쓰고 나머지는 진단으로 돌린다.
RAIN_PRIORITY = ("drain_piezo", "dailyrainin")

#: 작물 위험에는 안 쓰지만 전원·상태·참고로 남기는 필드.
#:
#: 시·주·월·연 누적 강수는 우리가 쓰지 않지만(일 누적에서 차분을 뽑으므로)
#: 받은 값을 버리지 않고 남긴다. 현장에서 '정말 그만큼 왔나'를 따질 때
#: 게이트웨이가 센 값과 우리가 센 값을 맞대 볼 수 있어야 한다.
DIAGNOSTICS: dict[str, tuple[str, str, Any]] = {
    "baromrelin": ("pressure_hpa", "hPa", _inhg_to_hpa),
    "baromabsin": ("pressure_abs_hpa", "hPa", _inhg_to_hpa),
    "ws90cap_volt": ("supercap_v", "V", None),
    "wh90batt": ("battery_v", "V", None),
    "wh90_ver": ("firmware", "", None),
    "tempinf": ("enclosure_temp", "degC", _f_to_c),
    "humidityin": ("enclosure_rh", "%", None),
    "maxdailygust": ("gust_daily_max", "m/s", _mph_to_ms),
    "windgustdir": ("gust_dir", "deg", None),
    "winddir_avg10m": ("wind_dir_avg10m", "deg", None),
    "vpd": ("vpd_gateway", "kPa", None),
    "hrain_piezo": ("rain_hour_total", "mm", _inch_to_mm),
    "erain_piezo": ("rain_event_total", "mm", _inch_to_mm),
    "wrain_piezo": ("rain_week_total", "mm", _inch_to_mm),
    "mrain_piezo": ("rain_month_total", "mm", _inch_to_mm),
    "yrain_piezo": ("rain_year_total", "mm", _inch_to_mm),
    "hourlyrainin": ("rain_hour_total", "mm", _inch_to_mm),
    "eventrainin": ("rain_event_total", "mm", _inch_to_mm),
    "weeklyrainin": ("rain_week_total", "mm", _inch_to_mm),
    "monthlyrainin": ("rain_month_total", "mm", _inch_to_mm),
    "yearlyrainin": ("rain_year_total", "mm", _inch_to_mm),
    "totalrainin": ("rain_all_total", "mm", _inch_to_mm),
    "lightning_num": ("lightning_count", "", None),
    "lightning": ("lightning_distance", "km", None),
    "console_batt": ("console_batt", "V", None),
    "wh25batt": ("battery_ok", "bool", None),
    "wh40batt": ("battery_v", "V", None),
    "wh57batt": ("battery_pct", "%", None),
    "wh65batt": ("battery_ok", "bool", None),
    "wh68batt": ("battery_v", "V", None),
    "wh80batt": ("battery_v", "V", None),
}

#: 번호가 붙어 오는 보조 채널. ``(정규식, 계열, 변수, 환산, 뜻이 갈리는가)``.
#:
#: **채널 번호는 센서 계열마다 따로 센다.** ``soilmoisture1`` 과
#: ``leafwetness_ch1`` 과 ``temp1f`` 은 모두 '1번'이지만 서로 다른 장치다 —
#: 토양수분계(WH51) 1번, 엽면습윤계(WH35) 1번, 온습도계(WH31) 1번. 번호만
#: 보고 묶으면 셋이 한 센서가 되어 배터리 값이 서로를 덮어쓴다. 그래서
#: 계열과 번호를 함께 열쇠로 쓴다(``soil1``·``leaf1``·``th1``).
#:
#: 센서 종류가 이름에 박혀 있는 것(토양수분·지온 프로브·엽면습윤)은 설치
#: 위치를 몰라도 무엇을 재는지 분명하다. 그러나 ``temp1f``·``humidity1`` 은
#: 범용 온습도계라 **어디에 달렸느냐에 따라 기온도 되고 초관부 온도도
#: 된다.** 그래서 설정이 말해 주지 않으면 버린다 — Zigbee 경로에서
#: ``temperature`` 를 다루는 원칙과 같다.
CHANNEL_PATTERNS: tuple[tuple[Any, str, str, Any, bool], ...] = (
    (re.compile(r"^soilmoisture(\d+)$"), "soil", "soil_moisture", None, False),
    (re.compile(r"^tf_ch(\d+)$"), "tf", "soil_temp", _f_to_c, False),
    (re.compile(r"^leafwetness_ch(\d+)$"), "leaf", "leaf_wetness", _pct_to_ratio, False),
    (re.compile(r"^temp(\d+)f$"), "th", "t_air", _f_to_c, True),
    (re.compile(r"^humidity(\d+)$"), "th", "rh", None, True),
)

#: 채널 진단 — 보조 센서의 배터리·원시값. 계열은 위와 같은 규칙으로 맞춘다.
CHANNEL_DIAGNOSTICS: tuple[tuple[Any, str, str, str], ...] = (
    (re.compile(r"^soilad(\d+)$"), "soil", "soil_raw_ad", ""),
    (re.compile(r"^soilbatt(\d+)$"), "soil", "battery_v", "V"),
    (re.compile(r"^tf_batt(\d+)$"), "tf", "battery_v", "V"),
    (re.compile(r"^leaf_batt(\d+)$"), "leaf", "battery_v", "V"),
    (re.compile(r"^batt(\d+)$"), "th", "battery_ok", "bool"),
    (re.compile(r"^pm25_ch(\d+)$"), "pm25", "pm25", "ug/m3"),
    (re.compile(r"^pm25_avg_24h_ch(\d+)$"), "pm25", "pm25_24h", "ug/m3"),
)

#: 계열 이름을 사람이 읽을 말로. 설정에 안 적힌 채널의 메모로 쓴다.
FAMILY_NAMES = {
    "soil": "토양수분(WH51)", "tf": "지온 프로브(WN34)",
    "leaf": "엽면습윤(WH35)", "th": "온습도(WH31)", "pm25": "미세먼지",
}

#: 게이트웨이에 붙은 센서가 **실제로 보내는** 필드.
#:
#: 게이트웨이는 중계기일 뿐이다. 무엇이 오는지는 거기에 무엇을 페어링했느냐로
#: 정해진다. WS90 만 붙여 놓고 '토양수분도 받는다'고 보고하면 안 되므로,
#: 수집 범위 점검(ofdf.data.coverage)은 이 표를 본다.
SENSOR_FIELDS: dict[str, frozenset[str]] = {
    #: WS90 7-in-1 — 기온·습도·풍속·풍향·강우(압전)·조도·UV.
    #: 게이트웨이가 기압(내장)과 실내 온습도(콘솔)도 함께 올린다.
    "WS90": frozenset({
        "tempf", "humidity", "windspeedmph", "windgustmph", "winddir",
        "solarradiation", "uv", "rrain_piezo", "srain_piezo", "drain_piezo",
        "hrain_piezo", "erain_piezo", "wrain_piezo", "mrain_piezo",
        "yrain_piezo", "ws90cap_volt", "wh90batt", "maxdailygust",
        "windgustdir", "windspdmph_avg10m", "winddir_avg10m",
        "baromrelin", "baromabsin", "tempinf", "humidityin", "vpd",
    }),
    #: WH40 티핑버킷 강우계를 따로 달았을 때
    "WH40": frozenset({
        "rainratein", "dailyrainin", "hourlyrainin", "eventrainin",
        "weeklyrainin", "monthlyrainin", "yearlyrainin", "totalrainin",
        "wh40batt",
    }),
    #: WH57 낙뢰
    "WH57": frozenset({"lightning_num", "lightning", "wh57batt"}),
}

#: 게이트웨이 설정에 센서 목록이 없을 때 가정하는 구성.
DEFAULT_SENSORS = ("WS90",)

#: 수치가 아니거나 우리가 안 쓰는 메타데이터. 경고를 띄우지 않는다.
IGNORED = frozenset({
    "PASSKEY", "stationtype", "stationname", "runtime", "dateutc", "freq",
    "model", "interval", "heap", "ws90_ver", "ws85_ver", "lightning_time",
    "rainratein_piezo", "rain_priority", "wh68batt_ok",
})


@dataclass
class Channel:
    """게이트웨이 보조 채널 한 칸의 설치 정보.

    열쇠는 계열과 번호다(``soil1``·``th2``). 번호만으로는 모자라다 —
    :data:`CHANNEL_PATTERNS` 설명 참고.
    """

    family: str
    number: int
    zone: str = ZONE_MAST
    sensor_id: str = ""
    note: str = ""
    #: 센서 종류 이름 -> 우리 변수. 기본값을 덮는다.
    #: 키는 ``soil_moisture``·``soil_temp``·``t_air``·``rh``·``leaf_wetness``.
    #: 빈 문자열을 적으면 그 값을 버린다.
    map: dict[str, str] = dc_field(default_factory=dict)
    #: 게이트웨이 칸에만 쓴다 — 거기 페어링한 센서 모델 목록.
    sensors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.sensor_id:
            self.sensor_id = f"ECO-{self.family.upper()}{self.number}"
        if not self.note:
            self.note = FAMILY_NAMES.get(self.family, self.family)

    @property
    def key(self) -> str:
        return f"{self.family}{self.number}"

    def variable_for(self, default_variable: str) -> str | None:
        if default_variable in self.map:
            return self.map[default_variable] or None
        return default_variable


def load_channels(path: str | Path) -> tuple[Channel | None, dict[str, Channel]]:
    """설치 정보 YAML 의 ``ecowitt:`` 칸을 읽는다.

    Zigbee·433MHz 장치와 **같은 파일**에 둔다. 포장에 무엇이 달려 있는지를
    여러 곳에 나눠 적으면 한쪽만 고치는 날이 온다::

        ecowitt:
          gateway:
            zone: mast
            sensor_id: WS90-01
            note: GW1200 · 폴대 상단 2m
          channels:
            soil1:
              zone: control
              sensor_id: SOIL-C2
              note: 비교구간 깊이 10cm
            th1:
              zone: control
              sensor_id: CN-02
              map:
                t_air: canopy_temp     # 지상 10cm 이므로 기온이 아니다

    열쇠는 계열+번호(``soil1``·``tf2``·``leaf1``·``th1``)다. 번호만 적으면
    토양수분 채널로 본다 — 실증포장에서 번호만 쓰는 채널은 그것뿐이다.

    Returns
    -------
    (게이트웨이 채널 또는 ``None``, {계열+번호: 채널})
    """
    import yaml

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section = raw.get("ecowitt") or {}

    gateway = None
    gw = section.get("gateway")
    if gw:
        gateway = Channel(
            family="gw", number=0, zone=gw.get("zone", ZONE_MAST),
            sensor_id=gw.get("sensor_id", "WS90-01"),
            note=gw.get("note", "에코윗 게이트웨이"),
            map=dict(gw.get("map") or {}),
            sensors=tuple(gw.get("sensors") or DEFAULT_SENSORS),
        )

    channels: dict[str, Channel] = {}
    for key, cfg in (section.get("channels") or {}).items():
        cfg = cfg or {}
        text = str(key)
        match = re.match(r"^([a-zA-Z_]*)(\d+)$", text)
        if not match:
            continue
        family = (match.group(1) or "soil").lower().strip("_")
        number = int(match.group(2))
        channel = Channel(
            family=family, number=number, zone=cfg.get("zone", ZONE_MAST),
            sensor_id=cfg.get("sensor_id", ""), note=cfg.get("note", ""),
            map=dict(cfg.get("map") or {}),
        )
        channels[channel.key] = channel
    return gateway, channels


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
    gateway: Channel | None = None,
    channels: dict[str, Channel] | None = None,
) -> list[dict[str, Any]]:
    """POST 본문 하나를 세로 형식 행들로 바꾼다.

    ``body`` 는 ``application/x-www-form-urlencoded`` 문자열이거나 이미
    풀어 놓은 사전이다.

    ``gateway``·``channels`` 를 넘기면 구간·센서이름·변수매핑을 설정에서
    가져온다. 보조 채널(토양수분·지온·온습도·엽면습윤)은 게이트웨이와
    **다른 구간**에 꽂힐 수 있으므로 채널마다 따로 적는다.
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

    if gateway is not None:
        zone, sensor_id = gateway.zone, gateway.sensor_id
    channels = channels or {}

    rows: list[dict[str, Any]] = []

    def add(variable: str, value: float, unit: str | None = None,
            *, row_zone: str | None = None, row_sensor: str | None = None) -> None:
        rows.append({
            "ts": iso, "zone": row_zone or zone,
            "sensor_id": row_sensor or sensor_id, "variable": variable,
            "value": float(value), "unit": unit if unit is not None else _unit(variable),
            "quality": "ok",
        })

    # 압전과 티핑버킷이 같이 달려 있으면 둘 다 rain 으로 들어와 같은 비를 두
    # 번 센다. 들어온 것 중 앞선 하나만 강수로 쓴다.
    rain_field = next((f for f in RAIN_PRIORITY if f in data), None)

    for key, raw in data.items():
        if key in IGNORED:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            if unknown is not None:
                unknown.append(f"에코윗 {key}: 수치가 아니다({raw!r})")
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
            if key != rain_field:
                # 쓰지 않는 쪽도 버리지 않고 남긴다. 두 계측기가 얼마나
                # 다르게 세는지는 현장에서 따질 거리가 된다.
                add(f"{variable}_alt_total", round(total, 3), "mm")
                continue
            counter = rain or RainAccumulator()
            add(variable, round(counter.step(f"{device}:{key}", total), 3))
            continue

        diag = DIAGNOSTICS.get(key)
        if diag:
            variable, unit, convert = diag
            add(variable, round(convert(value) if convert else value, 3), unit)
            continue

        if _add_channel(key, value, add, channels, unknown):
            continue

        if unknown is not None:
            unknown.append(f"에코윗 {key}: 매핑 없음(값 {raw!r})")
    return rows


def _add_channel(
    key: str,
    value: float,
    add: Any,
    channels: dict[str, Channel],
    unknown: list[str] | None,
) -> bool:
    """``soilmoisture3`` 처럼 번호가 붙은 채널 필드를 처리한다.

    처리했으면 ``True``. 센서 종류가 이름에 박힌 것은 설정이 없어도 받고,
    뜻이 갈리는 범용 온습도계(``temp1f``)는 설정이 없으면 버리고 경고한다.
    """
    for pattern, family, default_variable, convert, ambiguous in CHANNEL_PATTERNS:
        match = pattern.match(key)
        if not match:
            continue
        number = int(match.group(1))
        channel = channels.get(f"{family}{number}")
        if channel is None:
            if ambiguous:
                if unknown is not None:
                    unknown.append(
                        f"에코윗 {key}: 범용 온습도계는 어디에 달렸느냐로 뜻이 "
                        f"갈린다. 설정 ecowitt.channels.{family}{number} 에 "
                        f"어느 변수인지 적어야 한다"
                    )
                return True
            channel = Channel(family=family, number=number)
        variable = channel.variable_for(default_variable)
        if variable is None:
            return True                      # 설정이 일부러 버리라고 한 값
        add(variable, round(convert(value) if convert else value, 4),
            row_zone=channel.zone, row_sensor=channel.sensor_id)
        return True

    for pattern, family, variable, unit in CHANNEL_DIAGNOSTICS:
        match = pattern.match(key)
        if not match:
            continue
        number = int(match.group(1))
        channel = (channels.get(f"{family}{number}")
                   or Channel(family=family, number=number))
        add(variable, value, unit,
            row_zone=channel.zone, row_sensor=channel.sensor_id)
        return True
    return False


def reports_to_frame(bodies: Iterable[str | dict], **kwargs) -> pd.DataFrame:
    """여러 건을 한 번에 바꾼다. 강수 누적은 이어서 센다."""
    rain = kwargs.pop("rain", None) or RainAccumulator()
    rows: list[dict[str, Any]] = []
    for body in bodies:
        rows += report_to_rows(body, rain=rain, **kwargs)
    return rows_to_frame(rows)
