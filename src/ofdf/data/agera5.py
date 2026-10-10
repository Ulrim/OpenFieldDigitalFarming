"""AgERA5 일자료 파서.

``AgERA5_Naju-Daegyo_2x2mean_daily_2015-2026.csv`` 는 나주 대교동 격자
2x2 평균 일자료다. 컬럼명에 ``'항목 [단위]'`` 형태로 단위가 붙어 있다.

이 파일에서 시간단위 농업기상 관측망에 없는 두 가지를 가져온다.

* ``reference_evapotranspiration_all`` — 토양 물수지에 필요한 기준증발산량
* ``cloud_cover_24_hour_mean`` — 야간 복사냉각(서리) 판단 보조
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

#: 내부에서 쓰는 짧은 이름
RENAME = {
    "reference_evapotranspiration_all": "et0",
    "cloud_cover_24_hour_mean": "cloud_cover",
    "precipitation_flux_all": "rain_daily",
    "2m_temperature_24_hour_minimum": "t_min_daily",
    "2m_temperature_24_hour_maximum": "t_max_daily",
    "2m_temperature_night_time_minimum": "t_min_night",
    "solar_radiation_flux_all": "solar_daily",
    "10m_wind_speed_24_hour_mean": "wind_daily",
    "2m_relative_humidity_derived_24_hour_maximum": "rh_max_daily",
}


def _strip_units(name: str) -> str:
    return re.sub(r"\s*\[.*?\]\s*$", "", str(name)).strip()


def read_daily(path: str | Path) -> pd.DataFrame:
    """AgERA5 일자료 CSV를 읽어 날짜 인덱스 데이터프레임으로 돌려준다."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [_strip_units(c) for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.rename(columns=RENAME)
    return df.sort_values("date").reset_index(drop=True)


def hourly_et0(daily: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """일 기준증발산량을 시간별로 배분한다.

    증발산은 낮에 몰리므로 균등 분배 대신 낮 시간에 가중치를 둔 종 모양
    분포(06~18시 중심)를 쓴다. 가중치 합은 1이므로 하루 총량은 보존된다.
    """
    hours = pd.Series(index.hour, index=index)
    weight = ((hours - 6) / 12 * 3.14159265).apply(
        lambda x: max(0.0, __import__("math").sin(x))
    )
    daily_weight = weight.groupby(index.normalize()).transform("sum")
    share = (weight / daily_weight.replace(0, pd.NA)).fillna(1 / 24)

    lookup = daily.set_index("date")["et0"]
    daily_value = pd.Series(index.normalize(), index=index).map(lookup)
    return (daily_value * share).astype(float)
