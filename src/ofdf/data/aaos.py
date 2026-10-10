"""농업기상관측(AAOS) 시간자료 파서.

기상자료개방포털에서 내려받은 '농업기상관측자료 다운로드(시간 자료)' 파일은
확장자가 ``.xls`` 이지만 실제로는 HTML 테이블이다. 월별 파일 하나에 여러
관측지점이 세로로 쌓여 있다.

원본 컬럼
    도명, 지점명, 날짜, 기온(℃), 최고기온(℃), 최저기온(℃), 습도(%),
    풍향, 풍속(m/s), 강수량(mm), 일사량(MJ/m²), 일조시간(hr:mm), 토양수분(%)

주의 — 누적 컬럼
----------------
강수량·일사량·일조시간은 **그 시각의 값이 아니라 자정부터의 누적값**이다.
최고기온·최저기온도 자정 이후의 누적 최고/최저다. 예를 들어 2020-07-13
금천면 강수량은 1.5 -> 9.5 -> 21.5 ... -> 94.5(mm) 로 단조증가하다가
비가 그친 뒤 94.5 에서 멈춘다.

:func:`read_month` 는 이 컬럼들을 하루 단위로 차분해 **시간값**(``rain``,
``solar_mj``, ``sunshine``)으로 바꾸고, 누적 원본은 ``*_cum`` 이름으로
남겨 둔다. 차분하지 않고 쓰면 "일사 700 W/m^2 이상" 같은 기준이 전혀
다른 뜻이 되므로 반드시 시간값을 써야 한다.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

#: 원본 한글 컬럼 -> 내부 스네이크케이스 이름
COLUMN_MAP = {
    "도명": "province",
    "지점명": "station",
    "날짜": "ts",
    "기온": "t_air",
    "최고기온": "t_max",
    "최저기온": "t_min",
    "습도": "rh",
    "풍향": "wind_dir",
    "풍속": "wind_speed",
    "강수량": "rain",
    "일사량": "solar_mj",
    "일조시간": "sunshine",
    "토양수분": "soil_moisture",
}

NUMERIC_COLUMNS = [
    "t_air", "t_max", "t_min", "rh", "wind_speed",
    "rain", "solar_mj", "soil_moisture",
]

#: 자정부터 누적되는 컬럼 — 하루 단위로 차분해야 시간값이 된다
CUMULATIVE_COLUMNS = ["rain", "solar_mj", "sunshine"]

#: 자정 이후 누적 최고/최저인 컬럼 — 시간값이 아니므로 그대로 쓰면 안 된다
RUNNING_EXTREME_COLUMNS = ["t_max", "t_min"]

#: 결측을 뜻하는 표기
MISSING_TOKENS = {"-", "", "nan", "None", "null"}


def _normalise_header(name: str) -> str:
    """``'기온( ℃ )'`` 같은 헤더에서 단위를 떼고 내부 이름으로 바꾼다."""
    base = re.sub(r"\(.*?\)", "", str(name)).strip()
    return COLUMN_MAP.get(base, base)


def _to_numeric(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.where(~cleaned.isin(MISSING_TOKENS))
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_sunshine(series: pd.Series) -> pd.Series:
    """``'0:42'`` (시:분) 형태의 일조시간을 시간 단위 실수로 바꾼다."""
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.where(~cleaned.isin(MISSING_TOKENS))
    parts = cleaned.str.extract(r"^(\d+):(\d+)$")
    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    direct = pd.to_numeric(cleaned, errors="coerce")
    return (hours + minutes / 60).fillna(direct)


def decumulate(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """자정 기준 누적 컬럼을 시간값으로 차분한다.

    하루의 첫 관측은 누적값 그대로가 그 시간의 값이다. 음수 차분(관측기
    리셋이나 결측 복구로 생긴 것)은 0으로 막는다. 누적 원본은 ``*_cum``
    으로 보존한다.
    """
    cols = columns or CUMULATIVE_COLUMNS
    out = df.copy()
    day = out["ts"].dt.normalize()

    for col in cols:
        if col not in out.columns:
            continue
        out[f"{col}_cum"] = out[col]
        diff = out.groupby(["station", day])[col].diff()
        first = out.groupby(["station", day])[col].transform("first")
        out[col] = diff.fillna(first).clip(lower=0.0)

    return out


def read_month(path: str | Path) -> pd.DataFrame:
    """월별 AAOS 파일 하나를 표준 스키마 데이터프레임으로 읽는다."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    tables = pd.read_html(io.StringIO(raw))
    if not tables:
        raise ValueError(f"테이블을 찾지 못했습니다: {path}")

    df = tables[0]
    df.columns = [_normalise_header(c) for c in df.columns]

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = _to_numeric(df[col])
    if "sunshine" in df.columns:
        df["sunshine"] = _parse_sunshine(df["sunshine"])
    if "wind_dir" in df.columns:
        df["wind_dir"] = _to_numeric(df["wind_dir"])

    df = df.dropna(subset=["ts"]).sort_values(["station", "ts"])
    df = decumulate(df)

    keep = [
        "station", "ts", *NUMERIC_COLUMNS, "sunshine", "wind_dir",
        *[f"{c}_cum" for c in CUMULATIVE_COLUMNS],
    ]
    df = df[[c for c in keep if c in df.columns]]
    return df.reset_index(drop=True)


def load_directory(directory: str | Path, pattern: str = "*.xls") -> pd.DataFrame:
    """월별 파일이 모인 디렉터리를 한 장의 시계열로 합친다.

    같은 (지점, 시각)이 여러 파일에 중복돼 있으면 뒤에 읽은 값을 버린다.
    """
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"{directory} 에서 {pattern} 파일을 찾지 못했습니다")

    frames = [read_month(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["station", "ts"], keep="first")
    return df.sort_values(["station", "ts"]).reset_index(drop=True)


def availability_report(df: pd.DataFrame) -> pd.DataFrame:
    """지점 x 연도별 관측항목 가용률(%)을 집계한다.

    토양수분·풍속처럼 결측이 많은 항목을 모델 입력으로 쓸 수 있는지
    판단하기 위한 표다.
    """
    work = df.copy()
    work["year"] = work["ts"].dt.year
    cols = [c for c in NUMERIC_COLUMNS if c in work.columns]
    report = (
        work.groupby(["station", "year"])[cols]
        .apply(lambda g: g.notna().mean() * 100)
        .round(1)
    )
    report["n_hours"] = work.groupby(["station", "year"]).size()
    return report.reset_index()
