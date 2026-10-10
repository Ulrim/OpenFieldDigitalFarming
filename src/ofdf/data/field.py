"""실증포장 현장 센서 수집 데이터.

사업계획서 '착수 설계 확정 사양 ② 센서·통신·전장' 에 정의된 구성을 읽는다.

    기상 마스트(H 5.0m)  일사량·강우량·대기 온습도·토양 수분/온도/EC, 풍속(1초·3초 순간최대)
                          1분 수집·5분 집계, RS-485 -> Wi-Fi(Modbus TCP)
    적용구간 내부         캐노피 하부 온습도 2점, 엽면습윤·초관부(지상 10cm) 온도 각 2점,
                          토양수분·지온 2점(10·20cm), 조도, 강우 감지 — Zigbee 3.0 메시
    일반 노지 비교구간    적용구간과 대칭 위치에 같은 항목
    시설·안전             차광 위치, 개폐기·펌프 전류, 관수·살수 유량, 침수, 비상정지
                          이벤트·5초·1분

표준 입력 형식(long)
--------------------
MQTT·Zigbee 수집기에서 나오는 그대로의 세로 형식을 1차 형식으로 삼는다.
센서를 늘리거나 줄여도 스키마가 바뀌지 않고, 결측이 행 부재로 자연스럽게 표현된다::

    ts,zone,sensor_id,variable,value,unit,quality
    2026-10-01T00:00:00+09:00,mast,MAST-01,t_air,12.4,degC,ok
    2026-10-01T00:00:00+09:00,treatment,CN-01,canopy_temp,10.9,degC,ok
    2026-10-01T00:00:00+09:00,treatment,CN-02,canopy_temp,11.2,degC,ok

가로(wide) 형식도 읽을 수 있다. 컬럼명이 ``zone.variable`` 또는 ``variable`` 이면 된다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

#: 구간 이름
ZONE_MAST = "mast"              # 기상 마스트(두 구간 공통 기준)
ZONE_TREATMENT = "treatment"    # 시제품 적용구간
ZONE_CONTROL = "control"        # 일반 노지 비교구간
ZONE_FACILITY = "facility"      # 시설·안전

#: 수집 변수 -> (단위, 물리 유효범위). 품질관리와 집계 방식을 정하는 데 쓴다.
FIELD_VARIABLES: dict[str, tuple[str, float, float]] = {
    "t_air": ("degC", -25.0, 50.0),
    "rh": ("%", 0.0, 100.0),
    "solar": ("W/m2", 0.0, 2000.0),        # 마스트 사양 0~2,000 W/m²
    "rain": ("mm", 0.0, 50.0),             # 티핑 0.2mm, 1분 적산
    "rain_detect": ("bool", 0.0, 1.0),
    "wind_speed": ("m/s", 0.0, 60.0),
    "wind_gust_1s": ("m/s", 0.0, 60.0),
    "wind_gust_3s": ("m/s", 0.0, 60.0),    # 강풍 안전규칙 기준값
    "canopy_temp": ("degC", -25.0, 50.0),  # 지상 10cm 초관부
    "leaf_wetness": ("ratio", 0.0, 1.0),   # 0~1 또는 0/1
    "soil_moisture": ("%", 0.0, 100.0),
    "soil_temp": ("degC", -25.0, 50.0),
    "soil_ec": ("dS/m", 0.0, 20.0),
    "illuminance": ("lx", 0.0, 200000.0),
    # 시설·안전
    "shade_position": ("ratio", 0.0, 1.0),  # 0 완전회수 ~ 1 완전전개
    "pump_current": ("A", 0.0, 60.0),
    "motor_current": ("A", 0.0, 60.0),
    "irrigation_flow": ("L/min", 0.0, 500.0),
    "spray_flow": ("L/min", 0.0, 500.0),
    "enclosure_flood": ("bool", 0.0, 1.0),
    "emergency_stop": ("bool", 0.0, 1.0),
}

#: 구간 안에 여러 점이 설치된 변수를 합치는 방식.
#: 센서 1점이 고장나도 판단이 흔들리지 않도록 평균이 아니라 중앙값을 쓴다.
POINT_AGGREGATION = "median"

#: 시간 집계 방식. 강수는 적산, 돌풍은 최댓값, 나머지는 평균이다.
TIME_AGGREGATION: dict[str, str] = {
    "rain": "sum",
    "rain_detect": "max",
    "wind_gust_1s": "max",
    "wind_gust_3s": "max",
    "wind_speed": "mean",
    "leaf_wetness": "mean",
    "emergency_stop": "max",
    "enclosure_flood": "max",
    "irrigation_flow": "sum",
    "spray_flow": "sum",
}
DEFAULT_TIME_AGGREGATION = "mean"

#: 범위를 살짝 벗어난 값을 버리지 않고 경계로 자르는 변수와 허용 폭.
#:
#: 일사계는 야간에 열전대 오프셋 때문에 -20 W/m^2 정도의 음수를 정상적으로
#: 내보낸다. 이것을 범위이탈로 보고 지우면 야간 일사가 통째로 결측이 되어
#: (실측에서 전체의 37%), 야간을 보는 서리·병해 판단이 망가진다.
#: 습도·엽면습윤도 계기 특성상 0/100 을 조금 넘길 수 있다.
CLAMP_TOLERANCE: dict[str, float] = {
    "solar": 30.0,
    "rh": 3.0,
    "leaf_wetness": 0.05,
    "soil_moisture": 2.0,
    "shade_position": 0.05,
}

#: 품질 플래그 가운데 값을 버려야 하는 것
BAD_QUALITY = {"bad", "fault", "error", "stale", "nc"}

LONG_COLUMNS = {"ts", "zone", "variable", "value"}


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    """시각을 파싱하고 시간대를 떼어 낸다(현장은 KST 단일 시간대)."""
    ts = pd.to_datetime(series, errors="coerce", utc=False, format="mixed")
    try:
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return ts


def read_long(path: str | Path) -> pd.DataFrame:
    """세로 형식 수집 파일(CSV/JSONL)을 읽는다."""
    path = Path(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)

    missing = LONG_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"세로 형식에 필요한 컬럼이 없습니다: {sorted(missing)}")

    df["ts"] = _coerce_timestamp(df["ts"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    if "quality" in df.columns:
        bad = df["quality"].astype(str).str.lower().isin(BAD_QUALITY)
        df.loc[bad, "value"] = np.nan

    return df.dropna(subset=["ts"]).reset_index(drop=True)


def read_wide(path: str | Path, default_zone: str = ZONE_TREATMENT) -> pd.DataFrame:
    """가로 형식 파일을 세로 형식으로 바꿔 읽는다.

    컬럼명이 ``zone.variable`` 이면 구간을 그대로 쓰고, ``variable`` 뿐이면
    ``default_zone`` 으로 본다.
    """
    df = pd.read_csv(path)
    ts_col = next((c for c in df.columns if c.lower() in {"ts", "time", "timestamp", "datetime", "날짜"}), None)
    if ts_col is None:
        raise ValueError("시각 컬럼을 찾지 못했습니다 (ts/time/timestamp/datetime)")

    records = []
    stamps = _coerce_timestamp(df[ts_col])
    for column in df.columns:
        if column == ts_col:
            continue
        zone, _, variable = column.rpartition(".")
        records.append(
            pd.DataFrame({
                "ts": stamps,
                "zone": zone or default_zone,
                "sensor_id": column,
                "variable": variable,
                "value": pd.to_numeric(df[column], errors="coerce"),
            })
        )
    return pd.concat(records, ignore_index=True).dropna(subset=["ts"]).reset_index(drop=True)


def read_field(path: str | Path, default_zone: str = ZONE_TREATMENT) -> pd.DataFrame:
    """형식을 자동 판별해 읽는다."""
    path = Path(path)
    if path.is_dir():
        frames = [read_field(p, default_zone) for p in sorted(path.iterdir())
                  if p.suffix.lower() in {".csv", ".jsonl", ".ndjson"}]
        if not frames:
            raise FileNotFoundError(f"{path} 에 수집 파일이 없습니다")
        return pd.concat(frames, ignore_index=True)

    try:
        return read_long(path)
    except ValueError:
        return read_wide(path, default_zone)


def clamp_tolerated(df: pd.DataFrame) -> pd.DataFrame:
    """허용 폭 안에서 벗어난 값을 경계로 자른다(버리지 않는다)."""
    out = df.copy()
    for variable, tolerance in CLAMP_TOLERANCE.items():
        if variable not in FIELD_VARIABLES:
            continue
        _, low, high = FIELD_VARIABLES[variable]
        rows = out["variable"] == variable
        near_low = rows & out["value"].between(low - tolerance, low, inclusive="left")
        near_high = rows & out["value"].between(high, high + tolerance, inclusive="right")
        out.loc[near_low, "value"] = low
        out.loc[near_high, "value"] = high
    return out


def flag_out_of_range(df: pd.DataFrame) -> pd.Series:
    """물리 유효범위를 벗어난 값을 표시한다(허용 폭 적용 후)."""
    flags = pd.Series(False, index=df.index)
    for variable, (_, low, high) in FIELD_VARIABLES.items():
        rows = df["variable"] == variable
        flags |= rows & ((df["value"] < low) | (df["value"] > high))
    return flags


def reconcile_points(df: pd.DataFrame, *, spread_limit: float = 3.0) -> pd.DataFrame:
    """같은 구간·같은 변수의 여러 측정점을 하나로 합친다.

    적용구간은 엽면습윤·초관부 온도를 2점씩 둔다. 한 점이 고장나거나
    국소 그늘에 들어가도 판단이 흔들리지 않도록 평균이 아니라 중앙값을 쓰고,
    점 사이 벌어짐(spread)을 함께 남겨 센서 이상 신호로 쓴다.

    Returns
    -------
    pd.DataFrame
        ``ts, zone, variable, value, n_points, spread, spread_alarm`` 컬럼.
    """
    clean = clamp_tolerated(df)
    clean.loc[flag_out_of_range(clean), "value"] = np.nan

    grouped = clean.groupby(["ts", "zone", "variable"], sort=False)["value"]
    out = pd.DataFrame({
        "value": grouped.median() if POINT_AGGREGATION == "median" else grouped.mean(),
        "n_points": grouped.count(),
        "spread": grouped.max() - grouped.min(),
    }).reset_index()

    out["spread_alarm"] = (out["n_points"] >= 2) & (out["spread"] > spread_limit)
    return out


def to_wide(reconciled: pd.DataFrame) -> pd.DataFrame:
    """구간별 변수를 ``zone.variable`` 컬럼의 가로 표로 편다."""
    wide = reconciled.pivot_table(
        index="ts", columns=["zone", "variable"], values="value", aggfunc="first"
    )
    wide.columns = [f"{zone}.{variable}" for zone, variable in wide.columns]
    return wide.sort_index()


def resample(wide: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """변수 성격에 맞는 방식으로 시간 집계한다.

    사업계획서의 수집 설계는 '1분 수집·5분 집계'이고, AI 추론은 1시간
    격자에서 돈다. ``freq="5min"`` 으로 저장용 집계를, ``freq="h"`` 로
    모델 입력을 만든다.
    """
    rules = {}
    for column in wide.columns:
        variable = column.rpartition(".")[2]
        rules[column] = TIME_AGGREGATION.get(variable, DEFAULT_TIME_AGGREGATION)
    return wide.resample(freq).agg(rules)


def load(path: str | Path, freq: str = "h", *, spread_limit: float = 3.0) -> pd.DataFrame:
    """수집 파일 -> 품질관리 -> 측정점 통합 -> 시간 집계까지 한 번에."""
    raw = read_field(path)
    reconciled = reconcile_points(raw, spread_limit=spread_limit)
    return resample(to_wide(reconciled), freq)


def zone_frame(wide: pd.DataFrame, zone: str, *, fallback_zone: str = ZONE_MAST) -> pd.DataFrame:
    """한 구간의 데이터를 모델 입력 스키마(접두사 없는 컬럼)로 뽑는다.

    구간 내부에 없는 항목(일사·강우·풍속 등)은 기상 마스트 값으로 채운다.
    두 구간이 같은 마스트를 기준으로 쓰기 때문이다.
    """
    def pick(source_zone: str) -> pd.DataFrame:
        prefix = f"{source_zone}."
        columns = [c for c in wide.columns if c.startswith(prefix)]
        return wide[columns].rename(columns=lambda c: c[len(prefix):])

    frame = pick(zone)
    if fallback_zone and fallback_zone != zone:
        mast = pick(fallback_zone)
        for column in mast.columns:
            if column not in frame.columns:
                frame[column] = mast[column]
            else:
                frame[column] = frame[column].fillna(mast[column])
    return frame
