"""기상 시계열 입력 피처 생성.

단순 임계값 제어와 AI의 차이는 '지금 값이 기준을 넘었는가'가 아니라
'최근 몇 시간의 변화추세가 어디로 가고 있는가'를 본다는 데 있다.
그래서 현재값 외에 지연값, 구간 통계, 변화율, 지속시간을 만든다.

모든 피처는 **현재 시각 이하의 값만** 사용한다. 미래값은 라벨에만 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 구간 통계를 만들 기본 변수
BASE_VARIABLES = [
    "t_air", "rh", "rain", "solar_w", "wind_speed",
    "vpd", "dew_point", "canopy_temp", "soil_index", "leaf_wetness",
]

#: 되돌아볼 구간(시간)
WINDOWS = (1, 3, 6, 12, 24)


def _safe(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(np.nan, index=df.index, dtype=float)


def rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """구간 평균·최대·최소·합과 변화율을 만든다."""
    out: dict[str, pd.Series] = {}

    for col in BASE_VARIABLES:
        series = _safe(df, col)
        out[col] = series

        for w in WINDOWS:
            roll = series.rolling(w, min_periods=1)
            out[f"{col}_mean{w}"] = roll.mean()
            if col == "rain":
                out[f"{col}_sum{w}"] = roll.sum()
            else:
                out[f"{col}_max{w}"] = roll.max()
                out[f"{col}_min{w}"] = roll.min()

        # 변화추세: 현재값 - h시간 전 값
        for h in (1, 3, 6):
            out[f"{col}_delta{h}"] = series - series.shift(h)

    return pd.DataFrame(out, index=df.index)


def duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """조건이 현재까지 몇 시간 연속 유지됐는지 센다."""

    def run_length(mask: pd.Series) -> pd.Series:
        flag = mask.fillna(False).astype(bool)
        block = (~flag).cumsum()
        return flag.groupby(block).cumsum().astype(float)

    conditions = {
        "run_wet": _safe(df, "leaf_wetness") > 0.5,
        "run_rh90": _safe(df, "rh") >= 90.0,
        "run_rain": _safe(df, "rain") > 0.0,
        "run_dry": _safe(df, "rain").fillna(0.0) <= 0.0,
        "run_hot30": _safe(df, "t_air") >= 30.0,
        "run_cold4": _safe(df, "t_air") <= 4.0,
        "run_canopy2": _safe(df, "canopy_temp") <= 2.0,
    }
    return pd.DataFrame({k: run_length(v) for k, v in conditions.items()}, index=df.index)


def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """시각·계절을 주기함수로 넣는다. 야간/일출 전후 판단에 쓰인다."""
    hour = index.hour.to_numpy()
    doy = index.dayofyear.to_numpy()
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "is_night": ((hour >= 18) | (hour < 8)).astype(float),
            "month": index.month.to_numpy().astype(float),
        },
        index=index,
    )


def daily_context(df: pd.DataFrame) -> pd.DataFrame:
    """AgERA5 일자료에서 온 보조 컬럼을 그대로 피처로 넘긴다.

    예보 대체값 성격이라 그날 전체에 같은 값이 깔린다. 운량은 야간
    복사냉각(서리) 판단에, 기준증발산량은 건조 추세 판단에 쓰인다.
    """
    cols = [c for c in ("cloud_cover", "et0", "t_min_night", "rh_max_daily") if c in df.columns]
    return df[cols].copy() if cols else pd.DataFrame(index=df.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """지점 하나의 시간순 데이터에서 전체 입력 피처를 만든다."""
    parts = [
        rolling_features(df),
        duration_features(df),
        calendar_features(df.index),
        daily_context(df),
    ]
    features = pd.concat(parts, axis=1)
    return features.loc[:, ~features.columns.duplicated()]
