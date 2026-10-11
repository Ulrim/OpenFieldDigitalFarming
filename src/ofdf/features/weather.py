"""기상 시계열 입력 피처 생성.

단순 임계값 제어와 AI의 차이는 '지금 값이 기준을 넘었는가'가 아니라
'최근 몇 시간의 변화추세가 어디로 가고 있는가'를 본다는 데 있다.
그래서 현재값 외에 지연값, 구간 통계, 변화율, 지속시간을 만든다.

모든 피처는 **현재 시각 이하의 값만** 사용한다. 미래값은 라벨에만 쓴다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ofdf.labels.risk import LEAF_WETNESS_THRESHOLD, RiskThresholds

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

    th = RiskThresholds()
    t_air = _safe(df, "t_air")
    soil = _safe(df, "soil_index")

    conditions = {
        # 실측 엽면습윤센서는 연속값이라 바닥 잡음이 있다. 라벨과 같은 임계를 쓴다.
        "run_wet": _safe(df, "leaf_wetness") >= LEAF_WETNESS_THRESHOLD,
        "run_rh90": _safe(df, "rh") >= 90.0,
        "run_rain": _safe(df, "rain") > 0.0,
        "run_dry": _safe(df, "rain").fillna(0.0) <= 0.0,   # 무강수 지속(토양 건조와 다르다)
        "run_hot30": t_air >= 30.0,
        "run_cold4": t_air <= 4.0,
        "run_canopy2": _safe(df, "canopy_temp") <= 2.0,

        # 아래 셋은 **고온·건조 라벨이 실제로 쓰는 조건**의 지속시간이다.
        #
        # 그 라벨에서 주의와 경계를 가르는 것은 세기가 아니라 **지속시간**이다.
        # 건조 분기는 같은 조건이 1시간이면 주의, 2시간이면 경계다. 즉 주의는
        # 사실상 '건조 구간의 첫 1시간'이고, 그래서 주의 표본이 경계보다 드물다.
        #
        # 그런데 피처에는 그 지속시간이 없었다. run_dry 는 '비가 안 온 시간'
        # 이지 토양이 마른 시간이 아니고, run_hot30 은 기온만 보고 라벨이
        # 함께 요구하는 일사를 보지 않는다. 모델이 '이미 세 시간째 마른
        # 상태'인지 '방금 마르기 시작'인지 구분할 수단이 없으니, 정상을
        # 주의로 부르는 일이 잦았다(주의 정밀도 0.31).
        #
        # 모두 현재 시각까지의 관측으로만 만든다. 라벨은 미래 시각의 상태이므로
        # 누설이 아니다.
        "run_soil_low": soil <= th.soil_low,
        "run_dry_warm": (soil <= th.soil_low) & (t_air >= th.dry_min_temp),
        "run_hot_solar": (t_air >= th.heat_caution_temp)
                         & (_safe(df, "solar_w") >= th.heat_caution_solar),
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
