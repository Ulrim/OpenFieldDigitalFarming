"""농지별 센서편차·기상특성 보정.

위험판단 모델은 농업기상 관측소 자료로 학습했는데, 실제 판단은 실증포장
센서값으로 한다. 둘 사이에는 두 종류의 차이가 있고, 다루는 법이 다르다.

**센서 편차** — 계기 자체의 치우침, 설치 높이·차폐 차이.
    없애야 할 잡음이다. 관측소 기준으로 되돌려 모델이 학습 때와 같은
    눈금을 보게 한다.

**농지 미기상** — 그 포장이 실제로 더 춥거나 습한 것.
    지켜야 할 신호다. 이것까지 지우면 농지 단위 판단을 하는 의미가 없다.

둘을 데이터만으로 완전히 가를 수는 없다. 그래서 여기서는

* 시간대별 평균 차이를 **진단값으로 산출**하고(보정 전후 성능 비교 산출물),
* 모델 입력으로 쓸 때는 **되돌림 세기(strength)를 조절**할 수 있게 한다.
  ``strength=1.0`` 이면 관측소 눈금으로 완전히 되돌리고, ``0.0`` 이면
  현장값을 그대로 쓴다. 기본값 0.5 는 절반만 되돌린다는 뜻이 아니라
  "계기 편차는 지우되 미기상은 남긴다"는 절충의 출발점이며,
  실증 자료가 쌓이면 위험유형별 성능으로 정한다.

대리지표 교체
-------------
초관부 온도·엽면습윤·토양수분은 공공자료에 없어 대리지표로 만들어 썼다.
현장 실측이 붙으면 그대로 갈아끼우면 되지만, **토양수분만은 예외다.**
학습에 쓴 ``soil_index`` 는 물수지로 만든 0~100 지수이고 실측은 체적수분율(%)
이라 절대값이 다르다. 임계값을 분위수로 옮겨 맞춰야 한다
(:func:`match_threshold_by_quantile`).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

import numpy as np
import pandas as pd

#: 관측소와 현장에 모두 있어 편차를 잴 수 있는 변수
COMPARABLE_VARIABLES = ["t_air", "rh", "solar", "rain", "wind_speed"]

#: 현장에만 있어 대리지표를 대체하는 변수 (대체 대상 -> 실측 컬럼)
PROXY_REPLACEMENTS = {
    "canopy_temp": "canopy_temp",
    "leaf_wetness": "leaf_wetness",
}

#: 보정에 필요한 최소 중복 관측 시간. 이보다 적으면 편차를 믿지 않는다.
MIN_OVERLAP_HOURS = 72


@dataclass
class VariableBias:
    """변수 하나의 편차 추정."""

    variable: str
    offset: float                                   # 전체 중앙값 차이 (관측소 - 현장)
    scale: float = 1.0                              # 변동폭 비율
    hourly_offset: dict[int, float] = dataclass_field(default_factory=dict)
    n_overlap: int = 0
    reliable: bool = False

    def apply(self, series: pd.Series, index: pd.DatetimeIndex, strength: float = 1.0) -> pd.Series:
        """현장값을 관측소 눈금 쪽으로 ``strength`` 만큼 되돌린다."""
        if not self.reliable or strength <= 0:
            return series

        offsets = pd.Series(index.hour, index=series.index).map(self.hourly_offset)
        offsets = offsets.fillna(self.offset)

        centre = series.median()
        scaled = centre + (series - centre) * (1.0 + (self.scale - 1.0) * strength)
        return scaled + offsets * strength


@dataclass
class Calibration:
    """농지 하나에 대한 보정 묶음."""

    station: str
    biases: dict[str, VariableBias] = dataclass_field(default_factory=dict)
    strength: float = 0.5

    def apply(self, field_frame: pd.DataFrame) -> pd.DataFrame:
        """현장 관측 표에 보정을 적용한다."""
        out = field_frame.copy()
        for variable, bias in self.biases.items():
            if variable in out.columns:
                out[variable] = bias.apply(out[variable], out.index, self.strength)
        return out

    def summary(self) -> pd.DataFrame:
        rows = [
            {
                "변수": b.variable,
                "편차(관측소-현장)": round(b.offset, 3),
                "변동폭 비율": round(b.scale, 3),
                "중복 시간": b.n_overlap,
                "신뢰": "예" if b.reliable else "아니오(자료 부족)",
            }
            for b in self.biases.values()
        ]
        return pd.DataFrame(rows)


def _robust_scale(field: pd.Series, station: pd.Series) -> float:
    """사분위 범위 비로 변동폭 비율을 잰다(이상값에 둔감)."""
    field_iqr = field.quantile(0.75) - field.quantile(0.25)
    station_iqr = station.quantile(0.75) - station.quantile(0.25)
    if not np.isfinite(field_iqr) or field_iqr <= 1e-6:
        return 1.0
    return float(np.clip(station_iqr / field_iqr, 0.5, 2.0))


def fit(
    field_frame: pd.DataFrame,
    station_frame: pd.DataFrame,
    *,
    station_name: str = "",
    variables: list[str] | None = None,
    strength: float = 0.5,
    min_overlap: int = MIN_OVERLAP_HOURS,
) -> Calibration:
    """중복 기간에서 관측소-현장 편차를 추정한다.

    Parameters
    ----------
    field_frame, station_frame
        시각 인덱스를 갖는 시간 단위 표. 공통 시각만 쓴다.
    strength
        모델 입력에 적용할 되돌림 세기(0~1).
    min_overlap
        이보다 중복이 적은 변수는 ``reliable=False`` 로 두고 보정하지 않는다.

    Notes
    -----
    시간대별 편차를 따로 잡는다. 복사냉각은 야간에만 생기고 일사 차폐는
    주간에만 생겨서, 하루 평균 하나로는 둘 다 놓친다.
    """
    targets = variables or COMPARABLE_VARIABLES
    common = field_frame.index.intersection(station_frame.index)
    biases: dict[str, VariableBias] = {}

    for variable in targets:
        if variable not in field_frame.columns or variable not in station_frame.columns:
            continue

        pair = pd.DataFrame({
            "field": field_frame.loc[common, variable],
            "station": station_frame.loc[common, variable],
        }).dropna()

        if pair.empty:
            biases[variable] = VariableBias(variable, 0.0, n_overlap=0, reliable=False)
            continue

        difference = pair["station"] - pair["field"]
        hourly = difference.groupby(pair.index.hour).median().to_dict()

        biases[variable] = VariableBias(
            variable=variable,
            offset=float(difference.median()),
            scale=_robust_scale(pair["field"], pair["station"]),
            hourly_offset={int(k): float(v) for k, v in hourly.items()},
            n_overlap=len(pair),
            reliable=len(pair) >= min_overlap,
        )

    return Calibration(station=station_name, biases=biases, strength=strength)


def agreement_report(
    field_frame: pd.DataFrame,
    station_frame: pd.DataFrame,
    calibration: Calibration,
    *,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """보정 전후로 관측소와 얼마나 가까워졌는지 본다.

    사업계획서의 '현장 보정 전후 성능을 각각 제시' 요구에 쓰는 표다.
    관측소에 가까워지는 것 자체가 목표는 아니지만(농지 미기상은 실재한다),
    계기 편차가 줄었는지 확인하는 지표로는 쓸 수 있다.
    """
    targets = variables or list(calibration.biases)
    corrected = calibration.apply(field_frame)
    common = field_frame.index.intersection(station_frame.index)

    rows = []
    for variable in targets:
        if variable not in field_frame.columns or variable not in station_frame.columns:
            continue
        reference = station_frame.loc[common, variable]

        def stats(values: pd.Series) -> tuple[float, float]:
            delta = (values.loc[common] - reference).dropna()
            if delta.empty:
                return float("nan"), float("nan")
            return float(delta.abs().mean()), float(delta.mean())

        mae_before, bias_before = stats(field_frame[variable])
        mae_after, bias_after = stats(corrected[variable])

        rows.append({
            "변수": variable,
            "보정전 MAE": round(mae_before, 3),
            "보정후 MAE": round(mae_after, 3),
            "MAE 개선": round(mae_before - mae_after, 3),
            "보정전 편차": round(bias_before, 3),
            "보정후 편차": round(bias_after, 3),
            "신뢰": "예" if calibration.biases[variable].reliable else "아니오",
        })
    return pd.DataFrame(rows)


def quantile_map(series: pd.Series, reference: pd.Series) -> pd.Series:
    """측정값을 기준 척도 위로 옮긴다(분위수 사상).

    임계값 하나만 옮기는 :func:`match_threshold_by_quantile` 의 계열 전체판이다.
    **모델 입력**에는 이쪽이 필요하다.

    학습에 쓴 ``soil_index`` 는 물수지 지수(표준편차 23.8, 범위 0~94)이고
    현장 실측 체적수분율은 표준편차 2.4, 범위 25.6~38.2 다. 임계값만 맞추고
    측정값을 그대로 넣으면 모델은 거의 변하지 않는 상수를 보게 되어
    학습 분포 밖으로 나간다. 각 측정값이 자기 분포에서 놓인 분위수를 찾아
    기준 분포의 같은 분위수 값으로 바꾸면 척도가 맞는다.

    라벨 생성에는 쓰지 않는다. 라벨은 실측값과 정합된 임계값으로 만들어야
    '현장에서 실제로 일어난 일'을 뜻하기 때문이다.
    """
    source = series.dropna()
    target = reference.dropna()
    if source.empty or target.empty:
        return series

    quantiles = source.rank(pct=True).clip(0.0, 1.0)
    mapped = pd.Series(
        np.quantile(target.to_numpy(), quantiles.to_numpy()), index=source.index
    )
    return mapped.reindex(series.index)


def match_threshold_by_quantile(
    reference_series: pd.Series, threshold: float, target_series: pd.Series
) -> float:
    """임계값을 분위수 위치로 옮겨 다른 척도에 맞춘다.

    학습에 쓴 ``soil_index`` 는 물수지로 만든 0~100 지수이고 현장 실측은
    체적수분율(%)이다. 절대값이 달라 ``soil_index <= 30`` 을 그대로
    ``soil_moisture <= 30`` 으로 옮기면 뜻이 완전히 바뀐다.
    지수 분포에서 임계값이 놓인 분위수를 찾아, 실측 분포의 같은 분위수를
    새 임계값으로 쓴다.

    Examples
    --------
    >>> import pandas as pd
    >>> index = pd.Series(range(0, 101))
    >>> measured = pd.Series(range(10, 41))      # 체적수분율 10~40%
    >>> round(match_threshold_by_quantile(index, 30.0, measured), 1)
    19.2
    """
    reference = reference_series.dropna()
    target = target_series.dropna()
    if reference.empty or target.empty:
        return float(threshold)

    quantile = float((reference <= threshold).mean())
    return float(target.quantile(np.clip(quantile, 0.0, 1.0)))


def replace_proxies(
    model_frame: pd.DataFrame,
    field_frame: pd.DataFrame,
    *,
    replacements: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """대리지표 컬럼을 현장 실측으로 갈아끼운다.

    Returns
    -------
    (교체된 표, 실제로 교체된 컬럼 목록)
    """
    mapping = replacements or PROXY_REPLACEMENTS
    out = model_frame.copy()
    replaced = []

    for proxy_column, field_column in mapping.items():
        if proxy_column not in out.columns or field_column not in field_frame.columns:
            continue
        measured = field_frame[field_column].reindex(out.index)
        if measured.notna().sum() == 0:
            continue
        out[proxy_column] = measured.fillna(out[proxy_column])
        replaced.append(proxy_column)

    return out, replaced
