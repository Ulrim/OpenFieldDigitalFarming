"""위험 정답값(라벨) 생성.

사업계획서 '위험 정답값(라벨) 정의 및 확정 절차' 표를 그대로 코드로 옮긴
것이다. 위험유형 5종을 정상(0)/주의(1)/경계(2) 3등급으로 판정한다.

강풍·도복은 AI 판단 대상이 아니라 현장 즉응규칙(L1)이므로 별도 함수로
두고, 학습 라벨에는 넣지 않는다.

라벨 생성 원칙
--------------
* 시간 단위 관측에서 **그 시각의 상태등급**을 먼저 만든다.
* 예측 라벨은 ``(t, t+h]`` 구간의 최대 등급이다. 즉 "앞으로 h시간 안에
  도달할 가장 높은 위험등급"을 맞히는 문제가 된다.
* 라벨 계산에만 미래값을 쓰고, 입력 피처에는 절대 쓰지 않는다
  (사업계획서 '미래정보의 잘못된 입력 방지').

확인 필요 — 주의/경계 기준의 비대칭
-----------------------------------
사업계획서 표에서 강우·과습 주의는 "3시간 강우 20mm **그리고** 토양수분
상한 2시간"(AND)인데 경계는 "6시간 강우 30mm **또는** 토양수분 상한
6시간"(OR)이다. 조건이 더 엄격한 주의가 경계보다 드물게 발생한다.

나주 농업기상 10년 x 5지점(작기 9~11월, 109,200시간)으로 확인한 결과
주의 표본이 강우·과습 20건, 복합위험 9건까지 떨어졌다. 이 상태로는
위험유형별 종합점수(Macro F1)가 표본 수십 건짜리 '주의' 클래스에
좌우돼 성능 평가가 불안정해진다.

:class:`RiskThresholds` 의 ``rain_caution_mode`` 로 두 기준을 모두 쓸 수
있게 했다. ``"and"`` 가 문서 원문이고 ``"or"`` 가 권고안이다. 최종 기준은
3자 검토(실증농가·재배/병해 전문가·AI 담당)에서 확정한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NORMAL, CAUTION, WARNING = 0, 1, 2

#: 엽면습윤을 '젖음'으로 볼 기준값.
#:
#: 대리지표는 0.0/1.0 만 내보내지만 **실측 엽면습윤센서는 0~1 연속값**이고,
#: 마른 상태에서도 0.05 안팎의 바닥 잡음이 올라온다. 0 초과를 젖음으로 보면
#: 마른 날에도 습윤 지속시간이 끝없이 쌓여 병해 주의가 상시 발령된다.
#: 센서 제조사 권장 임계(보통 전체 범위의 30~50%)에 맞춰 현장에서 조정한다.
LEAF_WETNESS_THRESHOLD = 0.5

#: AI 판단 대상 위험유형
RISK_TYPES = ["heat_dry", "rain_wet", "disease", "frost", "compound"]

RISK_LABELS_KO = {
    "heat_dry": "고온·건조",
    "rain_wet": "강우·과습",
    "disease": "병해 유리환경",
    "frost": "저온·서리",
    "compound": "복합위험",
    "wind": "강풍·도복",
}


@dataclass
class RiskThresholds:
    """사업계획서 초기 기준값. 현장 보정 시 이 값만 바꾸면 된다."""

    # 고온·건조
    heat_caution_temp: float = 30.0
    heat_caution_solar: float = 700.0
    heat_warning_temp: float = 33.0
    heat_warning_solar: float = 800.0
    heat_caution_hours: int = 1
    heat_warning_hours: int = 1
    dry_caution_hours: int = 1          # 토양수분 하한 30분 -> 1시간 해상도
    dry_warning_hours: int = 2

    # 강우·과습
    #: 주의 판정 결합 방식.
    #: ``"and"`` = 사업계획서 원문(3시간 강우 20mm 그리고 토양수분 상한 2시간)
    #: ``"or"``  = 권고안(둘 중 하나만 성립해도 주의)
    #: 원문대로 두면 주의가 경계보다 엄격해져 주의 표본이 거의 생기지 않는다
    #: (작기 10년 x 5지점에서 20건). 등급별 균형 성능을 보려면 ``"or"`` 를 쓴다.
    rain_caution_mode: str = "and"
    rain_caution_3h_mm: float = 20.0
    rain_warning_6h_mm: float = 30.0
    wet_caution_hours: int = 2
    wet_warning_hours: int = 6

    # 병해 유리환경
    disease_caution_wet_hours: int = 6
    disease_caution_rh: float = 90.0
    disease_caution_temp: tuple[float, float] = (8.0, 24.0)
    disease_warning_wet_hours: int = 10
    disease_warning_rh: float = 95.0
    disease_warning_temp: tuple[float, float] = (12.0, 22.0)

    # 저온·서리
    frost_caution_tmin: float = 4.0
    frost_caution_wind: float = 2.0
    frost_caution_cloud: float = 0.5     # 맑음 판정(운량 0~1)
    frost_caution_canopy: float = 2.0
    frost_warning_canopy: float = 0.0
    frost_warning_canopy_hours: int = 1
    frost_warning_tmin: float = 0.0

    # 토양수분 상·하한(농가 기준). soil_index(0~100) 기준 절대값이다.
    # 분위수로 잡으면 겨울 건조까지 '하한 이탈'이 되어 계절과 무관하게
    # 위험이 깔리므로 절대값을 쓴다.
    soil_low: float = 30.0
    soil_high: float = 90.0

    #: 건조 분기를 고온·건조로 셀 최저 기온. 겨울 건조는 대파 고온스트레스가
    #: 아니므로 이 온도 미만이면 고온·건조 위험으로 세지 않는다.
    dry_min_temp: float = 20.0

    # 강풍(규칙 기반)
    wind_caution: float = 12.0
    wind_warning: float = 15.0

    #: 야간 시간대(저온·서리 판단 구간)
    night_hours: tuple[int, ...] = field(default=tuple(list(range(18, 24)) + list(range(0, 8))))


def _rolling_hours(mask: pd.Series, hours: int) -> pd.Series:
    """``mask`` 가 ``hours`` 시간 연속으로 참인지 표시한다."""
    if hours <= 1:
        return mask.astype(bool)
    return mask.rolling(hours, min_periods=hours).sum().ge(hours).fillna(False)


def _wet_run_length(wet: pd.Series, threshold: float = LEAF_WETNESS_THRESHOLD) -> pd.Series:
    """엽면습윤이 현재까지 몇 시간 연속됐는지 센다.

    ``threshold`` 이상을 젖음으로 본다. 실측센서의 바닥 잡음(마른 상태에서도
    0.05 안팎)을 젖음으로 세지 않기 위해 필요하다.
    """
    wet_bool = (wet.fillna(0) >= threshold)
    block = (~wet_bool).cumsum()
    return wet_bool.groupby(block).cumsum().astype(float)


def heat_dry_state(df: pd.DataFrame, th: RiskThresholds, soil_low: float) -> pd.Series:
    """고온·건조 상태등급."""
    hot_c = _rolling_hours(
        (df["t_air"] >= th.heat_caution_temp) & (df["solar_w"] >= th.heat_caution_solar),
        th.heat_caution_hours,
    )
    hot_w = _rolling_hours(
        (df["t_air"] >= th.heat_warning_temp) & (df["solar_w"] >= th.heat_warning_solar),
        th.heat_warning_hours,
    )
    # 건조 분기는 작물이 실제로 수분 스트레스를 받는 기온대에서만 센다
    dry = (df["soil_index"] <= soil_low) & (df["t_air"] >= th.dry_min_temp)
    dry_c = _rolling_hours(dry, th.dry_caution_hours)
    dry_w = _rolling_hours(dry, th.dry_warning_hours)

    state = pd.Series(NORMAL, index=df.index, dtype=int)
    state[hot_c | dry_c] = CAUTION
    state[hot_w | dry_w] = WARNING
    return state


def rain_wet_state(df: pd.DataFrame, th: RiskThresholds, soil_high: float) -> pd.Series:
    """강우·과습 상태등급."""
    rain3 = df["rain"].fillna(0).rolling(3, min_periods=1).sum()
    rain6 = df["rain"].fillna(0).rolling(6, min_periods=1).sum()
    saturated = df["soil_index"] >= soil_high

    heavy_rain = rain3 >= th.rain_caution_3h_mm
    soil_full = _rolling_hours(saturated, th.wet_caution_hours)
    caution = (heavy_rain & soil_full) if th.rain_caution_mode == "and" else (heavy_rain | soil_full)
    warning = (rain6 >= th.rain_warning_6h_mm) | _rolling_hours(saturated, th.wet_warning_hours)

    state = pd.Series(NORMAL, index=df.index, dtype=int)
    state[caution] = CAUTION
    state[warning] = WARNING
    return state


def disease_state(df: pd.DataFrame, th: RiskThresholds) -> pd.Series:
    """병해 유리환경 상태등급(노균병·잎마름병 발생 유리조건)."""
    run = _wet_run_length(df["leaf_wetness"])
    lo_c, hi_c = th.disease_caution_temp
    lo_w, hi_w = th.disease_warning_temp

    caution = (
        (run >= th.disease_caution_wet_hours)
        & (df["rh"] >= th.disease_caution_rh)
        & df["t_air"].between(lo_c, hi_c)
    )
    warning = (
        (run >= th.disease_warning_wet_hours)
        & (df["rh"] >= th.disease_warning_rh)
        & df["t_air"].between(lo_w, hi_w)
    )

    state = pd.Series(NORMAL, index=df.index, dtype=int)
    state[caution] = CAUTION
    state[warning] = WARNING
    return state


def frost_state(df: pd.DataFrame, th: RiskThresholds) -> pd.Series:
    """저온·서리 상태등급.

    대파 서리피해는 기온이 아니라 초관부(지상 10cm) 온도로 판단한다.
    야간 시간대에만 평가한다.
    """
    is_night = df.index.to_series().dt.hour.isin(th.night_hours).to_numpy()

    wind = df.get("wind_speed")
    calm = (wind <= th.frost_caution_wind) if wind is not None else pd.Series(True, index=df.index)
    calm = calm.fillna(True)

    cloud = df.get("cloud_cover")
    clear = (cloud <= th.frost_caution_cloud) if cloud is not None else pd.Series(True, index=df.index)
    clear = clear.fillna(True)

    radiative = (df["t_air"] <= th.frost_caution_tmin) & calm & clear
    caution = radiative | (df["canopy_temp"] <= th.frost_caution_canopy)
    warning = _rolling_hours(
        df["canopy_temp"] <= th.frost_warning_canopy, th.frost_warning_canopy_hours
    ) | (df["t_air"] <= th.frost_warning_tmin)

    state = pd.Series(NORMAL, index=df.index, dtype=int)
    state[caution.to_numpy() & is_night] = CAUTION
    state[warning.to_numpy() & is_night] = WARNING
    return state


def compound_state(states: dict[str, pd.Series]) -> pd.Series:
    """복합위험 상태등급.

    주의 = (강우·과습 주의 + 병해 주의) 또는 (과습 주의 + 저온·서리 주의)
    경계 = 구성 위험 가운데 하나라도 경계
    """
    wet, disease, frost = states["rain_wet"], states["disease"], states["frost"]

    pair_a = (wet >= CAUTION) & (disease >= CAUTION)
    pair_b = (wet >= CAUTION) & (frost >= CAUTION)
    any_pair = pair_a | pair_b

    state = pd.Series(NORMAL, index=wet.index, dtype=int)
    state[any_pair] = CAUTION
    state[any_pair & ((wet >= WARNING) | (disease >= WARNING) | (frost >= WARNING))] = WARNING
    return state


def compose_compound(
    rain_wet: np.ndarray | pd.Series,
    disease: np.ndarray | pd.Series,
    frost: np.ndarray | pd.Series,
) -> np.ndarray:
    """구성 위험의 **예측 등급**에서 복합위험을 산출한다.

    복합위험은 독립적인 기상현상이 아니라 '강우·과습 + 병해 유리환경' 또는
    '과습 + 저온·서리'가 겹친 상태를 뜻한다. 정의상 구성 위험의 함수이므로,
    별도 분류기를 학습하는 대신 구성 위험 예측을 조합해 만든다.

    이렇게 하면 두 가지가 좋아진다.

    * 표본 문제 — 복합위험 단독 학습은 작기 10년치에서도 위험표본이
      100건대라 성능이 불안정하다. 조합 방식은 구성 위험의 표본을 그대로 쓴다.
    * 설명 가능성 — "강우·과습 경계 + 병해 유리환경 주의가 겹쳤다"처럼
      판단근거가 구성 위험으로 바로 나온다(화면 M2 요구사항).
    """
    wet = np.asarray(rain_wet, dtype=int)
    dis = np.asarray(disease, dtype=int)
    frz = np.asarray(frost, dtype=int)

    paired = ((wet >= CAUTION) & (dis >= CAUTION)) | ((wet >= CAUTION) & (frz >= CAUTION))
    escalate = (wet >= WARNING) | (dis >= WARNING) | (frz >= WARNING)

    out = np.zeros(len(wet), dtype=int)
    out[paired] = CAUTION
    out[paired & escalate] = WARNING
    return out


def compound_reasons(
    rain_wet: int, disease: int, frost: int
) -> list[str]:
    """복합위험 판단근거 문구를 만든다(화면·로그 표시용)."""
    level_name = {CAUTION: "주의", WARNING: "경계"}
    reasons = []
    if rain_wet >= CAUTION and disease >= CAUTION:
        reasons.append(
            f"강우·과습 {level_name[min(rain_wet, WARNING)]} + "
            f"병해 유리환경 {level_name[min(disease, WARNING)]} 동시 발생"
        )
    if rain_wet >= CAUTION and frost >= CAUTION:
        reasons.append(
            f"강우·과습 {level_name[min(rain_wet, WARNING)]} + "
            f"저온·서리 {level_name[min(frost, WARNING)]} 동시 발생"
        )
    return reasons


def wind_state(df: pd.DataFrame, th: RiskThresholds) -> pd.Series:
    """강풍·도복 상태등급 — AI가 아니라 현장 즉응규칙(L1) 입력."""
    gust = df.get("wind_speed")
    if gust is None:
        return pd.Series(NORMAL, index=df.index, dtype=int)
    state = pd.Series(NORMAL, index=df.index, dtype=int)
    state[gust >= th.wind_caution] = CAUTION
    state[gust >= th.wind_warning] = WARNING
    return state


def build_states(
    df: pd.DataFrame, thresholds: RiskThresholds | None = None
) -> pd.DataFrame:
    """시각별 위험유형 상태등급 표를 만든다.

    ``df`` 는 ``ts`` 를 인덱스로 하는 지점 하나의 시간순 데이터로,
    :func:`ofdf.features.derived.add_derived` 를 거친 것이어야 한다.
    """
    th = thresholds or RiskThresholds()

    soil_low, soil_high = th.soil_low, th.soil_high

    states = {
        "heat_dry": heat_dry_state(df, th, soil_low),
        "rain_wet": rain_wet_state(df, th, soil_high),
        "disease": disease_state(df, th),
        "frost": frost_state(df, th),
    }
    states["compound"] = compound_state(states)
    states["wind"] = wind_state(df, th)
    return pd.DataFrame(states, index=df.index)


def future_labels(states: pd.DataFrame, horizons: tuple[int, ...] = (1, 3, 6)) -> pd.DataFrame:
    """``(t, t+h]`` 구간 최대 등급을 예측 라벨로 만든다.

    사업계획서의 1시간·3시간 판단이 성능목표이고, 6시간은 참고 알림이다.
    """
    out = {}
    for risk in states.columns:
        series = states[risk]
        for h in horizons:
            # shift(-1) 부터 shift(-h) 까지의 최대 = 미래 h시간 안의 최고등급
            window = pd.concat([series.shift(-i) for i in range(1, h + 1)], axis=1)
            out[f"{risk}_h{h}"] = window.max(axis=1)
    return pd.DataFrame(out, index=states.index)


def event_ids(state: pd.Series, gap_hours: int) -> pd.Series:
    """연속된 위험 상태를 하나의 '독립 이벤트'로 묶어 번호를 매긴다.

    사업계획서의 '위험유형별 독립 이벤트 산출' 규칙이다. 위험이 해소된 뒤
    ``gap_hours`` 이상 지나야 다음 발생을 별개 이벤트로 센다. 학습/검증/시험
    분할을 이벤트 단위로 해야 같은 기상사례가 두 셋에 겹치지 않는다.
    """
    active = (state > NORMAL).to_numpy()
    ids = np.full(len(active), -1, dtype=int)

    current = -1
    quiet = gap_hours + 1
    for i, on in enumerate(active):
        if on:
            if quiet > gap_hours:
                current += 1
            ids[i] = current
            quiet = 0
        else:
            quiet += 1
    return pd.Series(ids, index=state.index)


#: 위험유형별 독립 이벤트 분리 기준(시간) — 사업계획서 표 기준
EVENT_GAP_HOURS = {
    "heat_dry": 6,
    "rain_wet": 12,
    "disease": 4,
    "frost": 12,
    "compound": 12,
    "wind": 6,
}
