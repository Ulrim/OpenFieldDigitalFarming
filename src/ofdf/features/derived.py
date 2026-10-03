"""관측값에서 작물 위험판단에 필요한 물리 파생변수를 만든다.

노지 대파 위험판단에 필요한 값 가운데 상당수는 공공 관측망에 없다.
초관부(지상 10cm) 온도와 엽면습윤은 실증포장 센서로만 측정되고,
토양수분은 나주 농업기상 관측지점에서도 2025년부터만 기록된다.

여기서는 공공 관측값으로 계산 가능한 **대리지표(proxy)** 를 만든다.
현장 센서가 붙으면 같은 이름의 실측 컬럼(``canopy_temp``, ``leaf_wetness``,
``soil_index``)이 들어오고, 그 시각은 대리지표 대신 실측을 쓴다. 센서가
결측인 시각만 대리지표로 메우므로 통신이 끊겨도 판단이 비지 않는다.
어느 쪽을 썼는지는 ``{이름}_source`` 컬럼에 시각별로 남는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: 일사량 단위 환산 — 1 MJ/m^2 를 1시간에 받으면 평균 277.8 W/m^2
MJ_PER_HOUR_TO_W = 1e6 / 3600.0

#: 조도(lux) -> 일사강도(W/m^2) 환산 계수.
#:
#: WS90 같은 Fine Offset / Ecowitt 센서는 일사계가 아니라 조도계를 달고
#: 있어서, rtl_433 으로 직접 받으면 ``light_lux`` 만 나온다(에코윗 콘솔이
#: 보여주는 W/m^2 도 실은 콘솔이 이 계수로 나눈 값이다).
#:
#: 가시광 조도와 전파장 일사는 스펙트럼이 달라서 이 환산은 근사다. 맑은
#: 날 기준으로 맞춰진 값이라 흐린 날·일출일몰 무렵에는 오차가 커진다.
#: 그래서 환산값은 일사계 실측이 없을 때만 쓰는 대체 경로다.
LUX_PER_W_M2 = 126.7


def dew_point(t_air: pd.Series, rh: pd.Series) -> pd.Series:
    """Magnus-Tetens 식으로 이슬점온도(℃)를 구한다."""
    rh_clipped = rh.clip(lower=1.0, upper=100.0)
    a, b = 17.62, 243.12
    gamma = np.log(rh_clipped / 100.0) + (a * t_air) / (b + t_air)
    return (b * gamma) / (a - gamma)


def saturation_vapour_pressure(t_air: pd.Series) -> pd.Series:
    """포화수증기압(kPa)."""
    return 0.6108 * np.exp(17.27 * t_air / (t_air + 237.3))


def vapour_pressure_deficit(t_air: pd.Series, rh: pd.Series) -> pd.Series:
    """수증기압 부족분 VPD(kPa). 값이 클수록 건조 스트레스가 크다."""
    return saturation_vapour_pressure(t_air) * (1.0 - rh.clip(0, 100) / 100.0)


def solar_w_per_m2(solar_mj: pd.Series) -> pd.Series:
    """시간 일사량(MJ/m^2)을 평균 일사강도(W/m^2)로 바꾼다."""
    return solar_mj * MJ_PER_HOUR_TO_W


def leaf_wetness(
    t_air: pd.Series,
    rh: pd.Series,
    rain: pd.Series,
    solar_w: pd.Series,
    *,
    rh_threshold: float = 90.0,
    dew_depression: float = 2.0,
) -> pd.Series:
    """엽면습윤(잎 젖음) 여부 대리지표.

    실측 엽면습윤센서가 없을 때 쓰는 확장 임계값 모델이다. 다음 중
    하나라도 성립하면 잎이 젖은 것으로 본다.

    * 강우가 있었다
    * 상대습도가 ``rh_threshold`` 이상이다
    * 기온과 이슬점 차이가 ``dew_depression`` 이하이고 일사가 거의 없다(야간 결로)

    사업계획서의 병해 유리환경 판단(엽면습윤 지속시간 + RH + 기온)에서
    실측센서 설치 전 구간을 메우는 용도이며, 현장 엽면습윤센서가 붙으면
    이 컬럼을 실측값으로 대체한다.
    """
    wet_rain = rain.fillna(0) > 0.0
    wet_rh = rh >= rh_threshold
    depression = t_air - dew_point(t_air, rh)
    wet_dew = (depression <= dew_depression) & (solar_w.fillna(0) < 20.0)
    return (wet_rain | wet_rh | wet_dew).astype(float)


def canopy_temperature(
    t_air: pd.Series,
    rh: pd.Series,
    wind_speed: pd.Series,
    solar_w: pd.Series,
    cloud_cover: pd.Series | None = None,
    *,
    max_drop: float = 4.0,
) -> pd.Series:
    """초관부(지상 10cm) 온도 대리지표(℃).

    맑고 바람이 약한 밤에는 지면 복사냉각으로 초관부가 2m 기온보다
    낮아진다. 대파 서리피해는 기온이 아니라 이 초관부 온도로 결정되므로,
    실측센서가 붙기 전까지는 기온에서 복사냉각량을 빼서 추정한다.

    냉각량 = ``max_drop`` x 청명도 x 정온도 x 건조도 로 본다.

    * 청명도 : 구름이 적을수록 장파 복사로 열이 많이 빠진다(지배 인자).
      운량 자료가 없으면 0.5로 둔다.
    * 정온도 : 바람이 약할수록 대기 혼합이 적어 지면 근처가 더 식는다.
      ``exp(-u/1.5)`` 로 완만하게 감쇠시킨다.
    * 건조도 : 수증기가 적을수록 역복사가 약해 냉각이 커진다.
      이슬점차 0~8℃ 를 0.4~1.0 으로 사상한다.

    낮(일사가 있는 시간)에는 0으로 둔다. 노지 대파 캐노피는 성글어
    주간 가열 효과가 크지 않고, 서리 판단에 쓰이는 값은 야간이다.

    주의: 지면상태(피복·토양수분)와 지형 냉기호수를 반영하지 않은 1차
    근사다. 실증포장 초관부 온도센서 측정이 쌓이면 농지별 보정식으로
    교체해야 한다(사업계획서 '농지별 센서편차·기상특성 보정').
    """
    is_night = solar_w.fillna(0) < 20.0

    if cloud_cover is None:
        clearness = pd.Series(0.5, index=t_air.index)
    else:
        clearness = (1.0 - cloud_cover.clip(0.0, 1.0)).fillna(0.5)

    wind = wind_speed.fillna(wind_speed.median()).clip(lower=0.0)
    calmness = np.exp(-wind / 1.5)

    depression = (t_air - dew_point(t_air, rh)).clip(lower=0.0)
    dryness = 0.4 + 0.6 * (depression / 8.0).clip(upper=1.0)

    drop = max_drop * clearness * calmness * dryness
    return t_air - drop.where(is_night, 0.0)


def soil_water_index(
    rain: pd.Series,
    et0: pd.Series,
    *,
    capacity: float = 60.0,
    field_capacity_fraction: float = 0.75,
    stress_fraction: float = 0.5,
    drainage_rate: float = 0.25,
    initial_fraction: float = 0.6,
) -> pd.Series:
    """강우-증발산-배수 물수지로 만든 토양수분 지수(0~100).

    나주 농업기상 관측지점의 토양수분은 2025년부터만 기록돼 있어
    2016~2024년 학습구간에 쓸 수 없다. 대신 저수지(bucket) 모델로
    상대적인 건조/과습 추세를 만든다.

    배수가 없는 단순 bucket은 비가 한 번 오면 증발산만으로 물이 빠져
    며칠씩 포화에 고착된다(실측에서 중앙값 88, 85분위 99.9로 확인).
    그래서 포장용수량을 넘는 물은 중력배수로 매시간 일정 비율 빠지게 한다.

    * 포장용수량 이상 : 과습 구간. ``drainage_rate`` 비율로 시간당 배수
    * 포장용수량 이하 : 증발산으로만 감소
    * 스트레스점 이하 : 증발산에 FAO56 수분스트레스 계수를 곱해 감속

    Parameters
    ----------
    rain : 시간 강수량(mm)
    et0 : 시간 기준증발산량(mm)
    capacity : 포화 저수량(mm). 대파는 천근성이라 유효토심이 얕다.
    field_capacity_fraction : 포장용수량 / 포화 저수량
    stress_fraction : 이 수위 아래로 내려가면 실제 증발산이 줄기 시작한다
    drainage_rate : 포장용수량 초과분의 시간당 배수 비율

    Returns
    -------
    pd.Series
        0(완전 건조) ~ 100(포화). 실측 토양수분(%)과 절대값이 다르므로
        위험 임계값은 이 지수 기준으로 따로 정한다.
    """
    rain_mm = rain.fillna(0.0).to_numpy(dtype=float)
    pet_mm = et0.fillna(0.0).to_numpy(dtype=float)
    field_capacity = capacity * field_capacity_fraction
    stress_point = capacity * stress_fraction

    storage = np.empty(len(rain_mm), dtype=float)
    level = capacity * initial_fraction
    for i in range(len(rain_mm)):
        level = min(capacity, level + rain_mm[i])

        # 수분스트레스 계수(FAO56 Ks) — 토양이 마를수록 실제 증발산이 준다.
        # 이것이 없으면 저수지가 0까지 말라붙어 건조 신호가 포화된다.
        ks = 1.0 if level >= stress_point else max(0.0, level / stress_point)
        level = max(0.0, level - pet_mm[i] * ks)

        if level > field_capacity:
            level -= (level - field_capacity) * drainage_rate
        storage[i] = level

    return pd.Series(storage / capacity * 100.0, index=rain.index)


def _prefer_measured(out: pd.DataFrame, col: str, proxy: pd.Series) -> pd.Series:
    """실측 컬럼이 있으면 그것을 쓰고, 빈 구간만 대리지표로 메운다.

    실증포장 센서가 붙기 전까지 ``canopy_temp`` 같은 값은 공공 관측값으로
    계산한 대리지표였다. 센서가 붙은 뒤에도 대리지표로 덮어써 버리면
    장비를 사 놓고 추정값으로 판단하는 셈이 되므로, 실측이 있는 시각은
    실측을 쓰고 결측 시각만 대리지표로 메운다.

    실측인지 추정인지는 성능평가에서 구분해야 하므로 ``{col}_source``
    컬럼에 시각별로 ``measured`` / ``proxy`` 를 남긴다. 이 컬럼은
    :func:`ofdf.features.weather.build_features` 의 변수 목록에 없으므로
    모델 입력으로 새지 않는다.
    """
    if col not in out.columns:
        out[f"{col}_source"] = "proxy"
        return proxy

    measured = pd.to_numeric(out[col], errors="coerce")
    if not measured.notna().any():
        out[f"{col}_source"] = "proxy"
        return proxy

    out[f"{col}_source"] = np.where(measured.notna(), "measured", "proxy")
    return measured.where(measured.notna(), proxy)


def add_derived(df: pd.DataFrame, et0_col: str | None = "et0") -> pd.DataFrame:
    """관측 데이터프레임에 파생변수 컬럼을 붙여 돌려준다.

    ``df`` 는 지점 하나의 시간순 데이터여야 한다(물수지가 누적 계산이므로).
    """
    out = df.copy()

    # 일사 단위가 경로마다 다르다. 농업기상 관측자료는 시간 일사량(MJ/m^2)을
    # 주고, 현장 일사계는 일사강도(W/m^2)를 바로 준다. WS90 처럼 조도계만
    # 달린 센서는 lux 로 준다. 셋 다 받되 정확한 순서대로 고른다.
    if "solar_w" not in out.columns or out["solar_w"].isna().all():
        if "solar_mj" in out.columns:
            out["solar_w"] = solar_w_per_m2(out["solar_mj"])
        elif "solar" in out.columns:
            out["solar_w"] = out["solar"]
        elif "illuminance" in out.columns:
            # 조도계만 있는 경우(WS90 등). 근사 환산이라 마지막 순위다.
            out["solar_w"] = out["illuminance"] / LUX_PER_W_M2
        else:
            out["solar_w"] = np.nan
    out["dew_point"] = dew_point(out["t_air"], out["rh"])
    out["vpd"] = vapour_pressure_deficit(out["t_air"], out["rh"])
    out["leaf_wetness"] = _prefer_measured(
        out,
        "leaf_wetness",
        leaf_wetness(out["t_air"], out["rh"], out["rain"], out["solar_w"]),
    )
    out["canopy_temp"] = _prefer_measured(
        out,
        "canopy_temp",
        canopy_temperature(
            out["t_air"],
            out["rh"],
            out.get("wind_speed", pd.Series(np.nan, index=out.index)),
            out["solar_w"],
            out.get("cloud_cover"),
        ),
    )
    if et0_col and et0_col in out.columns:
        out["soil_index"] = _prefer_measured(
            out, "soil_index", soil_water_index(out["rain"], out[et0_col])
        )
    return out
