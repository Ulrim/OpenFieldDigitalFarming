"""기상청 예보 연계.

사업계획서의 AI 입력은 '기상청 예보 + 농지센서'다. 지금 구현은 관측값만
쓰고 있고, 그래서 강우·과습이 유일하게 목표에 못 미친다(§5.1). 앞으로
1~3시간 뒤 비가 올지를 과거 관측만으로 맞히는 데는 원리적인 한계가 있다.

이 모듈은 두 가지를 제공한다.

``KMAForecast``
    기상청 단기예보 API(동네예보) 연계. 발표시각 기준으로 예보를 받아
    시간 격자에 맞춘다. API 키가 있어야 동작한다.

``SimulatedForecast``
    예보 정확도를 **변수로 둔 모사 예보**. 실측 미래값에 정확도에 따른
    오차를 입혀 만든다. API 키 없이도 "예보를 붙이면 성능이 얼마나
    오르는가"를 미리 재서 연계 투자 여부를 판단할 수 있다.

    ``skill=1.0`` 은 완벽한 예보(이론적 상한), ``skill=0.0`` 은 예보가
    사실상 쓸모없는 경우다. 실제 기상청 초단기예보는 그 사이 어딘가이며,
    현장 연계 후 관측과 대조해 실측 skill 을 구해야 한다.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

#: 예보로 받는 항목과 예측 시계(시간)
FORECAST_VARIABLES = ["rain", "t_air", "rh", "wind_speed", "sky"]
FORECAST_HORIZONS = (1, 3, 6)

#: 기상청 동네예보 격자 — 나주시 남평읍 (실증포장). 설치 좌표로 확정해야 한다.
NAJU_NAMPYEONG_GRID = {"nx": 57, "ny": 71}

#: 동네예보 카테고리 -> 내부 이름
KMA_CATEGORY = {
    "T1H": "t_air",        # 기온
    "RN1": "rain",         # 1시간 강수량
    "REH": "rh",           # 습도
    "WSD": "wind_speed",   # 풍속
    "SKY": "sky",          # 하늘상태 1 맑음 3 구름많음 4 흐림
    "PTY": "precip_type",  # 강수형태
    "POP": "rain_prob",    # 강수확률(단기예보)
}


class ForecastProvider(Protocol):
    """예보 제공자 공통 규약."""

    def at(self, issued: pd.Timestamp, horizons: tuple[int, ...]) -> pd.DataFrame:
        """``issued`` 시각에 발표된 예보를 돌려준다.

        Returns
        -------
        pd.DataFrame
            시계(시간)를 인덱스로 하고 예보 항목을 컬럼으로 갖는 표.
        """
        ...


@dataclass
class KMAForecast:
    """기상청 단기예보 API 연계.

    초단기예보(getUltraSrtFcst)는 6시간, 단기예보(getVilageFcst)는 3일까지
    준다. 위험판단이 쓰는 1·3시간은 초단기예보 범위다.

    API 키는 환경변수나 설정으로 받고 **코드·저장소에 넣지 않는다.**
    """

    service_key: str
    nx: int = NAJU_NAMPYEONG_GRID["nx"]
    ny: int = NAJU_NAMPYEONG_GRID["ny"]
    base_url: str = (
        "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtFcst"
    )
    timeout: int = 10

    def _request(self, base_date: str, base_time: str) -> dict:
        import requests

        response = requests.get(
            self.base_url,
            params={
                "serviceKey": self.service_key, "dataType": "JSON",
                "numOfRows": 1000, "pageNo": 1,
                "base_date": base_date, "base_time": base_time,
                "nx": self.nx, "ny": self.ny,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def at(self, issued: pd.Timestamp, horizons: tuple[int, ...] = FORECAST_HORIZONS) -> pd.DataFrame:
        """발표시각 기준 예보를 받아 시계별로 정리한다.

        초단기예보는 매시 30분에 발표되고 10분 뒤부터 조회된다. 발표시각이
        아직 안 됐으면 직전 발표를 쓴다.
        """
        issue = issued.floor("h")
        if issued.minute < 45:
            issue -= pd.Timedelta(hours=1)

        payload = self._request(issue.strftime("%Y%m%d"), issue.strftime("%H30"))
        items = payload["response"]["body"]["items"]["item"]

        rows = []
        for item in items:
            name = KMA_CATEGORY.get(item["category"])
            if name is None:
                continue
            target = pd.Timestamp(f"{item['fcstDate']} {item['fcstTime'][:2]}:00")
            value = pd.to_numeric(item["fcstValue"], errors="coerce")
            # 강수없음은 '강수없음' 문자열로 온다
            if name == "rain" and pd.isna(value):
                value = 0.0
            rows.append({"target": target, "variable": name, "value": value})

        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(index=list(horizons))

        wide = frame.pivot_table(index="target", columns="variable", values="value")
        wanted = [issue + pd.Timedelta(hours=h) for h in horizons]
        out = wide.reindex(wanted)
        out.index = list(horizons)
        out.index.name = "horizon"
        return out


@dataclass
class SimulatedForecast:
    """정확도를 변수로 둔 모사 예보.

    실측 미래값에 오차를 입혀 만든다. 예보가 붙었을 때의 성능 상한과
    현실적인 기댓값을 미리 재는 용도다.

    Parameters
    ----------
    observations
        시각 인덱스를 갖는 실측 시계열(``rain``, ``t_air``, ``rh``, ``wind_speed``).
    skill
        0~1. 1 이면 완벽한 예보(이론적 상한), 0 이면 쓸모없는 예보.
    seed
        난수 씨앗. 같은 씨앗이면 같은 예보가 나온다(재현성).

    Notes
    -----
    강수는 '있다/없다'를 먼저 틀리고(탐지 실패·오경보), 양을 틀린다.
    기온·습도는 시계가 길수록 오차가 커진다. 실제 예보의 오차 구조와
    맞춘 단순화이며, 절대 수치를 기상청 실적으로 읽으면 안 된다.
    """

    observations: pd.DataFrame
    skill: float = 0.7
    seed: int = 42

    #: 예보가 완전히 무작위일 때의 강수 탐지율·오경보율
    base_hit_rate: float = 0.5
    base_false_alarm: float = 0.30
    #: 예보가 완전히 무작위일 때 1시간 시계의 기온 오차 표준편차(℃)
    base_temp_error: float = 3.0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _degrade_rain(self, truth: np.ndarray, lead: int) -> np.ndarray:
        """강수 예보 — 유무를 틀리고, 양을 틀린다."""
        skill = float(np.clip(self.skill, 0.0, 1.0))
        # 시계가 길수록 떨어진다
        effective = skill * (1.0 - 0.06 * (lead - 1))

        hit_rate = self.base_hit_rate + (1.0 - self.base_hit_rate) * effective
        false_alarm = self.base_false_alarm * (1.0 - effective)

        raining = truth > 0.0
        detected = raining & (self._rng.random(len(truth)) < hit_rate)
        spurious = ~raining & (self._rng.random(len(truth)) < false_alarm)

        out = np.zeros_like(truth, dtype=float)
        # 탐지한 비는 양을 로그정규 오차로 틀린다
        sigma = 1.2 * (1.0 - effective)
        noise = np.exp(self._rng.normal(0, sigma, len(truth)))
        out[detected] = truth[detected] * noise[detected]
        # 헛 예보는 가벼운 비를 예보한다
        out[spurious] = np.abs(self._rng.normal(1.0, 1.0, int(spurious.sum())))
        return out

    def _degrade_continuous(self, truth: np.ndarray, lead: int, base_error: float) -> np.ndarray:
        skill = float(np.clip(self.skill, 0.0, 1.0))
        effective = skill * (1.0 - 0.06 * (lead - 1))
        sigma = base_error * (1.0 - effective) * (1.0 + 0.15 * (lead - 1))
        return truth + self._rng.normal(0, sigma, len(truth))

    def build(self, horizons: tuple[int, ...] = FORECAST_HORIZONS) -> pd.DataFrame:
        """전 기간에 대한 예보 표를 한 번에 만든다.

        운영에서는 매 시각 API 를 부르지만, 실험에서는 전 구간을 한 번에
        만들어야 학습·평가가 된다. 컬럼 이름은 ``fc_<항목>_h<시계>`` 다.
        """
        out = pd.DataFrame(index=self.observations.index)
        index = pd.DatetimeIndex(self.observations.index)

        for lead in horizons:
            future = self.observations.shift(-lead)

            # 시간축이 끊긴 자리(작기만 남긴 자료의 11월 말 -> 이듬해 9월 초)에서
            # shift 는 엉뚱한 해의 값을 끌어온다. 실제 시각 차이가 lead 와
            # 다른 행은 예보가 없는 것으로 둔다.
            actual_gap = pd.Series(index, index=index).shift(-lead) - index
            valid = (actual_gap == pd.Timedelta(hours=lead)).to_numpy()

            if "rain" in future.columns:
                truth = future["rain"].fillna(0.0).to_numpy()
                forecast_rain = self._degrade_rain(truth, lead)
                out[f"fc_rain_h{lead}"] = np.where(valid, forecast_rain, np.nan)
                # 강수확률 — 예보한 비의 양에서 되짚는다
                out[f"fc_rain_prob_h{lead}"] = np.clip(
                    out[f"fc_rain_h{lead}"] / 5.0, 0.0, 1.0
                )

            for name, base_error in (
                ("t_air", self.base_temp_error), ("rh", 12.0), ("wind_speed", 1.5)
            ):
                if name in future.columns:
                    truth = future[name].to_numpy(dtype=float)
                    degraded = self._degrade_continuous(truth, lead, base_error)
                    out[f"fc_{name}_h{lead}"] = np.where(valid, degraded, np.nan)

        # 누적 예보 강수 — 사업계획서의 '3시간 예보강우 20mm' 기준에 쓰인다
        rain_columns = [c for c in out.columns if c.startswith("fc_rain_h")]
        if rain_columns:
            out["fc_rain_cum"] = out[rain_columns].sum(axis=1)
        return out

    def at(self, issued: pd.Timestamp, horizons: tuple[int, ...] = FORECAST_HORIZONS) -> pd.DataFrame:
        table = self.build(horizons)
        if issued not in table.index:
            return pd.DataFrame(index=list(horizons))
        row = table.loc[issued]
        rows = {}
        for lead in horizons:
            rows[lead] = {
                name: row.get(f"fc_{name}_h{lead}")
                for name in ("rain", "t_air", "rh", "wind_speed")
            }
        return pd.DataFrame(rows).T


def observed_skill(forecast: pd.Series, observed: pd.Series, *, threshold: float = 0.1) -> dict:
    """예보와 실측을 대조해 실제 정확도를 잰다.

    현장 연계 뒤 이 함수로 기상청 예보의 실측 skill 을 구해, 모사 실험에서
    쓴 skill 값과 맞춰 본다.
    """
    pair = pd.DataFrame({"forecast": forecast, "observed": observed}).dropna()
    if pair.empty:
        return {}

    predicted = pair["forecast"] > threshold
    actual = pair["observed"] > threshold

    hits = int((predicted & actual).sum())
    misses = int((~predicted & actual).sum())
    false_alarms = int((predicted & ~actual).sum())

    pod = hits / (hits + misses) if (hits + misses) else float("nan")
    far = false_alarms / (hits + false_alarms) if (hits + false_alarms) else float("nan")
    denominator = hits + misses + false_alarms
    return {
        "n": len(pair),
        "탐지율(POD)": round(pod, 3),
        "오경보율(FAR)": round(far, 3),
        "임계성공지수(CSI)": round(hits / denominator, 3) if denominator else float("nan"),
        "양적 MAE": round(float((pair["forecast"] - pair["observed"]).abs().mean()), 3),
    }
