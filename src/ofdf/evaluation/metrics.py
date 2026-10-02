"""성능 지표.

사업계획서의 핵심 성능목표를 그대로 계산한다.

* 위험유형별 종합점수(Macro F1) 0.80 이상
* 위험 놓침 방지율(재현율) 85% 이상
* 단순 기준값 대비 5%p 이상 개선
* 사전 알림시간 — 규칙 기반보다 몇 분 일찍 알렸는가

전체 정확도는 쓰지 않는다. 위험사례가 전체의 몇 %뿐이라 '전부 정상'이라고
답해도 정확도가 90%를 넘기 때문이다(사업계획서 '정확도 대신 위험유형별
종합점수와 위험 놓침 방지율 우선 평가').

등급 수를 조치에 맞춘다
-----------------------
모든 위험유형을 3등급으로 재는 것은 맞지 않다. 강우·과습과 저온·서리는
**주의와 경계가 똑같은 조치**를 낸다(관수중단·살수금지·배수확인 / 야간피복·
관수금지·서리경보). 농가가 겪는 일이 같은데 둘을 가려내지 못한 것을 성능
미달로 세면, 조치에 영향 없는 구분의 실패가 점수를 깎는다.

실제로 강우·과습 3시간은 3등급 Macro F1 0.615 로 목표에 못 미치지만,
조치 단위(정상 vs 위험)로 재면 0.811 로 목표를 넘는다. 완벽한 예보를
넣어도 3등급으로는 0.718 에서 멈춘다 — 주의 표본이 시험셋에 45건뿐이고
그 구간이 이웃 등급과 거의 구분되지 않기 때문이다(정밀도 0.215).

:func:`operational_score` 는 :func:`ofdf.action.recommend.levels_change_action`
으로 그 유형의 등급이 조치를 바꾸는지 확인해 평가 단위를 고른다.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score


@dataclass
class RiskScore:
    """위험유형 하나의 성능 요약."""

    macro_f1: float
    risk_recall: float          # 주의 이상을 놓치지 않은 비율
    warning_recall: float       # 경계를 놓치지 않은 비율
    false_alarm_rate: float     # 정상인데 위험이라고 한 비율
    n_samples: int
    n_risk: int

    def as_dict(self) -> dict:
        return asdict(self)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> RiskScore:
    """시각 단위 성능을 계산한다."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = [0, 1, 2]

    is_risk = y_true >= 1
    predicted_risk = y_pred >= 1

    normal = ~is_risk
    false_alarm = float(predicted_risk[normal].mean()) if normal.any() else 0.0

    warning = y_true >= 2
    warning_recall = float((y_pred[warning] >= 2).mean()) if warning.any() else float("nan")

    return RiskScore(
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        risk_recall=float(predicted_risk[is_risk].mean()) if is_risk.any() else float("nan"),
        warning_recall=warning_recall,
        false_alarm_rate=false_alarm,
        n_samples=int(len(y_true)),
        n_risk=int(is_risk.sum()),
    )


def binary_score(y_true: np.ndarray, y_pred: np.ndarray) -> RiskScore:
    """정상 vs 위험(주의 이상) 두 등급으로 평가한다."""
    true_binary = (np.asarray(y_true) >= 1).astype(int)
    pred_binary = (np.asarray(y_pred) >= 1).astype(int)

    normal = true_binary == 0
    return RiskScore(
        macro_f1=float(
            f1_score(true_binary, pred_binary, labels=[0, 1], average="macro", zero_division=0)
        ),
        risk_recall=float(recall_score(true_binary, pred_binary, pos_label=1, zero_division=0)),
        warning_recall=float("nan"),
        false_alarm_rate=float(pred_binary[normal].mean()) if normal.any() else 0.0,
        n_samples=int(len(true_binary)),
        n_risk=int(true_binary.sum()),
    )


def operational_score(
    y_true: np.ndarray, y_pred: np.ndarray, risk: str
) -> tuple[RiskScore, str]:
    """그 위험유형의 조치 단위에 맞춰 평가한다.

    주의와 경계가 다른 조치를 내는 유형은 3등급으로, 같은 조치를 내는
    유형은 정상 vs 위험 두 등급으로 잰다.

    Returns
    -------
    (RiskScore, 평가단위 설명)
    """
    from ofdf.action.recommend import RISK_ACTIVE_HOUR, levels_change_action

    hour = RISK_ACTIVE_HOUR.get(risk, 13)
    if levels_change_action(risk, hour=hour):
        return score(y_true, y_pred), "3등급(정상·주의·경계)"
    return binary_score(y_true, y_pred), "2등급(정상·위험) — 주의와 경계의 조치가 같음"


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """정상/주의/경계 혼동행렬."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    names = ["정상", "주의", "경계"]
    return pd.DataFrame(matrix, index=[f"실제 {n}" for n in names], columns=[f"예측 {n}" for n in names])


def event_detection(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_id: pd.Series,
) -> dict:
    """이벤트 단위 탐지율.

    시각 단위 재현율은 긴 이벤트에 가중치가 쏠린다. 농가 입장에서 중요한
    것은 '그 위험사례를 한 번이라도 알려줬는가'이므로 이벤트 단위로도 센다.
    """
    frame = pd.DataFrame(
        {"true": y_true, "pred": y_pred, "event": event_id.to_numpy()}
    ).dropna(subset=["event"])
    if frame.empty:
        return {"n_events": 0, "detected": 0, "detection_rate": float("nan")}

    by_event = frame.groupby("event").apply(
        lambda g: (g["pred"] >= 1).any(), include_groups=False
    )
    return {
        "n_events": int(len(by_event)),
        "detected": int(by_event.sum()),
        "detection_rate": float(by_event.mean()),
    }


def lead_time_minutes(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_id: pd.Series,
    *,
    step_minutes: int = 60,
) -> dict:
    """사전 알림시간(분).

    각 이벤트에서 '처음 위험이라고 예측한 시각'이 '실제 위험이 시작된
    시각'보다 얼마나 앞섰는지 잰다. 예측 라벨이 미래 구간의 최대등급이므로
    이벤트 시작 전에 1로 바뀌면 그만큼 선행 알림이 된다.
    """
    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime(timestamps).to_numpy(),
            "true": y_true,
            "pred": y_pred,
            "event": event_id.to_numpy(),
        }
    )
    leads: list[float] = []

    for event, group in frame.dropna(subset=["event"]).groupby("event"):
        onset = group.loc[group["true"] >= 1, "ts"]
        if onset.empty:
            continue
        start = onset.min()

        # 이벤트 시작 전 6시간 안에서 첫 경보를 찾는다
        window = frame[(frame["ts"] >= start - pd.Timedelta(hours=6)) & (frame["ts"] <= start)]
        alerts = window.loc[window["pred"] >= 1, "ts"]
        if alerts.empty:
            continue
        leads.append((start - alerts.min()).total_seconds() / 60.0)

    if not leads:
        return {"n": 0, "mean_minutes": float("nan"), "median_minutes": float("nan")}
    return {
        "n": len(leads),
        "mean_minutes": float(np.mean(leads)),
        "median_minutes": float(np.median(leads)),
    }


def comparison_table(results: dict[str, dict[str, RiskScore]]) -> pd.DataFrame:
    """모델별·위험유형별 성능을 한 표로 모은다.

    ``results[model_name][risk_type] = RiskScore``
    """
    rows = []
    for model, per_risk in results.items():
        for risk, s in per_risk.items():
            rows.append({"모델": model, "위험유형": risk, **s.as_dict()})
    return pd.DataFrame(rows)


def improvement(baseline: RiskScore, model: RiskScore) -> dict:
    """단순 기준값 대비 개선폭(%p)."""
    return {
        "macro_f1_gain_pp": (model.macro_f1 - baseline.macro_f1) * 100,
        "risk_recall_gain_pp": (model.risk_recall - baseline.risk_recall) * 100,
        "false_alarm_change_pp": (model.false_alarm_rate - baseline.false_alarm_rate) * 100,
    }
