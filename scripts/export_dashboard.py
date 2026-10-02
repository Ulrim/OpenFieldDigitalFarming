"""현장 모니터링 화면용 데이터 산출.

사업계획서의 모니터링 화면 M1~M4 가 쓰는 자료를 하나의 JSON 으로 뽑는다.
실제 관측값에 학습된 모델을 돌리고 조치 엔진을 거친 결과이며, 꾸며 낸
숫자가 아니다.

    M1 실시간 대시보드  위험카드(1·3시간), 환경 현재값, 시설상태, 적용↔비교구간 24시간
    M2 AI 판단 상세     위험수준·예상시점·판단근거 3개·추천조치·신뢰도·모델버전
    M3 제어 이력        명령 계층(L0~L3)·시각·대상·승인여부·실행결과
    M4 이벤트·알림      등급별 목록

사용 예::

    python scripts/export_dashboard.py \
        --cache artifacts/dataset.pkl --models artifacts/weather \
        --out dashboard/data.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf.action.recommend import SensorState, decide  # noqa: E402
from ofdf.labels.risk import (  # noqa: E402
    RISK_LABELS_KO,
    compose_compound,
    compound_reasons,
)

BASE_RISKS = ["heat_dry", "rain_wet", "disease", "frost"]
LEVEL_NAMES = {0: "정상", 1: "주의", 2: "경계"}
MODEL_VERSION = "ofdf-weather-0.1.0"

#: 화면에 띄울 환경 항목 (키, 이름, 단위, 소수점)
ENVIRONMENT_FIELDS = [
    ("t_air", "기온", "℃", 1),
    ("canopy_temp", "초관부 온도", "℃", 1),
    ("rh", "상대습도", "%", 0),
    ("leaf_wetness", "엽면습윤", "", 2),
    ("solar_w", "일사", "W/m²", 0),
    ("rain", "시간 강수", "mm", 1),
    ("soil_index", "토양수분 지수", "", 0),
    ("wind_speed", "풍속", "m/s", 1),
]

#: 판단근거 변수의 사람이 읽는 이름
FEATURE_LABELS = {
    "t_air": "기온", "canopy_temp": "초관부 온도", "rh": "상대습도",
    "solar_w": "일사", "rain": "시간 강수", "soil_index": "토양수분 지수",
    "wind_speed": "풍속", "vpd": "수증기압 부족", "dew_point": "이슬점",
    "leaf_wetness": "엽면습윤", "cloud_cover": "운량", "et0": "기준증발산",
    "run_wet": "엽면습윤 연속시간", "run_rh90": "고습 연속시간",
    "run_rain": "강우 연속시간", "run_dry": "무강우 연속시간",
    "run_hot30": "30℃ 이상 연속", "run_cold4": "4℃ 이하 연속",
    "run_canopy2": "초관부 2℃ 이하 연속", "is_night": "야간 여부",
    "hour_sin": "시각(주기)", "hour_cos": "시각(주기)",
    "doy_sin": "계절(주기)", "doy_cos": "계절(주기)", "month": "월",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--out", default="dashboard/data.json")
    p.add_argument("--station", default="나주시 봉황면")
    p.add_argument("--risk", default="frost", help="시연할 위험유형")
    p.add_argument("--hours", type=int, default=24, help="보여줄 과거 시간")
    return p.parse_args()


def readable(name: str) -> str:
    """피처 이름을 사람이 읽는 말로 바꾼다."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for suffix, text in (
        ("_mean", "{}시간 평균"), ("_max", "{}시간 최고"), ("_min", "{}시간 최저"),
        ("_sum", "{}시간 합계"), ("_delta", "{}시간 변화"),
    ):
        if suffix in name:
            base, _, window = name.partition(suffix)
            label = FEATURE_LABELS.get(base, base)
            return f"{label} {text.format(window)}"
    return FEATURE_LABELS.get(name, name)


def pick_moment(dataset, risk: str, hours: int) -> int:
    """위험이 실제로 발생한 시점을 고른다 — 가장 긴 이벤트의 시작 직전."""
    events = dataset.events[f"{risk}_event"].reset_index(drop=True)
    states = dataset.states[risk].reset_index(drop=True)
    sizes = events.value_counts()
    for event_id in sizes.index:
        rows = np.where(events.to_numpy() == event_id)[0]
        start = rows.min()
        if start > hours and (states.iloc[rows] >= 2).any():
            return int(start)
    return int(len(states) // 2)


def main() -> int:
    args = parse_args()
    import lightgbm as lgb

    dataset = pickle.loads(Path(args.cache).read_bytes())
    features = dataset.features.reset_index(drop=True)
    meta = dataset.meta.reset_index(drop=True)
    labels = dataset.labels.reset_index(drop=True)

    models = {}
    for risk in BASE_RISKS:
        for horizon in (1, 3):
            path = Path(args.models) / f"model_{risk}_h{horizon}.txt"
            if path.exists():
                models[(risk, horizon)] = lgb.Booster(model_file=str(path))
    if not models:
        print(f"[오류] {args.models} 에 모델이 없습니다")
        return 1

    now = pick_moment(dataset, args.risk, args.hours)
    station = meta.loc[now, "station"]
    timestamp = pd.Timestamp(meta.loc[now, "ts"])
    print(f"[시점] {station} {timestamp:%Y-%m-%d %H시}")

    # ---- 같은 지점의 최근 구간 ----
    same = np.where(meta["station"].to_numpy() == station)[0]
    window = [i for i in same if now - args.hours < i <= now]

    # ---- M1 위험카드 + M2 판단근거 ----
    cards, judgements = [], []
    for risk in BASE_RISKS:
        entry = {"risk": risk, "name": RISK_LABELS_KO[risk], "horizons": {}}
        for horizon in (1, 3):
            model = models.get((risk, horizon))
            if model is None:
                continue
            row = features.iloc[[now]][model.feature_name()]
            proba = model.predict(row)[0]
            level = int(np.argmax(proba))
            truth = int(labels.loc[now, f"{risk}_h{horizon}"] or 0)

            entry["horizons"][str(horizon)] = {
                "level": level, "levelName": LEVEL_NAMES[level],
                "confidence": round(float(proba[level]), 3),
                "probabilities": [round(float(p), 3) for p in proba],
                "actual": truth, "actualName": LEVEL_NAMES[truth],
                "correct": level == truth,
            }

            if horizon == 3 and level > 0:
                contrib = model.predict(row, pred_contrib=True)[0]
                n = len(model.feature_name())
                values = np.array(contrib[level * (n + 1) : level * (n + 1) + n])
                order = np.argsort(-np.abs(values))[:3]
                judgements.append({
                    "risk": risk, "name": RISK_LABELS_KO[risk],
                    "level": level, "levelName": LEVEL_NAMES[level],
                    "confidence": round(float(proba[level]), 3),
                    "modelVersion": MODEL_VERSION,
                    "expectedOnset": (timestamp + pd.Timedelta(hours=horizon)).isoformat(),
                    "evidence": [
                        {
                            "feature": model.feature_name()[i],
                            "label": readable(model.feature_name()[i]),
                            "value": round(float(row.iloc[0, i]), 2),
                            "contribution": round(float(values[i]), 3),
                        }
                        for i in order
                    ],
                })
        cards.append(entry)

    levels = {
        c["risk"]: c["horizons"].get("3", c["horizons"].get("1", {})).get("level", 0)
        for c in cards
    }
    confidences = {
        c["risk"]: c["horizons"].get("3", {}).get("confidence", 1.0) for c in cards
    }
    compound = int(compose_compound(
        [levels.get("rain_wet", 0)], [levels.get("disease", 0)], [levels.get("frost", 0)]
    )[0])
    reasons = compound_reasons(
        levels.get("rain_wet", 0), levels.get("disease", 0), levels.get("frost", 0)
    )
    levels["compound"] = compound
    cards.append({
        "risk": "compound", "name": RISK_LABELS_KO["compound"],
        "horizons": {"3": {
            "level": compound, "levelName": LEVEL_NAMES[compound],
            "confidence": round(min(confidences.values()), 3),
            "composed": True, "reasons": reasons,
        }},
    })

    # ---- 조치 ----
    sensors = SensorState(
        gust_3s=round(float(features.loc[now, "wind_speed"]) * 1.8, 1),
        rain_detected=bool(features.loc[now, "rain"] > 0),
        soil_above_limit=bool(features.loc[now, "soil_index"] >= 90),
        hour=int(timestamp.hour),
    )
    decision = decide(levels, sensors, confidence=confidences, compound_reasons=reasons)

    # ---- M3 제어 이력 (최근 구간을 되짚어 생성) ----
    history = []
    for index in window[-12:]:
        row_time = pd.Timestamp(meta.loc[index, "ts"])
        step_levels = {}
        for risk in BASE_RISKS:
            model = models.get((risk, 3))
            if model is None:
                continue
            proba = model.predict(features.iloc[[index]][model.feature_name()])[0]
            step_levels[risk] = int(np.argmax(proba))
        step_sensors = SensorState(
            gust_3s=round(float(features.loc[index, "wind_speed"]) * 1.8, 1),
            rain_detected=bool(features.loc[index, "rain"] > 0),
            soil_above_limit=bool(features.loc[index, "soil_index"] >= 90),
            hour=int(row_time.hour),
        )
        step = decide(step_levels, step_sensors)
        for action in step.actions:
            history.append({
                "time": row_time.isoformat(), "layer": int(action.layer),
                "device": action.device, "command": action.command,
                "reason": action.reason, "riskType": action.risk_type,
                "executed": not action.rejected_by and not action.advisory,
                "rejectedBy": action.rejected_by, "advisory": action.advisory,
            })

    # ---- 시계열 ----
    series = {
        "time": [pd.Timestamp(meta.loc[i, "ts"]).isoformat() for i in window],
        **{
            key: [
                None if pd.isna(features.loc[i, key]) else round(float(features.loc[i, key]), digits)
                for i in window
            ]
            for key, _, _, digits in ENVIRONMENT_FIELDS
            if key in features.columns
        },
    }

    payload = {
        "generated": pd.Timestamp.now().isoformat(),
        "station": station,
        "now": timestamp.isoformat(),
        "modelVersion": MODEL_VERSION,
        "mode": decision.mode,
        "environment": [
            {
                "key": key, "name": name, "unit": unit,
                "value": (None if pd.isna(features.loc[now, key])
                          else round(float(features.loc[now, key]), digits)),
            }
            for key, name, unit, digits in ENVIRONMENT_FIELDS
            if key in features.columns
        ],
        "facility": {
            "gust3s": sensors.gust_3s,
            "rainDetected": sensors.rain_detected,
            "soilAboveLimit": sensors.soil_above_limit,
            "shadePosition": 0.0,
        },
        "cards": cards,
        "judgements": judgements,
        "actions": [
            {
                "device": a.device, "command": a.command, "layer": int(a.layer),
                "reason": a.reason, "riskType": a.risk_type,
                "executed": not a.rejected_by and not a.advisory,
                "rejectedBy": a.rejected_by, "advisory": a.advisory,
            }
            for a in decision.actions
        ],
        "history": history,
        "series": series,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    correct = sum(
        h.get("correct", False) for c in cards for h in c["horizons"].values()
    )
    total = sum(1 for c in cards for h in c["horizons"].values() if "correct" in h)
    print(f"[판단] {correct}/{total} 적중 / 운전모드 {decision.mode}")
    print(f"[조치] 실행 {len(decision.executable())} / 거부 "
          f"{sum(1 for a in decision.actions if a.rejected_by)}")
    print(f"[출력] {out_path} ({out_path.stat().st_size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
