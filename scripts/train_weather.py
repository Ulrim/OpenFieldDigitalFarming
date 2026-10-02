"""기상 기반 농지위험 판단 모델 학습·평가.

사용 예::

    python scripts/train_weather.py \
        --aaos-dir data/raw/aaos \
        --agera5 data/raw/agera5_naju_daily.csv \
        --out artifacts/weather

위험유형 x 예측시계마다 LightGBM 을 학습하고, 같은 시험데이터에서
단순 기준값 규칙 및 로지스틱 회귀와 성능을 비교한다.
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

from ofdf.data.dataset import build  # noqa: E402
from ofdf.evaluation import metrics  # noqa: E402
from ofdf.labels.risk import RISK_LABELS_KO, RISK_TYPES, compose_compound  # noqa: E402
from ofdf.models import weather_risk  # noqa: E402
from ofdf.models.baseline import LogisticBaseline, RuleBaseline  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aaos-dir", required=True, help="농업기상 월별 파일 디렉터리")
    p.add_argument("--agera5", required=True, help="AgERA5 일자료 CSV")
    p.add_argument("--out", default="artifacts/weather", help="산출물 디렉터리")
    p.add_argument("--cache", default=None, help="데이터셋 캐시 pickle 경로")
    p.add_argument("--horizons", default="1,3", help="예측시계(시간), 쉼표 구분")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--season", default="9,10,11", help="대상 월. 'all' 이면 연중 전체"
    )
    return p.parse_args()


def load_dataset(args: argparse.Namespace):
    cache = Path(args.cache) if args.cache else None
    if cache and cache.exists():
        print(f"[데이터] 캐시 사용: {cache}")
        return pickle.loads(cache.read_bytes())

    months = None if args.season == "all" else tuple(int(m) for m in args.season.split(","))
    print("[데이터] 원본에서 생성 중 ...")
    ds = build(args.aaos_dir, args.agera5, season_months=months)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(ds))
    return ds


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(h) for h in args.horizons.split(",")]

    ds = load_dataset(args)
    print(f"[데이터] {len(ds):,} 시간 x {ds.features.shape[1]} 피처")
    if ds.qc is not None:
        print(f"[품질] 이상표시 {sum(ds.qc.flagged.values()):,}건 / "
              f"중복 {ds.qc.duplicates} / 결측시각 {ds.qc.gaps}")

    X = ds.features.reset_index(drop=True)
    meta = ds.meta.reset_index(drop=True)

    results: dict[str, dict[str, metrics.RiskScore]] = {
        "단순 기준값 규칙": {}, "로지스틱 회귀(현재값)": {}, "LightGBM(AI)": {},
    }
    rows, explanations = [], {}
    # 복합위험은 구성 위험 예측을 조합해 만든다(독립 학습하지 않는다)
    base_risks = [r for r in RISK_TYPES if r != "compound"]
    component_preds: dict[int, dict[str, np.ndarray]] = {h: {} for h in horizons}

    # 모든 위험유형이 같은 시험셋을 쓰도록 공통 분할을 한 번만 만든다.
    # 그래야 복합위험을 구성 위험 예측으로 조합할 수 있다.
    states = ds.states.reset_index(drop=True)
    shared_groups = weather_risk.shared_group_keys(meta, states)
    train_idx, valid_idx, test_idx = weather_risk.split_by_event(shared_groups, seed=args.seed)
    print(f"[분할] 학습 {len(train_idx):,} / 검증 {len(valid_idx):,} / 시험 {len(test_idx):,} "
          f"(기상사례 블록 {shared_groups.nunique():,}개 단위)")

    for risk in base_risks:
        events = ds.events[f"{risk}_event"].reset_index(drop=True)

        for horizon in horizons:
            key = f"{risk}_h{horizon}"
            y = ds.labels[key].reset_index(drop=True).fillna(0).astype(int).to_numpy()
            tag = f"{RISK_LABELS_KO[risk]} {horizon}시간"

            if (y[test_idx] >= 1).sum() == 0:
                print(f"  [건너뜀] {tag}: 시험셋에 위험사례 없음")
                continue

            # 1) 단순 기준값 규칙 — 현재 상태등급을 그대로 예측으로 쓴다
            pred_rule = RuleBaseline().predict(states[risk].iloc[test_idx])

            # 2) 로지스틱 회귀 — 현재값만
            logistic = LogisticBaseline().fit(X.iloc[train_idx], y[train_idx])
            pred_logit = logistic.predict(X.iloc[test_idx])

            # 3) LightGBM — 변화추세·지속시간 포함 전체 피처
            model = weather_risk.train_one(
                X, y, train_idx, valid_idx, risk=risk, horizon=horizon
            )
            pred_ai = model.predict(X.iloc[test_idx])

            y_test = y[test_idx]
            s_rule = metrics.score(y_test, pred_rule)
            s_logit = metrics.score(y_test, pred_logit)
            s_ai = metrics.score(y_test, pred_ai)

            results["단순 기준값 규칙"][tag] = s_rule
            results["로지스틱 회귀(현재값)"][tag] = s_logit
            results["LightGBM(AI)"][tag] = s_ai

            gain = metrics.improvement(s_rule, s_ai)
            detection = metrics.event_detection(y_test, pred_ai, events.iloc[test_idx])
            lead = metrics.lead_time_minutes(
                meta.iloc[test_idx]["ts"], y_test, pred_ai, events.iloc[test_idx]
            )

            rows.append(
                {
                    "위험유형": RISK_LABELS_KO[risk],
                    "시계(h)": horizon,
                    "시험 표본": s_ai.n_samples,
                    "위험 표본": s_ai.n_risk,
                    "규칙 MacroF1": round(s_rule.macro_f1, 3),
                    "로지스틱 MacroF1": round(s_logit.macro_f1, 3),
                    "AI MacroF1": round(s_ai.macro_f1, 3),
                    "AI 놓침방지율": round(s_ai.risk_recall, 3),
                    "규칙 놓침방지율": round(s_rule.risk_recall, 3),
                    "AI 오경보율": round(s_ai.false_alarm_rate, 3),
                    "MacroF1 개선(%p)": round(gain["macro_f1_gain_pp"], 1),
                    "놓침방지 개선(%p)": round(gain["risk_recall_gain_pp"], 1),
                    "이벤트 탐지율": round(detection["detection_rate"], 3),
                    "사전알림(분)": round(lead["mean_minutes"], 0) if lead["n"] else None,
                }
            )

            component_preds[horizon][risk] = pred_ai

            weather_risk.save(model, out_dir / f"model_{key}.txt")
            explanations[key] = weather_risk.feature_importance(model).to_dict("records")
            print(f"  [완료] {tag}: AI MacroF1 {s_ai.macro_f1:.3f} "
                  f"(규칙 {s_rule.macro_f1:.3f}) 놓침방지 {s_ai.risk_recall:.3f}")

    # ---- 복합위험: 구성 위험 예측 조합 ----
    for horizon in horizons:
        parts = component_preds[horizon]
        if not {"rain_wet", "disease", "frost"} <= parts.keys():
            continue
        y_test = ds.labels[f"compound_h{horizon}"].reset_index(drop=True).fillna(0).astype(int).to_numpy()[test_idx]
        if (y_test >= 1).sum() == 0:
            continue

        pred_ai = compose_compound(parts["rain_wet"], parts["disease"], parts["frost"])
        pred_rule = states["compound"].iloc[test_idx].to_numpy()

        s_rule = metrics.score(y_test, pred_rule)
        s_ai = metrics.score(y_test, pred_ai)
        tag = f"{RISK_LABELS_KO['compound']} {horizon}시간"
        results["단순 기준값 규칙"][tag] = s_rule
        results["LightGBM(AI)"][tag] = s_ai

        events = ds.events["compound_event"].reset_index(drop=True)
        gain = metrics.improvement(s_rule, s_ai)
        detection = metrics.event_detection(y_test, pred_ai, events.iloc[test_idx])
        lead = metrics.lead_time_minutes(meta.iloc[test_idx]["ts"], y_test, pred_ai, events.iloc[test_idx])
        rows.append({
            "위험유형": RISK_LABELS_KO["compound"] + "(조합)", "시계(h)": horizon,
            "시험 표본": s_ai.n_samples, "위험 표본": s_ai.n_risk,
            "규칙 MacroF1": round(s_rule.macro_f1, 3), "로지스틱 MacroF1": None,
            "AI MacroF1": round(s_ai.macro_f1, 3),
            "AI 놓침방지율": round(s_ai.risk_recall, 3), "규칙 놓침방지율": round(s_rule.risk_recall, 3),
            "AI 오경보율": round(s_ai.false_alarm_rate, 3),
            "MacroF1 개선(%p)": round(gain["macro_f1_gain_pp"], 1),
            "놓침방지 개선(%p)": round(gain["risk_recall_gain_pp"], 1),
            "이벤트 탐지율": round(detection["detection_rate"], 3),
            "사전알림(분)": round(lead["mean_minutes"], 0) if lead["n"] else None,
        })
        print(f"  [완료] {tag}(조합): AI MacroF1 {s_ai.macro_f1:.3f} (규칙 {s_rule.macro_f1:.3f})")

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "performance.csv", index=False, encoding="utf-8-sig")
    (out_dir / "feature_importance.json").write_text(
        json.dumps(explanations, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pd.set_option("display.width", 250)
    print("\n===== 성능 비교 =====")
    print(table.to_string(index=False))
    print(f"\n산출물: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
