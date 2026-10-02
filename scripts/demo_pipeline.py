"""시제품 전체 흐름 시연.

사업계획서의 핵심 작동 구조를 그대로 한 번 돌려 본다.

    ① 데이터 입력 → ② 데이터 정리 → ③ AI 위험판단 → ④ 조치 결정 → ⑤ 현장 실행·기록

실제 관측 데이터에서 위험 상황이 있었던 시각을 골라, 그때 시스템이 무엇을
판단하고 어떤 조치를 내렸을지 보여 준다. 화면 M1(실시간 대시보드)과
M2(AI 판단 상세)에 올라갈 내용이 그대로 나온다.

사용 예::

    python scripts/demo_pipeline.py --cache artifacts/dataset.pkl --models artifacts/weather
"""

from __future__ import annotations

import argparse
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
from ofdf.models.fusion import VisionFinding, fuse_disease_risk  # noqa: E402

BASE_RISKS = ["heat_dry", "rain_wet", "disease", "frost"]
LEVEL_NAMES = {0: "정상", 1: "주의", 2: "경계"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, help="데이터셋 pickle")
    p.add_argument("--models", required=True, help="학습된 모델 디렉터리")
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--cases", type=int, default=4, help="보여줄 상황 수")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def load_models(model_dir: Path, horizon: int) -> dict:
    import lightgbm as lgb

    models = {}
    for risk in BASE_RISKS:
        path = model_dir / f"model_{risk}_h{horizon}.txt"
        if path.exists():
            models[risk] = lgb.Booster(model_file=str(path))
    return models


def pick_cases(ds, horizon: int, n: int, seed: int) -> list[int]:
    """위험유형이 골고루 섞이도록 상황을 고른다."""
    rng = np.random.default_rng(seed)
    labels = ds.labels.reset_index(drop=True)
    chosen = []
    for risk in BASE_RISKS:
        hits = np.where(labels[f"{risk}_h{horizon}"].fillna(0).to_numpy() >= 2)[0]
        if len(hits):
            chosen.append(int(rng.choice(hits)))
        if len(chosen) >= n:
            break
    return chosen[:n]


def main() -> int:
    args = parse_args()
    ds = pickle.loads(Path(args.cache).read_bytes())
    models = load_models(Path(args.models), args.horizon)
    if not models:
        print(f"[오류] {args.models} 에 모델이 없습니다. 먼저 train_weather.py 를 실행하세요.")
        return 1

    features = ds.features.reset_index(drop=True)
    meta = ds.meta.reset_index(drop=True)
    labels = ds.labels.reset_index(drop=True)

    for row in pick_cases(ds, args.horizon, args.cases, args.seed):
        ts = meta.loc[row, "ts"]
        station = meta.loc[row, "station"]
        sample = features.iloc[[row]]

        print("\n" + "=" * 78)
        print(f"■ {station}  {ts:%Y-%m-%d %H시}  —  {args.horizon}시간 후 위험판단")
        print("=" * 78)

        # ① 데이터 입력 / ② 정리
        print("\n[① 입력·정리] 현재 환경")
        for name, col, unit in [
            ("기온", "t_air", "℃"), ("상대습도", "rh", "%"),
            ("일사", "solar_w", "W/m²"), ("시간강수", "rain", "mm"),
            ("초관부온도", "canopy_temp", "℃"), ("토양수분지수", "soil_index", ""),
            ("엽면습윤 연속", "run_wet", "시간"),
        ]:
            if col in sample:
                print(f"    {name:12s} {float(sample[col].iloc[0]):7.1f} {unit}")

        # ③ AI 위험판단
        print(f"\n[③ AI 위험판단] {args.horizon}시간 이내 도달 등급")
        levels, confidences = {}, {}
        for risk, booster in models.items():
            proba = booster.predict(sample[booster.feature_name()])[0]
            level = int(np.argmax(proba))
            levels[risk] = level
            confidences[risk] = float(proba[level])
            truth = int(labels.loc[row, f"{risk}_h{args.horizon}"] or 0)
            mark = "O" if level == truth else "X"
            print(
                f"    {RISK_LABELS_KO[risk]:8s} {LEVEL_NAMES[level]:3s} "
                f"(신뢰도 {proba[level]:.2f})   실제 {LEVEL_NAMES[truth]} [{mark}]"
            )

        compound = int(
            compose_compound([levels.get("rain_wet", 0)], [levels.get("disease", 0)],
                             [levels.get("frost", 0)])[0]
        )
        levels["compound"] = compound
        reasons = compound_reasons(
            levels.get("rain_wet", 0), levels.get("disease", 0), levels.get("frost", 0)
        )
        if compound:
            print(f"    {RISK_LABELS_KO['compound']:8s} {LEVEL_NAMES[compound]:3s} ← {' / '.join(reasons)}")

        # 판단근거 상위 3개(화면 M2 요구사항)
        top_risk = max(levels, key=lambda r: levels[r]) if any(levels.values()) else None
        if top_risk in models:
            booster = models[top_risk]
            contrib = booster.predict(sample[booster.feature_name()], pred_contrib=True)[0]
            n_feat = len(booster.feature_name())
            start = levels[top_risk] * (n_feat + 1)
            values = np.array(contrib[start : start + n_feat])
            order = np.argsort(-np.abs(values))[:3]
            print(f"\n[판단근거] {RISK_LABELS_KO[top_risk]} 상위 3개")
            for i in order:
                name = booster.feature_name()[i]
                print(f"    {name:22s} 값 {float(sample[name].iloc[0]):8.2f}  기여도 {values[i]:+.3f}")

        # 비전 융합 — 병해 위험
        if levels.get("disease", 0) >= 1:
            vision = VisionFinding(disease_code=18, severity=1, confidence=0.81,
                                   inspected_hours_ago=6.0)
            fused = fuse_disease_risk(
                levels["disease"], vision,
                favourable_hours_7d=float(features.loc[row, "run_wet"]) * 3,
            )
            print(f"\n[비전 융합] 병해 종합등급: {fused.level_name}")
            for reason in fused.reasons:
                print(f"    · {reason}")

        # ④ 조치 결정 / ⑤ 현장 실행
        sensors = SensorState(
            gust_3s=float(features.loc[row, "wind_speed"]) * 1.8,  # 시간평균 -> 순간최대 근사
            rain_detected=float(features.loc[row, "rain"]) > 0,
            soil_above_limit=float(features.loc[row, "soil_index"]) >= 90,
            hour=int(ts.hour),
        )
        decision = decide(levels, sensors, confidence=confidences, compound_reasons=reasons)
        print(f"\n[④⑤ 조치·실행] 운전모드: {decision.mode}")
        if not decision.actions:
            print("    조치 없음")
        for action in decision.actions:
            print(f"    {action.describe()}")

    print("\n" + "=" * 78)
    print("주의 — 3초 순간최대풍속은 현장 풍속계 값이다. 위 시연은 공공 관측의")
    print("시간평균 풍속을 1.8배 한 근사이며, 실제 시제품은 현장 센서를 쓴다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
