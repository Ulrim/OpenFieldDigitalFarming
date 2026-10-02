"""기상청 예보 연계의 가치 측정.

강우·과습은 유일하게 성능목표에 못 미치는 위험유형이다. 원인은 모델이
아니라 입력이다 — 과거 관측만으로 1~3시간 뒤 비를 맞히는 데는 원리적
한계가 있고, 사업계획서 설계에는 기상청 예보가 입력으로 들어가 있다.

API 키 없이도 연계 투자 여부를 판단할 수 있도록, **예보 정확도를 변수로
두고 성능 곡선**을 그린다. 연계 후 :func:`ofdf.data.forecast.observed_skill`
로 실제 예보의 정확도를 재면, 이 곡선에서 기대 성능을 바로 읽을 수 있다.

사용 예::

    python scripts/forecast_value.py \
        --cache artifacts/dataset.pkl --out artifacts/forecast
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

from ofdf.data.forecast import SimulatedForecast, observed_skill  # noqa: E402
from ofdf.evaluation import metrics  # noqa: E402
from ofdf.labels.risk import RISK_LABELS_KO  # noqa: E402
from ofdf.models import weather_risk  # noqa: E402

#: 예보가 가장 크게 기여할 위험유형. 고온·건조·병해는 현재값 추세가 더 중요하다.
TARGET_RISKS = ["rain_wet", "heat_dry", "frost", "disease"]

#: 예보 생성에 쓰는 관측 항목
OBSERVATION_COLUMNS = ["rain", "t_air", "rh", "wind_speed"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--out", default="artifacts/forecast")
    p.add_argument("--horizons", default="1,3")
    p.add_argument("--skills", default="0.0,0.3,0.5,0.7,0.85,1.0")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_forecast_features(dataset, skills: list[float], seed: int) -> dict[float, pd.DataFrame]:
    """지점별로 모사 예보를 만들어 하나의 표로 모은다.

    예보는 지점마다 따로 만든다. 지점을 섞으면 shift 가 다른 지점 값을
    끌어온다.
    """
    meta = dataset.meta.reset_index(drop=True)
    features = dataset.features.reset_index(drop=True)

    tables: dict[float, list[pd.DataFrame]] = {s: [] for s in skills}
    positions: list[np.ndarray] = []

    for station in meta["station"].unique():
        rows = np.where(meta["station"].to_numpy() == station)[0]
        observations = features.iloc[rows][OBSERVATION_COLUMNS].copy()
        observations.index = pd.DatetimeIndex(meta["ts"].iloc[rows].to_numpy())
        positions.append(rows)

        for skill in skills:
            provider = SimulatedForecast(observations, skill=skill, seed=seed)
            tables[skill].append(provider.build())

    order = np.concatenate(positions)
    out = {}
    for skill, parts in tables.items():
        frame = pd.concat(parts, ignore_index=True)
        frame.index = order
        out[skill] = frame.sort_index()
    return out


def measure_forecast_quality(
    dataset, forecasts: dict[float, pd.DataFrame], horizon: int
) -> pd.DataFrame:
    """각 skill 의 예보가 실제로 어느 정도 품질인지 적어 둔다."""
    features = dataset.features.reset_index(drop=True)
    meta = dataset.meta.reset_index(drop=True)

    rows = []
    for skill, table in forecasts.items():
        observed_future = []
        for station in meta["station"].unique():
            index = np.where(meta["station"].to_numpy() == station)[0]
            series = features.iloc[index]["rain"].reset_index(drop=True).shift(-horizon)
            series.index = index
            observed_future.append(series)
        truth = pd.concat(observed_future).sort_index()

        quality = observed_skill(table[f"fc_rain_h{horizon}"], truth)
        rows.append({"skill": skill, **quality})
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizons = [int(h) for h in args.horizons.split(",")]
    skills = [float(s) for s in args.skills.split(",")]

    dataset = pickle.loads(Path(args.cache).read_bytes())
    features = dataset.features.reset_index(drop=True)
    meta = dataset.meta.reset_index(drop=True)
    states = dataset.states.reset_index(drop=True)

    print(f"[데이터] {len(features):,} 시간 / 기본 피처 {features.shape[1]}개")
    print(f"[설정] 예보 정확도 {skills}")

    groups = weather_risk.shared_group_keys(meta, states)
    train_idx, valid_idx, test_idx = weather_risk.split_by_event(groups, seed=args.seed)
    print(f"[분할] 학습 {len(train_idx):,} / 검증 {len(valid_idx):,} / 시험 {len(test_idx):,}")

    print("[예보] 모사 예보 생성 중 ...")
    forecasts = build_forecast_features(dataset, skills, args.seed)
    n_forecast = forecasts[skills[0]].shape[1]
    print(f"    예보 피처 {n_forecast}개 추가")

    quality = measure_forecast_quality(dataset, forecasts, horizons[-1])
    print(f"\n[예보 품질] {horizons[-1]}시간 강수 예보")
    print(quality.to_string(index=False))

    rows = []
    for risk in TARGET_RISKS:
        for horizon in horizons:
            key = f"{risk}_h{horizon}"
            y = dataset.labels[key].reset_index(drop=True).fillna(0).astype(int).to_numpy()
            if (y[test_idx] >= 1).sum() == 0:
                continue

            # 기준선 — 예보 없음
            base_model = weather_risk.train_one(
                features, y, train_idx, valid_idx, risk=risk, horizon=horizon
            )
            base = metrics.score(y[test_idx], base_model.predict(features.iloc[test_idx]))

            for skill in skills:
                combined = pd.concat([features, forecasts[skill]], axis=1)
                model = weather_risk.train_one(
                    combined, y, train_idx, valid_idx, risk=risk, horizon=horizon
                )
                score = metrics.score(y[test_idx], model.predict(combined.iloc[test_idx]))

                importance = weather_risk.feature_importance(model, top=50)
                forecast_share = (
                    importance[importance["feature"].str.startswith("fc_")]["gain"].sum()
                    / max(importance["gain"].sum(), 1e-9)
                )

                rows.append({
                    "위험유형": RISK_LABELS_KO[risk],
                    "시계(h)": horizon,
                    "예보 정확도": skill,
                    "MacroF1": round(score.macro_f1, 3),
                    "기준선 MacroF1": round(base.macro_f1, 3),
                    "F1 개선(%p)": round((score.macro_f1 - base.macro_f1) * 100, 1),
                    "재현율": round(score.risk_recall, 3),
                    "기준선 재현율": round(base.risk_recall, 3),
                    "재현율 개선(%p)": round((score.risk_recall - base.risk_recall) * 100, 1),
                    "오경보율": round(score.false_alarm_rate, 3),
                    "예보 기여도(%)": round(forecast_share * 100, 1),
                })
            print(f"  [완료] {RISK_LABELS_KO[risk]} {horizon}시간 "
                  f"(기준선 F1 {base.macro_f1:.3f})")

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 240)
    print("\n===== 예보 정확도별 성능 =====")
    print(table.to_string(index=False))

    print("\n===== 요약: 예보가 목표 달성에 필요한 정확도 =====")
    for (risk, horizon), group in table.groupby(["위험유형", "시계(h)"], sort=False):
        meets = group[group["MacroF1"] >= 0.80]
        baseline = group["기준선 MacroF1"].iloc[0]
        if meets.empty:
            print(f"  {risk} {horizon}h: 기준선 {baseline:.3f} -> "
                  f"완벽예보에서도 {group['MacroF1'].max():.3f} (목표 0.80 미달)")
        else:
            need = meets["예보 정확도"].min()
            print(f"  {risk} {horizon}h: 기준선 {baseline:.3f} -> "
                  f"예보 정확도 {need:.2f} 이상이면 목표 0.80 달성")

    table.to_csv(out_dir / "forecast_value.csv", index=False, encoding="utf-8-sig")
    quality.to_csv(out_dir / "forecast_quality.csv", index=False, encoding="utf-8-sig")
    (out_dir / "summary.json").write_text(
        json.dumps({"skills": skills, "horizons": horizons,
                    "rows": table.to_dict("records")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n산출물: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
