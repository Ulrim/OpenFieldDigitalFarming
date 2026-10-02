"""현장 보정 — 관측소 학습모델을 실증포장 센서에 맞춘다.

사업계획서의 '현장 보정·현장시험(10.1~10.30)' 단계를 수행한다.

    ① 현장 수집 데이터 적재·품질관리·측정점 통합
    ② 가장 가까운 관측지점과 겹치는 기간에서 센서편차 추정
    ③ 대리지표(초관부 온도·엽면습윤)를 현장 실측으로 교체
    ④ 토양수분 임계값을 분위수로 정합(지수 0~100 vs 체적수분율 %)
    ⑤ 보정 전후 위험판단을 비교해 성능을 각각 제시

사용 예::

    python scripts/calibrate_field.py \
        --field data/field/sim_field.csv \
        --aaos-dir data/raw/aaos --agera5 data/raw/agera5.csv \
        --station '나주시 봉황면' --models artifacts/weather \
        --out artifacts/calibration
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf.data import aaos, agera5, field as field_data, quality  # noqa: E402
from ofdf.features import calibration as calib  # noqa: E402
from ofdf.features.derived import add_derived  # noqa: E402
from ofdf.features.weather import build_features  # noqa: E402
from ofdf.labels.risk import (  # noqa: E402
    RISK_LABELS_KO,
    RiskThresholds,
    build_states,
    future_labels,
)

BASE_RISKS = ["heat_dry", "rain_wet", "disease", "frost"]
LEVEL_NAMES = {0: "정상", 1: "주의", 2: "경계"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field", required=True, help="현장 수집 파일 또는 디렉터리")
    p.add_argument("--aaos-dir", required=True)
    p.add_argument("--agera5", required=True)
    p.add_argument("--station", default="나주시 봉황면", help="기준 관측지점")
    p.add_argument("--zone", default="treatment", help="보정 대상 구간")
    p.add_argument("--models", default=None, help="학습된 모델 디렉터리(있으면 판단 비교)")
    p.add_argument("--out", default="artifacts/calibration")
    p.add_argument("--horizon", type=int, default=3)
    p.add_argument("--strength", type=float, default=0.5, help="되돌림 세기 0~1")
    return p.parse_args()


def load_station(args: argparse.Namespace) -> pd.DataFrame:
    """관측소 시간자료를 모델 입력 스키마로 만든다."""
    hourly = aaos.load_directory(args.aaos_dir)
    flagged, _ = quality.run(hourly)
    hourly = quality.apply_mask(flagged)

    daily = agera5.read_daily(args.agera5)
    wind = hourly[hourly["station"].isin(["나주시 금천면", "나주시 산포면"])]
    wind = wind.groupby("ts")["wind_speed"].mean()

    one = hourly[hourly["station"] == args.station].drop_duplicates("ts").set_index("ts").sort_index()
    one = one.reindex(pd.date_range(one.index.min(), one.index.max(), freq="h"))
    one["wind_speed"] = one["wind_speed"].fillna(wind.reindex(one.index))

    day = one.index.normalize()
    for column in ("et0", "cloud_cover", "t_min_night", "rh_max_daily"):
        if column in daily.columns:
            one[column] = pd.Series(day, index=one.index).map(daily.set_index("date")[column])
    one["et0"] = one["et0"] / 24.0
    return add_derived(one)


def load_field(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """현장 수집 데이터를 1시간 격자로 올린다."""
    hourly = field_data.load(args.field, freq="h")
    zone = field_data.zone_frame(hourly, args.zone)
    # 관측소 스키마에 맞춰 이름을 맞춘다
    if "solar" in zone.columns:
        zone["solar_w"] = zone["solar"]
    return hourly, zone


def check_reference_station(args: argparse.Namespace, index: pd.DatetimeIndex):
    """기준으로 삼을 관측소의 센서 건전성을 본다."""
    hourly = aaos.load_directory(args.aaos_dir)
    window = hourly[
        (hourly["ts"] >= index.min()) & (hourly["ts"] <= index.max())
    ]
    if window.empty:
        return None
    health = quality.station_health(window)
    return health[health["station"] == args.station] if not health.empty else None


def compare_states(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """보정 전후 위험등급이 얼마나 달라졌는지 센다."""
    rows = []
    for risk in BASE_RISKS:
        a, b = before[risk], after[risk]
        changed = (a != b)
        rows.append({
            "위험유형": RISK_LABELS_KO[risk],
            "보정전 주의↑": int((a >= 1).sum()),
            "보정후 주의↑": int((b >= 1).sum()),
            "보정전 경계": int((a >= 2).sum()),
            "보정후 경계": int((b >= 2).sum()),
            "등급 변경": int(changed.sum()),
            "변경률(%)": round(changed.mean() * 100, 2),
            "상향": int((b > a).sum()),
            "하향": int((b < a).sum()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- ① 현장 데이터 ----
    print(f"[①] 현장 수집 적재: {args.field}")
    field_wide, zone = load_field(args)
    print(f"    {len(field_wide):,} 시간 / 구간 '{args.zone}' 변수 {len(zone.columns)}개")
    print(f"    기간 {zone.index.min()} ~ {zone.index.max()}")

    print(f"[①] 관측소 적재: {args.station}")
    station = load_station(args)

    # 기준 관측소가 성한지 먼저 본다. 고장난 지점을 기준으로 보정하면
    # 그 고장을 현장 데이터에 그대로 옮겨 심는다.
    health = check_reference_station(args, zone.index)
    if health is not None and not health.empty:
        print("\n[①] 기준 관측소 건전성 (대상 기간)")
        print(health.to_string(index=False))
        bad = health[health["verdict"] != "정상"]
        if not bad.empty:
            print(f"    [경고] {args.station} 에 이상이 있습니다 — "
                  f"다른 지점을 기준으로 쓰거나 해당 항목은 보정에서 제외하세요")
    overlap = zone.index.intersection(station.index)
    print(f"    중복 {len(overlap):,} 시간")
    if len(overlap) < calib.MIN_OVERLAP_HOURS:
        print(f"    [경고] 중복이 {calib.MIN_OVERLAP_HOURS}시간 미만이라 보정을 신뢰할 수 없습니다")

    # ---- ② 센서편차 추정 ----
    print(f"\n[②] 센서편차 추정 (되돌림 세기 {args.strength})")
    calibration = calib.fit(zone, station, station_name=args.station, strength=args.strength)
    print(calibration.summary().to_string(index=False))

    report = calib.agreement_report(zone, station, calibration)
    print("\n[②] 보정 전후 관측소 일치도")
    print(report.to_string(index=False))

    corrected = calibration.apply(zone)

    # ---- ③ 대리지표 -> 실측 교체 ----
    print("\n[③] 대리지표 교체")
    station_window = station.loc[overlap].copy()
    proxy_frame = station_window.copy()
    measured_frame = station_window.copy()

    for column in ("t_air", "rh", "solar_w", "rain", "wind_speed"):
        if column in corrected.columns:
            measured_frame[column] = corrected[column].reindex(measured_frame.index)

    measured_frame, replaced = calib.replace_proxies(measured_frame, corrected)
    print(f"    교체된 컬럼: {replaced or '없음'}")
    for column in replaced:
        delta = (measured_frame[column] - proxy_frame[column]).dropna()
        if not delta.empty:
            print(f"      {column:14s} 실측-대리 평균 {delta.mean():+.2f} / 표준편차 {delta.std():.2f}")

    # ---- ④ 토양수분 임계 정합 ----
    print("\n[④] 토양수분 임계값 분위수 정합")
    thresholds = RiskThresholds(rain_caution_mode="or")
    if "soil_moisture" in corrected.columns:
        measured_soil = corrected["soil_moisture"].reindex(measured_frame.index)
        low = calib.match_threshold_by_quantile(
            proxy_frame["soil_index"], thresholds.soil_low, measured_soil
        )
        high = calib.match_threshold_by_quantile(
            proxy_frame["soil_index"], thresholds.soil_high, measured_soil
        )
        print(f"    지수 하한 {thresholds.soil_low:.0f} -> 실측 {low:.1f}%  "
              f"/ 지수 상한 {thresholds.soil_high:.0f} -> 실측 {high:.1f}%")
        # 라벨용: 실측값 + 정합된 임계값 — '현장에서 실제로 일어난 일'
        measured_frame["soil_index"] = measured_soil
        field_thresholds = RiskThresholds(
            rain_caution_mode="or", soil_low=low, soil_high=high
        )
        # 모델 입력용: 학습 척도로 분위수 사상 — 분포 밖 입력을 막는다
        model_soil = calib.quantile_map(measured_soil, proxy_frame["soil_index"])
        print(f"    모델 입력용 분위수 사상: 실측 평균 {measured_soil.mean():.1f}% / "
              f"표준편차 {measured_soil.std():.1f} -> 지수 평균 {model_soil.mean():.1f} / "
              f"표준편차 {model_soil.std():.1f}")
    else:
        field_thresholds = thresholds
        model_soil = None
        print("    현장 토양수분 없음 — 지수 임계값을 그대로 사용")

    # ---- ⑤ 보정 전후 판단 비교 ----
    print("\n[⑤] 보정 전후 위험판단 비교")
    before_states = build_states(proxy_frame, thresholds)
    after_states = build_states(measured_frame, field_thresholds)
    table = compare_states(before_states, after_states)
    print(table.to_string(index=False))

    if args.models:
        print(f"\n[⑤] 모델 예측 비교 ({args.horizon}시간)")
        # 모델에 넣을 표는 라벨용과 다르다. 토양수분은 학습 척도로 되돌린다.
        model_input = measured_frame.copy()
        if model_soil is not None:
            model_input["soil_index"] = model_soil
        rows = predict_compare(proxy_frame, model_input, after_states, args)
        if rows is not None:
            print(rows.to_string(index=False))
            rows.to_csv(out_dir / "prediction_compare.csv", index=False, encoding="utf-8-sig")

    # ---- 산출물 ----
    calibration_payload = {
        "station": calibration.station,
        "zone": args.zone,
        "strength": calibration.strength,
        "overlap_hours": int(len(overlap)),
        "biases": {
            name: {
                "offset": bias.offset, "scale": bias.scale,
                "hourly_offset": bias.hourly_offset,
                "n_overlap": bias.n_overlap, "reliable": bias.reliable,
            }
            for name, bias in calibration.biases.items()
        },
        "soil_threshold": {"low": field_thresholds.soil_low, "high": field_thresholds.soil_high},
        "replaced_proxies": replaced,
    }
    (out_dir / "calibration.json").write_text(
        json.dumps(calibration_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report.to_csv(out_dir / "agreement.csv", index=False, encoding="utf-8-sig")
    table.to_csv(out_dir / "state_compare.csv", index=False, encoding="utf-8-sig")
    print(f"\n산출물: {out_dir}")
    return 0


def predict_compare(proxy_frame, measured_frame, truth_states, args) -> pd.DataFrame | None:
    """보정 전후 입력으로 모델을 돌려 성능을 비교한다."""
    import lightgbm as lgb
    from ofdf.evaluation import metrics

    model_dir = Path(args.models)
    truth = future_labels(truth_states, horizons=(args.horizon,))

    before_features = build_features(proxy_frame)
    after_features = build_features(measured_frame)

    rows = []
    for risk in BASE_RISKS:
        path = model_dir / f"model_{risk}_h{args.horizon}.txt"
        if not path.exists():
            continue
        booster = lgb.Booster(model_file=str(path))
        columns = booster.feature_name()

        y = truth[f"{risk}_h{args.horizon}"].fillna(0).astype(int).to_numpy()
        if (y >= 1).sum() == 0:
            continue

        def predict(frame: pd.DataFrame) -> np.ndarray:
            aligned = frame.reindex(columns=columns)
            return booster.predict(aligned).argmax(axis=1)

        before_score = metrics.score(y, predict(before_features))
        after_score = metrics.score(y, predict(after_features))
        rows.append({
            "위험유형": RISK_LABELS_KO[risk],
            "위험 표본": before_score.n_risk,
            "보정전 MacroF1": round(before_score.macro_f1, 3),
            "보정후 MacroF1": round(after_score.macro_f1, 3),
            "F1 개선(%p)": round((after_score.macro_f1 - before_score.macro_f1) * 100, 1),
            "보정전 재현율": round(before_score.risk_recall, 3),
            "보정후 재현율": round(after_score.risk_recall, 3),
            "재현율 개선(%p)": round((after_score.risk_recall - before_score.risk_recall) * 100, 1),
        })
    return pd.DataFrame(rows) if rows else None


if __name__ == "__main__":
    raise SystemExit(main())
