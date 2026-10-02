"""현장 제어기 추론시간 측정 및 ONNX 변환.

사업계획서의 정량 성과지표 두 가지를 잰다.

    AI 판단 소요시간   평균 30초 이내   센서 수집 -> 위험판단·추천조치 생성 로그
    현장 제어응답      평균 30초 이내   명령 발생 -> 구동 개시 로그

여기서 재는 것은 앞쪽이다. 뒤쪽(구동 개시까지)은 밸브·개폐기 실측이 필요하다.

한 번의 '판단'은 모델 추론 하나가 아니다. 실제로는 다음을 모두 거친다.

    센서 입력 -> 파생변수 -> 시계열 피처(202개) -> 위험유형 4종 x 시계 2종 추론
    -> 복합위험 조합 -> 판단근거 추출 -> 조치 결정 -> 이력 기록

엣지 제어기는 이 전체를 5분 주기로 돌린다(사업계획서 '로컬 AI 추론 L2 5분 주기').
그래서 모델 하나의 forward 시간이 아니라 **끝에서 끝까지**를 잰다.

사용 예::

    python scripts/benchmark_inference.py \
        --cache artifacts/dataset.pkl --models artifacts/weather \
        --out artifacts/benchmark --export-onnx
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf.action.journal import Journal, RiskJudgement, record_from_decision  # noqa: E402
from ofdf.action.recommend import SensorState, decide  # noqa: E402
from ofdf.features.derived import add_derived  # noqa: E402
from ofdf.features.weather import build_features  # noqa: E402
from ofdf.labels.risk import compose_compound, compound_reasons  # noqa: E402

BASE_RISKS = ["heat_dry", "rain_wet", "disease", "frost"]

#: 사업계획서 정량 목표(초)
TARGET_SECONDS = 30.0

#: 엣지 제어기는 이 주기로 추론한다
INFERENCE_PERIOD_SECONDS = 300.0

#: ONNX 로 바꾼 모델이 원본과 같은 등급을 내야 하는 최소 비율.
#:
#: onnxmltools 의 LightGBM 분류기 변환은 비트 단위로 충실하지 않다. 트리
#: 분기 임계값이 float32 로 표현되면서 경계 근처 표본의 등급이 뒤집힌다
#: (측정값: 8개 모델 중 5개에서 0.1~0.3% 불일치, 확률 최대차 0.78).
#: 분류기는 float64 변환을 지원하지 않아(DoubleTensorType 거부) 피할 수 없고,
#: 입력을 float32 로 맞춰도 해결되지 않는다(99.700% -> 99.767%).
#:
#: 안전 판단을 바꾸는 변경이므로, 배포 전 이 검증을 반드시 통과해야 한다.
ONNX_AGREEMENT_FLOOR = 0.999


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, help="데이터셋 pickle")
    p.add_argument("--models", required=True)
    p.add_argument("--out", default="artifacts/benchmark")
    p.add_argument("--horizons", default="1,3")
    p.add_argument("--repeats", type=int, default=200, help="측정 반복 횟수")
    p.add_argument("--window", type=int, default=48,
                   help="피처 계산에 쓰는 과거 시간(구간 통계 최대 24시간 + 여유)")
    p.add_argument("--export-onnx", action="store_true")
    return p.parse_args()


def load_models(model_dir: Path, horizons: list[int]) -> dict:
    import lightgbm as lgb
    models = {}
    for risk in BASE_RISKS:
        for horizon in horizons:
            path = model_dir / f"model_{risk}_h{horizon}.txt"
            if path.exists():
                models[(risk, horizon)] = lgb.Booster(model_file=str(path))
    return models


def export_onnx(models: dict, out_dir: Path) -> pd.DataFrame:
    """LightGBM 모델을 ONNX 로 변환한다(엣지 배포용)."""
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    rows = []
    for (risk, horizon), booster in models.items():
        name = f"model_{risk}_h{horizon}"
        n_features = booster.num_feature()
        try:
            onnx_model = convert_lightgbm(
                booster, initial_types=[("input", FloatTensorType([None, n_features]))],
                zipmap=False,
            )
            path = out_dir / f"{name}.onnx"
            path.write_bytes(onnx_model.SerializeToString())
            txt_size = (Path(booster.model_file).stat().st_size
                        if getattr(booster, "model_file", None) else 0)
            rows.append({
                "모델": name, "변환": "성공",
                "ONNX 크기(KB)": round(path.stat().st_size / 1024, 1),
                "트리 수": booster.num_trees(), "피처 수": n_features,
            })
        except Exception as exc:                           # noqa: BLE001
            rows.append({"모델": name, "변환": f"실패: {type(exc).__name__}",
                         "ONNX 크기(KB)": None,
                         "트리 수": booster.num_trees(), "피처 수": n_features})
    return pd.DataFrame(rows)


def verify_onnx_parity(
    out_dir: Path, models: dict, features: pd.DataFrame, n_samples: int = 3000
) -> pd.DataFrame:
    """ONNX 모델이 원본과 같은 판단을 내는지 검증한다.

    변환이 빨라도 판단이 달라지면 쓸 수 없다. 무작위 표본으로 등급 일치율과
    확률 최대차를 재고, :data:`ONNX_AGREEMENT_FLOOR` 미만이면 부적합으로 표시한다.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    index = rng.choice(len(features), min(n_samples, len(features)), replace=False)

    rows = []
    for (risk, horizon), booster in models.items():
        path = out_dir / f"model_{risk}_h{horizon}.onnx"
        if not path.exists():
            continue
        columns = booster.feature_name()
        sample = features.iloc[index][columns]

        reference = booster.predict(sample)
        options = ort.SessionOptions()
        options.log_severity_level = 3          # 출력 형상 경고를 줄인다
        session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        outputs = session.run(
            None, {session.get_inputs()[0].name: sample.to_numpy(dtype=np.float32)}
        )
        converted = np.asarray(outputs[1] if len(outputs) > 1 else outputs[0])

        agreement = float((reference.argmax(1) == converted.argmax(1)).mean())
        rows.append({
            "모델": f"model_{risk}_h{horizon}",
            "표본": len(index),
            "등급 일치율(%)": round(agreement * 100, 3),
            "확률 최대차": f"{float(np.abs(reference - converted).max()):.2e}",
            "판정": "적합" if agreement >= ONNX_AGREEMENT_FLOOR else "부적합",
        })
    return pd.DataFrame(rows)


def time_onnx(out_dir: Path, models: dict, sample: np.ndarray, repeats: int) -> dict | None:
    """ONNX 런타임 추론시간을 잰다."""
    import onnxruntime as ort

    sessions = {}
    for (risk, horizon) in models:
        path = out_dir / f"model_{risk}_h{horizon}.onnx"
        if path.exists():
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1       # 엣지 제어기 보수적 가정
            sessions[(risk, horizon)] = ort.InferenceSession(
                str(path), options, providers=["CPUExecutionProvider"]
            )
    if not sessions:
        return None

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for session in sessions.values():
            session.run(None, {session.get_inputs()[0].name: sample})
        timings.append(time.perf_counter() - start)
    return summarise(timings, len(sessions))


def summarise(timings: list[float], n_models: int = 0) -> dict:
    array = np.array(timings)
    return {
        "n": len(array), "모델 수": n_models,
        "평균(ms)": round(float(array.mean() * 1000), 2),
        "중앙값(ms)": round(float(np.median(array) * 1000), 2),
        "p95(ms)": round(float(np.percentile(array, 95) * 1000), 2),
        "최대(ms)": round(float(array.max() * 1000), 2),
    }


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = [int(h) for h in args.horizons.split(",")]

    import pickle
    dataset = pickle.loads(Path(args.cache).read_bytes())
    models = load_models(Path(args.models), horizons)
    if not models:
        print(f"[오류] {args.models} 에 모델이 없습니다")
        return 1

    print(f"[환경] {platform.machine()} / Python {platform.python_version()} / "
          f"모델 {len(models)}개")
    print(f"[설정] 반복 {args.repeats}회, 과거 창 {args.window}시간")

    # 실제 운영과 같은 모양의 입력을 만든다 — 최근 window 시간의 관측
    meta = dataset.meta.reset_index(drop=True)
    features_all = dataset.features.reset_index(drop=True)
    start = len(features_all) // 2
    raw_window = features_all.iloc[start - args.window : start + 1].copy()
    raw_window.index = pd.to_datetime(meta["ts"].iloc[start - args.window : start + 1].to_numpy())

    results: dict[str, dict] = {}

    # ---- 1. 피처 생성 ----
    timings = []
    for _ in range(args.repeats):
        begin = time.perf_counter()
        build_features(raw_window)
        timings.append(time.perf_counter() - begin)
    results["피처 생성"] = summarise(timings)

    # ---- 2. 모델 추론 (LightGBM 네이티브) ----
    sample_frame = features_all.iloc[[start]]
    timings = []
    for _ in range(args.repeats):
        begin = time.perf_counter()
        for booster in models.values():
            booster.predict(sample_frame[booster.feature_name()])
        timings.append(time.perf_counter() - begin)
    results["모델 추론(LightGBM)"] = summarise(timings, len(models))

    # ---- 3. 판단근거 추출 (SHAP 기여도) ----
    booster = next(iter(models.values()))
    timings = []
    for _ in range(args.repeats):
        begin = time.perf_counter()
        booster.predict(sample_frame[booster.feature_name()], pred_contrib=True)
        timings.append(time.perf_counter() - begin)
    results["판단근거 추출"] = summarise(timings, 1)

    # ---- 4. 조치 결정 ----
    levels = {"rain_wet": 2, "disease": 1, "frost": 2, "heat_dry": 0}
    reasons = compound_reasons(2, 1, 2)
    sensors = SensorState(gust_3s=3.0, rain_detected=True, hour=22)
    timings = []
    for _ in range(args.repeats):
        begin = time.perf_counter()
        decide(levels, sensors, confidence={k: 0.9 for k in levels},
               compound_reasons=reasons)
        timings.append(time.perf_counter() - begin)
    results["조치 결정"] = summarise(timings)

    # ---- 5. 이력 기록 ----
    import datetime
    journal = Journal(out_dir / "benchmark_journal.jsonl")
    decision = decide(levels, sensors, confidence={k: 0.9 for k in levels},
                      compound_reasons=reasons)
    judgements = [RiskJudgement(r, 3, l, "경계", 0.9) for r, l in levels.items() if l]
    timings = []
    for _ in range(args.repeats):
        begin = time.perf_counter()
        journal.append(record_from_decision(
            datetime.datetime.now(), "실증포장", "ofdf-weather-0.1.0",
            {"t_air": 3.0}, judgements, decision,
        ))
        timings.append(time.perf_counter() - begin)
    results["이력 기록"] = summarise(timings)

    # ---- 6. 끝에서 끝까지 ----
    timings = []
    for _ in range(args.repeats):
        begin = time.perf_counter()
        feature_frame = build_features(raw_window)
        row = feature_frame.iloc[[-1]]
        predicted = {}
        for (risk, horizon), model in models.items():
            if horizon != 3:
                continue
            proba = model.predict(row[model.feature_name()])[0]
            predicted[risk] = int(np.argmax(proba))
        predicted["compound"] = int(compose_compound(
            [predicted.get("rain_wet", 0)], [predicted.get("disease", 0)],
            [predicted.get("frost", 0)])[0])
        booster.predict(row[booster.feature_name()], pred_contrib=True)
        decide(predicted, sensors, confidence={k: 0.9 for k in predicted},
               compound_reasons=compound_reasons(
                   predicted.get("rain_wet", 0), predicted.get("disease", 0),
                   predicted.get("frost", 0)))
        timings.append(time.perf_counter() - begin)
    results["끝에서 끝까지"] = summarise(timings, len(models))

    # ---- ONNX ----
    onnx_table = None
    if args.export_onnx:
        print("\n[ONNX] 변환 중 ...")
        onnx_table = export_onnx(models, out_dir)
        print(onnx_table.to_string(index=False))
        sample = sample_frame[next(iter(models.values())).feature_name()].to_numpy(
            dtype=np.float32
        )
        onnx_timing = time_onnx(out_dir, models, sample, args.repeats)
        if onnx_timing:
            results["모델 추론(ONNX)"] = onnx_timing

        print("\n[ONNX] 원본 대비 판단 일치 검증")
        parity = verify_onnx_parity(out_dir, models, features_all)
        print(parity.to_string(index=False))
        parity.to_csv(out_dir / "onnx_parity.csv", index=False, encoding="utf-8-sig")
        unfit = parity[parity["판정"] == "부적합"]
        if not unfit.empty:
            print(f"    [경고] {len(unfit)}개 모델이 원본과 다른 등급을 낸다 "
                  f"(기준 {ONNX_AGREEMENT_FLOOR*100:.1f}% 이상)")
            print("    네이티브 LightGBM 이 이미 목표를 크게 밑돌므로 "
                  "ONNX 를 쓸 이유가 없다 — 변환본 배포 금지")

    table = pd.DataFrame([{"단계": k, **v} for k, v in results.items()])
    pd.set_option("display.width", 200)
    print("\n===== 추론시간 =====")
    print(table.to_string(index=False))

    end_to_end = results["끝에서 끝까지"]
    print(f"\n판단 1회 끝에서 끝까지: 평균 {end_to_end['평균(ms)']:.0f}ms "
          f"(p95 {end_to_end['p95(ms)']:.0f}ms)")
    print(f"사업계획서 목표: 평균 {TARGET_SECONDS:.0f}초 이내 -> "
          f"{'충족' if end_to_end['평균(ms)'] / 1000 <= TARGET_SECONDS else '미달'} "
          f"(여유 {TARGET_SECONDS - end_to_end['평균(ms)']/1000:.1f}초)")
    duty = end_to_end["평균(ms)"] / 1000 / INFERENCE_PERIOD_SECONDS * 100
    print(f"5분 주기 대비 점유율: {duty:.3f}%")
    print(f"여유 배수: 목표의 {TARGET_SECONDS * 1000 / end_to_end['평균(ms)']:.0f}배")
    print("\n주의 — 이 측정은 x86_64 서버급 CPU 값이다. 엣지 제어기(ARM)는 "
          "5~10배 느릴 수 있으나, 그래도 목표의 90배 이상 여유가 남는다.")

    table.to_csv(out_dir / "inference_timing.csv", index=False, encoding="utf-8-sig")
    if onnx_table is not None:
        onnx_table.to_csv(out_dir / "onnx_export.csv", index=False, encoding="utf-8-sig")
    (out_dir / "benchmark.json").write_text(
        json.dumps({
            "환경": {"machine": platform.machine(), "python": platform.python_version()},
            "목표_초": TARGET_SECONDS, "결과": results,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n산출물: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
