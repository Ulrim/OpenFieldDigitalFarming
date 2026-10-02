"""핵심 실증 시나리오 8종 시험 실행.

사업계획서 '핵심 실증 시나리오' 표의 판정기준으로 자동 시험하고,
외부 공인시험 성적서에 쓸 결과표와 판단·조치 이력을 남긴다.

자연 기상사례가 나지 않아도 검증할 수 있도록 센서 입력 모사로 돌린다.
자연 발생 사례로 다시 돌릴 때는 ``--source 자연사례`` 로 구분 표기한다.

사용 예::

    python scripts/run_scenarios.py --out artifacts/scenarios
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf.action.journal import Journal, RiskJudgement, record_from_decision  # noqa: E402
from ofdf.evaluation import scenario  # noqa: E402
from ofdf.labels.risk import RISK_LABELS_KO  # noqa: E402

LEVEL_NAMES = {0: "정상", 1: "주의", 2: "경계"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="artifacts/scenarios")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", default=scenario.SOURCE_SIMULATED,
                   choices=[scenario.SOURCE_SIMULATED, scenario.SOURCE_NATURAL])
    p.add_argument("--station", default="나주시 남평읍 실증포장")
    p.add_argument("--model-version", default="ofdf-weather-0.1.0")
    p.add_argument("--verbose", action="store_true", help="실패한 실행의 상세를 보인다")
    return p.parse_args()


def write_journal(results: list[scenario.ScenarioResult], path: Path, args) -> int:
    """시나리오 실행 전체를 판단 이력으로 남긴다."""
    journal = Journal(path)
    base = datetime.datetime(2026, 10, 12, 0, 0)
    written = 0

    for result in results:
        for run in result.runs:
            for step in run.steps:
                if step.decision is None:
                    continue
                judgements = [
                    RiskJudgement(
                        risk_type=risk, horizon_hours=3, level=level,
                        level_name=LEVEL_NAMES.get(level, str(level)),
                        confidence=step.confidence.get(risk, 1.0),
                    )
                    for risk, level in step.risk_levels.items() if level > 0
                ]
                record = record_from_decision(
                    base + datetime.timedelta(hours=written % 24, days=written // 24),
                    args.station, args.model_version,
                    {
                        "gust_3s": step.sensors.gust_3s or 0.0,
                        "rain_detected": float(step.sensors.rain_detected),
                        "soil_above_limit": float(step.sensors.soil_above_limit),
                    },
                    judgements, step.decision,
                )
                journal.append(record)
                written += 1
    return written


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[시험] 핵심 실증 시나리오 8종 — 구분 '{args.source}'")
    results = scenario.run_all(seed=args.seed)
    for result in results:
        result.source = args.source

    table = scenario.results_table(results)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 46)
    print()
    print(table.to_string(index=False))

    total = sum(r.repeats for r in results)
    passed = sum(r.passed for r in results)
    met = sum(r.meets_criterion for r in results)
    print(f"\n총 {total}회 중 {passed}회 통과 ({passed/total*100:.1f}%) / "
          f"시나리오 {met}/{len(results)} 적합")

    failures = [r for r in results if not r.meets_criterion]
    if failures:
        print("\n===== 부적합 시나리오 =====")
        for result in failures:
            print(f"\n■ {result.name}  ({result.passed}/{result.repeats} 통과, "
                  f"기준 {result.required})")
            print(f"  판정기준: {result.criterion}")
            for run in result.runs:
                if not run.passed:
                    print(f"    [{run.index}] {run.detail}")
                    if args.verbose:
                        for step in run.steps:
                            actions = ", ".join(
                                f"{a.device} {a.command}" for a in step.executed()
                            ) or "조치 없음"
                            print(f"        {step.hour:02d}시 {step.risk_levels} -> {actions}")
                    break

    written = write_journal(results, out_dir / "scenario_journal.jsonl", args)
    table.to_csv(out_dir / "scenario_results.csv", index=False, encoding="utf-8-sig")

    lead = [m for r in results for m in r.lead_times()]
    if lead:
        print(f"\n사전알림 선행시간: 평균 {sum(lead)/len(lead):.0f}분 / 최소 {min(lead):.0f}분")

    print(f"\n판단 이력 {written:,}건 기록")
    print(f"산출물: {out_dir}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
