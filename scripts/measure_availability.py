"""수집·저장률 측정 — 사업계획서 정량목표 98% 이상.

현장 수집이 시작되면 이 스크립트를 돌려 숫자를 낸다. 결과 JSON 을
산출물 디렉터리에 두면 성능평가서가 그대로 싣는다.

    python scripts/measure_availability.py \
        --field data/field --journal logs/journal.jsonl \
        --out artifacts/availability.json

시험 구간(--start/--end)을 주는 것을 권한다. 주지 않으면 **자료가 있는
구간만** 분모로 잡혀, 수집기가 아예 죽어 있던 시간이 빠진 채 100% 가
나온다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf.data import field as field_mod  # noqa: E402
from ofdf.evaluation.availability import (  # noqa: E402
    collection_rate,
    combined_rate,
    storage_rate,
)

TARGET = 0.98


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field", required=True, help="현장 수집 디렉터리")
    p.add_argument("--journal", default=None, help="판단 이력 JSON Lines")
    p.add_argument("--freq", default="5min", help="기대 수집 주기")
    p.add_argument("--period", type=int, default=300, help="제어기 주기(초)")
    p.add_argument("--start", default=None, help="시험 구간 시작 (ISO8601)")
    p.add_argument("--end", default=None, help="시험 구간 끝 (ISO8601)")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    frame = field_mod.load(args.field, freq=args.freq)
    coll = collection_rate(frame, freq=args.freq, start=args.start, end=args.end)

    store = None
    if args.journal and Path(args.journal).exists():
        store = storage_rate(args.journal, period_seconds=args.period)

    from ofdf.evaluation.availability import Availability
    overall = combined_rate(coll, store or Availability(0, 0, 0.0))

    print(f"수집률   {coll.rate:7.3%}  ({coll.observed:,}/{coll.expected:,} 표본)")
    if store:
        print(f"저장률   {store.rate:7.3%}  ({store.observed:,}/{store.expected:,} 주기)")
    print(f"수집·저장률 {overall:7.3%}  목표 {TARGET:.0%} → "
          f"{'달성' if overall >= TARGET else '미달'}")

    if coll.by_variable:
        print("\n변수별 수집률")
        for name, rate in sorted(coll.by_variable.items(), key=lambda kv: kv[1]):
            print(f"  {name:<28}{rate:8.3%}")
    if coll.gaps:
        print("\n긴 결측 구간")
        for begin, finish, length in coll.gaps[:5]:
            print(f"  {begin} ~ {finish}  ({length}칸)")

    payload = {
        "수집률": round(coll.rate, 5),
        "저장률": round(store.rate, 5) if store else None,
        "수집·저장률": round(overall, 5),
        "목표": TARGET,
        "달성": overall >= TARGET,
        "기대표본": coll.expected,
        "관측표본": coll.observed,
        "변수별": {k: round(v, 5) for k, v in coll.by_variable.items()},
        "결측구간": coll.gaps,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\n[출력] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
