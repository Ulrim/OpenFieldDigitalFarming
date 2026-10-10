"""수집·저장률 산출.

사업계획서의 정량목표 가운데 **데이터 정상 수집·저장률 98% 이상**은
센서가 설치되어야 실제 값이 나온다. 그런데 "설치 후에 재겠다"로 두면
설치하고 나서도 재지 못한다 — 재는 방법이 정해져 있지 않으면 시험을
해도 증빙이 남지 않기 때문이다.

그래서 **산출 방법을 먼저 코드로 고정**한다. 현장 수집이 시작되는 날
같은 함수에 자료를 넣으면 숫자가 나오고, 성능평가서가 그 숫자를 그대로
싣는다.

두 가지를 따로 센다.

수집률
    기대한 시각·측정점·변수의 격자 가운데 실제로 값이 들어온 비율.
    센서가 죽었거나 통신이 끊긴 것을 잡는다.

저장률
    제어기가 돈 주기 가운데 판단 이력이 실제로 남은 비율. 수집은 됐는데
    기록이 유실된 것을 잡는다. 둘을 합쳐야 '수집·저장률'이 된다.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import pandas as pd


@dataclass
class Availability:
    """수집률 산출 결과."""

    expected: int                       #: 기대 표본 수
    observed: int                       #: 실제로 들어온 표본 수
    rate: float                         #: observed / expected
    by_variable: dict[str, float] = dc_field(default_factory=dict)
    gaps: list[tuple[str, str, int]] = dc_field(default_factory=list)
    """연속 결측 구간 ``(시작, 끝, 길이)``. 긴 것부터."""

    def meets(self, target: float = 0.98) -> bool:
        return self.rate >= target


def collection_rate(
    frame: pd.DataFrame,
    *,
    freq: str = "5min",
    variables: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    max_gaps: int = 10,
) -> Availability:
    """시각 x 변수 격자에서 실제로 값이 들어온 비율을 센다.

    ``frame`` 은 시각을 인덱스로 갖는 넓은 형식이어야 한다
    (:func:`ofdf.data.field.load` 의 출력).

    기대 격자를 **관측된 시각의 최소~최대**가 아니라 ``start``/``end`` 로
    받을 수 있게 한 것은, 수집기가 아예 죽어 있던 구간을 분모에서 빠뜨리지
    않기 위해서다. 그 구간이 빠지면 수집률이 100%로 나온다.
    """
    if frame.empty:
        return Availability(expected=0, observed=0, rate=0.0)

    index = pd.DatetimeIndex(frame.index)
    first = pd.Timestamp(start) if start is not None else index.min()
    last = pd.Timestamp(end) if end is not None else index.max()
    grid = pd.date_range(first, last, freq=freq)

    cols = [c for c in (variables or frame.columns) if c in frame.columns]
    if not cols or len(grid) == 0:
        return Availability(expected=0, observed=0, rate=0.0)

    aligned = frame.reindex(grid)[cols]
    expected = len(grid) * len(cols)
    observed = int(aligned.notna().to_numpy().sum())

    by_variable = {c: float(aligned[c].notna().mean()) for c in cols}

    # 한 변수라도 들어온 시각은 '수집된 시각'으로 본다. 전부 빈 시각이
    # 이어지는 구간이 통신 두절이나 수집기 정지에 해당한다.
    alive = aligned.notna().any(axis=1)
    gaps: list[tuple[str, str, int]] = []
    run_start: pd.Timestamp | None = None
    for ts, ok in alive.items():
        if not ok and run_start is None:
            run_start = ts
        elif ok and run_start is not None:
            gaps.append((run_start.isoformat(), ts.isoformat(),
                         int((ts - run_start) / pd.Timedelta(freq))))
            run_start = None
    if run_start is not None:
        gaps.append((run_start.isoformat(), last.isoformat(),
                     int((last - run_start) / pd.Timedelta(freq)) + 1))
    gaps.sort(key=lambda g: -g[2])

    return Availability(
        expected=expected,
        observed=observed,
        rate=observed / expected if expected else 0.0,
        by_variable=by_variable,
        gaps=gaps[:max_gaps],
    )


def storage_rate(
    journal_path: str | Path,
    *,
    period_seconds: int = 300,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> Availability:
    """판단 이력이 주기마다 빠짐없이 남았는지 센다.

    이력은 JSON Lines 라 한 줄이 판단 하나다. 중간이 깨져도 나머지를
    읽을 수 있으므로, 읽히지 않는 줄은 유실로 세고 넘어간다.
    """
    path = Path(journal_path)
    if not path.exists():
        return Availability(expected=0, observed=0, rate=0.0)

    stamps: list[dt.datetime] = []
    broken = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            stamps.append(dt.datetime.fromisoformat(json.loads(line)["time"]))
        except (ValueError, KeyError, json.JSONDecodeError):
            broken += 1

    if not stamps:
        return Availability(expected=0, observed=0, rate=0.0)

    first = start or min(stamps)
    last = end or max(stamps)
    span = (last - first).total_seconds()
    expected = int(span // period_seconds) + 1
    observed = len(stamps)

    return Availability(
        expected=expected,
        observed=observed,
        # 중복 기록이 있어도 1을 넘기지 않는다
        rate=min(1.0, observed / expected) if expected else 0.0,
        by_variable={"읽히지 않은 줄": float(broken)},
    )


def combined_rate(collection: Availability, storage: Availability) -> float:
    """수집률과 저장률을 곱해 '수집·저장률' 하나로 만든다.

    둘은 서로 다른 고장이다. 센서가 멀쩡해도 기록이 유실되면 자료는 없고,
    기록이 멀쩡해도 센서가 죽으면 빈 기록이 남는다. 그래서 곱한다.
    """
    if collection.expected == 0:
        return storage.rate
    if storage.expected == 0:
        return collection.rate
    return collection.rate * storage.rate
