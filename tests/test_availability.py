"""수집·저장률 산출 검증.

사업계획서 정량목표 '데이터 정상 수집·저장률 98% 이상'을 재는 방법이
센서 설치 전에 고정돼 있어야 한다. 설치하고 나서 정하면 시험을 해도
증빙이 남지 않는다.
"""

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import pytest

from ofdf.evaluation.availability import (
    collection_rate,
    combined_rate,
    storage_rate,
)


def frame(n: int, freq: str = "5min") -> pd.DataFrame:
    index = pd.date_range("2026-10-01", periods=n, freq=freq)
    return pd.DataFrame({"t_air": 12.0, "rh": 70.0}, index=index)


def test_full_collection_is_one():
    got = collection_rate(frame(12))
    assert got.rate == pytest.approx(1.0)
    assert got.expected == 12 * 2 and got.observed == 12 * 2
    assert got.meets(0.98)


def test_missing_values_lower_the_rate_per_variable():
    f = frame(10)
    f.iloc[0:2, f.columns.get_loc("rh")] = None
    got = collection_rate(f)
    assert got.rate == pytest.approx(18 / 20)
    assert got.by_variable["t_air"] == pytest.approx(1.0)
    assert got.by_variable["rh"] == pytest.approx(0.8)
    assert not got.meets(0.98)


def test_dead_collector_window_is_not_silently_dropped():
    """수집기가 죽어 있던 구간이 분모에서 빠지면 수집률이 100%로 나온다.

    관측된 시각의 최소~최대만 보면 '없는 시각'은 애초에 세지 않는다.
    시험 구간을 밖에서 받아야 그 구멍이 분모에 들어온다.
    """
    f = frame(6)                       # 00:00~00:25 만 수집됨
    end = f.index[-1] + pd.Timedelta("30min")   # 실제 시험은 30분 더 이어졌다

    naive = collection_rate(f)
    honest = collection_rate(f, start=f.index[0], end=end)

    assert naive.rate == pytest.approx(1.0)
    assert honest.rate < 0.6
    assert honest.gaps and honest.gaps[0][2] >= 6


def test_gap_list_reports_the_longest_outage_first():
    f = frame(20)
    f.iloc[3:5] = None            # 10분 두절
    f.iloc[10:16] = None          # 30분 두절
    got = collection_rate(f)
    assert got.gaps[0][2] == 6
    assert got.gaps[1][2] == 2


def test_storage_rate_counts_missing_cycles(tmp_path):
    path = tmp_path / "journal.jsonl"
    base = dt.datetime(2026, 10, 1, 0, 0)
    # 5분 주기로 12회 돌아야 하는데 9회만 남았다
    kept = [0, 1, 2, 3, 4, 5, 6, 7, 11]
    path.write_text("\n".join(
        json.dumps({"time": (base + dt.timedelta(minutes=5 * i)).isoformat()})
        for i in kept
    ), encoding="utf-8")

    got = storage_rate(path, period_seconds=300)
    assert got.expected == 12
    assert got.observed == 9
    assert got.rate == pytest.approx(0.75)


def test_broken_lines_are_counted_not_crashed(tmp_path):
    """한 줄이 깨져도 나머지를 읽는다 — JSON Lines 를 쓴 이유다."""
    path = tmp_path / "journal.jsonl"
    base = dt.datetime(2026, 10, 1, 0, 0)
    lines = [json.dumps({"time": base.isoformat()}),
             "{깨진 줄",
             json.dumps({"time": (base + dt.timedelta(minutes=5)).isoformat()})]
    path.write_text("\n".join(lines), encoding="utf-8")

    got = storage_rate(path, period_seconds=300)
    assert got.observed == 2
    assert got.by_variable["읽히지 않은 줄"] == 1.0


def test_missing_journal_is_zero_not_an_error(tmp_path):
    assert storage_rate(tmp_path / "없음.jsonl").rate == 0.0


def test_combined_multiplies_the_two_failures():
    """센서가 멀쩡해도 기록이 유실되면 자료는 없다. 둘은 곱해진다."""
    from ofdf.evaluation.availability import Availability

    f = frame(10)
    f.iloc[0:1] = None
    coll = collection_rate(f)                       # 수집 0.9
    store = Availability(expected=10, observed=9, rate=0.9)

    assert combined_rate(coll, store) == pytest.approx(coll.rate * 0.9)
    # 한쪽만 재어진 경우에는 재어진 쪽을 그대로 쓴다
    assert combined_rate(coll, Availability(0, 0, 0.0)) == pytest.approx(coll.rate)
    assert combined_rate(Availability(0, 0, 0.0), store) == pytest.approx(0.9)


def test_empty_input_does_not_divide_by_zero():
    got = collection_rate(pd.DataFrame())
    assert got.rate == 0.0 and got.expected == 0
