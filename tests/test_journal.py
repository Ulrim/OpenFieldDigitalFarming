"""판단 이력 기록 검증 — AI 설명 기능 요구사항."""

from __future__ import annotations

import datetime
import json

import pytest

from ofdf.action.journal import Evidence, Journal, RiskJudgement, record_from_decision
from ofdf.action.recommend import SensorState, decide


@pytest.fixture
def record():
    decision = decide({"frost": 2}, SensorState(hour=23), confidence={"frost": 0.91})
    judgements = [
        RiskJudgement(
            "frost", 3, 2, "경계", 0.91, [0.02, 0.07, 0.91],
            expected_onset="2025-11-02T02:00:00",
            evidence=[
                Evidence("canopy_temp", -0.4, 1.82, "초관부 온도 0℃ 이하"),
                Evidence("run_canopy2", 3.0, 0.95, "초관부 2℃ 이하 3시간 연속"),
                Evidence("cloud_cover", 0.08, 0.61, "맑음"),
            ],
        )
    ]
    return record_from_decision(
        datetime.datetime(2025, 11, 1, 23, 0), "나주시 봉황면", "ofdf-weather-0.1.0",
        {"t_air": 2.1, "canopy_temp": -0.4}, judgements, decision,
    )


def test_record_carries_required_fields(record):
    """사업계획서가 요구하는 출력 항목이 모두 들어 있다."""
    data = json.loads(record.to_json())
    judgement = data["judgements"][0]
    assert judgement["level"] == 2 and judgement["confidence"] == 0.91
    assert judgement["expected_onset"]                       # 예상 발생시점
    assert len(judgement["evidence"]) >= 3                   # 주요 판단근거 3개 이상
    assert data["recommended"] and data["executed"]          # 추천조치·실행결과
    assert data["model_version"] and data["schema_version"]  # 재현을 위한 버전


def test_journal_roundtrip(tmp_path, record):
    journal = Journal(tmp_path / "journal.jsonl")
    journal.append(record)
    journal.append(record)
    assert len(journal.read()) == 2


def test_journal_skips_corrupt_lines(tmp_path, record):
    """통신 장애로 줄이 깨져도 나머지는 읽을 수 있어야 한다."""
    path = tmp_path / "journal.jsonl"
    journal = Journal(path)
    journal.append(record)
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"timestamp": "2025-11-01T23:0\n')     # 중간에 끊긴 줄
    journal.append(record)
    assert len(journal.read()) == 2


def test_monthly_report_counts_control_commands(tmp_path, record):
    journal = Journal(tmp_path / "journal.jsonl")
    for _ in range(3):
        journal.append(record)
    report = journal.monthly_report(2025, 11)
    assert report["판단 건수"] == 3
    assert report["유효 제어명령"] == 9                       # 3회 x 3개 명령
    assert report["위험등급 발생"]["frost_2"] == 3


def test_monthly_report_tracks_advisory_ratio(tmp_path):
    """저신뢰 판단은 권고모드로 집계된다."""
    low = decide({"frost": 2}, SensorState(hour=23), confidence={"frost": 0.5})
    high = decide({"frost": 2}, SensorState(hour=23), confidence={"frost": 0.95})
    journal = Journal(tmp_path / "journal.jsonl")
    for decision in (low, high):
        journal.append(
            record_from_decision(
                datetime.datetime(2025, 11, 1, 23, 0), "A", "v1", {}, [], decision
            )
        )
    assert journal.monthly_report(2025, 11)["권고모드 비율"] == 0.5


def test_rejected_actions_keep_their_reason(tmp_path):
    """상위 계층에 막힌 조치도 사유와 함께 남는다."""
    decision = decide({"frost": 2}, SensorState(hour=23, gust_3s=12.0))
    record = record_from_decision(
        datetime.datetime(2025, 11, 1, 23, 0), "A", "v1", {}, [], decision
    )
    assert record.rejected
    assert all(item["rejected_by"] for item in record.rejected)
