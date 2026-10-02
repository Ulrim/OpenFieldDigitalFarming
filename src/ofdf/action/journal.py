"""판단·조치 이력 기록.

사업계획서의 'AI 설명 기능' 요구사항을 구현한다.

    AI 출력 : 1시간·3시간 위험수준, 예상 발생시점, 주요 판단근거 3개,
              추천조치, 판단 신뢰도
    AI 이력 : 모델 버전, 판단에 사용한 입력값, 위험확률, 설명요인, 권고조치,
              사용자 승인 여부, 실제 제어·환경결과를 개별 기상사례 단위로 저장

기록은 JSON Lines 로 남긴다. 한 줄이 판단 하나라서 통신이 끊긴 동안
로컬에 쌓아 두었다가 재접속 시 시간순으로 올리기 쉽고, 중간이 깨져도
나머지를 읽을 수 있다(사업계획서 '로컬 72시간 독립운전 후 순서 보장 동기화').
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass
class Evidence:
    """판단근거 하나 — 어떤 변수가 어느 값일 때 얼마나 기여했는가."""

    feature: str
    value: float
    contribution: float
    description: str = ""


@dataclass
class RiskJudgement:
    """위험유형 하나에 대한 판단."""

    risk_type: str
    horizon_hours: int
    level: int
    level_name: str
    confidence: float
    probabilities: list[float] = field(default_factory=list)
    expected_onset: str | None = None     # 예상 발생시점(ISO8601)
    evidence: list[Evidence] = field(default_factory=list)


@dataclass
class DecisionRecord:
    """판단 한 건의 전체 기록.

    이 한 줄만 있으면 '그때 왜 그렇게 판단하고 무엇을 실행했는지'를
    다시 설명할 수 있어야 한다.
    """

    timestamp: str
    station: str
    model_version: str
    schema_version: str = SCHEMA_VERSION

    inputs: dict[str, float] = field(default_factory=dict)
    judgements: list[RiskJudgement] = field(default_factory=list)
    operating_mode: str = "자동"

    recommended: list[dict[str, Any]] = field(default_factory=list)
    executed: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)

    user_approved: bool | None = None      # 권고모드일 때 농가 승인 여부
    vision: dict[str, Any] | None = None   # 비전 진단 결과(있으면)
    outcome: dict[str, Any] | None = None  # 실제 제어·환경 결과(사후 기록)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def record_from_decision(
    timestamp: datetime,
    station: str,
    model_version: str,
    inputs: dict[str, float],
    judgements: list[RiskJudgement],
    decision,
    *,
    vision: dict[str, Any] | None = None,
) -> DecisionRecord:
    """:class:`ofdf.action.recommend.Decision` 을 기록 형식으로 바꾼다."""

    def describe(action) -> dict[str, Any]:
        return {
            "device": action.device,
            "command": action.command,
            "layer": int(action.layer),
            "reason": action.reason,
            "risk_type": action.risk_type,
            "advisory": action.advisory,
        }

    recommended = [describe(a) for a in decision.actions]
    executed = [describe(a) for a in decision.executable()]
    rejected = [
        {**describe(a), "rejected_by": a.rejected_by}
        for a in decision.actions
        if a.rejected_by
    ]

    return DecisionRecord(
        timestamp=timestamp.isoformat(),
        station=station,
        model_version=model_version,
        inputs=inputs,
        judgements=judgements,
        operating_mode=decision.mode,
        recommended=recommended,
        executed=executed,
        rejected=rejected,
        vision=vision,
    )


class Journal:
    """판단 이력을 JSON Lines 로 쌓는다."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: DecisionRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")

    def read(self) -> list[dict]:
        """기록을 읽는다. 깨진 줄은 건너뛴다(통신 장애 중 부분 기록 대비)."""
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def monthly_report(self, year: int, month: int) -> dict:
        """월간 리포트(화면 M7) 집계.

        위험사례 요약, 유효 제어명령 집계, 권고모드 비율을 센다.
        '유효 제어명령 60회 이상'은 사업계획서의 정량 성과목표다.
        """
        prefix = f"{year:04d}-{month:02d}"
        records = [r for r in self.read() if r.get("timestamp", "").startswith(prefix)]

        risk_counts: dict[str, int] = {}
        for record in records:
            for judgement in record.get("judgements", []):
                if judgement.get("level", 0) >= 1:
                    key = f"{judgement['risk_type']}_{judgement['level']}"
                    risk_counts[key] = risk_counts.get(key, 0) + 1

        executed = sum(len(r.get("executed", [])) for r in records)
        rejected = sum(len(r.get("rejected", [])) for r in records)
        advisory = sum(1 for r in records if r.get("operating_mode", "").startswith("권고"))

        return {
            "기간": prefix,
            "판단 건수": len(records),
            "위험등급 발생": risk_counts,
            "유효 제어명령": executed,
            "상위계층 거부": rejected,
            "권고모드 비율": round(advisory / len(records), 3) if records else 0.0,
        }
