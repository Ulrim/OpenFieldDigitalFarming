"""비전 병징 + 기상 유리환경 융합 판단.

두 신호는 서로 다른 것을 본다.

* 기상 모델 — 앞으로 1~3시간 **병이 생기기 좋은 환경**이 되는가(선행)
* 비전 모델 — 지금 포장에 **실제 병징이 있는가**, 어느 병해이고 얼마나 번졌는가(현재)

환경만 보면 "유리환경이었지만 균이 없어 발병하지 않은" 경우까지 경보가
나가고, 영상만 보면 이미 병징이 보인 뒤라 늦는다. 둘을 합치면 '지금 병이
있는데 앞으로 환경까지 유리해진다'는 가장 급한 상황을 가려낼 수 있다.

노균병·잎마름병은 감염에서 병징까지 잠복기가 있다. 그래서 과거 유리환경
누적을 함께 본다(사업계획서 '병 진단은 범위 제외'이므로, 여기서는 진단
결과를 방제 시급도 판정의 보조 입력으로만 쓴다).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ofdf.labels.risk import CAUTION, NORMAL, WARNING

#: 융합 등급
LEVEL_NAMES = {0: "정상", 1: "주의", 2: "경계", 3: "긴급"}
URGENT = 3

#: 병징 심각도(비전 risk 코드) -> 설명
SEVERITY_NAMES = {0: "병징 없음", 1: "초기", 2: "중기", 3: "말기"}


@dataclass
class VisionFinding:
    """비전 모델이 내놓은 포장 상태."""

    disease_code: int = 0
    severity: int = 0
    confidence: float = 1.0
    inspected_hours_ago: float = 0.0

    @property
    def has_symptom(self) -> bool:
        return self.disease_code > 0 and self.severity > 0


@dataclass
class FusionResult:
    """융합 판단 결과."""

    level: int
    reasons: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    confidence: float = 1.0

    @property
    def level_name(self) -> str:
        return LEVEL_NAMES.get(self.level, str(self.level))


def fuse_disease_risk(
    weather_level: int,
    vision: VisionFinding | None,
    *,
    favourable_hours_7d: float = 0.0,
    stale_after_hours: float = 72.0,
) -> FusionResult:
    """병해 종합 위험을 판정한다.

    Parameters
    ----------
    weather_level
        기상 모델의 병해 유리환경 등급(0 정상 / 1 주의 / 2 경계).
    vision
        최근 촬영 영상의 진단 결과. 없으면 환경만으로 판단한다.
    favourable_hours_7d
        최근 7일 누적 유리환경 시간. 잠복기를 감안한 감염 압력 대리값.
    stale_after_hours
        영상이 이보다 오래됐으면 현재 상태로 믿지 않는다.

    Returns
    -------
    FusionResult
        등급, 판단근거, 추천조치. 판단근거는 화면 M2 와 로그에 그대로 쓴다.
    """
    reasons: list[str] = []
    actions: list[str] = []
    level = max(NORMAL, min(weather_level, WARNING))
    confidence = 1.0

    if weather_level >= WARNING:
        reasons.append("기상: 병해 유리환경 경계 (엽면습윤 지속 + 고습 + 적온)")
    elif weather_level >= CAUTION:
        reasons.append("기상: 병해 유리환경 주의")

    if favourable_hours_7d >= 48:
        reasons.append(f"최근 7일 유리환경 누적 {favourable_hours_7d:.0f}시간 — 감염압 높음")
        level = max(level, CAUTION)

    fresh = vision is not None and vision.inspected_hours_ago <= stale_after_hours
    if vision is not None and not fresh:
        reasons.append(
            f"영상 진단이 {vision.inspected_hours_ago:.0f}시간 전 — 현재 상태로 보지 않음"
        )

    if fresh and vision.has_symptom:
        confidence = min(confidence, vision.confidence)
        reasons.append(
            f"영상: 질병코드 {vision.disease_code} "
            f"{SEVERITY_NAMES.get(vision.severity, vision.severity)} 병징 확인"
            f" (신뢰도 {vision.confidence:.2f})"
        )
        # 병징이 이미 있는데 환경까지 유리해지면 확산이 빠르다
        if weather_level >= CAUTION:
            level = URGENT
            reasons.append("병징 발생 + 유리환경 중복 — 확산 위험")
        else:
            level = max(level, CAUTION if vision.severity == 1 else WARNING)

        if vision.severity >= 2:
            actions.append("방제 즉시 실행 — 병반 확산 단계")
        else:
            actions.append("방제 실행 — 초기 병징 단계")
        actions.append("발병 구역 표시 및 인접 이랑 예찰")
    elif fresh:
        reasons.append("영상: 병징 미확인")
        if weather_level >= WARNING:
            actions.append("예방 방제 준비 — 병징은 없으나 환경이 유리함")

    if level >= CAUTION:
        actions.append("살수 금지 — 잎 젖음 시간 연장 방지")
        actions.append("일몰 전 잎 건조 2시간 확보")
    if level >= WARNING:
        actions.append("농가 알림 발송")

    # 중복 제거(순서 보존)
    actions = list(dict.fromkeys(actions))
    return FusionResult(level=level, reasons=reasons, actions=actions, confidence=confidence)


def fuse_growth_check(
    vision_severity: int,
    frost_level: int,
    *,
    hours_since_frost: float | None = None,
) -> FusionResult:
    """서리 피해 확인 — 서리 경보 다음날 영상으로 실제 피해를 확인한다.

    사업계획서의 '익일 동해 예찰'을 자동화한 것이다. 서리 경계가 났던
    다음 날 아침 영상에서 잎 끝 황화·동해가 보이면 실제 피해로 기록하고,
    보이지 않으면 경보가 과했는지 기록한다. 이 기록이 서리 모델의 농지별
    보정 데이터가 된다.
    """
    reasons: list[str] = []
    actions: list[str] = []
    level = NORMAL

    if frost_level >= WARNING:
        reasons.append("전일 저온·서리 경계 발생")
    elif frost_level >= CAUTION:
        reasons.append("전일 저온·서리 주의 발생")

    if hours_since_frost is not None and hours_since_frost > 36:
        reasons.append(f"서리 발생 후 {hours_since_frost:.0f}시간 경과 — 예찰 시기 지남")

    if vision_severity > 0:
        level = WARNING if vision_severity >= 2 else CAUTION
        reasons.append(f"영상: 잎 피해 {SEVERITY_NAMES.get(vision_severity)} 확인")
        actions.append("피해주율 조사 및 생육조사표 기록")
        actions.append("서리 판단 기준 농지별 보정자료로 축적")
    elif frost_level >= CAUTION:
        reasons.append("영상: 피해 미확인 — 경보 대비 실제 피해 없음")
        actions.append("오경보 사례로 기록 — 초관부 온도 임계값 재검토")

    return FusionResult(level=level, reasons=reasons, actions=actions)
