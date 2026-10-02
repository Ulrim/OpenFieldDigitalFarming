"""조치추천 및 제어 우선순위 엔진.

사업계획서의 '제어 우선순위 4계층'과 '충돌 해소 및 반복작동 방지 규칙'을
구현한다. AI는 앞으로의 위험과 필요한 조치를 **추천**할 뿐이고, 실제로
무엇이 실행되는지는 계층 우선순위가 정한다.

    L0 하드웨어 안전 : 비상정지·과전류 차단. 전원·접점 단독 작동(소프트웨어 무관)
    L1 현장 즉응 규칙 : 강풍 회수, 강우 시 관수·살수 중단. AI가 멈춰도 작동
    L2 AI 선행판단   : 서리 야간 피복, 강우 전 관수 중단, 고온 차광 한정
    L3 사용자 제어   : 농가 승인·거부, 수동 조작, 임계값 설정

상위 계층이 금지한 동작은 하위 계층이 요청해도 거부하고 사유를 남긴다.
판단 신뢰도가 기준 미만이면 자동 실행하지 않고 농가 승인을 받는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from ofdf.labels.risk import CAUTION, NORMAL, WARNING

#: 자동 실행을 허용하는 최소 판단 신뢰도. 미만이면 권고모드로 떨어진다.
CONFIDENCE_THRESHOLD = 0.70

#: 차광 제한 운영 — 대파는 양광성 작물이라 과도한 차광이 연약 생육을 부른다.
#: 사업계획서: '10시 이전·16시 이후 차광 금지', '고온 경계 30분 지속 시에만
#: 최대 2시간 차광', 해제조건 '2시간·16시 도달'.
#: 사유 문구로만 적어 두면 지켜지지 않는다. 엔진이 직접 막는다.
SHADE_EARLIEST_HOUR = 10
SHADE_LATEST_HOUR = 16
SHADE_MAX_MINUTES = 120


class Layer(IntEnum):
    """제어 계층. 숫자가 작을수록 우선한다."""

    HARDWARE = 0
    FIELD_RULE = 1
    AI = 2
    USER = 3


@dataclass
class Action:
    """장치 하나에 대한 조치."""

    device: str
    command: str
    layer: Layer
    reason: str
    risk_type: str = ""
    advisory: bool = False      # True 면 농가 승인 후 실행(권고모드)
    rejected_by: str = ""       # 상위 계층에 막힌 경우 그 사유

    def describe(self) -> str:
        mark = "권고" if self.advisory else "실행"
        if self.rejected_by:
            return f"[거부] {self.device} {self.command} — {self.rejected_by}"
        return f"[{mark}] {self.device} {self.command} — {self.reason}"


@dataclass
class SensorState:
    """현장 즉응규칙(L1)이 보는 실측값."""

    gust_3s: float | None = None        # 3초 순간최대풍속 m/s
    rain_detected: bool = False
    soil_above_limit: bool = False      # 토양수분 농가 상한 초과
    spraying: bool = False              # 살수 가동 중
    hour: int = 12
    emergency_stop: bool = False
    overcurrent: bool = False
    comms_down: bool = False
    #: 오늘 차광막을 전개한 **누적** 시간(분). 자정에 0으로 돌아간다.
    #: 2시간 제한은 하루 총량이다. 걷었다가 다시 펴는 것으로 늘릴 수 없다.
    shade_deployed_minutes: float = 0.0
    #: 차광막이 지금 펴져 있는지. 누적시간과 다른 값이다 —
    #: 2시간을 다 쓰고 걷은 상태는 누적 120분이면서 열려 있지 않다.
    shade_open: bool = False


@dataclass
class Decision:
    """판단 결과 묶음."""

    actions: list[Action] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    mode: str = "자동"

    def executable(self) -> list[Action]:
        return [a for a in self.actions if not a.rejected_by and not a.advisory]


# --------------------------------------------------------------------------
# 위험유형별 추천 조치 (사업계획서 '위험 정답값 정의' 표의 추천조치 열)
# --------------------------------------------------------------------------

def _heat_dry_actions(
    level: int, hour: int, shade_minutes: float, shade_open: bool
) -> list[Action]:
    if level < CAUTION:
        return []
    actions = [
        Action("관수밸브", "관수 준비·실행", Layer.AI, "고온·건조로 수분 스트레스 예상", "heat_dry"),
        Action("알림", "고온·건조 주의 알림", Layer.AI, "농가 확인 요청", "heat_dry"),
    ]

    within_window = SHADE_EARLIEST_HOUR <= hour < SHADE_LATEST_HOUR
    within_budget = shade_minutes < SHADE_MAX_MINUTES
    allowed = within_window and within_budget

    if shade_open and not allowed:
        reason = (
            f"{SHADE_LATEST_HOUR}시 도달 — 차광 종료" if not within_window
            else f"당일 누적 {shade_minutes:.0f}분 — 최대 {SHADE_MAX_MINUTES}분 소진"
        )
        actions.append(Action("차광막", "차광 회수", Layer.AI, reason, "heat_dry"))
    elif level >= WARNING and allowed and not shade_open:
        remaining = SHADE_MAX_MINUTES - shade_minutes
        actions.append(
            Action("차광막", "차광 한정 전개", Layer.AI,
                   f"고온 경계 — {SHADE_EARLIEST_HOUR}~{SHADE_LATEST_HOUR}시 한정, "
                   f"당일 잔여 {remaining:.0f}분", "heat_dry")
        )
    return actions


def _rain_wet_actions(level: int) -> list[Action]:
    if level < CAUTION:
        return []
    return [
        Action("관수밸브", "관수 중단", Layer.AI, "강우·과습 예상 — 과습 가중 방지", "rain_wet"),
        Action("살수", "살수 금지", Layer.AI, "강우 중 살수 중복 방지", "rain_wet"),
        Action("알림", "배수로·고랑 확인, 무름·뿌리피해 예찰", Layer.AI,
               "과습 지속 시 뿌리활력 저하", "rain_wet"),
    ]


def _disease_actions(level: int) -> list[Action]:
    if level < CAUTION:
        return []
    actions = [
        Action("살수", "살수 금지", Layer.AI, "잎 젖음 시간 연장 방지", "disease"),
        Action("알림", "노균병·잎마름병 예찰·방제 준비", Layer.AI,
               "엽면습윤 지속 + 고습 — 병해 유리환경", "disease"),
    ]
    if level >= WARNING:
        actions.append(
            Action("알림", "방제 일정 조정 권고", Layer.AI, "경계 등급 도달", "disease")
        )
    return actions


def _frost_actions(level: int, hour: int) -> list[Action]:
    if level < CAUTION:
        return []
    actions = [
        Action("야간피복", "야간 피복 전개", Layer.AI, "초관부 복사냉각으로 서리 예상", "frost"),
        Action("관수밸브", "야간·새벽 관수 금지", Layer.AI, "관수가 동결피해를 키움", "frost"),
        Action("알림", "서리 경보 — 익일 동해 예찰", Layer.AI, "잎 끝 황화·동해 확인", "frost"),
    ]
    if hour is not None and not (hour >= 18 or hour < 8):
        # 주간에는 피복을 전개하지 않는다(광합성 저해)
        actions[0] = Action("야간피복", "일몰 후 피복 준비", Layer.AI,
                            "주간 전개는 광 부족을 유발", "frost")
    return actions


def _compound_actions(level: int, reasons: list[str]) -> list[Action]:
    if level < CAUTION:
        return []
    detail = " / ".join(reasons) if reasons else "구성 위험 동시 발생"
    return [
        Action("관수밸브", "관수 중단 유지", Layer.AI, detail, "compound"),
        Action("살수", "살수 금지", Layer.AI, detail, "compound"),
        Action("알림", "방제 준비·배수 확인·야간 피복 권고 동시 발송", Layer.AI, detail, "compound"),
    ]


RISK_ACTION_BUILDERS = {
    "heat_dry": lambda level, ctx: _heat_dry_actions(
        level, ctx.get("hour", 12), ctx.get("shade_deployed_minutes", 0.0),
        ctx.get("shade_open", False),
    ),
    "rain_wet": lambda level, ctx: _rain_wet_actions(level),
    "disease": lambda level, ctx: _disease_actions(level),
    "frost": lambda level, ctx: _frost_actions(level, ctx.get("hour", 12)),
    "compound": lambda level, ctx: _compound_actions(level, ctx.get("compound_reasons", [])),
}


# --------------------------------------------------------------------------
# L0 / L1 안전규칙
# --------------------------------------------------------------------------

def hardware_actions(sensors: SensorState) -> list[Action]:
    """L0 — 소프트웨어와 무관하게 작동하는 하드웨어 안전."""
    actions = []
    if sensors.emergency_stop:
        actions.append(Action("전 구동부", "전원 차단", Layer.HARDWARE,
                              "비상정지 접점 개방 — 물리 복구 + 리셋 필요"))
    if sensors.overcurrent:
        actions.append(Action("해당 구동부", "정지·경보", Layer.HARDWARE,
                              "정격 전류 1.3배 3초 초과 — 수동 확인 후 리셋"))
    return actions


def field_rule_actions(sensors: SensorState) -> list[Action]:
    """L1 — AI가 멈춰도 작동하는 현장 즉응규칙."""
    actions = []
    gust = sensors.gust_3s

    if gust is not None:
        if gust >= 14.0:
            actions.append(Action("전 구동부", "경보·구동 잠금", Layer.FIELD_RULE,
                                  f"3초 순간최대풍속 {gust:.1f}m/s — 현장 확인 후 수동 해제"))
        if gust >= 9.0:
            actions.append(Action("차광막·피복", "즉시 완전 회수(5분 이내)", Layer.FIELD_RULE,
                                  f"3초 순간최대풍속 {gust:.1f}m/s"))
        elif gust >= 7.0:
            actions.append(Action("차광막·피복", "전개 금지", Layer.FIELD_RULE,
                                  f"3초 순간최대풍속 {gust:.1f}m/s"))

    if sensors.rain_detected:
        actions.append(Action("관수밸브", "관수 즉시 중단", Layer.FIELD_RULE, "강우 감지"))
        actions.append(Action("살수", "살수 즉시 중단", Layer.FIELD_RULE, "강우 감지"))
        actions.append(Action("차광막", "차광 회수", Layer.FIELD_RULE,
                              "젖은 차광망은 풍하중이 커짐"))

    if sensors.soil_above_limit:
        actions.append(Action("관수밸브", "관수 금지", Layer.FIELD_RULE, "토양수분 농가 상한 초과"))
        actions.append(Action("알림", "배수 상태 확인", Layer.FIELD_RULE, "토양수분 상한 초과"))

    if sensors.spraying and (sensors.gust_3s or 0) > 4.0:
        actions.append(Action("살수", "살수 즉시 중지", Layer.FIELD_RULE,
                              "풍속 4m/s 초과 — 도로·인근 주택 비산 방지"))

    if sensors.comms_down:
        actions.append(Action("제어기", "로컬 독립운전", Layer.FIELD_RULE,
                              "서버 응답 3분 미수신 — 데이터 큐 적재"))
    return actions


#: 장치 묶음 — 같은 구동부를 공유하거나 한 명령이 함께 거는 장치들.
#: 차광막과 야간피복은 같은 권취 구동부를 쓰므로(차광 스크린 겸용 피복)
#: 강풍 회수 명령 하나가 둘 다 건다.
DEVICE_GROUPS = {
    "차광막": {"차광막", "차광막·피복", "야간피복", "전 구동부"},
    "야간피복": {"야간피복", "차광막·피복", "차광막", "전 구동부"},
    "차광막·피복": {"차광막", "야간피복", "차광막·피복", "전 구동부"},
    "관수밸브": {"관수밸브", "전 구동부"},
    "살수": {"살수", "전 구동부"},
}

#: 상위 계층 금지 명령이 하위 계층 명령을 막는 규칙.
#:
#: ``(차단하는 명령 키워드, 막히는 장치, 막히는 명령 키워드)``
#:
#: 중요 — 관수(점적)와 살수(증발냉각 미세살수)는 다른 장치다. 병해
#: 유리환경에서 금지되는 것은 잎을 적시는 **살수**이고, 잎을 적시지 않는
#: 점적관수는 막지 않는다. 그래서 막히는 장치를 규칙에 명시한다.
BLOCKING_RULES = [
    ("차광 회수", "차광막", ["전개"]),
    ("회수", "차광막", ["전개"]),
    ("회수", "야간피복", ["전개"]),
    ("전개 금지", "차광막", ["전개"]),
    ("전개 금지", "야간피복", ["전개"]),
    ("구동 잠금", "차광막", ["전개"]),
    ("구동 잠금", "야간피복", ["전개"]),
    ("관수 즉시 중단", "관수밸브", ["관수 준비", "관수 실행"]),
    ("관수 중단", "관수밸브", ["관수 준비", "관수 실행"]),
    ("관수 금지", "관수밸브", ["관수 준비", "관수 실행"]),
    ("살수 즉시 중단", "살수", ["살수 시작"]),
    ("살수 즉시 중지", "살수", ["살수 시작"]),
    ("살수 중단", "살수", ["살수 시작"]),
    ("살수 금지", "살수", ["살수 시작"]),
    ("전원 차단", "*", ["전개", "관수 준비", "관수 실행", "살수 시작", "피복"]),
]


def _same_device(a: str, b: str) -> bool:
    """두 장치 이름이 같은 구동부를 가리키는지 본다."""
    return a == b or b in DEVICE_GROUPS.get(a, {a}) or a in DEVICE_GROUPS.get(b, {b})


def _is_blocked(action: Action, higher: list[Action]) -> str:
    """상위 계층 명령에 막히는지 확인하고 사유를 돌려준다.

    같은 장치(또는 같은 구동부를 공유하는 장치)에 대한 금지 명령만 막는다.
    알림은 어떤 경우에도 막지 않는다(위험을 알리는 것 자체는 안전하다).
    """
    if action.device == "알림":
        return ""

    for blocker in higher:
        if blocker.rejected_by or blocker.layer >= action.layer:
            continue
        for forbid, blocked_device, blocked_commands in BLOCKING_RULES:
            if forbid not in blocker.command:
                continue
            if blocked_device != "*" and not _same_device(action.device, blocked_device):
                continue
            if blocked_device == "*" and not _same_device(action.device, blocker.device):
                continue
            if any(k in action.command for k in blocked_commands):
                return (
                    f"L{int(blocker.layer)} '{blocker.device} {blocker.command}' 우선 "
                    f"({blocker.reason})"
                )
    return ""


def _deduplicate(actions: list[Action]) -> list[Action]:
    """같은 장치에 같은 명령이 여러 위험유형에서 나오면 하나로 합친다.

    복합위험과 구성 위험이 같은 조치를 요구하는 경우가 많다. 제어기에
    같은 명령을 두 번 보내지 않도록 합치고, 사유는 함께 남긴다.
    """
    merged: dict[tuple[str, str], Action] = {}
    for action in actions:
        key = (action.device, action.command)
        if key in merged:
            existing = merged[key]
            if action.risk_type and action.risk_type not in existing.risk_type:
                existing.risk_type = f"{existing.risk_type}+{action.risk_type}".strip("+")
            # 하나라도 자동 실행이면 자동으로 둔다(더 낮은 계층이 이미 요구)
            existing.advisory = existing.advisory and action.advisory
            continue
        merged[key] = action
    return list(merged.values())


def decide(
    risk_levels: dict[str, int],
    sensors: SensorState,
    *,
    confidence: dict[str, float] | None = None,
    compound_reasons: list[str] | None = None,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Decision:
    """위험판단 결과와 현장 실측값으로 최종 조치를 정한다.

    Parameters
    ----------
    risk_levels : 위험유형 -> 등급(0 정상 / 1 주의 / 2 경계)
    sensors : 현장 즉응규칙이 보는 실측값
    confidence : 위험유형 -> AI 판단 신뢰도(0~1)
    compound_reasons : 복합위험 판단근거 문구

    Returns
    -------
    Decision
        계층 우선순위와 충돌 규칙을 적용한 조치 목록. 상위 계층에 막힌
        조치도 사유와 함께 남겨 운영기록에 쓴다.
    """
    confidence = confidence or {}
    context = {
        "hour": sensors.hour,
        "shade_deployed_minutes": sensors.shade_deployed_minutes,
        "shade_open": sensors.shade_open,
        "compound_reasons": compound_reasons or [],
    }

    hardware = hardware_actions(sensors)
    field_rules = field_rule_actions(sensors)

    ai_actions: list[Action] = []
    alerts: list[str] = []
    low_confidence = False

    for risk, level in risk_levels.items():
        if level <= NORMAL or risk not in RISK_ACTION_BUILDERS:
            continue
        score = confidence.get(risk, 1.0)
        advisory = score < confidence_threshold
        low_confidence |= advisory

        for action in RISK_ACTION_BUILDERS[risk](level, context):
            action.advisory = advisory
            if advisory:
                action.reason += f" (신뢰도 {score:.2f} — 농가 승인 후 실행)"
            ai_actions.append(action)

        if level >= WARNING:
            alerts.append(f"{risk} 경계")
        else:
            alerts.append(f"{risk} 주의")

    # 상위 계층부터 차례로 쌓으면서 충돌을 거른다
    resolved: list[Action] = list(hardware)
    for action in _deduplicate(field_rules):
        action.rejected_by = _is_blocked(action, resolved)
        resolved.append(action)
    for action in _deduplicate(ai_actions):
        action.rejected_by = _is_blocked(action, resolved)
        resolved.append(action)

    mode = "권고(농가 승인 필요)" if low_confidence else "자동"
    if sensors.emergency_stop or sensors.overcurrent:
        mode = "안전정지"
    elif sensors.comms_down:
        mode = "로컬 독립운전"

    return Decision(actions=resolved, alerts=alerts, mode=mode)
