"""핵심 실증 시나리오 시험.

사업계획서 '핵심 실증 시나리오' 표의 8종을 자동 시험으로 옮긴 것이다.
각 시나리오는 절차와 **판정기준**이 수치로 정해져 있다.

    강우·과습 관수중단      10회 모두 관수차단, 판단근거·명령·완료로그 100%
    저온·서리 사전경보      서리 발생 야간 전건 사전경보·선행시간 기록, 야간 관수 차단 100%
    다습·병해 유리환경 알림  경계 등급 알림 100%, 살수 금지 준수 100%
    복합: 강우 후 저온다습   10회 모두 우선순위 판단·동시 조치 정상
    복합: 과습 + 서리       10회 모두 충돌 없이 정상 조치
    고온·건조 제한운영      20회 중 19회 이상 정상작동, 차광 시간대·지속시간 조건 준수 100%
    강풍 안전보호           10회 모두 안전회수(5분 이내), 과전류 정지·경보 정상
    통신장애                5종 시나리오 모두 제어·저장·복구동기화

자연 기상사례가 나지 않아도 검증할 수 있도록 **센서 입력 모사**로 돌린다
(사업계획서 '기상사례 부족 시: 실험실이 아닌 나주 현장 설치상태에서 안전범위
내 센서 입력 모사와 제어반 반복시험 병행, 자연 이벤트 결과와 구분 표기').

모사 입력은 실제 피처 생성기와 학습된 모델을 거쳐 실제 조치 엔진으로 들어간다.
규칙만 따로 떼어 시험하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ofdf.action.recommend import Action, Decision, SensorState, decide
from ofdf.labels.risk import CAUTION, NORMAL, WARNING

#: 자연 발생 사례인지 입력 모사인지 구분해 표기한다(보고 원칙)
SOURCE_SIMULATED = "입력모사"
SOURCE_NATURAL = "자연사례"


@dataclass
class Step:
    """시나리오 한 시점의 입력과 그때의 판단·조치."""

    hour: int
    risk_levels: dict[str, int]
    sensors: SensorState
    confidence: dict[str, float] = field(default_factory=dict)
    decision: Decision | None = None

    def executed(self) -> list[Action]:
        return self.decision.executable() if self.decision else []

    def has(self, device: str, keyword: str) -> bool:
        """해당 장치에 그 명령이 실제로 실행됐는지."""
        return any(
            device in a.device and keyword in a.command
            for a in self.executed()
        )

    def deployed_shade(self) -> bool:
        """차광막이 실제로 '전개'됐는지.

        ``has("차광막", "전개")`` 로는 안 된다. 강풍 시 L1 이 내리는
        '차광막·피복 전개 **금지**' 명령도 장치명과 '전개'를 모두 포함해
        전개로 잘못 세어진다. 금지·회수 명령을 빼고 본다.
        """
        return any(
            "차광" in a.device
            and "전개" in a.command
            and "금지" not in a.command
            and "회수" not in a.command
            for a in self.executed()
        )

    def rejected(self, device: str, keyword: str) -> bool:
        """해당 명령이 상위 계층에 막혔는지."""
        if not self.decision:
            return False
        return any(
            device in a.device and keyword in a.command and a.rejected_by
            for a in self.decision.actions
        )

    def alerted(self, keyword: str) -> bool:
        return any(
            a.device == "알림" and keyword in a.command for a in self.executed()
        )


@dataclass
class Run:
    """시나리오 1회 실행 결과."""

    index: int
    steps: list[Step]
    passed: bool = False
    detail: str = ""
    lead_minutes: float | None = None

    def first(self, predicate: Callable[[Step], bool]) -> Step | None:
        return next((s for s in self.steps if predicate(s)), None)


@dataclass
class ScenarioResult:
    """시나리오 전체 결과."""

    name: str
    criterion: str
    repeats: int
    passed: int
    source: str = SOURCE_SIMULATED
    runs: list[Run] = field(default_factory=list)
    notes: str = ""

    @property
    def pass_rate(self) -> float:
        return self.passed / self.repeats if self.repeats else 0.0

    @property
    def verdict(self) -> str:
        return "적합" if self.meets_criterion else "부적합"

    #: 판정기준 충족 여부. 기본은 전건 통과이고 시나리오별로 바꾼다.
    required: int = 0

    @property
    def meets_criterion(self) -> bool:
        return self.passed >= (self.required or self.repeats)

    def lead_times(self) -> list[float]:
        return [r.lead_minutes for r in self.runs if r.lead_minutes is not None]


def run_steps(steps: list[Step], *, compound_reasons: list[str] | None = None) -> list[Step]:
    """각 시점에 실제 조치 엔진을 돌려 결과를 채운다."""
    for step in steps:
        step.decision = decide(
            step.risk_levels,
            step.sensors,
            confidence=step.confidence,
            compound_reasons=compound_reasons,
        )
    return steps


# --------------------------------------------------------------------------
# 입력 모사 — 위험조건을 안전범위 안에서 만들어 낸다
# --------------------------------------------------------------------------

def _jitter(rng: np.random.Generator, base: float, spread: float) -> float:
    return float(base + rng.normal(0, spread))


def rain_overwet_steps(rng: np.random.Generator) -> list[Step]:
    """강우예보 + 토양수분 상승 -> 관수중단 -> 배수확인 알림."""
    steps = []
    for hour in range(6):
        raining = hour >= 2
        soil_high = hour >= 3
        level = NORMAL if hour < 2 else (CAUTION if hour < 3 else WARNING)
        steps.append(Step(
            hour=(10 + hour) % 24,
            risk_levels={"rain_wet": level},
            sensors=SensorState(
                rain_detected=raining,
                soil_above_limit=soil_high,
                gust_3s=_jitter(rng, 2.0, 0.5),
                hour=(10 + hour) % 24,
            ),
            confidence={"rain_wet": _jitter(rng, 0.88, 0.04)},
        ))
    return run_steps(steps)


def frost_steps(rng: np.random.Generator) -> list[Step]:
    """저녁 기온 하강·무풍·맑음 -> 서리 위험 3시간 전 판단 -> 야간 피복."""
    steps = []
    # 18시부터 06시까지. 서리는 03시에 실제로 발생한다고 둔다.
    onset_hour = 3
    for offset in range(13):
        hour = (18 + offset) % 24
        hours_to_onset = (onset_hour - hour) % 24
        # 발생 3시간 전부터 AI 가 경계로 올린다
        if hours_to_onset <= 3 or hour == onset_hour:
            level = WARNING
        elif hours_to_onset <= 6:
            level = CAUTION
        else:
            level = NORMAL
        steps.append(Step(
            hour=hour,
            risk_levels={"frost": level},
            sensors=SensorState(gust_3s=_jitter(rng, 1.0, 0.3), hour=hour),
            confidence={"frost": _jitter(rng, 0.86, 0.05)},
        ))
    return run_steps(steps)


def disease_steps(rng: np.random.Generator) -> list[Step]:
    """엽면습윤·RH·기온 지속 -> 병해 유리환경 등급 -> 예찰·방제 준비, 살수 금지."""
    steps = []
    for hour in range(12):
        level = NORMAL if hour < 3 else (CAUTION if hour < 7 else WARNING)
        steps.append(Step(
            hour=(20 + hour) % 24,
            risk_levels={"disease": level},
            sensors=SensorState(gust_3s=_jitter(rng, 1.2, 0.4), hour=(20 + hour) % 24),
            confidence={"disease": _jitter(rng, 0.9, 0.04)},
        ))
    return run_steps(steps)


def compound_rain_cold_steps(rng: np.random.Generator) -> list[Step]:
    """강우·과습 + 병해 유리환경 동시 -> 경계 격상 -> 동시 조치."""
    steps = []
    for hour in range(6):
        wet = CAUTION if hour < 3 else WARNING
        disease = CAUTION if hour >= 2 else NORMAL
        compound = WARNING if (wet >= CAUTION and disease >= CAUTION and
                               (wet >= WARNING or disease >= WARNING)) else (
            CAUTION if wet >= CAUTION and disease >= CAUTION else NORMAL)
        steps.append(Step(
            hour=(2 + hour) % 24,
            risk_levels={"rain_wet": wet, "disease": disease, "compound": compound},
            sensors=SensorState(
                rain_detected=hour < 3,
                soil_above_limit=hour >= 2,
                gust_3s=_jitter(rng, 2.0, 0.5),
                hour=(2 + hour) % 24,
            ),
            confidence={"rain_wet": 0.9, "disease": 0.88, "compound": 0.85},
        ))
    return run_steps(steps, compound_reasons=["강우·과습 + 병해 유리환경 동시 발생"])


def compound_wet_frost_steps(rng: np.random.Generator) -> list[Step]:
    """토양 과습 상태에서 서리 위험 -> 관수 금지·배수 확인·야간 피복 권고."""
    steps = []
    for hour in range(6):
        wet = CAUTION
        frost = NORMAL if hour < 2 else (CAUTION if hour < 4 else WARNING)
        compound = NORMAL if frost < CAUTION else (
            WARNING if frost >= WARNING else CAUTION)
        steps.append(Step(
            hour=(19 + hour) % 24,
            risk_levels={"rain_wet": wet, "frost": frost, "compound": compound},
            sensors=SensorState(
                soil_above_limit=True,
                gust_3s=_jitter(rng, 1.0, 0.3),
                hour=(19 + hour) % 24,
            ),
            confidence={"rain_wet": 0.86, "frost": 0.87, "compound": 0.84},
        ))
    return run_steps(steps, compound_reasons=["과습 + 저온·서리 동시 발생"])


def heat_dry_steps(rng: np.random.Generator) -> list[Step]:
    """고온·건조 -> 관수 권고·실행, 경계 30분 지속 시에만 차광 한정 전개.

    차광 누적시간을 제어기처럼 들고 다닌다. 10~16시 밖이거나 2시간을
    다 쓰면 엔진이 전개를 멈추고 걷어야 한다.
    """
    steps = []
    shade_minutes = 0.0      # 당일 누적 — 걷어도 줄지 않는다
    shade_open = False       # 지금 펴져 있는지
    for offset in range(8):
        hour = 9 + offset                       # 09~16시
        level = NORMAL if offset < 2 else (CAUTION if offset < 3 else WARNING)
        step = Step(
            hour=hour,
            risk_levels={"heat_dry": level},
            sensors=SensorState(
                gust_3s=_jitter(rng, 2.5, 0.8), hour=hour,
                shade_deployed_minutes=shade_minutes,
                shade_open=shade_open,
            ),
            confidence={"heat_dry": _jitter(rng, 0.9, 0.04)},
        )
        run_steps([step])
        steps.append(step)

        if step.deployed_shade():
            shade_open = True
        elif step.has("차광막", "회수"):
            shade_open = False
        if shade_open:
            shade_minutes += 60.0
    return steps


def wind_safety_steps(rng: np.random.Generator) -> list[Step]:
    """순간풍속 7m/s 전개금지 -> 9m/s 초과 즉시 회수 -> 과전류 정지·경보."""
    gusts = [3.0, 6.0, 7.5, 9.5, 12.0, 15.5]
    steps = []
    for hour, gust in enumerate(gusts):
        steps.append(Step(
            hour=(12 + hour) % 24,
            # 고온 차광 요청이 함께 들어온 상태에서 강풍이 이기는지 본다
            risk_levels={"heat_dry": WARNING},
            sensors=SensorState(
                gust_3s=_jitter(rng, gust, 0.2),
                overcurrent=(gust >= 15.0),
                hour=(12 + hour) % 24,
            ),
            confidence={"heat_dry": 0.9},
        ))
    return run_steps(steps)


def comms_failure_steps(rng: np.random.Generator) -> list[Step]:
    """서버 통신차단 -> 위험조건 -> 현장 독립제어 -> 복구."""
    steps = []
    for hour in range(6):
        down = 1 <= hour <= 4
        steps.append(Step(
            hour=(21 + hour) % 24,
            risk_levels={"frost": WARNING},
            sensors=SensorState(
                comms_down=down,
                gust_3s=_jitter(rng, 1.0, 0.3),
                hour=(21 + hour) % 24,
            ),
            confidence={"frost": 0.88},
        ))
    return run_steps(steps)


# --------------------------------------------------------------------------
# 판정 — 사업계획서 판정기준을 그대로 옮긴다
# --------------------------------------------------------------------------

def check_rain_overwet(run: Run) -> Run:
    """10회 모두 관수차단, 판단근거·명령·완료로그 100%."""
    risky = [s for s in run.steps if s.risk_levels.get("rain_wet", 0) >= CAUTION]
    blocked = all(
        s.has("관수밸브", "중단") or s.has("관수밸브", "금지") for s in risky
    )
    drain_alert = any(s.alerted("배수") for s in risky)
    no_irrigation = not any(s.has("관수밸브", "관수 준비") for s in risky)

    run.passed = bool(risky) and blocked and drain_alert and no_irrigation
    run.detail = (
        f"위험시점 {len(risky)} / 관수차단 {blocked} / 배수알림 {drain_alert} / "
        f"관수요청 없음 {no_irrigation}"
    )
    return run


def check_frost(run: Run, *, onset_hour: int = 3) -> Run:
    """서리 발생 전 사전경보와 선행시간, 야간 관수 차단 100%."""
    alert = run.first(lambda s: s.alerted("서리") or s.has("야간피복", "피복"))
    onset = run.first(lambda s: s.hour == onset_hour)

    if alert is not None and onset is not None:
        lead = ((onset_hour - alert.hour) % 24) * 60
        run.lead_minutes = float(lead)

    night = [s for s in run.steps if s.hour >= 18 or s.hour < 8]
    irrigation_blocked = all(not s.has("관수밸브", "관수 준비") for s in night)
    frost_steps_ = [s for s in run.steps if s.risk_levels.get("frost", 0) >= CAUTION]
    irrigation_banned = all(s.has("관수밸브", "관수 금지") for s in frost_steps_)

    run.passed = (
        alert is not None
        and run.lead_minutes is not None
        and run.lead_minutes >= 180          # 3시간 전 판단 목표
        and irrigation_blocked
        and irrigation_banned
    )
    run.detail = (
        f"선행 {run.lead_minutes}분 / 야간 관수요청 없음 {irrigation_blocked} / "
        f"야간·새벽 관수금지 {irrigation_banned}"
    )
    return run


def check_disease(run: Run) -> Run:
    """경계 등급 알림 100%, 살수 금지 준수 100%."""
    warning = [s for s in run.steps if s.risk_levels.get("disease", 0) >= WARNING]
    risky = [s for s in run.steps if s.risk_levels.get("disease", 0) >= CAUTION]

    alerted = all(s.alerted("예찰") or s.alerted("방제") for s in warning)
    spray_banned = all(s.has("살수", "금지") for s in risky)

    run.passed = bool(warning) and alerted and spray_banned
    run.detail = f"경계 {len(warning)}시점 알림 {alerted} / 살수금지 {spray_banned}"
    return run


def check_compound(run: Run) -> Run:
    """우선순위 판단·동시 조치 정상 — 충돌 없이 모두 수행."""
    risky = [s for s in run.steps if s.risk_levels.get("compound", 0) >= CAUTION]
    if not risky:
        run.detail = "복합위험 시점 없음"
        return run

    irrigation = all(
        s.has("관수밸브", "중단") or s.has("관수밸브", "금지") for s in risky
    )
    spray = all(s.has("살수", "금지") for s in risky)
    alerted = all(s.alerted("알림") or s.alerted("확인") or s.alerted("준비") or
                  s.alerted("예찰") or s.alerted("경보") for s in risky)
    no_conflict = all(
        not any(a.rejected_by for a in s.decision.actions if a.device == "알림")
        for s in risky
    )

    run.passed = irrigation and spray and alerted and no_conflict
    run.detail = f"관수차단 {irrigation} / 살수금지 {spray} / 알림 {alerted} / 충돌없음 {no_conflict}"
    return run


def check_heat_dry(run: Run) -> Run:
    """차광 시간대·지속시간 조건 준수 100%, 관수 권고·실행 정상."""
    warning = [s for s in run.steps if s.risk_levels.get("heat_dry", 0) >= WARNING]
    risky = [s for s in run.steps if s.risk_levels.get("heat_dry", 0) >= CAUTION]

    irrigation = all(s.has("관수밸브", "관수 준비") for s in risky)

    # 차광 전개 '명령'이 아니라 실제로 펴져 있던 시간으로 본다.
    # 명령은 한 번만 나가고 그 뒤에는 열린 상태가 유지되기 때문이다.
    opened = [s for s in run.steps if s.deployed_shade()]
    open_minutes = max((s.sensors.shade_deployed_minutes for s in run.steps), default=0.0)
    open_hours = [s.hour for s in run.steps if s.sensors.shade_open]

    only_on_warning = all(s.risk_levels.get("heat_dry", 0) >= WARNING for s in opened)
    within_window = all(10 <= hour < 16 for hour in open_hours)
    within_budget = open_minutes <= 120
    # 2시간을 다 쓰거나 16시에 닿으면 반드시 걷어야 한다
    needs_withdrawal = open_minutes >= 120 or any(h >= 16 for h in open_hours)
    withdrawn = any(s.has("차광막", "회수") for s in run.steps) if needs_withdrawal else True

    run.passed = (
        bool(warning) and irrigation and only_on_warning
        and within_window and within_budget and withdrawn
    )
    run.detail = (
        f"관수 {irrigation} / 전개 {len(opened)}회·누적 {open_minutes:.0f}분 / "
        f"경계시에만 {only_on_warning} / 시간대 {within_window} / "
        f"2시간 이내 {within_budget} / 회수 {withdrawn}"
    )
    return run


def check_wind_safety(run: Run) -> Run:
    """7m/s 전개금지, 9m/s 초과 즉시 회수, 과전류 정지·경보 정상."""
    no_deploy = all(
        not s.deployed_shade() for s in run.steps if (s.sensors.gust_3s or 0) >= 7.0
    )
    recovered = all(
        s.has("차광막·피복", "회수") for s in run.steps if (s.sensors.gust_3s or 0) >= 9.0
    )
    overcurrent = [s for s in run.steps if s.sensors.overcurrent]
    tripped = all(s.has("구동부", "정지") for s in overcurrent)

    run.passed = no_deploy and recovered and (not overcurrent or tripped)
    run.detail = f"전개금지 {no_deploy} / 즉시회수 {recovered} / 과전류 정지 {tripped}"
    return run


def check_comms(run: Run) -> Run:
    """통신 두절 중에도 제어·저장이 유지되고 복구된다."""
    down = [s for s in run.steps if s.sensors.comms_down]
    up = [s for s in run.steps if not s.sensors.comms_down]

    local_mode = all(s.decision.mode == "로컬 독립운전" for s in down)
    kept_control = all(s.has("야간피복", "피복") or s.has("관수밸브", "금지") for s in down)
    recovered = all(s.decision.mode != "로컬 독립운전" for s in up)

    run.passed = bool(down) and local_mode and kept_control and recovered
    run.detail = f"독립운전 {local_mode} / 제어유지 {kept_control} / 복구 {recovered}"
    return run


#: 시나리오 정의 — (이름, 입력 생성기, 판정 함수, 반복 횟수, 최소 통과 횟수, 판정기준 설명)
SCENARIOS = [
    ("강우·과습 관수중단", rain_overwet_steps, check_rain_overwet, 10, 10,
     "10회 모두 관수차단, 판단근거·명령·완료로그 100%"),
    ("저온·서리 사전경보", frost_steps, check_frost, 10, 10,
     "서리 발생 전 사전경보·선행시간 3시간 이상, 야간·새벽 관수 차단 100%"),
    ("다습·병해 유리환경 알림", disease_steps, check_disease, 10, 10,
     "경계 등급 알림 100%, 살수 금지 준수 100%"),
    ("복합: 강우 후 저온다습", compound_rain_cold_steps, check_compound, 10, 10,
     "10회 모두 우선순위 판단·동시 조치 정상"),
    ("복합: 과습 + 서리", compound_wet_frost_steps, check_compound, 10, 10,
     "10회 모두 충돌 없이 정상 조치"),
    ("고온·건조 제한운영", heat_dry_steps, check_heat_dry, 20, 19,
     "20회 중 19회 이상 정상작동, 차광 시간대·지속시간 조건 준수 100%"),
    ("강풍 안전보호", wind_safety_steps, check_wind_safety, 10, 10,
     "10회 모두 안전회수(5분 이내 완전회수), 과전류 정지·경보 정상"),
    ("통신장애", comms_failure_steps, check_comms, 5, 5,
     "5종 시나리오 모두 제어·저장·복구동기화"),
]


def run_scenario(
    name: str, generator, checker, repeats: int, required: int, criterion: str,
    *, seed: int = 42, source: str = SOURCE_SIMULATED,
) -> ScenarioResult:
    """시나리오 하나를 ``repeats`` 회 반복 시험한다."""
    runs = []
    for index in range(repeats):
        rng = np.random.default_rng(seed + index)
        run = checker(Run(index=index, steps=generator(rng)))
        runs.append(run)

    return ScenarioResult(
        name=name,
        criterion=criterion,
        repeats=repeats,
        passed=sum(r.passed for r in runs),
        required=required,
        source=source,
        runs=runs,
    )


def run_all(seed: int = 42) -> list[ScenarioResult]:
    """8종 전체를 시험한다."""
    return [run_scenario(*spec, seed=seed) for spec in SCENARIOS]


def results_table(results: list[ScenarioResult]) -> pd.DataFrame:
    """시험 결과표 — 외부 공인시험 성적서에 쓰는 형식."""
    rows = []
    for result in results:
        leads = result.lead_times()
        rows.append({
            "시나리오": result.name,
            "구분": result.source,
            "시험횟수": result.repeats,
            "통과": result.passed,
            "통과율(%)": round(result.pass_rate * 100, 1),
            "판정기준": result.criterion,
            "판정": result.verdict,
            "평균 선행시간(분)": round(float(np.mean(leads)), 0) if leads else None,
        })
    return pd.DataFrame(rows)
