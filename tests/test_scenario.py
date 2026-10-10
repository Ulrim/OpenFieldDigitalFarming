"""핵심 실증 시나리오 시험 하네스 검증.

시험 하네스가 무조건 통과하면 쓸모가 없다. 판정기준이 **고장을 실제로
잡아내는지**를 함께 본다.
"""

from __future__ import annotations

import numpy as np
import pytest

from ofdf.action.recommend import SHADE_MAX_MINUTES, Layer, SensorState, decide
from ofdf.evaluation import scenario
from ofdf.evaluation.scenario import Run, Step
from ofdf.labels.risk import CAUTION, NORMAL, WARNING


def test_all_scenarios_meet_criteria():
    """핵심 실증 시나리오 전체가 판정기준을 충족한다."""
    results = scenario.run_all()
    # 사업계획서 8종 가운데 통신장애가 5종으로 나뉘므로 7 + 5 = 12 이다.
    assert len(results) == 12
    failed = [r.name for r in results if not r.meets_criterion]
    assert not failed, f"부적합: {failed}"


def test_comms_failure_is_five_distinct_scenarios():
    """통신장애는 같은 시험의 반복이 아니라 서로 다른 고장 양상이어야 한다.

    판정기준에 '5종 시나리오' 라고 적어 두고 같은 생성기를 난수만 바꿔
    5회 돌리고 있었다. 반복시험이지 5종 시험이 아니다.
    """
    assert len(scenario.COMMS_VARIANTS) == 5
    generators = {gen for _, gen in scenario.COMMS_VARIANTS}
    assert len(generators) == 5, "서로 다른 생성기여야 한다"

    # 구분 기준은 두절 패턴만이 아니다. '두절 중 안전사건' 은 두절 모양이
    # 단기 두절과 같고 과전류가 끼어든다는 점이 다르다 — 그것도 엄연히
    # 다른 고장이다. 그래서 입력 전체를 보고 겹치는지 센다.
    import numpy as np
    signatures = {
        tuple((s.sensors.comms_down, s.sensors.overcurrent, s.sensors.emergency_stop)
              for s in gen(np.random.default_rng(7)))
        for _, gen in scenario.COMMS_VARIANTS
    }
    assert len(signatures) == 5, f"입력이 겹치는 시나리오가 있다: {len(signatures)}종"

    # 두절 길이도 서로 달라야 같은 것을 되풀이하지 않는다
    lengths = {
        sum(s.sensors.comms_down for s in gen(np.random.default_rng(7)))
        for _, gen in scenario.COMMS_VARIANTS
    }
    assert len(lengths) >= 3, f"두절 길이가 사실상 한 종류다: {lengths}"

    names = [r.name for r in scenario.run_all() if r.name.startswith("통신장애")]
    assert len(names) == 5


def test_results_table_is_reportable():
    table = scenario.results_table(scenario.run_all())
    assert set(table.columns) >= {"시나리오", "구분", "시험횟수", "통과", "판정기준", "판정"}
    assert (table["판정"] == "적합").all()


def test_frost_lead_time_meets_three_hour_target():
    """서리는 발생 3시간 전에 알려야 한다."""
    result = scenario.run_scenario(*scenario.SCENARIOS[1])
    leads = result.lead_times()
    assert leads and min(leads) >= 180


# --------------------------------------------------------------------------
# 판정기준이 고장을 잡는지
# --------------------------------------------------------------------------

def _steps(levels, sensors_list, confidence=None):
    steps = [
        Step(hour=s.hour, risk_levels=lv, sensors=s, confidence=confidence or {})
        for lv, s in zip(levels, sensors_list)
    ]
    return scenario.run_steps(steps)


def test_rain_criterion_fails_without_irrigation_block():
    """관수가 차단되지 않으면 강우·과습 판정이 떨어져야 한다."""
    # 위험등급은 올렸지만 현장 즉응규칙 입력(강우·토양상한)이 없어
    # 관수 중단 명령이 나가지 않는 상태를 만든다
    steps = _steps(
        [{"rain_wet": NORMAL}] * 3,
        [SensorState(hour=10), SensorState(hour=11), SensorState(hour=12)],
    )
    assert not scenario.check_rain_overwet(Run(0, steps)).passed


def test_disease_criterion_fails_without_spray_ban():
    steps = _steps([{"disease": NORMAL}] * 3, [SensorState(hour=h) for h in (20, 21, 22)])
    assert not scenario.check_disease(Run(0, steps)).passed


def test_wind_criterion_fails_if_shade_deploys_in_wind():
    """강풍에 차광이 펴지면 판정이 떨어진다."""
    # 엔진을 거치지 않고 '전개가 실행된' 상태를 직접 만든다
    step = Step(hour=13, risk_levels={"heat_dry": WARNING},
                sensors=SensorState(gust_3s=12.0, hour=13))
    step.decision = decide({"heat_dry": WARNING}, SensorState(hour=13))   # 강풍 없이 판단
    step.sensors = SensorState(gust_3s=12.0, hour=13)                      # 강풍으로 바꿔치기
    assert step.deployed_shade()
    assert not scenario.check_wind_safety(Run(0, [step])).passed


def test_comms_criterion_fails_without_local_mode():
    steps = _steps([{"frost": WARNING}] * 2, [SensorState(hour=22), SensorState(hour=23)])
    assert not scenario.check_comms(Run(0, steps)).passed   # 통신두절 시점이 없다


# --------------------------------------------------------------------------
# 차광 제한 운영 — 사업계획서 '10~16시, 최대 2시간'
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hour", [9, 16, 17, 20])
def test_shade_never_deploys_outside_window(hour):
    decision = decide(
        {"heat_dry": WARNING}, SensorState(hour=hour, shade_deployed_minutes=0.0)
    )
    assert not any("전개" in a.command for a in decision.executable())


@pytest.mark.parametrize("hour", [10, 12, 15])
def test_shade_deploys_inside_window(hour):
    decision = decide({"heat_dry": WARNING}, SensorState(hour=hour))
    assert any("전개" in a.command for a in decision.executable())


def test_shade_withdraws_when_daily_budget_spent():
    decision = decide(
        {"heat_dry": WARNING},
        SensorState(hour=13, shade_deployed_minutes=SHADE_MAX_MINUTES, shade_open=True),
    )
    commands = [a.command for a in decision.executable() if "차광" in a.device]
    assert any("회수" in c for c in commands)
    assert not any("전개" in c for c in commands)


def test_shade_does_not_redeploy_after_budget_spent():
    """걷었다가 다시 펴는 것으로 2시간 제한을 늘릴 수 없다."""
    decision = decide(
        {"heat_dry": WARNING},
        SensorState(hour=14, shade_deployed_minutes=SHADE_MAX_MINUTES, shade_open=False),
    )
    assert not any("전개" in a.command for a in decision.executable())


def test_shade_withdraws_at_closing_hour():
    decision = decide(
        {"heat_dry": WARNING},
        SensorState(hour=16, shade_deployed_minutes=60.0, shade_open=True),
    )
    assert any("회수" in a.command for a in decision.executable() if "차광" in a.device)


def test_no_repeat_deploy_command_while_open():
    """이미 펴져 있으면 같은 명령을 다시 내지 않는다(반복작동 방지)."""
    decision = decide(
        {"heat_dry": WARNING},
        SensorState(hour=13, shade_deployed_minutes=60.0, shade_open=True),
    )
    assert not any("전개" in a.command for a in decision.executable())


def test_deployed_shade_ignores_prohibition_commands():
    """'전개 금지'는 전개가 아니다."""
    step = Step(hour=13, risk_levels={"heat_dry": WARNING},
                sensors=SensorState(gust_3s=8.0, hour=13))
    step.decision = decide({"heat_dry": WARNING}, SensorState(gust_3s=8.0, hour=13))
    assert any("전개 금지" in a.command for a in step.executed())
    assert not step.deployed_shade()
