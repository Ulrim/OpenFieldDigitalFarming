"""제어 우선순위와 충돌 해소 규칙 검증.

사업계획서 '충돌 해소 및 반복작동 방지 규칙' 표의 각 행이 그대로
재현되는지 본다.
"""

from __future__ import annotations

from ofdf.action.recommend import Layer, SensorState, decide
from ofdf.labels.risk import WARNING


def commands(decision, device_keyword: str) -> list:
    return [a for a in decision.actions if device_keyword in a.device]


def test_wind_overrides_shading_request():
    """강풍 + 고온 → 강풍 우선, 차광 전개 거부."""
    d = decide({"heat_dry": 2}, SensorState(gust_3s=10.0, hour=14))
    deploy = [a for a in d.actions if "전개" in a.command and "금지" not in a.command]
    assert deploy and all(a.rejected_by for a in deploy)
    assert any("회수" in a.command and not a.rejected_by for a in d.actions)


def test_wind_overrides_frost_cover_request():
    """강풍 + 서리 → 피복 전개 거부, 서리 알림은 유지."""
    d = decide({"frost": 2}, SensorState(gust_3s=10.0, hour=22))
    cover = [a for a in d.actions if a.device == "야간피복"]
    assert cover and all(a.rejected_by for a in cover)
    assert any(a.device == "알림" and not a.rejected_by for a in d.actions)


def test_rain_overrides_irrigation_request():
    """강우 + 고온 → 관수 요청 거부, 살수·차광 회수 실행."""
    d = decide({"heat_dry": 2}, SensorState(rain_detected=True, hour=13))
    requests = [a for a in d.actions if "관수 준비" in a.command]
    assert requests and all(a.rejected_by for a in requests)


def test_disease_blocks_misting_but_not_drip_irrigation():
    """병해 유리환경은 잎을 적시는 살수만 막고, 점적관수는 막지 않는다."""
    d = decide({"disease": 2, "heat_dry": 1}, SensorState(hour=11))
    assert any(a.device == "살수" and "금지" in a.command for a in d.actions)
    drip = [a for a in d.actions if "관수 준비" in a.command]
    assert drip and not any(a.rejected_by for a in drip)


def test_spraying_stops_above_wind_limit():
    """살수 중 풍속 4m/s 초과 → 즉시 중지(비산 방지)."""
    d = decide({}, SensorState(gust_3s=5.0, spraying=True))
    assert any("살수" in a.device and "중지" in a.command for a in d.actions)


def test_emergency_stop_blocks_everything():
    """비상정지는 모든 구동 요청을 막고 운전모드를 안전정지로 바꾼다."""
    d = decide({"heat_dry": 2}, SensorState(emergency_stop=True, hour=14))
    assert d.mode == "안전정지"
    drives = [a for a in d.actions if a.layer == Layer.AI and a.device != "알림"]
    assert drives and all(a.rejected_by for a in drives)


def test_low_confidence_switches_to_advisory():
    """신뢰도 기준 미만이면 자동 실행하지 않고 농가 승인을 받는다."""
    d = decide({"frost": 2}, SensorState(hour=23), confidence={"frost": 0.55})
    assert d.mode.startswith("권고")
    assert all(a.advisory for a in d.actions if a.layer == Layer.AI)
    assert d.executable() == []


def test_high_confidence_runs_automatically():
    d = decide({"frost": 2}, SensorState(hour=23), confidence={"frost": 0.9})
    assert d.mode == "자동"
    assert d.executable()


def test_comms_failure_switches_to_local_mode():
    d = decide({"frost": 1}, SensorState(comms_down=True, hour=21))
    assert d.mode == "로컬 독립운전"
    assert any("독립운전" in a.command for a in d.actions)


def test_alerts_are_never_blocked():
    """알림은 상위 계층이 막지 않는다 — 위험을 알리는 것 자체는 안전하다."""
    d = decide({"heat_dry": 2, "frost": 2}, SensorState(emergency_stop=True, gust_3s=16.0))
    alerts = [a for a in d.actions if a.device == "알림"]
    assert alerts and not any(a.rejected_by for a in alerts)


def test_normal_conditions_produce_no_action():
    d = decide({"heat_dry": 0, "frost": 0}, SensorState(hour=12))
    assert d.actions == []
    assert d.mode == "자동"


def test_duplicate_commands_are_merged():
    """복합위험과 구성 위험이 같은 조치를 내면 하나로 합친다."""
    d = decide({"compound": 2, "rain_wet": 2}, SensorState(hour=20))
    sprays = [a for a in d.actions if a.device == "살수" and "금지" in a.command]
    assert len(sprays) == 1


# --------------------------------------------------------------------------
# 구동부 설치 여부
# --------------------------------------------------------------------------

def test_missing_actuator_becomes_advice_not_execution():
    """구동부가 없는 장치는 '실행'으로 기록하면 안 된다.

    조치엔진은 장치 7종에 명령하는데 1차 구매분은 관수밸브와 전원차단만
    덮는다. 이것을 모른 채 돌리면 '차광막 차광 한정 전개'가 실행으로 남는다.
    실행된 것은 없는데 제어 이력에는 남고, 그 이력이 그대로 성능평가의
    제어응답 측정 대상이 된다.
    """
    installed = {"관수밸브", "전 구동부"}
    decision = decide({"frost": WARNING}, SensorState(hour=20),
                      installed_devices=installed)

    cover = [a for a in decision.actions if a.device == "야간피복"]
    assert cover, "조치 자체는 사라지면 안 된다 — 손이 바뀔 뿐이다"
    assert all(a.advisory for a in cover)
    assert all(not a.rejected_by for a in cover), "거부가 아니라 권고다"
    assert all("구동부 미설치" in a.reason for a in cover)

    valve = [a for a in decision.actions if a.device == "관수밸브"]
    assert valve and all(not a.advisory for a in valve), "달린 장치는 그대로 실행"

    assert "구동부 미설치" in decision.mode


def test_alerts_run_even_when_device_list_is_narrow():
    """알림은 구동부가 아니다. 좁게 적어도 농가에게는 가야 한다."""
    decision = decide({"frost": WARNING}, SensorState(hour=20),
                      installed_devices={"관수밸브"})
    alerts = [a for a in decision.actions if a.device == "알림"]
    assert alerts and all(not a.advisory for a in alerts)


def test_full_hardware_is_the_default():
    """판단은 하드웨어 사정과 무관해야 한다.

    차광막이 아직 없다고 해서 '지금 차광이 필요하다'는 판단이 달라지지는
    않는다. 설정이 좁히지 않으면 엔진은 전부 달린 것으로 본다. 기본을
    '아무것도 없다'로 두면 설정을 깜빡한 제어기가 조용히 아무것도 실행하지
    않는데, 그쪽이 더 위험하다.
    """
    decision = decide({"heat_dry": WARNING}, SensorState(hour=12))
    assert any("전개" in a.command for a in decision.executable())
    assert decision.mode == "자동"


def test_installed_devices_does_not_change_the_judgement():
    """구동부 유무로 조치 목록 자체가 달라지면 안 된다."""
    args = ({"frost": WARNING, "heat_dry": WARNING}, SensorState(hour=12))
    full = decide(*args)
    narrow = decide(*args, installed_devices={"관수밸브"})
    assert [(a.device, a.command) for a in full.actions] \
        == [(a.device, a.command) for a in narrow.actions]
