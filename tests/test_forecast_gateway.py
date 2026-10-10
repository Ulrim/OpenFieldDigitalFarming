"""예보 장애 시 대체 로직 검증.

사업계획서는 기상청 단기예보 연동을 요구하지만, 제어기는 통신이 끊긴
채로도 72시간을 돌아야 한다. 예보를 못 받아서 서리 판단이 멈추면 그것이
더 큰 사고다. 그래서 예보는 '있으면 쓰고 없으면 없는 대로' 가는 보조
입력이어야 한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ofdf.data.forecast import (
    FORECAST_STALE_MINUTES,
    ForecastGateway,
    build_provider,
)

NOW = pd.Timestamp("2026-10-10 20:00")


class Stub:
    """원하는 대로 성공·실패하는 예보 제공자."""

    def __init__(self, frames=None, error=None):
        self.frames = list(frames or [])
        self.error = error
        self.calls = 0

    def at(self, issued, horizons=(1, 3)):
        self.calls += 1
        if self.error:
            raise self.error
        return self.frames.pop(0) if self.frames else None


def frame(t_air: float) -> pd.DataFrame:
    return pd.DataFrame({"t_air": [t_air, t_air - 1]}, index=[1, 3])


def test_no_service_key_means_no_provider():
    """키를 저장소에 두지 않으므로, 없을 때 조용히 꺼져야 한다."""
    assert build_provider(None) is None
    assert build_provider("") is None


def test_provider_uses_naju_grid_by_default():
    from ofdf.data.forecast import NAJU_NAMPYEONG_GRID

    provider = build_provider("키")
    assert provider.nx == NAJU_NAMPYEONG_GRID["nx"] == 57
    assert provider.ny == NAJU_NAMPYEONG_GRID["ny"] == 71


def test_successful_fetch_is_passed_through():
    gate = ForecastGateway(Stub([frame(2.0)]))
    got = gate.fetch(NOW)
    assert got is not None and got.loc[1, "t_air"] == 2.0
    assert gate.status == "정상"


def test_failure_falls_back_to_the_last_forecast():
    """한 번 실패했다고 판단을 멈추면 안 된다."""
    stub = Stub([frame(2.0)])
    gate = ForecastGateway(stub)
    gate.fetch(NOW)

    stub.error = ConnectionError("통신 두절")
    got = gate.fetch(NOW + pd.Timedelta(minutes=60))

    assert got is not None and got.loc[1, "t_air"] == 2.0
    assert "직전 예보" in gate.status
    assert "ConnectionError" in gate.last_error


def test_stale_forecast_is_dropped_not_reused_forever():
    """몇 시간 지난 예보를 계속 쓰면 틀린 미래를 보는 셈이 된다."""
    stub = Stub([frame(2.0)])
    gate = ForecastGateway(stub)
    gate.fetch(NOW)

    stub.error = ConnectionError("통신 두절")
    got = gate.fetch(NOW + pd.Timedelta(minutes=FORECAST_STALE_MINUTES + 1))

    assert got is None
    assert "만료" in gate.status


def test_failure_without_any_history_returns_none():
    """예보가 없어도 관측만으로 판단을 이어갈 수 있어야 한다."""
    gate = ForecastGateway(Stub(error=TimeoutError("응답 없음")))
    assert gate.fetch(NOW) is None
    assert gate.status == "예보 없음"


def test_empty_or_all_nan_response_counts_as_failure():
    """200 을 받아도 내용이 비면 예보가 아니다."""
    empty = ForecastGateway(Stub([pd.DataFrame()]))
    assert empty.fetch(NOW) is None

    blank = ForecastGateway(Stub([pd.DataFrame({"t_air": [None, None]}, index=[1, 3])]))
    assert blank.fetch(NOW) is None


@pytest.mark.parametrize("error", [
    ConnectionError("끊김"), TimeoutError("느림"),
    ValueError("형식 오류"), KeyError("response"),
])
def test_every_failure_kind_is_handled_the_same(error):
    """통신·인증·형식 무엇이든 판단을 멈추게 해서는 안 된다."""
    gate = ForecastGateway(Stub(error=error))
    assert gate.fetch(NOW) is None
    assert gate.status == "예보 없음"


def test_recovery_after_failure():
    stub = Stub([frame(2.0)], error=ConnectionError("끊김"))
    gate = ForecastGateway(stub)
    assert gate.fetch(NOW) is None

    stub.error = None
    got = gate.fetch(NOW + pd.Timedelta(minutes=60))
    assert got is not None and gate.status == "정상" and gate.last_error == ""
