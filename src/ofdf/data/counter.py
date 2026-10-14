"""적산 계측기를 구간값으로 바꾼다.

세 수집 경로가 모두 적산값을 만난다.

    rtl_433 ``rain_mm``           초기화 이후 누적 강수
    에코윗 ``dailyrainin``        그날 누적 강수(자정에 0)
    Zigbee ``energy``             설치 이후 누적 전력량
    Zigbee ``water_consumed``     설치 이후 누적 관수량

그대로 넣으면 5분 강수가 수백 mm 로, 5분 관수량이 수백 L 로 잡힌다. 셋이
같은 문제라 셈도 한 곳에 둔다.

값이 **줄어드는** 경우를 특별히 다룬다. 자정 초기화, 센서 교체, 통신 복구
뒤의 재집계가 모두 여기 해당한다. 차이를 그대로 쓰면 음수가 나오고, 음수
강수나 음수 관수량은 물수지를 망가뜨린다. 그래서 그 구간은 0 으로 둔다 —
그 사이에 내린 비를 한 번 놓치는 편이, 안 내린 비를 음수로 기록하는 것보다
낫다.
"""

from __future__ import annotations


class CumulativeCounter:
    """누적 총량에서 구간 증가분을 만든다.

    ``key`` 마다 따로 센다. 장치가 둘이면 서로의 값을 섞지 않아야 한다.

    >>> c = CumulativeCounter()
    >>> c.step("pump", 10.0)          # 처음 본 값은 기준점이다
    0.0
    >>> c.step("pump", 12.5)
    2.5
    >>> c.step("pump", 0.0)           # 초기화 — 음수를 내보내지 않는다
    0.0
    """

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def step(self, key: str, total: float) -> float:
        previous = self._last.get(key)
        self._last[key] = total
        if previous is None or total < previous:
            return 0.0
        return total - previous

    def reset(self, key: str | None = None) -> None:
        """기준점을 지운다. ``key`` 가 없으면 전부."""
        if key is None:
            self._last.clear()
        else:
            self._last.pop(key, None)
