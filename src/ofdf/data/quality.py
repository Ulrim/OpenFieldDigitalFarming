"""관측 데이터 품질관리.

사업계획서의 '데이터 품질관리 절차' 가운데 자동 탐지 항목을 구현한다.

* 범위이탈 — 물리적으로 불가능한 값
* 급변 — 1시간 변화가 물리적 한계를 넘는 값
* 고정값 — 센서가 멈춰 같은 값이 계속 나오는 구간
* 영점 채움 — 통신이 끊긴 구간을 0 으로 메운 자리
* 교차 모순 — 변수끼리 같이 성립할 수 없는 조합
* 시간중복 / 결측 — 같은 시각이 두 번 들어오거나 비는 구간

**원본은 바꾸지 않는다.** 대신 ``qc_*`` 플래그 컬럼을 붙이고,
:func:`apply_mask` 로 이상값을 결측 처리한 사본을 따로 만든다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: (최솟값, 최댓값) — 나주 지역 농업기상 관측 기준 물리 범위
VALID_RANGE: dict[str, tuple[float, float]] = {
    "t_air": (-25.0, 45.0),
    "t_max": (-25.0, 45.0),
    "t_min": (-25.0, 45.0),
    "rh": (0.0, 100.0),
    "rain": (0.0, 150.0),          # 시간 강수량
    "solar_mj": (0.0, 4.5),        # 시간 일사량(맑은 날 정오 약 3.6)
    "wind_speed": (0.0, 40.0),
    "soil_moisture": (0.0, 100.0),
}

#: 1시간 사이에 날 수 있는 최대 변화폭
MAX_HOURLY_CHANGE: dict[str, float] = {
    "t_air": 10.0,
    "rh": 60.0,
    "wind_speed": 20.0,
}

#: 영점 채움으로 보는 채널 묶음.
#:
#: 통신이 끊기면 수집기가 빈 자리를 0 으로 메운다. 값 하나하나는 물리
#: 범위 안이라 범위검사로는 잡히지 않는데, 모델에는 '기온 0℃·습도 0%·
#: 일사 0' 이라는 또렷한 신호로 들어간다.
#:
#: 2024-10-23 봉황면은 14시부터 기온·습도·강수·일사가 모두 정확히 0.0 이고
#: 최고/최저기온은 비었다가 23시에 -50.0/50.0 이 찍혔다. 그 구간에서 모델은
#: 신뢰도 0.99 로 저온·서리 경계를 냈다. 라벨도 같은 자료에서 나오므로
#: '적중' 으로 집계돼 스스로를 검증해 버린다. 현장이라면 야간 피복이 전개되고
#: 관수가 금지되는 오경보다.
#:
#: 상대습도가 정확히 0% 인 자연 상태는 없다. 전 기간 7,826건이 나왔고
#: 문평면이 6,948건으로 대부분이다.
NULL_FILL_CHANNELS = ["t_air", "rh", "solar_mj"]
NULL_FILL_MIN_CHANNELS = 2

#: 상대습도가 이보다 낮으면 관측이 아니라 결측으로 본다.
RH_IMPLAUSIBLE_BELOW = 1.0

#: 강우를 인정할 최소 상대습도(%).
#:
#: 비가 내리는 동안 상대습도가 이보다 낮을 수는 없다. 2025-10-09 14시
#: 봉황면에서 누적 강수가 0 -> 158.5 -> 0 으로 튀었는데 그 시각 습도는
#: 43.7%, 기온은 27.8도로 오르는 중이었다. 비가 온 적이 없는 전도계
#: 오작동이다. 범위검사만으로는 더 작은 값의 같은 오류를 못 잡는다.
RAIN_MIN_HUMIDITY = 60.0

#: 야간으로 보는 시각(이 사이에 일사가 잡히면 모순)
#:
#: 봉황면 일사계는 야간에도 누적값이 시간당 0.2 MJ 씩 올라간다
#: (2025-10-15 20시 5.6 -> 21시 5.8 -> 22시 6.0 -> 23시 6.2).
#: 0.2 MJ/h 는 55 W/m^2 로, 한밤중에 나올 수 없는 값이다. 계기 영점 드리프트다.
#: 야간 일사는 물리적으로 0 이므로 결측이 아니라 0 으로 되돌린다.
NIGHT_HOURS = (22, 23, 0, 1, 2, 3)
NIGHT_MAX_SOLAR_MJ = 0.02

#: 상대습도 센서가 정상이면 가을·겨울에 한 번은 이 값에 닿는다.
#: 한 해 최댓값이 이보다 낮으면 상한 포화·영점 드리프트를 의심한다.
#:
#: 나주 봉황면은 10~11월 최대습도가 2016~2024년 내내 93~100% 였는데
#: 2025년만 88.8% 였고, 같은 기간 공산·문평·금천은 99.8~100% 였다.
#: 이 상태에서는 병해 유리환경 기준(RH 90/95% 이상)이 영원히 성립하지 않아
#: 그 지점을 기준으로 삼으면 병해 판단이 통째로 죽는다.
RH_CEILING_EXPECTED = 93.0

#: 고정값으로 보기 시작하는 연속 시간. 야간 일사 0 처럼 정상적으로 상수인
#: 구간이 있는 변수는 제외한다.
STUCK_HOURS = 24
STUCK_EXEMPT = {"rain", "solar_mj", "sunshine"}


@dataclass
class QCReport:
    """품질관리 결과 요약."""

    n_rows: int
    flagged: dict[str, int] = field(default_factory=dict)
    duplicates: int = 0
    gaps: int = 0

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {"항목": k, "이상건수": v, "비율(%)": round(v / self.n_rows * 100, 4)}
            for k, v in sorted(self.flagged.items(), key=lambda x: -x[1])
        ]
        return pd.DataFrame(rows)


def flag_range(df: pd.DataFrame) -> pd.DataFrame:
    """물리 범위를 벗어난 값을 표시한다."""
    flags = {}
    for col, (lo, hi) in VALID_RANGE.items():
        if col in df.columns:
            flags[f"qc_range_{col}"] = (df[col] < lo) | (df[col] > hi)
    return pd.DataFrame(flags, index=df.index).fillna(False)


def flag_spike(df: pd.DataFrame) -> pd.DataFrame:
    """1시간 변화폭이 물리 한계를 넘는 값을 표시한다."""
    flags = {}
    for col, limit in MAX_HOURLY_CHANGE.items():
        if col in df.columns:
            flags[f"qc_spike_{col}"] = df[col].diff().abs() > limit
    return pd.DataFrame(flags, index=df.index).fillna(False)


def flag_stuck(df: pd.DataFrame, hours: int = STUCK_HOURS) -> pd.DataFrame:
    """같은 값이 ``hours`` 이상 이어지는 구간을 표시한다."""
    flags = {}
    for col in VALID_RANGE:
        if col not in df.columns or col in STUCK_EXEMPT:
            continue
        series = df[col]
        block = (series != series.shift()).cumsum()
        run = series.groupby(block).transform("size")
        flags[f"qc_stuck_{col}"] = (run >= hours) & series.notna()
    return pd.DataFrame(flags, index=df.index).fillna(False)


def flag_null_fill(df: pd.DataFrame) -> pd.DataFrame:
    """통신 두절 구간을 0 으로 메운 자리를 표시한다.

    여러 채널이 **동시에 정확히 0** 이면 관측이 아니라 빈 자리다. 실제
    기상에서 기온과 일사와 습도가 같은 시각에 정확히 0 이 될 수는 없다.
    """
    present = [c for c in NULL_FILL_CHANNELS if c in df.columns]
    flags = {}

    if len(present) >= NULL_FILL_MIN_CHANNELS:
        zeros = sum((df[c] == 0.0).fillna(False).astype(int) for c in present)
        flags["qc_null_fill"] = zeros >= NULL_FILL_MIN_CHANNELS

    if "rh" in df.columns:
        flags["qc_rh_zero"] = (df["rh"] < RH_IMPLAUSIBLE_BELOW).fillna(False)

    if not flags:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(flags, index=df.index).fillna(False)


def flag_inconsistent(df: pd.DataFrame, ts_col: str = "ts") -> pd.DataFrame:
    """변수끼리 같이 성립할 수 없는 조합을 표시한다.

    단일 변수 범위검사가 놓치는 센서 오작동을 잡는다. 전도형 우량계가
    튀어 만든 가짜 강수는 값 자체는 범위 안이어도 그 시각 습도와 모순된다.
    """
    flags = {}

    if {"rain", "rh"} <= set(df.columns):
        flags["qc_cross_rain_dry"] = (df["rain"] > 0) & (df["rh"] < RAIN_MIN_HUMIDITY)

    if {"solar_mj"} <= set(df.columns) and ts_col in df.columns:
        night = pd.to_datetime(df[ts_col], errors="coerce").dt.hour.isin(NIGHT_HOURS)
        flags["qc_cross_night_solar"] = night & (df["solar_mj"] > NIGHT_MAX_SOLAR_MJ)

    if {"t_air", "t_max"} <= set(df.columns):
        flags["qc_cross_tmax"] = df["t_air"] > df["t_max"] + 0.1
    if {"t_air", "t_min"} <= set(df.columns):
        flags["qc_cross_tmin"] = df["t_air"] < df["t_min"] - 0.1

    if not flags:
        return pd.DataFrame(index=df.index)
    return pd.DataFrame(flags, index=df.index).fillna(False)


def check_index(
    df: pd.DataFrame, ts_col: str = "ts", group_col: str | None = "station"
) -> tuple[int, int]:
    """시간중복 건수와 결측 시각 수를 센다.

    여러 관측지점이 한 표에 세로로 쌓여 있으므로 지점별로 따로 센다.
    지점을 구분하지 않으면 같은 시각의 다른 지점 관측이 전부 중복으로
    잡힌다.
    """
    if group_col and group_col in df.columns:
        groups = [g for _, g in df.groupby(group_col, sort=False)]
    else:
        groups = [df]

    duplicates = gaps = 0
    for g in groups:
        ts = g[ts_col].dropna()
        if ts.empty:
            continue
        duplicates += int(ts.duplicated().sum())
        expected = pd.date_range(ts.min(), ts.max(), freq="h")
        gaps += max(int(len(expected) - ts.nunique()), 0)
    return duplicates, gaps


def station_health(
    df: pd.DataFrame,
    *,
    station_col: str = "station",
    ts_col: str = "ts",
    months: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """지점x연도별 센서 건전성을 진단한다.

    행 단위 플래그로는 잡히지 않는 **계통 고장**을 찾는다. 값 하나하나는
    물리 범위 안이지만 분포 자체가 틀어진 경우다. 기준 관측소를 고르거나
    현장 보정의 기준으로 삼을 지점을 정할 때 먼저 본다.

    Parameters
    ----------
    months
        이 월만 본다. 작기(9~11월)로 좁히면 해당 시기의 건전성을 본다.

    Returns
    -------
    pd.DataFrame
        지점·연도별 ``rh_max``, ``rh_ceiling_ok``, ``rain_dry_hours``,
        ``night_solar_hours``, ``stuck_ratio``, ``verdict``.
    """
    work = df.copy()
    if months:
        work = work[pd.to_datetime(work[ts_col]).dt.month.isin(months)]
    if work.empty:
        return pd.DataFrame()

    work["_year"] = pd.to_datetime(work[ts_col]).dt.year
    work["_hour"] = pd.to_datetime(work[ts_col]).dt.hour

    rows = []
    for (station, year), group in work.groupby([station_col, "_year"], sort=True):
        rh_max = float(group["rh"].max()) if "rh" in group else float("nan")
        ceiling_ok = bool(rh_max >= RH_CEILING_EXPECTED) if pd.notna(rh_max) else False

        rain_dry = 0
        if {"rain", "rh"} <= set(group.columns):
            rain_dry = int(((group["rain"] > 0) & (group["rh"] < RAIN_MIN_HUMIDITY)).sum())

        night_solar = 0
        if "solar_mj" in group.columns:
            night = group["_hour"].isin(NIGHT_HOURS)
            night_solar = int((night & (group["solar_mj"] > NIGHT_MAX_SOLAR_MJ)).sum())

        null_fill = 0
        present = [c for c in NULL_FILL_CHANNELS if c in group.columns]
        if len(present) >= NULL_FILL_MIN_CHANNELS:
            zeros = sum((group[c] == 0.0).fillna(False).astype(int) for c in present)
            null_fill = int((zeros >= NULL_FILL_MIN_CHANNELS).sum())

        stuck_ratio = 0.0
        if "t_air" in group.columns:
            series = group["t_air"]
            block = (series != series.shift()).cumsum()
            stuck_ratio = float((series.groupby(block).transform("size") >= STUCK_HOURS).mean())

        problems = []
        if not ceiling_ok:
            problems.append(f"습도 상한 포화(최대 {rh_max:.1f}%)")
        if rain_dry > 0:
            problems.append(f"가짜 강수 {rain_dry}건")
        if night_solar > len(group) * 0.01:
            problems.append(f"야간 일사 드리프트 {night_solar}건")
        if stuck_ratio > 0.05:
            problems.append(f"고정값 {stuck_ratio*100:.0f}%")
        if null_fill > 0:
            problems.append(f"영점 채움 {null_fill}건")

        rows.append({
            "station": station,
            "year": int(year),
            "n_hours": len(group),
            "rh_max": round(rh_max, 1) if pd.notna(rh_max) else None,
            "rh_ceiling_ok": ceiling_ok,
            "rain_dry_hours": rain_dry,
            "night_solar_hours": night_solar,
            "stuck_ratio": round(stuck_ratio, 3),
            "null_fill_hours": null_fill,
            "verdict": "정상" if not problems else " / ".join(problems),
        })
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, ts_col: str = "ts") -> tuple[pd.DataFrame, QCReport]:
    """전체 품질관리를 수행하고 플래그가 붙은 데이터와 요약을 돌려준다."""
    flags = pd.concat(
        [
            flag_range(df), flag_spike(df), flag_stuck(df),
            flag_null_fill(df), flag_inconsistent(df, ts_col),
        ],
        axis=1,
    )
    flagged = pd.concat([df, flags], axis=1)
    flagged["qc_any"] = flags.any(axis=1)

    duplicates, gaps = check_index(df, ts_col) if ts_col in df.columns else (0, 0)
    report = QCReport(
        n_rows=len(df),
        flagged={c: int(flags[c].sum()) for c in flags.columns if flags[c].any()},
        duplicates=duplicates,
        gaps=gaps,
    )
    return flagged, report


def apply_mask(df: pd.DataFrame) -> pd.DataFrame:
    """플래그가 붙은 값을 결측으로 바꾼 사본을 만든다.

    원본 보존 원칙에 따라 입력은 수정하지 않는다. 범위이탈·급변·교차모순만
    지우고 고정값은 지우지 않는다(장기 편차는 보정 대상이지 삭제 대상이 아니다).
    """
    out = df.copy()
    for col in VALID_RANGE:
        marks = [f"qc_range_{col}", f"qc_spike_{col}"]
        present = [m for m in marks if m in out.columns]
        if col in out.columns and present:
            bad = out[present].any(axis=1)
            out.loc[bad, col] = np.nan

    # 영점 채움 구간은 관측이 없는 것이다. 해당 채널을 전부 지운다.
    null_fill = out.get("qc_null_fill")
    if null_fill is not None:
        rows = null_fill.fillna(False)
        for col in (*NULL_FILL_CHANNELS, "rain", "wind_speed", "t_max", "t_min"):
            if col in out.columns:
                out.loc[rows, col] = np.nan
    rh_zero = out.get("qc_rh_zero")
    if rh_zero is not None and "rh" in out.columns:
        out.loc[rh_zero.fillna(False), "rh"] = np.nan

    # 교차모순은 모순을 일으킨 쪽 변수를 바로잡는다.
    # 가짜 강수는 참값을 알 수 없으므로 지우고, 야간 일사는 참값이 0 이므로
    # 0 으로 되돌린다(지우면 야간을 보는 서리·병해 판단에 구멍이 생긴다).
    if "qc_cross_rain_dry" in out.columns and "rain" in out.columns:
        out.loc[out["qc_cross_rain_dry"].fillna(False), "rain"] = np.nan
    if "qc_cross_night_solar" in out.columns and "solar_mj" in out.columns:
        out.loc[out["qc_cross_night_solar"].fillna(False), "solar_mj"] = 0.0
    return out
