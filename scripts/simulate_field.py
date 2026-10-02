"""실증포장 현장 센서 수집 데이터 모사.

실제 현장 수집이 시작되기 전에 **수집 형식과 보정 파이프라인을 먼저
검증하기 위한** 도구다. 나주 농업기상 관측값을 바탕으로, 사업계획서
'착수 설계 확정 사양 ② 센서·통신·전장' 의 센서 구성대로 1분 간격
수집 기록을 만든다.

관측소 값에 다음을 입혀 현장값을 만든다.

* **농지 미기상** — 실재하는 차이. 보정으로 지우면 안 되는 신호.
  캐노피 하부는 주간에 덜 뜨겁고 야간에 더 식으며, 습도는 높다.
  초관부(지상 10cm)는 맑고 바람 약한 밤에 기온보다 뚜렷이 낮다.
* **센서 편차** — 계기 치우침. 보정으로 지워야 할 잡음.
* **측정점 간 산포** — 같은 구간 2점 사이의 자연스러운 차이.
* **센서 고장** — 고정값·범위이탈·결측. 품질관리가 잡아내는지 확인용.

실제 수집이 시작되면 이 스크립트 대신 수집기 출력을 쓰면 되고,
형식이 같으므로 뒷단은 바꿀 필요가 없다.

사용 예::

    python scripts/simulate_field.py \
        --aaos-dir data/raw/aaos --agera5 data/raw/agera5.csv \
        --station '나주시 봉황면' --start 2025-10-01 --end 2025-11-20 \
        --out data/field/sim_field.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf.data import aaos, agera5  # noqa: E402
from ofdf.data.field import ZONE_CONTROL, ZONE_FACILITY, ZONE_MAST, ZONE_TREATMENT  # noqa: E402
from ofdf.features.derived import add_derived  # noqa: E402

#: 계기 편차 — 보정이 찾아내야 할 값. (변수, 구간) -> 더할 양
SENSOR_BIAS = {
    ("t_air", ZONE_MAST): -0.4,
    ("rh", ZONE_MAST): +2.5,
    ("solar", ZONE_MAST): -18.0,
    ("t_air", ZONE_TREATMENT): -0.7,
    ("t_air", ZONE_CONTROL): -0.6,
    ("rh", ZONE_TREATMENT): +3.0,
    ("rh", ZONE_CONTROL): +2.8,
}

#: 측정점 간 산포(표준편차)
POINT_NOISE = {
    "t_air": 0.25, "rh": 1.5, "canopy_temp": 0.4,
    "leaf_wetness": 0.08, "soil_moisture": 1.2, "soil_temp": 0.3,
}

#: 구간별 설치 측정점 수 (사업계획서: 적용구간·비교구간 각 2점)
POINT_COUNT = {
    ZONE_MAST: {"t_air": 1, "rh": 1, "solar": 1, "rain": 1, "wind_speed": 1,
                "wind_gust_3s": 1, "soil_moisture": 1, "soil_temp": 1, "soil_ec": 1},
    ZONE_TREATMENT: {"t_air": 2, "rh": 2, "canopy_temp": 2, "leaf_wetness": 2,
                     "soil_moisture": 2, "soil_temp": 2, "rain_detect": 1, "illuminance": 1},
    ZONE_CONTROL: {"t_air": 2, "rh": 2, "canopy_temp": 2, "leaf_wetness": 2,
                   "soil_moisture": 2, "soil_temp": 2, "rain_detect": 1},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--aaos-dir", required=True)
    p.add_argument("--agera5", required=True)
    p.add_argument("--station", default="나주시 봉황면", help="기준으로 삼을 관측지점")
    p.add_argument("--start", default="2025-10-01")
    p.add_argument("--end", default="2025-11-20")
    p.add_argument("--out", required=True, help="출력 CSV (세로 형식)")
    p.add_argument("--interval", default="1min", help="수집 주기")
    p.add_argument("--fault-rate", type=float, default=0.002, help="센서 고장 발생 비율")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def station_hourly(args: argparse.Namespace) -> pd.DataFrame:
    """관측소 시간자료를 읽어 파생변수까지 붙인다."""
    hourly = aaos.load_directory(args.aaos_dir)
    daily = agera5.read_daily(args.agera5)

    wind = hourly[hourly["station"].isin(["나주시 금천면", "나주시 산포면"])]
    wind = wind.groupby("ts")["wind_speed"].mean()

    one = hourly[hourly["station"] == args.station].drop_duplicates("ts").set_index("ts").sort_index()
    one = one.loc[args.start : args.end]
    one["wind_speed"] = one["wind_speed"].fillna(wind.reindex(one.index))

    day = one.index.normalize()
    for column in ("et0", "cloud_cover"):
        if column in daily.columns:
            one[column] = pd.Series(day, index=one.index).map(daily.set_index("date")[column])
    one["et0"] = one["et0"] / 24.0
    return add_derived(one)


def microclimate(frame: pd.DataFrame, zone: str, rng: np.random.Generator) -> pd.DataFrame:
    """관측소 값에 실재하는 농지 미기상 차이를 입힌다.

    적용구간은 차광·관수 시설이 있어 비교구간보다 주간 기온이 낮고
    습도가 높다. 두 구간 모두 캐노피 하부라 관측소(노출 1.5m)보다
    주간에 덜 뜨겁고 야간에 더 식는다.
    """
    out = pd.DataFrame(index=frame.index)
    is_day = frame["solar_w"] > 50

    shelter = 0.6 if zone == ZONE_TREATMENT else 0.3      # 시설에 의한 주간 냉각
    out["t_air"] = frame["t_air"] - np.where(is_day, shelter, -0.5)
    out["rh"] = (frame["rh"] + np.where(is_day, 2.0, 4.0)).clip(0, 100)

    # 초관부는 복사냉각을 더 받는다. 관측소 기온이 아니라 **그 구간 기온**에서
    # 빼야 한다. 구간 기온만 식히고 초관부는 관측소 기준으로 두면 주간에
    # 초관부가 기온보다 높아지는 모순이 생긴다.
    cooling = (frame["t_air"] - frame["canopy_temp"]).clip(lower=0.0)
    extra_cooling = np.where(frame["solar_w"] < 20, 0.8, 0.0)
    out["canopy_temp"] = out["t_air"] - cooling - extra_cooling

    # 엽면습윤 실측센서는 0~1 연속값이다. 젖으면 0.8~1.0, 마르면 0.02~0.08 의
    # 바닥 잡음이 올라온다. 라벨·피처는 LEAF_WETNESS_THRESHOLD 로 이분한다.
    wet = frame["leaf_wetness"].to_numpy()
    if zone == ZONE_TREATMENT:
        wet = wet * (rng.random(len(wet)) > 0.12)
    out["leaf_wetness"] = np.where(
        wet > 0.5,
        np.clip(0.90 + rng.normal(0, 0.04, len(wet)), 0, 1),
        np.clip(0.05 + rng.normal(0, 0.015, len(wet)), 0, 1),
    )

    # 체적수분율(%) — 물수지 지수와 척도가 다르다(보정 시 분위수 정합 필요)
    out["soil_moisture"] = 12.0 + frame["soil_index"] * 0.28
    out["soil_temp"] = frame["t_air"].rolling(24, min_periods=1).mean() + 0.8
    out["soil_ec"] = 0.8 + rng.normal(0, 0.05, len(frame))
    out["rain_detect"] = (frame["rain"] > 0).astype(float)
    out["illuminance"] = frame["solar_w"] * 110.0
    return out


def mast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """기상 마스트(H 5.0m) 채널."""
    out = pd.DataFrame(index=frame.index)
    out["t_air"] = frame["t_air"]
    out["rh"] = frame["rh"]
    out["solar"] = frame["solar_w"]
    out["rain"] = frame["rain"]
    out["wind_speed"] = frame["wind_speed"]
    # 3초 순간최대풍속 — 시간평균의 1.6~2.2배로 모사한다.
    # 공공자료에는 없는 채널이고, 강풍 안전규칙이 쓰는 값이다.
    gust_ratio = 1.6 + 0.6 * np.clip(frame["wind_speed"] / 6.0, 0, 1)
    out["wind_gust_3s"] = frame["wind_speed"] * gust_ratio
    out["soil_moisture"] = 12.0 + frame["soil_index"] * 0.28
    out["soil_temp"] = frame["t_air"].rolling(24, min_periods=1).mean() + 0.8
    out["soil_ec"] = 0.85
    return out


def to_minute(frame: pd.DataFrame, interval: str, rng: np.random.Generator) -> pd.DataFrame:
    """시간 값을 수집 주기로 보간한다(완만한 변수는 선형, 강수는 분배)."""
    index = pd.date_range(frame.index.min(), frame.index.max(), freq=interval)
    out = frame.reindex(frame.index.union(index)).interpolate("time").reindex(index)

    if "rain" in frame.columns:
        per_hour = frame["rain"].reindex(index, method="ffill")
        steps = pd.Series(index, index=index).dt.floor("h").map(
            pd.Series(index, index=index).dt.floor("h").value_counts()
        )
        out["rain"] = (per_hour / steps).fillna(0.0)
    return out


def emit(
    frame: pd.DataFrame, zone: str, counts: dict[str, int],
    rng: np.random.Generator, fault_rate: float,
) -> pd.DataFrame:
    """구간 하나를 세로 형식 기록으로 편다."""
    records = []
    for variable, n_points in counts.items():
        if variable not in frame.columns:
            continue
        base = frame[variable].to_numpy(dtype=float)
        bias = SENSOR_BIAS.get((variable, zone), 0.0)

        for point in range(1, n_points + 1):
            noise = rng.normal(0, POINT_NOISE.get(variable, 0.0), len(base))
            values = base + bias + noise
            quality = np.full(len(base), "ok", dtype=object)

            if fault_rate > 0:
                # 고정값 — 센서가 멈춰 같은 값이 이어진다
                if rng.random() < 0.25:
                    start = rng.integers(0, max(1, len(values) - 180))
                    values[start : start + 180] = values[start]
                # 범위이탈 — 통신 오류로 터무니없는 값이 섞인다
                broken = rng.random(len(values)) < fault_rate
                values[broken] = -999.0
                quality[broken] = "bad"
                # 결측 — 무선 구간 유실
                dropped = rng.random(len(values)) < fault_rate
                values[dropped] = np.nan

            records.append(pd.DataFrame({
                "ts": frame.index,
                "zone": zone,
                "sensor_id": f"{zone[:2].upper()}-{variable[:4]}-{point:02d}",
                "variable": variable,
                "value": np.round(values, 3),
                "quality": quality,
            }))
    return pd.concat(records, ignore_index=True)


def facility_frame(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """시설·안전 채널 — 차광 위치, 전류, 유량, 침수, 비상정지."""
    n = len(frame)
    shade = np.zeros(n)
    # 맑고 더운 낮에만 차광이 전개된다
    deploy = (frame["solar"] > 700) & (frame["t_air"] > 28)
    shade[deploy.to_numpy()] = 1.0

    return pd.DataFrame({
        "shade_position": shade,
        "motor_current": np.where(np.abs(np.diff(shade, prepend=0)) > 0, 3.2, 0.0),
        "pump_current": np.zeros(n),
        "irrigation_flow": np.zeros(n),
        "spray_flow": np.zeros(n),
        "enclosure_flood": np.zeros(n),
        "emergency_stop": np.zeros(n),
    }, index=frame.index)


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"[관측소] {args.station} {args.start} ~ {args.end}")
    hourly = station_hourly(args)
    print(f"    {len(hourly):,} 시간")

    zones = {
        ZONE_MAST: mast_frame(hourly),
        ZONE_TREATMENT: microclimate(hourly, ZONE_TREATMENT, rng),
        ZONE_CONTROL: microclimate(hourly, ZONE_CONTROL, rng),
    }

    print(f"[모사] 수집 주기 {args.interval}, 고장률 {args.fault_rate:.3%}")
    parts = []
    for zone, frame in zones.items():
        minute = to_minute(frame, args.interval, rng)
        parts.append(emit(minute, zone, POINT_COUNT[zone], rng, args.fault_rate))

    mast_minute = to_minute(zones[ZONE_MAST], args.interval, rng)
    facility = facility_frame(mast_minute, rng)
    parts.append(emit(facility, ZONE_FACILITY, {c: 1 for c in facility.columns}, rng, 0.0))

    records = pd.concat(parts, ignore_index=True).sort_values(["ts", "zone", "variable"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records.to_csv(out_path, index=False, encoding="utf-8")

    print(f"[출력] {out_path}  {len(records):,} 행 / {out_path.stat().st_size/1e6:.1f} MB")
    print(f"    구간: {sorted(records['zone'].unique())}")
    print(f"    변수: {len(records['variable'].unique())}종")
    print(f"    품질 bad: {(records['quality'] == 'bad').sum():,} / 결측: {records['value'].isna().sum():,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
