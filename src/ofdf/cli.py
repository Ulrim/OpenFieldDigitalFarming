"""엣지 제어기 실행 진입점.

현장 제어함의 라즈베리파이에서 도는 실행 루프다. 사업계획서의 핵심 작동구조
다섯 단계를 5분 주기로 되풀이한다.

    ① 데이터 입력 → ② 데이터 정리 → ③ AI 위험판단 → ④ 조치 결정 → ⑤ 현장 실행·기록

통신이 끊겨도 혼자 돌아야 하므로, 서버를 부르지 않고 로컬에 쌓는다. 판단
이력은 JSON Lines 로 한 줄에 하나씩 적어 두었다가 복구되면 시간순으로 올린다.

사용 예::

    python -m ofdf.cli edge  --config configs/weather.yaml      # 상주 실행
    python -m ofdf.cli once  --config configs/weather.yaml      # 한 번만
"""

from __future__ import annotations

import argparse
import datetime
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ofdf import __version__
from ofdf.action.journal import Journal, RiskJudgement, record_from_decision
from ofdf.action.recommend import SensorState, decide
from ofdf.data import field as field_data
from ofdf.features.derived import add_derived
from ofdf.features.weather import build_features
from ofdf.labels.risk import RISK_LABELS_KO, compose_compound, compound_reasons

LOG = logging.getLogger("ofdf.edge")

BASE_RISKS = ["heat_dry", "rain_wet", "disease", "frost"]
LEVEL_NAMES = {0: "정상", 1: "주의", 2: "경계"}

#: 추론 주기(초). 사업계획서의 'AI 선행판단 L2 5분 주기'.
DEFAULT_PERIOD = 300

#: 피처 계산에 필요한 과거 시간. 구간 통계가 최대 24시간이라 여유를 둔다.
WINDOW_HOURS = 48

#: 예측 시계
#: 엣지가 쓰는 예측시계(시간). 짧은 것부터 적는다 — 예상시점을 '위험이
#: 처음 넘어서는 시계'로 잡기 때문에 순서가 뜻을 가진다.
#:
#: 전에는 3시간 하나만 읽었다. 1시간 모델을 학습해 놓고 엣지가 쓰지 않아,
#: 화면이 1·3시간 두 칸을 보여주게 돼 있는데도 3시간만 채워졌고 예상시점은
#: 언제나 '지금+3시간' 이었다.
HORIZONS = (1, 3)
HORIZON = HORIZONS[-1]      # 조치 판단에 쓰는 대표 시계


@dataclass
class EdgeConfig:
    """엣지 실행 설정."""

    field_dir: Path
    model_dir: Path
    journal_path: Path
    dashboard_path: Path | None = None
    zone: str = "treatment"
    period: int = DEFAULT_PERIOD
    station: str = "실증포장"
    #: 구동부가 실제로 달린 장치. ``None`` 이면 전부 달린 것으로 본다.
    #:
    #: 1차 구매분은 관수밸브와 전원차단(스마트 차단기)만 덮는다. 차광막·
    #: 야간피복·살수 구동부는 별도 발주다. 이것을 적어 두지 않으면 엔진이
    #: '차광막 차광 한정 전개'를 **실행**으로 기록한다. 실행된 것은 없는데
    #: 제어 이력에는 남고, 그 이력이 그대로 성능평가의 제어응답 측정
    #: 대상이 된다. 구동부가 들어오면 설정에 이름을 더하면 된다.
    installed_devices: frozenset[str] | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ofdf.cli", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    for name, help_text in (("edge", "5분 주기로 상주 실행"), ("once", "한 번만 실행")):
        s = sub.add_parser(name, help=help_text)
        s.add_argument("--config", default=None, help="설정 YAML (없으면 아래 인자를 쓴다)")
        s.add_argument("--field", default="data/field", help="현장 수집 디렉터리")
        s.add_argument("--models", default="artifacts/weather")
        s.add_argument("--journal", default="logs/journal.jsonl")
        s.add_argument("--dashboard", default="dashboard/data.json",
                       help="화면 데이터 출력 경로. 빈 문자열이면 쓰지 않는다")
        s.add_argument("--zone", default="treatment")
        s.add_argument("--period", type=int, default=DEFAULT_PERIOD)
        s.add_argument("--station", default="실증포장")
        s.add_argument("--installed-devices", default=None,
                       help="구동부가 달린 장치를 쉼표로. 생략하면 설정파일, "
                            "그것도 없으면 전부 달린 것으로 본다")
        s.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def load_config(args: argparse.Namespace) -> EdgeConfig:
    """명령행 인자와 설정 파일을 합친다. 명령행이 이긴다."""
    settings: dict = {}
    if args.config and Path(args.config).exists():
        try:
            import yaml
            settings = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        except ImportError:
            LOG.warning("pyyaml 이 없어 설정 파일을 건너뛴다")

    data = settings.get("data", {})
    return EdgeConfig(
        field_dir=Path(args.field or data.get("field_dir", "data/field")),
        model_dir=Path(args.models),
        journal_path=Path(args.journal),
        dashboard_path=Path(args.dashboard) if args.dashboard else None,
        zone=args.zone,
        period=args.period,
        station=args.station,
        installed_devices=_installed_devices(args, settings),
    )


def _installed_devices(
    args: argparse.Namespace, settings: dict
) -> frozenset[str] | None:
    """구동부가 달린 장치 목록을 정한다. 명령행이 설정파일을 이긴다.

    아무 데도 적혀 있지 않으면 ``None`` 을 돌려 전부 달린 것으로 본다.
    현장에 내보낼 때는 반드시 적어야 하는 값이지만, 기본값을 '아무것도
    없다'로 두면 설정을 깜빡한 제어기가 조용히 아무것도 실행하지 않는다.
    그쪽이 더 위험하므로 기본은 '다 있다'로 두고 문서에서 못을 박는다.
    """
    raw = getattr(args, "installed_devices", None)
    if raw is None:
        raw = settings.get("edge", {}).get("installed_devices")
    if raw is None:
        return None
    names = raw.split(",") if isinstance(raw, str) else list(raw)
    return frozenset(n.strip() for n in names if n.strip())


def load_models(model_dir: Path, horizons: tuple[int, ...] = HORIZONS) -> dict:
    """위험유형별 모델을 예측시계마다 읽는다. ``{시계: {위험: 모델}}``."""
    import lightgbm as lgb

    models: dict[int, dict] = {}
    for horizon in horizons:
        bucket = {}
        for risk in BASE_RISKS:
            path = model_dir / f"model_{risk}_h{horizon}.txt"
            if path.exists():
                bucket[risk] = lgb.Booster(model_file=str(path))
            else:
                LOG.warning("모델 없음: %s", path)
        if bucket:
            models[horizon] = bucket
    return models


def read_window(config: EdgeConfig) -> pd.DataFrame | None:
    """현장 수집에서 최근 관측창을 만든다."""
    try:
        hourly = field_data.load(config.field_dir, freq="h")
    except FileNotFoundError:
        LOG.error("수집 파일이 없다: %s", config.field_dir)
        return None

    zone = field_data.zone_frame(hourly, config.zone)
    if zone.empty:
        LOG.error("구간 '%s' 자료가 비었다", config.zone)
        return None

    # 일사 단위는 add_derived 가 solar_w / solar_mj / solar 중에서 고른다
    for required in ("t_air", "rh", "rain", "wind_speed"):
        if required not in zone.columns:
            zone[required] = np.nan

    window = zone.tail(WINDOW_HOURS + 1)
    return add_derived(window)


def judge_all(models: dict, features: pd.DataFrame) -> dict[int, tuple[dict, dict, dict]]:
    """예측시계마다 판단한다. ``{시계: (등급, 신뢰도, 판단근거)}``."""
    return {h: judge(bucket, features) for h, bucket in sorted(models.items())}


def earliest_onset(
    by_horizon: dict[int, tuple[dict, dict, dict]], risk: str, now: pd.Timestamp
) -> str | None:
    """위험이 처음 주의 이상으로 올라오는 시계를 예상시점으로 돌려준다.

    1시간 모델이 이미 주의라면 한 시간 안의 일이고, 1시간은 정상인데
    3시간이 경계라면 그 사이에 온다. 전에는 시계와 무관하게 '지금+3시간'
    을 적었다 — 예상이 아니라 상수였다.
    """
    for horizon in sorted(by_horizon):
        levels, _, _ = by_horizon[horizon]
        if levels.get(risk, 0) > 0:
            return (now + pd.Timedelta(hours=horizon)).isoformat()
    return None


def judge(models: dict, features: pd.DataFrame) -> tuple[dict, dict, dict]:
    """위험등급·신뢰도·판단근거를 낸다."""
    row = features.iloc[[-1]]
    levels, confidences, evidence = {}, {}, {}

    for risk, model in models.items():
        columns = model.feature_name()
        proba = model.predict(row.reindex(columns=columns))[0]
        level = int(np.argmax(proba))
        levels[risk] = level
        confidences[risk] = float(proba[level])

        if level > 0:
            contrib = model.predict(row.reindex(columns=columns), pred_contrib=True)[0]
            n = len(columns)
            values = np.asarray(contrib)[level * (n + 1) : level * (n + 1) + n]
            order = np.argsort(-np.abs(values))[:3]
            evidence[risk] = [
                {
                    "feature": columns[i],
                    "value": float(row.iloc[0].get(columns[i], float("nan"))),
                    "contribution": float(values[i]),
                }
                for i in order
            ]

    levels["compound"] = int(compose_compound(
        [levels.get("rain_wet", 0)], [levels.get("disease", 0)], [levels.get("frost", 0)]
    )[0])
    return levels, confidences, evidence


def sensors_from(features: pd.DataFrame, now: pd.Timestamp) -> SensorState:
    """현장 즉응규칙(L1)이 보는 실측값을 모은다."""
    last = features.iloc[-1]

    def value(name: str, default: float = 0.0) -> float:
        got = last.get(name)
        return default if got is None or pd.isna(got) else float(got)

    # 3초 순간최대풍속은 풍속계가 직접 준다. 없으면 강풍 판정을 하지 않는다
    # (평균 풍속으로는 순간최대 기준을 판정할 수 없다).
    gust = last.get("wind_gust_3s")
    return SensorState(
        gust_3s=None if gust is None or pd.isna(gust) else float(gust),
        rain_detected=value("rain") > 0 or value("rain_detect") > 0.5,
        soil_above_limit=value("soil_moisture", -1) >= 0 and value("soil_moisture") >= 90,
        hour=int(now.hour),
        shade_deployed_minutes=value("shade_deployed_minutes"),
        shade_open=value("shade_position") > 0.5,
    )


def write_dashboard(path: Path, payload: dict) -> None:
    """화면이 읽을 파일을 통째로 바꿔 쓴다(읽는 중 깨지지 않게 임시파일 경유)."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def run_once(config: EdgeConfig, models: dict, journal: Journal) -> bool:
    """한 주기를 수행한다. 성공하면 True."""
    observations = read_window(config)
    if observations is None or observations.empty:
        return False

    features = build_features(observations)
    now = pd.Timestamp(features.index[-1])

    by_horizon = judge_all(models, features)
    # 조치는 대표 시계(가장 먼 것)로 정한다. 더 멀리 보는 쪽이 선행시간을
    # 확보하고, 짧은 시계는 예상시점을 좁히는 데 쓴다.
    levels, confidences, evidence = by_horizon[max(by_horizon)]
    reasons = compound_reasons(
        levels.get("rain_wet", 0), levels.get("disease", 0), levels.get("frost", 0)
    )
    sensors = sensors_from(observations, now)
    decision = decide(levels, sensors, confidence=confidences, compound_reasons=reasons,
                      installed_devices=config.installed_devices)

    judgements = [
        RiskJudgement(
            risk_type=risk, horizon_hours=HORIZON, level=level,
            level_name=LEVEL_NAMES[level], confidence=confidences.get(risk, 1.0),
            expected_onset=earliest_onset(by_horizon, risk, now),
            evidence=evidence.get(risk, []),
        )
        for risk, level in levels.items() if level > 0
    ]
    journal.append(record_from_decision(
        now.to_pydatetime(), config.station, f"ofdf-weather-{__version__}",
        {k: float(observations.iloc[-1][k])
         for k in ("t_air", "rh", "canopy_temp", "soil_index")
         if k in observations.columns and pd.notna(observations.iloc[-1][k])},
        judgements, decision,
    ))

    active = [f"{RISK_LABELS_KO[r]} {LEVEL_NAMES[l]}" for r, l in levels.items() if l > 0]
    LOG.info(
        "%s · %s · 조치 %d건(거부 %d) · %s",
        now.strftime("%m-%d %H시"), ", ".join(active) or "전 유형 정상",
        len(decision.executable()),
        sum(1 for a in decision.actions if a.rejected_by), decision.mode,
    )

    if config.dashboard_path:
        write_dashboard(config.dashboard_path, {
            "generated": datetime.datetime.now().isoformat(),
            "station": config.station, "now": now.isoformat(),
            "modelVersion": f"ofdf-weather-{__version__}", "mode": decision.mode,
            "environment": [
                {"key": k, "name": k, "unit": "",
                 "value": round(float(observations.iloc[-1][k]), 2)}
                for k in ("t_air", "canopy_temp", "rh", "leaf_wetness",
                          "solar_w", "rain", "soil_index", "wind_speed")
                if k in observations.columns and pd.notna(observations.iloc[-1][k])
            ],
            "cards": [
                {"risk": r, "name": RISK_LABELS_KO[r],
                 "horizons": {
                     str(h): {
                         "level": by_horizon[h][0].get(r, 0),
                         "levelName": LEVEL_NAMES[by_horizon[h][0].get(r, 0)],
                         "confidence": round(by_horizon[h][1].get(r, 1.0), 3),
                     }
                     for h in sorted(by_horizon) if r in by_horizon[h][0]
                 }}
                for r, l in levels.items()
            ],
            "judgements": [], "actions": [
                {"device": a.device, "command": a.command, "layer": int(a.layer),
                 "reason": a.reason, "riskType": a.risk_type,
                 "executed": not a.rejected_by and not a.advisory,
                 "rejectedBy": a.rejected_by, "advisory": a.advisory}
                for a in decision.actions
            ],
            "history": [], "series": {"time": [], "t_air": [], "canopy_temp": []},
        })
    return True


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args)

    models = load_models(config.model_dir)
    if not models:
        LOG.error("모델이 하나도 없다: %s", config.model_dir)
        return 1
    LOG.info("ofdf %s · 모델 %d개 · 구간 '%s' · 주기 %d초",
             __version__, len(models), config.zone, config.period)

    journal = Journal(config.journal_path)

    if args.command == "once":
        return 0 if run_once(config, models, journal) else 1

    stopping = False

    def stop(signum, _frame):
        nonlocal stopping
        LOG.info("신호 %d 수신 — 이번 주기를 마치고 멈춘다", signum)
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    while not stopping:
        started = time.monotonic()
        try:
            run_once(config, models, journal)
        except Exception:                                  # noqa: BLE001
            # 한 주기가 실패해도 멈추지 않는다. 통신이 끊겨도 혼자 돌아야 한다.
            LOG.exception("주기 수행 실패 — 다음 주기에 다시 시도한다")

        remaining = config.period - (time.monotonic() - started)
        while remaining > 0 and not stopping:
            time.sleep(min(1.0, remaining))
            remaining -= 1.0
    LOG.info("정지")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
