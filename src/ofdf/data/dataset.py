"""학습용 데이터셋 조립.

농업기상 시간자료 + AgERA5 일자료 -> 파생변수 -> 피처/라벨 테이블.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ofdf.data import aaos, agera5, quality
from ofdf.features.derived import add_derived
from ofdf.features.weather import build_features
from ofdf.labels.risk import EVENT_GAP_HOURS, RiskThresholds, build_states, event_ids, future_labels

#: 풍속이 기록되는 지점. 나머지 지점은 이 값을 지역 대표값으로 빌려 쓴다.
WIND_SOURCE_STATIONS = ("나주시 금천면", "나주시 산포면")

#: 노지 대파 실증 대상 기간(월). 사업계획서의 '9~11월 복합 기상위험',
#: '10~11월 저온·서리' 구간이다. 겨울을 포함해 학습하면 한겨울 서리가
#: 라벨을 지배해(1월 야간 대부분이 주의) 가을 서리 신호가 묻힌다.
CROP_SEASON_MONTHS = (9, 10, 11)


@dataclass
class Dataset:
    features: pd.DataFrame
    labels: pd.DataFrame
    states: pd.DataFrame
    events: pd.DataFrame
    meta: pd.DataFrame
    qc: "quality.QCReport | None" = None

    def __len__(self) -> int:
        return len(self.features)


def _regional_wind(hourly: pd.DataFrame) -> pd.Series:
    """풍속 관측지점들의 시간별 평균을 지역 대표 풍속으로 만든다."""
    source = hourly[hourly["station"].isin(WIND_SOURCE_STATIONS)]
    return source.groupby("ts")["wind_speed"].mean()


def build(
    aaos_dir: str | Path,
    agera5_csv: str | Path,
    *,
    horizons: tuple[int, ...] = (1, 3, 6),
    thresholds: RiskThresholds | None = None,
    stations: tuple[str, ...] | None = None,
    season_months: tuple[int, ...] | None = CROP_SEASON_MONTHS,
) -> Dataset:
    """원본 파일에서 학습 가능한 데이터셋을 만든다.

    Parameters
    ----------
    season_months
        학습·평가 대상 월. 피처와 라벨은 연중 전체로 계산한 뒤 마지막에
        걸러낸다. 그래야 9월 1일의 24시간 구간 통계가 8월 말 관측값을
        제대로 쓴다. ``None`` 이면 연중 전체를 쓴다.
    """
    hourly = aaos.load_directory(aaos_dir)
    daily = agera5.read_daily(agera5_csv)

    # 품질관리: 범위이탈·급변을 결측 처리한다(원본은 그대로 둔다)
    flagged, qc_report = quality.run(hourly)
    hourly = quality.apply_mask(flagged)

    wind = _regional_wind(hourly)
    targets = stations or tuple(hourly["station"].unique())

    feature_parts, label_parts, state_parts, event_parts, meta_parts = [], [], [], [], []

    for station in targets:
        one = (
            hourly[hourly["station"] == station]
            .drop_duplicates(subset="ts")
            .set_index("ts")
            .sort_index()
        )
        # 시간 격자를 메워 rolling 구간이 실제 시간을 뜻하도록 맞춘다
        one = one.reindex(pd.date_range(one.index.min(), one.index.max(), freq="h"))
        one["station"] = station

        # 풍속 결측 지점은 지역 대표 풍속으로 채운다
        one["wind_speed"] = one["wind_speed"].fillna(wind.reindex(one.index))

        # 일자료 병합(그날 값을 하루 전체에 깔아 준다)
        day = one.index.normalize()
        for col in ("et0", "cloud_cover", "t_min_night", "rh_max_daily"):
            if col in daily.columns:
                one[col] = pd.Series(day, index=one.index).map(daily.set_index("date")[col])
        one["et0"] = one["et0"] / 24.0  # 일 총량을 시간 평균으로

        one = add_derived(one)

        states = build_states(one, thresholds)
        labels = future_labels(states, horizons)
        features = build_features(one)
        # 이벤트 번호는 지점 단위로 매겨지므로 지점명을 붙여 전역 고유화한다
        events = pd.DataFrame(
            {
                f"{r}_event": event_ids(states[r], EVENT_GAP_HOURS[r]).map(
                    lambda i, st=station: f"{st}#{i}" if i >= 0 else None
                )
                for r in states.columns
            },
            index=states.index,
        )
        meta = pd.DataFrame({"station": station, "ts": one.index}, index=one.index)

        feature_parts.append(features)
        label_parts.append(labels)
        state_parts.append(states)
        event_parts.append(events)
        meta_parts.append(meta)

    dataset = Dataset(
        qc=qc_report,
        features=pd.concat(feature_parts),
        labels=pd.concat(label_parts),
        states=pd.concat(state_parts),
        events=pd.concat(event_parts),
        meta=pd.concat(meta_parts),
    )
    if season_months:
        dataset = filter_season(dataset, season_months)
    return dataset


def filter_season(dataset: Dataset, months: tuple[int, ...]) -> Dataset:
    """대상 월만 남긴다. 구간 통계는 이미 계산된 뒤이므로 경계가 끊기지 않는다."""
    keep = dataset.meta["ts"].dt.month.isin(months).to_numpy()
    return Dataset(
        features=dataset.features[keep],
        labels=dataset.labels[keep],
        states=dataset.states[keep],
        events=dataset.events[keep],
        meta=dataset.meta[keep],
        qc=dataset.qc,
    )
