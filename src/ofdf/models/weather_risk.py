"""기상 기반 농지위험 판단 모델.

위험유형 x 예측시계(1시간/3시간)마다 LightGBM 다중분류기를 하나씩 학습해
정상/주의/경계를 판정한다. 사업계획서가 '현장 적용이 쉬운 모델 우선'으로
LightGBM·XGBoost를 1순위 후보로 둔 설계를 따른다.

위험사례가 전체의 몇 %뿐이라 클래스 가중치를 넣고, 성능은 정확도가 아니라
위험유형별 Macro F1과 위험 놓침 방지율로 본다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None


DEFAULT_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,
}


@dataclass
class TrainedRisk:
    """위험유형 x 시계 하나에 대한 학습 결과."""

    risk: str
    horizon: int
    booster: object
    columns: list[str]
    best_iteration: int
    class_weight: dict[int, float] = field(default_factory=dict)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return proba.argmax(axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(
            X[self.columns], num_iteration=self.best_iteration or None
        )


def group_keys(meta: pd.DataFrame, events: pd.Series) -> pd.Series:
    """누수 없는 분할을 위한 그룹 키를 만든다.

    사업계획서의 '학습·검증·시험 데이터는 날짜가 아닌 개별 기상사례 단위로
    분리' 규칙이다. 위험 구간은 이벤트 번호로, 평상 구간은 (지점, 날짜)
    블록으로 묶는다. 같은 기상사례가 학습셋과 시험셋에 쪼개져 들어가면
    성능이 부풀려진다.
    """
    normal_key = meta["station"] + "#평상#" + meta["ts"].dt.date.astype(str)
    return events.where(events.notna(), normal_key)


def shared_group_keys(
    meta: pd.DataFrame,
    states: pd.DataFrame,
    *,
    risks: list[str] | None = None,
    gap_hours: int = 12,
) -> pd.Series:
    """위험유형 전체에 공통으로 쓸 그룹 키를 만든다.

    위험유형마다 따로 분할하면 유형별 시험셋이 서로 달라서, 복합위험처럼
    여러 유형의 예측을 조합해야 하는 산출을 만들 수 없다. 그래서 어느
    유형이든 위험이 걸린 구간을 **합집합**으로 묶어 하나의 기상사례
    블록으로 보고, 그 블록 단위로 나눈다.

    블록 앞뒤 ``gap_hours`` 시간은 같은 블록에 포함시킨다. 사건 직전·직후의
    평상 시간이 다른 분할로 새어 나가면 선행 신호가 누수되기 때문이다.
    """
    from ofdf.labels.risk import RISK_TYPES

    targets = risks or [r for r in RISK_TYPES if r in states.columns]
    keys = pd.Series(index=meta.index, dtype=object)

    for station, rows in meta.groupby("station", sort=False).groups.items():
        active = (states.loc[rows, targets] > 0).any(axis=1)

        # 사건 전후 gap_hours 를 같은 블록으로 끌어들인다(팽창)
        padded = (
            active.rolling(2 * gap_hours + 1, center=True, min_periods=1).max().astype(bool)
        )
        block = (padded & ~padded.shift(fill_value=False)).cumsum()

        dates = meta.loc[rows, "ts"].dt.date.astype(str)
        keys.loc[rows] = np.where(
            padded.to_numpy(),
            station + "#사례" + block.astype(str),
            station + "#평상" + dates,
        )
    return keys


def split_by_event(
    groups: pd.Series, *, test_size: float = 0.2, valid_size: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """그룹 단위로 학습/검증/시험 인덱스를 나눈다."""
    idx = np.arange(len(groups))
    g = groups.to_numpy()

    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(outer.split(idx, groups=g))

    inner = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
    train_rel, valid_rel = next(inner.split(rest_idx, groups=g[rest_idx]))
    return rest_idx[train_rel], rest_idx[valid_rel], test_idx


def balanced_weights(y: np.ndarray) -> dict[int, float]:
    """클래스 가중치 — 드문 위험등급에 비중을 준다."""
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    return {int(c): float(total / (len(classes) * n)) for c, n in zip(classes, counts)}


def train_one(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    *,
    risk: str,
    horizon: int,
    params: dict | None = None,
    num_boost_round: int = 600,
    early_stopping_rounds: int = 50,
) -> TrainedRisk:
    """위험유형 하나, 시계 하나에 대한 모델을 학습한다."""
    if lgb is None:
        raise ImportError("lightgbm 이 필요합니다: pip install lightgbm")

    columns = list(X.columns)
    weights = balanced_weights(y[train_idx])

    settings = {**DEFAULT_PARAMS, **(params or {})}
    # 실제로 나타난 클래스 수에 맞춘다(드문 등급이 아예 없는 분할 대비)
    settings["num_class"] = 3

    train_set = lgb.Dataset(
        X.iloc[train_idx][columns],
        label=y[train_idx],
        weight=np.array([weights.get(int(v), 1.0) for v in y[train_idx]]),
    )
    valid_set = lgb.Dataset(
        X.iloc[valid_idx][columns],
        label=y[valid_idx],
        weight=np.array([weights.get(int(v), 1.0) for v in y[valid_idx]]),
        reference=train_set,
    )

    booster = lgb.train(
        settings,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    return TrainedRisk(
        risk=risk,
        horizon=horizon,
        booster=booster,
        columns=columns,
        best_iteration=booster.best_iteration,
        class_weight=weights,
    )


def feature_importance(model: TrainedRisk, top: int = 15) -> pd.DataFrame:
    """판단근거 화면에 올릴 주요 영향변수를 뽑는다.

    사업계획서는 '주요 판단근거 3개 이상 화면·로그 표시'를 요구한다.
    전역 중요도는 모델 설명서용이고, 개별 판단의 근거는 SHAP 등으로
    따로 뽑는다(:func:`explain_one`).
    """
    gain = model.booster.feature_importance(importance_type="gain")
    return (
        pd.DataFrame({"feature": model.columns, "gain": gain})
        .sort_values("gain", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )


def explain_one(model: TrainedRisk, X: pd.DataFrame, row: int, top: int = 3) -> pd.DataFrame:
    """개별 판단의 주요 근거를 기여도 순으로 뽑는다.

    LightGBM 의 ``pred_contrib`` 로 각 변수가 예측 확률에 얼마나 기여했는지
    계산한다(SHAP 값). 예측된 등급 쪽 기여도 상위 ``top`` 개를 돌려준다.
    """
    contrib = model.booster.predict(
        X.iloc[[row]][model.columns], num_iteration=model.best_iteration or None,
        pred_contrib=True,
    )
    n_features = len(model.columns)
    predicted = int(model.predict(X.iloc[[row]])[0])

    # pred_contrib 은 클래스마다 (n_features + 1) 길이로 이어 붙어 나온다
    start = predicted * (n_features + 1)
    values = np.asarray(contrib).reshape(-1)[start : start + n_features]

    frame = pd.DataFrame(
        {
            "feature": model.columns,
            "contribution": values,
            "value": X.iloc[row][model.columns].to_numpy(),
        }
    )
    frame["abs"] = frame["contribution"].abs()
    return frame.sort_values("abs", ascending=False).head(top).drop(columns="abs").reset_index(drop=True)


def save(model: TrainedRisk, path: str | Path) -> None:
    """모델을 텍스트 포맷으로 저장한다(버전 관리·재현 목적)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.booster.save_model(str(path), num_iteration=model.best_iteration or None)
