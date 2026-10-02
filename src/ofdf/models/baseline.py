"""비교 기준 모델.

AI가 실제로 나아졌는지 보려면 같은 시험데이터에서 '지금 쓰는 방식'과
비교해야 한다. 사업계획서의 비교 기준은 두 가지다.

* 단순 기준값 규칙 — 센서값이 임계를 넘으면 작동하는 사후 대응 방식
* 로지스틱 회귀 — 현재값만 보는 단순 통계 모델
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#: 로지스틱 회귀가 보는 '현재값' 변수. 구간 통계·변화율은 주지 않는다.
CURRENT_VALUE_FEATURES = [
    "t_air", "rh", "rain", "solar_w", "wind_speed",
    "vpd", "dew_point", "canopy_temp", "soil_index", "leaf_wetness",
]


class RuleBaseline:
    """단순 임계값 규칙.

    '지금 이 순간의 상태등급'을 그대로 미래 예측으로 내놓는다. 임계를
    넘어야 비로소 작동하므로 구조적으로 선행 대응이 불가능하다. 이것이
    AI가 넘어야 할 기준선이다.
    """

    name = "단순 기준값 규칙"

    def fit(self, *_args, **_kwargs) -> "RuleBaseline":
        return self

    def predict(self, current_state: pd.Series) -> np.ndarray:
        return np.asarray(current_state, dtype=int)


class LogisticBaseline:
    """현재값만 쓰는 다항 로지스틱 회귀."""

    name = "로지스틱 회귀(현재값)"

    def __init__(self, **kwargs):
        self.pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    # scikit-learn 1.5 이후 다중분류는 multinomial 이 기본이라
                    # multi_class 인자를 따로 주지 않는다(1.9에서 제거됨).
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        **kwargs,
                    ),
                ),
            ]
        )
        self.columns: list[str] = []

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogisticBaseline":
        self.columns = [c for c in CURRENT_VALUE_FEATURES if c in X.columns]
        self.pipeline.fit(X[self.columns].fillna(0.0), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X[self.columns].fillna(0.0))
