"""카메라 비전 기반 대파 병해 진단 모델.

입력은 라벨의 병반 위치(bbox)로 잘라낸 잎 영역이고, 출력은 두 가지다.

* 병해 종류 — 정상 / 질병코드 16 / 17 / 18
* 심각도    — 정상(0) / 초기(1) / 중기(2) / 말기(3)

두 가지 경로를 지원한다.

``descriptor``
    색·질감 기술자 + LightGBM. 사전학습 가중치가 필요 없어 폐쇄망이나
    가중치 호스트가 막힌 환경에서도 학습·추론이 된다.
``cnn``
    torchvision 백본 + 분류 헤드. 사전학습 가중치를 쓸 수 있으면 성능이
    더 좋고, 전체 5만 장 학습 시 이 경로를 쓴다.

분할은 **촬영 세션 단위**로 한다. 같은 포장에서 같은 날 찍은 사진이
학습셋과 시험셋에 갈라져 들어가면 성능이 부풀려진다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ofdf.features.image import describe

#: 병해 분류에 쓰는 질병코드 순서(모델 출력 인덱스 = 이 리스트의 위치)
DISEASE_CLASSES = [0, 16, 17, 18]

#: 심각도 등급
RISK_CLASSES = [0, 1, 2, 3]


def load_crop(row: pd.Series, size: int = 224, pad: float = 0.08) -> np.ndarray:
    """라벨 bbox 로 병반 영역을 잘라 정사각형으로 맞춘다.

    bbox 를 ``pad`` 비율만큼 넓혀 병반 주변 건전조직을 함께 담는다.
    병반만 보면 '얼마나 번졌는가'(심각도)를 알 수 없기 때문이다.
    """
    image = Image.open(row["path"]).convert("RGB")
    width, height = image.size

    if row.get("has_bbox") and pd.notna(row.get("xtl")):
        x0, y0 = float(row["xtl"]), float(row["ytl"])
        x1, y1 = float(row["xbr"]), float(row["ybr"])
        mx, my = (x1 - x0) * pad, (y1 - y0) * pad
        box = (
            max(0, int(x0 - mx)), max(0, int(y0 - my)),
            min(width, int(x1 + mx)), min(height, int(y1 + my)),
        )
        if box[2] > box[0] and box[3] > box[1]:
            image = image.crop(box)

    image = image.resize((size, size), Image.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def precompute_crops(
    index: pd.DataFrame, cache_path: str | Path, size: int = 224, verbose: int = 500
) -> np.ndarray:
    """모든 이미지를 잘라 배열로 캐시한다.

    원본이 최대 4032x3024 라 매 에폭 디코딩하면 CPU 학습이 디스크에 묶인다.
    한 번만 잘라 두고 재사용한다.
    """
    cache = Path(cache_path)
    if cache.exists():
        return np.load(cache, mmap_mode="r")

    cache.parent.mkdir(parents=True, exist_ok=True)
    # uint8 로 저장한다. float32 로 두면 3,742장 x 224px 만 2.2GB 를 먹는다.
    out = np.zeros((len(index), size, size, 3), dtype=np.uint8)
    for i, (_, row) in enumerate(index.iterrows()):
        out[i] = (load_crop(row, size=size) * 255).astype(np.uint8)
        if verbose and i % verbose == 0:
            print(f"    전처리 {i:,}/{len(index):,}")
    np.save(cache, out)
    return np.load(cache, mmap_mode="r")


def descriptor_matrix(crops: np.ndarray, verbose: int = 500) -> pd.DataFrame:
    """잘라낸 영상 배열(uint8)에서 색·질감 기술자 행렬을 만든다."""
    rows = []
    for i in range(len(crops)):
        rows.append(describe(np.asarray(crops[i], dtype=np.float32) / 255.0))
        if verbose and i % verbose == 0:
            print(f"    기술자 {i:,}/{len(crops):,}")
    return pd.DataFrame(rows)


def geometry_features(index: pd.DataFrame) -> pd.DataFrame:
    """병반 bbox 의 크기·위치 특징.

    심각도(초기/중기/말기)는 '병반이 전체에서 얼마나 번졌는가'인데,
    bbox 로 잘라낸 영상만 보면 그 정보가 사라진다. 잘라내기 전의 상대
    크기를 따로 넣어 준다.
    """
    width = index["width"].astype(float)
    height = index["height"].astype(float)
    box_w = (index["xbr"].astype(float) - index["xtl"].astype(float)).clip(lower=0)
    box_h = (index["ybr"].astype(float) - index["ytl"].astype(float)).clip(lower=0)

    out = pd.DataFrame(index=index.index)
    out["box_w_frac"] = (box_w / width).fillna(0.0)
    out["box_h_frac"] = (box_h / height).fillna(0.0)
    out["box_area_frac"] = (box_w * box_h / (width * height)).fillna(0.0)
    out["box_aspect"] = (box_w / box_h.replace(0, np.nan)).fillna(1.0)
    out["box_cx"] = ((index["xtl"].astype(float) + box_w / 2) / width).fillna(0.5)
    out["box_cy"] = ((index["ytl"].astype(float) + box_h / 2) / height).fillna(0.5)
    out["img_megapixels"] = (width * height / 1e6).fillna(0.0)
    out["grow_stage"] = index["grow"].astype(float)
    return out.reset_index(drop=True)


def session_split(
    index: pd.DataFrame,
    *,
    test_size: float = 0.25,
    valid_size: float = 0.15,
    seed: int = 42,
    stratify: str | None = "disease",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """촬영 세션 단위로 학습/검증/시험을 나눈다.

    이 데이터셋은 촬영 세션 하나가 사실상 병해 하나다(세션 32개 중 29개가
    단일 질병코드). 세션을 그냥 무작위로 나누면 질병코드 17처럼 세션이
    3개뿐인 병해가 시험셋에서 통째로 빠진다.

    그래서 ``stratify`` 열의 값마다 세션을 따로 모아, 이미지 수가 큰
    세션부터 '목표 대비 가장 모자란 분할'에 넣는 방식으로 배분한다.
    이러면 각 분할이 모든 병해를 갖게 되고 표본 비율도 목표에 가까워진다.

    세션 단위를 유지하는 이유는 그대로다. 같은 포장에서 같은 날 찍은
    사진이 학습셋과 시험셋에 갈라지면 성능이 부풀려진다.
    """
    rng = np.random.default_rng(seed)
    targets = {"train": 1.0 - test_size - valid_size, "valid": valid_size, "test": test_size}

    if stratify and stratify in index.columns:
        session_class = index.groupby("session")[stratify].agg(
            lambda s: s.value_counts().idxmax()
        )
        class_groups = [
            session_class[session_class == c].index.to_numpy()
            for c in sorted(session_class.unique())
        ]
    else:
        class_groups = [index["session"].unique()]

    counts = index["session"].value_counts()
    assigned: dict[str, set[str]] = {k: set() for k in targets}

    for sessions in class_groups:
        sessions = list(sessions)
        rng.shuffle(sessions)
        # 큰 세션부터 배분해야 마지막에 비율이 크게 틀어지지 않는다
        sessions.sort(key=lambda s: -counts[s])
        totals = {k: 0 for k in targets}
        pool = sum(counts[s] for s in sessions)

        for session in sessions:
            # 목표 대비 가장 모자란 분할에 넣는다
            split = min(targets, key=lambda k: totals[k] - targets[k] * pool)
            assigned[split].add(session)
            totals[split] += counts[session]

    session = index["session"].to_numpy()
    return tuple(  # type: ignore[return-value]
        np.where(np.isin(session, list(assigned[k])))[0] for k in ("train", "valid", "test")
    )


# --------------------------------------------------------------------------
# CNN 경로
# --------------------------------------------------------------------------

@dataclass
class CNNConfig:
    backbone: str = "resnet18"
    pretrained: bool = True
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2


def build_cnn(config: CNNConfig, n_disease: int, n_risk: int):
    """병해 종류와 심각도를 함께 내보내는 2헤드 분류기를 만든다.

    두 과제는 같은 병징을 보므로 백본을 공유하는 쪽이 표본을 아낀다.
    사전학습 가중치를 받을 수 없으면 ``pretrained=False`` 로 두고
    처음부터 학습한다(성능은 떨어진다).
    """
    import torch
    from torch import nn
    import torchvision

    factory = getattr(torchvision.models, config.backbone)
    weights = "DEFAULT" if config.pretrained else None
    backbone = factory(weights=weights)

    if hasattr(backbone, "fc"):
        n_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif hasattr(backbone, "classifier"):
        last = backbone.classifier[-1]
        n_features = last.in_features
        backbone.classifier = nn.Identity()
    else:
        raise ValueError(f"지원하지 않는 백본입니다: {config.backbone}")

    class TwoHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.drop = nn.Dropout(config.dropout)
            self.disease = nn.Linear(n_features, n_disease)
            self.risk = nn.Linear(n_features, n_risk)

        def forward(self, x):
            h = self.drop(self.backbone(x))
            return self.disease(h), self.risk(h)

    return TwoHead()


#: ImageNet 정규화 상수
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor_batch(crops: np.ndarray, indices: np.ndarray, augment: bool = False):
    """캐시된 (N,H,W,3) uint8 배열을 정규화된 (B,3,H,W) 텐서로 바꾼다."""
    import torch

    batch = np.asarray(crops[indices], dtype=np.float32) / 255.0
    if augment:
        flip = np.random.rand(len(batch)) < 0.5
        batch[flip] = batch[flip, :, ::-1]
        flip = np.random.rand(len(batch)) < 0.5
        batch[flip] = batch[flip, ::-1]
        batch *= np.random.uniform(0.85, 1.15, size=(len(batch), 1, 1, 1)).astype(np.float32)
        np.clip(batch, 0.0, 1.0, out=batch)

    batch = (batch - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(batch.transpose(0, 3, 1, 2)))
