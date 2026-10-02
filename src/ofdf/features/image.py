"""병징 영상 기술자(descriptor).

대파 잎 병해는 색(담갈색·황화·괴사)과 질감(병반 경계, 반점 밀도)으로
구분된다. 사전학습 가중치를 받을 수 없는 환경에서도 동작하도록,
색/질감 통계만으로 만든 기술자를 제공한다.

사전학습 백본을 쓸 수 있으면 :mod:`ofdf.models.vision` 의 CNN 경로가
성능이 더 좋다. 이 모듈은 가중치 없이도 돌아가는 경로이자, CNN 특징과
이어 붙여 쓸 수 있는 보조 특징이다.
"""

from __future__ import annotations

import numpy as np

EPSILON = 1e-8


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """0~1 RGB 배열을 HSV(H는 0~360)로 바꾼다."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    diff = mx - mn

    hue = np.zeros_like(mx)
    mask = diff > EPSILON
    idx = (mx == r) & mask
    hue[idx] = (60 * ((g[idx] - b[idx]) / diff[idx])) % 360
    idx = (mx == g) & mask
    hue[idx] = 60 * ((b[idx] - r[idx]) / diff[idx]) + 120
    idx = (mx == b) & mask
    hue[idx] = 60 * ((r[idx] - g[idx]) / diff[idx]) + 240

    sat = np.where(mx > EPSILON, diff / (mx + EPSILON), 0.0)
    return np.stack([hue, sat, mx], axis=-1)


def _hist(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins, range=value_range)
    return counts / (counts.sum() + EPSILON)


def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    """Sobel 근사 기울기 크기 — 병반 경계와 반점 밀도를 잡는다."""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.sqrt(gx**2 + gy**2)


def _region_descriptor(rgb: np.ndarray, prefix: str) -> dict[str, float]:
    """영역 하나의 색·질감 기술자."""
    hsv = rgb_to_hsv(rgb)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    gray = rgb @ np.array([0.299, 0.587, 0.114])

    out: dict[str, float] = {}

    # 색 분포 히스토그램
    for i, v in enumerate(_hist(hue, 18, (0.0, 360.0))):
        out[f"{prefix}_hue{i}"] = float(v)
    for i, v in enumerate(_hist(sat, 10, (0.0, 1.0))):
        out[f"{prefix}_sat{i}"] = float(v)
    for i, v in enumerate(_hist(val, 10, (0.0, 1.0))):
        out[f"{prefix}_val{i}"] = float(v)

    # 건전 잎 / 병징 색 비율
    healthy = (hue >= 70) & (hue <= 160) & (sat > 0.25)
    chlorotic = (hue >= 35) & (hue < 70) & (sat > 0.2)      # 황화
    necrotic = (hue >= 10) & (hue < 35) & (sat > 0.15)      # 담갈색 괴사
    bleached = (sat <= 0.15) & (val > 0.4)                  # 회백색 변색
    out[f"{prefix}_healthy_frac"] = float(healthy.mean())
    out[f"{prefix}_chlorotic_frac"] = float(chlorotic.mean())
    out[f"{prefix}_necrotic_frac"] = float(necrotic.mean())
    out[f"{prefix}_bleached_frac"] = float(bleached.mean())
    out[f"{prefix}_lesion_ratio"] = float(
        (chlorotic.sum() + necrotic.sum() + bleached.sum()) / (healthy.sum() + EPSILON)
    )

    # 채널 통계
    for name, channel in (("hue", hue), ("sat", sat), ("val", val), ("gray", gray)):
        out[f"{prefix}_{name}_mean"] = float(channel.mean())
        out[f"{prefix}_{name}_std"] = float(channel.std())
    for i, c in enumerate("rgb"):
        out[f"{prefix}_{c}_mean"] = float(rgb[..., i].mean())
        out[f"{prefix}_{c}_std"] = float(rgb[..., i].std())

    # 식생지수 — 잎 활력
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = 2 * g - r - b
    out[f"{prefix}_exg_mean"] = float(exg.mean())
    out[f"{prefix}_exg_std"] = float(exg.std())
    out[f"{prefix}_gli_mean"] = float(((2 * g - r - b) / (2 * g + r + b + EPSILON)).mean())

    # 질감
    grad = _gradient_magnitude(gray)
    out[f"{prefix}_grad_mean"] = float(grad.mean())
    out[f"{prefix}_grad_std"] = float(grad.std())
    out[f"{prefix}_grad_p90"] = float(np.percentile(grad, 90))
    for i, v in enumerate(_hist(grad, 8, (0.0, 0.5))):
        out[f"{prefix}_grad{i}"] = float(v)

    return out


def describe(rgb: np.ndarray) -> dict[str, float]:
    """잘라낸 병반 영역에서 전체 기술자를 만든다.

    Parameters
    ----------
    rgb : (H, W, 3) 0~1 실수 배열

    Returns
    -------
    dict
        전체 영역과 중앙부(병징이 몰린 곳) 기술자를 합친 것.
    """
    features = _region_descriptor(rgb, "all")

    h, w = rgb.shape[:2]
    center = rgb[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if center.size:
        features.update(_region_descriptor(center, "mid"))

    features["aspect_ratio"] = float(w / (h + EPSILON))
    return features
