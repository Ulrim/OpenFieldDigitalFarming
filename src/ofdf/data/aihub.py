"""AI-Hub '노지작물 질병 진단 이미지' 라벨 파서 (작물코드 9 = 파/대파).

파일 구성
---------
``[원천]`` zip 에 이미지, ``[라벨]`` zip 에 같은 이름 + ``.json`` 이 들어 있다.

라벨 JSON::

    {"description": {"image", "date", "height", "width", "task", "type", "region"},
     "annotations": {"disease", "crop", "area", "grow", "risk", "points": [{xtl,ytl,xbr,ybr}]}}

파일명도 같은 정보를 담는다::

    V006_79_1_16_09_03_12_1_0827e_20200916_31.jpg
     |    |  | |  |  |  |  |  |      |        |
     |    |  | |  |  |  |  |  |      |        +- 일련번호
     |    |  | |  |  |  |  |  |      +---------- 촬영일자
     |    |  | |  |  |  |  |  +----------------- 촬영 세션(농가) 코드
     |    |  | |  |  |  |  +-------------------- 위험도(심각도) 0~3
     |    |  | |  |  |  +----------------------- 생육단계
     |    |  | |  |  +-------------------------- 부위
     |    |  | |  +----------------------------- 작물(09=파)
     |    |  | +-------------------------------- 질병코드
     |    |  +---------------------------------- 정상(0)/질병(1)
     |    +------------------------------------- 과제코드
     +------------------------------------------ 데이터 버전

촬영 세션 코드는 학습/시험 분할을 나눌 때 쓴다. 같은 포장에서 같은 날
찍은 사진이 학습셋과 시험셋에 갈라져 들어가면 성능이 부풀려진다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

#: 작물코드
CROP_WELSH_ONION = 9

#: 위험도(심각도) 코드
RISK_NAMES = {0: "정상", 1: "초기", 2: "중기", 3: "말기"}

#: 질병코드 -> 병명.
#:
#: 확인 필요 — 아래 한글 병명은 영상 병징과 사업계획서가 지목한 대파 병해
#: (노균병·녹병·잎마름병·무름병)를 대조한 **잠정** 매핑이다. 코드와 병명의
#: 공식 대응은 AI-Hub 데이터 설명서로 확정해야 하며, 모델 학습·평가는
#: 병명이 아니라 코드로 하므로 매핑이 바뀌어도 성능에는 영향이 없다.
DISEASE_NAMES = {
    0: "정상",
    16: "질병코드16 (광범위 담갈색 병반형 — 노균병 추정, 확인 필요)",
    17: "질병코드17 (미세 반점형 — 확인 필요)",
    18: "질병코드18 (방추형 갈색 병반형 — 잎마름병 추정, 확인 필요)",
}


@dataclass
class ImageRecord:
    """이미지 한 장의 라벨."""

    path: Path
    image: str
    disease: int
    risk: int
    crop: int
    area: int
    grow: int
    session: str
    date: str
    bbox: tuple[int, int, int, int]
    width: int
    height: int


def parse_filename(name: str) -> dict:
    """파일명에서 메타데이터를 뽑는다."""
    stem = name.split(".")[0]
    parts = stem.split("_")
    if len(parts) < 10:
        return {}
    return {
        "type": int(parts[2]),
        "disease": int(parts[3]),
        "crop": int(parts[4]),
        "area": int(parts[5]),
        "grow": int(parts[6]),
        "risk": int(parts[7]),
        "session": parts[8],
        "date": parts[9],
    }


def read_label(path: str | Path) -> dict:
    """라벨 JSON 하나를 읽는다."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ann, desc = data["annotations"], data["description"]
    point = (ann.get("points") or [{}])[0]
    return {
        "image": desc["image"],
        "width": desc["width"],
        "height": desc["height"],
        "date": desc.get("date", ""),
        "disease": ann["disease"],
        "crop": ann["crop"],
        "area": ann["area"],
        "grow": ann["grow"],
        "risk": ann["risk"],
        "xtl": point.get("xtl"),
        "ytl": point.get("ytl"),
        "xbr": point.get("xbr"),
        "ybr": point.get("ybr"),
    }


def build_index(
    image_dirs: list[str | Path],
    label_dirs: list[str | Path],
    *,
    crop: int = CROP_WELSH_ONION,
) -> pd.DataFrame:
    """이미지 디렉터리와 라벨 디렉터리를 맞춰 색인 표를 만든다.

    라벨이 없는 이미지는 파일명에서 메타데이터를 복원하되, 병반 위치
    (bbox)가 없으므로 전체 이미지를 쓴다.
    """
    labels: dict[str, dict] = {}
    for d in label_dirs:
        for p in Path(d).glob("*.json"):
            try:
                record = read_label(p)
            except (KeyError, json.JSONDecodeError):
                continue
            labels[record["image"]] = record

    rows = []
    for d in image_dirs:
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            meta = parse_filename(p.name)
            label = labels.get(p.name, {})
            if not meta and not label:
                continue
            if (label.get("crop", meta.get("crop")) or crop) != crop:
                continue

            rows.append(
                {
                    "path": str(p),
                    "image": p.name,
                    "disease": label.get("disease", meta.get("disease", -1)),
                    "risk": label.get("risk", meta.get("risk", -1)),
                    "grow": label.get("grow", meta.get("grow", -1)),
                    "area": label.get("area", meta.get("area", -1)),
                    "session": meta.get("session", "unknown"),
                    "date": meta.get("date", ""),
                    "width": label.get("width"),
                    "height": label.get("height"),
                    "xtl": label.get("xtl"),
                    "ytl": label.get("ytl"),
                    "xbr": label.get("xbr"),
                    "ybr": label.get("ybr"),
                    "has_bbox": label.get("xtl") is not None,
                }
            )

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["is_disease"] = (frame["disease"] > 0).astype(int)
    return frame


def summarise(index: pd.DataFrame) -> pd.DataFrame:
    """질병코드 x 위험도 교차표."""
    return pd.crosstab(
        index["disease"].map(lambda d: f"{d} {DISEASE_NAMES.get(d, '')}".strip()),
        index["risk"].map(lambda r: f"{r} {RISK_NAMES.get(r, '')}".strip()),
        margins=True,
        margins_name="합계",
    )
