"""카메라 비전 기반 대파 병해 진단 모델 학습·평가.

사용 예::

    python scripts/train_vision.py \
        --images data/vision/train_disease data/vision/val_normal \
        --labels data/vision/label_train_disease data/vision/label_val_normal \
        --out artifacts/vision --path descriptor

``--path descriptor`` 는 색·질감 기술자 + LightGBM 으로, 사전학습 가중치가
필요 없다. ``--path cnn`` 은 torchvision 백본을 쓴다(가중치 내려받기 필요).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf.data.aihub import DISEASE_NAMES, RISK_NAMES, build_index, summarise  # noqa: E402
from ofdf.models import vision  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", nargs="+", required=True, help="원천 이미지 디렉터리")
    p.add_argument("--labels", nargs="+", required=True, help="라벨 JSON 디렉터리")
    p.add_argument("--out", default="artifacts/vision")
    p.add_argument("--cache", default="artifacts/vision/crops224.npy")
    p.add_argument("--path", choices=["descriptor", "cnn"], default="descriptor")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--no-pretrained", action="store_true", help="사전학습 가중치 사용 안 함")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, names: dict, title: str) -> dict:
    from sklearn.metrics import classification_report, f1_score

    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    accuracy = float((y_true == y_pred).mean())
    print(f"\n--- {title} ---")
    print(f"Macro F1 {macro:.3f} / 정확도 {accuracy:.3f}")
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    print(
        classification_report(
            y_true, y_pred, labels=labels,
            target_names=[str(names.get(l, l))[:22] for l in labels],
            zero_division=0, digits=3,
        )
    )
    return {"macro_f1": float(macro), "accuracy": accuracy}


def train_descriptor(features, labels, train_idx, valid_idx, test_idx, n_class, seed):
    """색·질감 기술자 + LightGBM."""
    import lightgbm as lgb

    classes, counts = np.unique(labels[train_idx], return_counts=True)
    total = counts.sum()
    weight_map = {int(c): total / (len(classes) * n) for c, n in zip(classes, counts)}
    weights = np.array([weight_map.get(int(v), 1.0) for v in labels[train_idx]])

    train_set = lgb.Dataset(features.iloc[train_idx], label=labels[train_idx], weight=weights)
    valid_set = lgb.Dataset(features.iloc[valid_idx], label=labels[valid_idx], reference=train_set)

    booster = lgb.train(
        {
            "objective": "multiclass", "num_class": n_class, "learning_rate": 0.05,
            "num_leaves": 31, "min_data_in_leaf": 20, "feature_fraction": 0.7,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
            "verbose": -1, "seed": seed, "num_threads": 0,
        },
        train_set, num_boost_round=800, valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    pred = booster.predict(features.iloc[test_idx], num_iteration=booster.best_iteration).argmax(1)
    return booster, pred


def train_cnn(crops, disease_y, risk_y, train_idx, valid_idx, test_idx, args):
    """torchvision 백본 2헤드 학습."""
    import torch
    from torch import nn

    config = vision.CNNConfig(
        pretrained=not args.no_pretrained, image_size=args.size, epochs=args.epochs
    )
    model = vision.build_cnn(config, len(vision.DISEASE_CLASSES), len(vision.RISK_CLASSES))
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    def class_weights(y):
        classes, counts = np.unique(y, return_counts=True)
        w = np.ones(max(len(vision.DISEASE_CLASSES), len(vision.RISK_CLASSES)))
        for c, n in zip(classes, counts):
            w[c] = counts.sum() / (len(classes) * n)
        return torch.tensor(w, dtype=torch.float32)

    loss_disease = nn.CrossEntropyLoss(weight=class_weights(disease_y[train_idx])[: len(vision.DISEASE_CLASSES)])
    loss_risk = nn.CrossEntropyLoss(weight=class_weights(risk_y[train_idx])[: len(vision.RISK_CLASSES)])

    best_state, best_score = None, -1.0
    rng = np.random.default_rng(args.seed)

    for epoch in range(config.epochs):
        model.train()
        order = rng.permutation(train_idx)
        running = 0.0
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            x = vision.to_tensor_batch(crops, batch, augment=True)
            yd = torch.from_numpy(disease_y[batch]).long()
            yr = torch.from_numpy(risk_y[batch]).long()

            optimiser.zero_grad()
            pd_logit, pr_logit = model(x)
            loss = loss_disease(pd_logit, yd) + 0.5 * loss_risk(pr_logit, yr)
            loss.backward()
            optimiser.step()
            running += float(loss.detach()) * len(batch)

        model.eval()
        with torch.no_grad():
            preds = []
            for start in range(0, len(valid_idx), 64):
                batch = valid_idx[start : start + 64]
                preds.append(model(vision.to_tensor_batch(crops, batch))[0].argmax(1).numpy())
            valid_pred = np.concatenate(preds)
        from sklearn.metrics import f1_score
        score = f1_score(disease_y[valid_idx], valid_pred, average="macro", zero_division=0)
        print(f"    epoch {epoch+1}/{config.epochs} loss {running/len(order):.3f} 검증 MacroF1 {score:.3f}")
        if score > best_score:
            best_score, best_state = score, {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        dis, rsk = [], []
        for start in range(0, len(test_idx), 64):
            batch = test_idx[start : start + 64]
            a, b = model(vision.to_tensor_batch(crops, batch))
            dis.append(a.argmax(1).numpy())
            rsk.append(b.argmax(1).numpy())
    return model, np.concatenate(dis), np.concatenate(rsk)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = build_index(args.images, args.labels)
    print(f"[데이터] {len(index):,}장 / 촬영 세션 {index['session'].nunique()}개")
    print(summarise(index).to_string())

    print("[전처리] 병반 영역 크롭 ...")
    crops = vision.precompute_crops(index, args.cache, size=args.size)

    disease_y = index["disease"].map({c: i for i, c in enumerate(vision.DISEASE_CLASSES)}).to_numpy()
    risk_y = index["risk"].to_numpy()

    train_idx, valid_idx, test_idx = vision.session_split(index, seed=args.seed)
    print(f"[분할] 학습 {len(train_idx):,} / 검증 {len(valid_idx):,} / 시험 {len(test_idx):,} (촬영 세션 단위)")

    disease_names = {i: DISEASE_NAMES.get(c, c) for i, c in enumerate(vision.DISEASE_CLASSES)}
    results = {}

    if args.path == "descriptor":
        print("[특징] 색·질감 기술자 계산 ...")
        features = vision.descriptor_matrix(crops)
        # 병반의 상대 크기·위치 — 심각도 판정에 꼭 필요하다
        features = pd.concat(
            [features.reset_index(drop=True), vision.geometry_features(index)], axis=1
        )
        print(f"    기술자 {features.shape[1]}개 (색·질감 + 병반 기하)")

        booster_d, pred_d = train_descriptor(
            features, disease_y, train_idx, valid_idx, test_idx, len(vision.DISEASE_CLASSES), args.seed
        )
        results["disease"] = evaluate(disease_y[test_idx], pred_d, disease_names, "병해 종류 진단")
        booster_d.save_model(str(out_dir / "vision_disease.txt"))

        booster_r, pred_r = train_descriptor(
            features, risk_y, train_idx, valid_idx, test_idx, len(vision.RISK_CLASSES), args.seed
        )
        results["risk"] = evaluate(risk_y[test_idx], pred_r, RISK_NAMES, "심각도 판정")
        booster_r.save_model(str(out_dir / "vision_risk.txt"))

        importance = pd.DataFrame({
            "feature": features.columns,
            "gain": booster_d.feature_importance("gain"),
        }).sort_values("gain", ascending=False).head(20)
        print("\n--- 병해 진단 주요 영향 기술자 ---")
        print(importance.to_string(index=False))
        importance.to_csv(out_dir / "vision_importance.csv", index=False, encoding="utf-8-sig")
    else:
        print(f"[학습] CNN (사전학습 {'사용 안 함' if args.no_pretrained else '사용'}) ...")
        model, pred_d, pred_r = train_cnn(
            crops, disease_y, risk_y, train_idx, valid_idx, test_idx, args
        )
        results["disease"] = evaluate(disease_y[test_idx], pred_d, disease_names, "병해 종류 진단")
        results["risk"] = evaluate(risk_y[test_idx], pred_r, RISK_NAMES, "심각도 판정")
        import torch
        torch.save(model.state_dict(), out_dir / "vision_cnn.pt")

    (out_dir / "performance.json").write_text(
        json.dumps({"path": args.path, **results}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n산출물: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
