"""제출물 생성과 화면 데이터 검증."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts(tmp_path):
    """최소한의 산출물 묶음을 만든다."""
    weather = tmp_path / "weather"
    weather.mkdir()
    pd.DataFrame([{
        "위험유형": "저온·서리", "시계(h)": 3, "시험 표본": 100, "위험 표본": 20,
        "평가단위": "2등급", "규칙 조치단위F1": 0.86, "AI 조치단위F1": 0.93,
        "전용학습 조치단위F1": 0.94, "조치단위 개선(%p)": 7.6,
        "AI 놓침방지율": 0.96, "규칙 놓침방지율": 0.66, "AI 오경보율": 0.02,
        "놓침방지 개선(%p)": 30.3, "사전알림(분)": 248.0,
    }]).to_csv(weather / "performance.csv", index=False, encoding="utf-8-sig")
    (weather / "feature_importance.json").write_text(
        json.dumps({"frost_h3": [{"feature": "canopy_temp", "gain": 1.0}]}), encoding="utf-8"
    )

    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    pd.DataFrame([{
        "시나리오": "저온·서리 사전경보", "구분": "입력모사", "시험횟수": 10,
        "통과": 10, "통과율(%)": 100.0, "판정기준": "선행 3시간 이상", "판정": "적합",
    }]).to_csv(scenarios / "scenario_results.csv", index=False, encoding="utf-8-sig")

    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    (benchmark / "benchmark.json").write_text(json.dumps({
        "환경": {"machine": "x86_64"}, "목표_초": 30.0,
        "결과": {"끝에서 끝까지": {"평균(ms)": 37.8, "중앙값(ms)": 33.6,
                             "p95(ms)": 65.9, "최대(ms)": 101.3}},
    }, ensure_ascii=False), encoding="utf-8")

    vision = tmp_path / "vision"
    vision.mkdir()
    (vision / "performance.json").write_text(json.dumps({
        "path": "descriptor", "disease": {"macro_f1": 0.818, "accuracy": 0.86},
        "risk": {"macro_f1": 0.55, "accuracy": 0.653},
    }), encoding="utf-8")
    return tmp_path


def build(artifacts_dir: Path, out_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_report.py"),
         "--artifacts", str(artifacts_dir), "--out", str(out_dir)],
        capture_output=True, text=True, cwd=ROOT,
    )


def test_builds_both_reports(artifacts, tmp_path):
    out = tmp_path / "reports"
    result = build(artifacts, out)
    assert result.returncode == 0, result.stderr
    assert (out / "성능평가서.md").exists()
    assert (out / "AI모델설명서.md").exists()


def test_performance_report_carries_measurements(artifacts, tmp_path):
    out = tmp_path / "reports"
    build(artifacts, out)
    text = (out / "성능평가서.md").read_text(encoding="utf-8")

    assert "0.93" in text and "0.96" in text          # 측정값이 그대로 실린다
    assert "저온·서리 사전경보" in text                  # 시나리오 결과
    assert "37.8" in text                             # 추론시간
    assert "0.818" in text                            # 비전 성능
    assert "달성" in text                              # 목표 대조
    assert "측정하지 못한 항목" in text                   # 빠진 것을 밝힌다


def test_report_states_evaluation_unit_per_risk(artifacts, tmp_path):
    """평가 단위가 조치에서 나온다는 것을 보고서가 밝힌다."""
    out = tmp_path / "reports"
    build(artifacts, out)
    text = (out / "성능평가서.md").read_text(encoding="utf-8")
    assert "주의·경계 조치" in text
    assert "| 강우·과습 | 같음 | 2등급 |" in text
    assert "| 고온·건조 | 다름 | 3등급 |" in text


def test_model_card_states_limits_and_scope(artifacts, tmp_path):
    out = tmp_path / "reports"
    build(artifacts, out)
    text = (out / "AI모델설명서.md").read_text(encoding="utf-8")
    assert "용도 밖 사용 금지" in text
    assert "알려진 한계" in text
    assert "ONNX" in text                              # 변환본 금지가 적혀 있다
    assert "누적값" in text                             # 자료 함정이 적혀 있다
    assert "재현" in text


def test_missing_artifacts_do_not_crash(tmp_path):
    """산출물이 없어도 보고서는 만들어지고, 없다는 사실을 적는다."""
    out = tmp_path / "reports"
    result = build(tmp_path / "empty", out)
    assert result.returncode == 0, result.stderr
    text = (out / "성능평가서.md").read_text(encoding="utf-8")
    assert "없어" in text or "없다" in text


# --------------------------------------------------------------------------
# 화면
# --------------------------------------------------------------------------

def test_dashboard_page_follows_artifact_contract():
    page = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    for tag in ("<!DOCTYPE", "<html", "<head>", "<body"):
        assert tag not in page, f"{tag} 는 게시 시 자동으로 감싸진다"
    assert "<title>" in page
    # 테마 3상태: 기본 :root + 매체질의 + data-theme
    assert "prefers-color-scheme: dark" in page
    assert ':root[data-theme="dark"]' in page
    assert ':root:not([data-theme="light"])' in page


def test_dashboard_data_has_screen_fields():
    data = json.loads((ROOT / "dashboard" / "data.json").read_text(encoding="utf-8"))
    assert {"station", "now", "mode", "cards", "environment", "actions", "series"} <= data.keys()
    assert len(data["cards"]) >= 4
    for card in data["cards"]:
        for view in card["horizons"].values():
            assert view["level"] in (0, 1, 2)
            assert 0.0 <= view["confidence"] <= 1.0
    assert len(data["series"]["time"]) == len(data["series"]["t_air"])
