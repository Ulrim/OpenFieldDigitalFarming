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

def test_dashboard_page_is_a_standalone_document():
    """화면은 라즈베리파이가 그대로 내보내는 파일이다.

    한때 이 파일은 게시용 미리보기를 기준으로 삼아 문서 뼈대를 일부러
    빼 두었다. 게시 경로는 뼈대를 자동으로 감싸 주기 때문이다. 그런데
    현장 제어기는 같은 파일을 ``http.server`` 로 그냥 내보내고, 거기에는
    감싸 주는 주체가 없다. 그 결과

    * ``<meta charset>`` 이 없어 한글이 전부 깨지고,
    * ``<!doctype>`` 이 없어 쿼크스 모드로 배치가 틀어지고,
    * ``<meta name=viewport>`` 가 없어 휴대폰이 축소해서 띄운다.

    제품은 제어기 쪽이다. 미리보기가 아니라 포장에서 도는 화면을 기준으로
    맞춘다.
    """
    page = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert page.lstrip().lower().startswith("<!doctype html>")
    assert 'lang="ko"' in page
    assert '<meta charset="utf-8">' in page
    assert '<meta name="viewport"' in page and "width=device-width" in page
    assert "<title>" in page


def test_dashboard_page_survives_without_network():
    """통신이 끊긴 포장에서 원격 글꼴이 화면을 붙잡으면 안 된다."""
    page = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    for line in page.splitlines():
        if "fonts.googleapis.com" in line and "<link" in line:
            # 비차단(media=print → onload 교체)이거나 noscript 안이어야 한다
            assert 'media="print"' in line or "<noscript>" in line, line


def test_dashboard_page_refreshes_itself():
    """벽패널은 아무도 새로고침하지 않는다. 1회 fetch 로 되돌리면 안 된다."""
    page = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
    assert "setInterval(load" in page, "주기 갱신이 사라졌다"
    assert "STALE_MS" in page, "오래된 자료 경고가 사라졌다"
    assert 'id="stale"' in page


def test_dashboard_page_keeps_three_state_theming():
    page = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")
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


def test_artifact_page_build_strips_only_the_document_shell():
    """게시용 미리보기는 현장 화면에서 만들어 낸다.

    두 경로의 요구가 반대다. 제어기는 완전한 문서여야 하고, 게시 경로는
    뼈대를 자기가 감싸므로 들어 있으면 중첩된다. 원본을 제어기 쪽으로 두고
    게시용을 만들어 내는데, 그 변환이 내용까지 깎아 내면 안 된다.
    """
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "page.html"
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_artifact_page.py"),
             "--out", str(out)],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr
        built = out.read_text(encoding="utf-8")

    source = (ROOT / "dashboard" / "index.html").read_text(encoding="utf-8")

    # 게시 경로가 넣어 주는 것은 빠져야 한다
    for tag in ("<!doctype", "<html", "<head>", "<body", "<meta charset"):
        assert tag not in built.lower(), tag

    # 화면을 이루는 것은 그대로 남아야 한다
    for keep in ("<title>", "<style>", "<script>", 'id="stale"',
                 "setInterval(load", "STALE_MS", "prefers-color-scheme: dark"):
        assert keep in built, keep

    # 떼어낸 줄 말고는 손대지 않았는지 — 길이가 크게 줄면 뭔가 더 깎인 것이다
    assert len(built) > len(source) * 0.95


def test_weather_artifacts_resolve_to_operational_results(artifacts, tmp_path):
    """조치단위 결과가 있으면 그쪽을 쓰고, 없으면 있는 것을 쓴다.

    제출물은 조치단위 평가를 싣는다(README 3.13). 그 열은 운영 평가단위로
    다시 돌린 산출물에만 있다. 경로 이름을 기본값으로 박아 두면 다른 쪽을
    쓰는 사람에게 **조용히 빈 보고서**가 나가는데, 제출 문서에서는 그게
    예외로 끝나는 것보다 나쁘다.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_report import resolve_weather_dir

    # 픽스처에는 weather 만 있다 — 그것을 찾아내야 한다
    assert resolve_weather_dir(artifacts, None).name == "weather"

    # 둘 다 있으면 조치단위 쪽이 이긴다
    (artifacts / "weather_or").mkdir()
    (artifacts / "weather_or" / "performance.csv").write_text("위험유형\n", encoding="utf-8")
    assert resolve_weather_dir(artifacts, None).name == "weather_or"

    # 명시 지정은 언제나 이긴다
    assert resolve_weather_dir(artifacts, "weather").name == "weather"


def test_report_refuses_results_without_action_unit_columns(tmp_path):
    """조치단위 열이 없으면 무엇이 없는지 말하고 끝낸다."""
    stale = tmp_path / "weather"
    stale.mkdir()
    pd.DataFrame([{"위험유형": "저온·서리", "시계(h)": 3, "AI MacroF1": 0.93}]).to_csv(
        stale / "performance.csv", index=False, encoding="utf-8-sig")

    result = build(tmp_path, tmp_path / "reports")
    assert result.returncode != 0
    assert "조치단위" in result.stderr and "README 3.13" in result.stderr
