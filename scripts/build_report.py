"""제출물 생성 — 성능평가서와 AI 모델 설명서.

사업계획서가 요구하는 산출물을 학습·시험 결과에서 자동으로 만든다.
숫자를 사람이 옮겨 적지 않으므로 보고서와 실제 결과가 어긋나지 않는다.

    성능평가서    외부 공인시험 신청(10.15)과 최종보고에 쓴다.
                  위험유형별 종합점수·놓침 방지율·단순 기준값 대비 개선,
                  시나리오 시험 결과, 추론시간
    모델 설명서    AI 모델 1종의 입력·출력·학습·한계를 적는다.
                  '동일 결과를 다시 확인할 수 있도록' 하는 재현 정보 포함

사용 예::

    python scripts/build_report.py --artifacts artifacts --out reports
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf import __version__  # noqa: E402
from ofdf.action.recommend import (  # noqa: E402
    CONFIDENCE_THRESHOLD, SHADE_MAX_MINUTES, action_distinct_levels,
)
from ofdf.data.quality import VALID_RANGE  # noqa: E402
from ofdf.labels.risk import RISK_LABELS_KO, RiskThresholds  # noqa: E402

#: 사업계획서 정량 성능목표
TARGETS = {
    "macro_f1": 0.80,
    "risk_recall": 0.85,
    "improvement_pp": 5.0,
    "inference_seconds": 30.0,
    "storage_rate": 0.98,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifacts", default="artifacts", help="산출물 루트")
    p.add_argument("--out", default="reports")
    p.add_argument("--weather", default="weather", help="기상 모델 산출물 하위 경로")
    p.add_argument("--project", default="노지작물 재해위험 판단·조치추천 AI 현장제어 시제품 개발")
    p.add_argument("--org", default="주식회사 컬리버")
    p.add_argument("--site", default="전라남도 나주시 남평읍 대교리 (노지 대파)")
    return p.parse_args()


def read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:                                  # noqa: BLE001
        return "(저장소 정보 없음)"


def table(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    """데이터프레임을 마크다운 표로 바꾼다."""
    view = frame[columns] if columns else frame
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    rule = "|" + "|".join("---" for _ in view.columns) + "|"
    rows = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in view.itertuples(index=False)
    ]
    return "\n".join([header, rule, *rows])


def verdict(value: float | None, target: float, higher_is_better: bool = True) -> str:
    if value is None or pd.isna(value):
        return "미측정"
    ok = value >= target if higher_is_better else value <= target
    return "달성" if ok else "미달"


def performance_report(args, data: dict) -> str:
    weather = data["weather"]
    scenarios = data["scenarios"]
    timing = data["timing"]
    vision = data["vision"]

    lines = [
        "# 성능평가서",
        "",
        f"- 과제명 : {args.project}",
        f"- 수행기관 : {args.org}",
        f"- 실증지 : {args.site}",
        f"- 작성일 : {date.today():%Y-%m-%d}",
        f"- 소프트웨어 버전 : ofdf {__version__} ({git_revision()})",
        "",
        "## 1. 평가 설계",
        "",
        "정확도는 쓰지 않는다. 위험사례가 전체의 몇 %뿐이라 '전부 정상'이라고 답해도",
        "정확도가 90%를 넘기 때문이다. 위험유형별 **종합점수(Macro F1)** 와",
        "**위험 놓침 방지율(재현율)** 을 본다.",
        "",
        "평가 단위는 조치에 맞춘다. 주의와 경계가 **다른 조치**를 내는 위험유형은",
        "3등급으로, **같은 조치**를 내는 유형은 정상 대 위험 2등급으로 잰다.",
        "농가가 겪는 일이 같은데 둘을 가려내지 못한 것을 성능 미달로 세지 않기 위해서다.",
        "",
    ]

    distinct = action_distinct_levels()
    unit_rows = pd.DataFrame([
        {
            "위험유형": RISK_LABELS_KO[risk],
            "주의·경계 조치": "다름" if ok else "같음",
            "평가 단위": "3등급" if ok else "2등급",
        }
        for risk, ok in distinct.items()
    ])
    lines += [table(unit_rows), ""]

    lines += [
        "학습·검증·시험은 **기상사례 블록 단위**로 나눈다. 어느 위험유형이든 위험이",
        "걸린 구간을 합집합으로 묶고 앞뒤 12시간을 같은 블록에 포함시켜, 같은 기상사례가",
        "학습셋과 시험셋에 쪼개지지 않게 한다.",
        "",
        "## 2. 기상 기반 위험판단 성능",
        "",
    ]

    if weather is None:
        lines += ["> 학습 산출물(performance.csv)이 없어 측정값을 채우지 못했다.", ""]
    else:
        cols = [c for c in [
            "위험유형", "시계(h)", "시험 표본", "위험 표본", "평가단위",
            "규칙 조치단위F1", "AI 조치단위F1", "전용학습 조치단위F1",
            "AI 놓침방지율", "규칙 놓침방지율", "AI 오경보율", "사전알림(분)",
        ] if c in weather.columns]
        lines += [table(weather, cols), ""]

        def best(row) -> float:
            direct = row.get("전용학습 조치단위F1")
            return float(direct) if pd.notna(direct) else float(row["AI 조치단위F1"])

        scores = weather.apply(best, axis=1)
        recalls = weather["AI 놓침방지율"].astype(float)
        summary = pd.DataFrame([
            {
                "성능목표": "위험유형별 종합점수 0.80 이상",
                "결과": f"{int((scores >= TARGETS['macro_f1']).sum())}/{len(scores)} 항목 달성",
                "최저값": round(float(scores.min()), 3),
                "판정": verdict(float(scores.min()), TARGETS["macro_f1"]),
            },
            {
                "성능목표": "위험 놓침 방지율 85% 이상",
                "결과": f"{int((recalls >= TARGETS['risk_recall']).sum())}/{len(recalls)} 항목 달성",
                "최저값": round(float(recalls.min()), 3),
                "판정": verdict(float(recalls.min()), TARGETS["risk_recall"]),
            },
        ])
        lines += ["### 2.1 성능목표 대조", "", table(summary), ""]

        short = weather[scores < TARGETS["macro_f1"]]
        if not short.empty:
            lines += [
                "미달 항목:",
                "",
                table(short, ["위험유형", "시계(h)", "AI 조치단위F1", "AI 놓침방지율"]),
                "",
            ]

        lines += [
            "### 2.2 단순 기준값 대비 개선",
            "",
            "단순 기준값 규칙은 현재 센서값이 임계를 넘은 뒤에 작동하므로 구조적으로",
            "선행 대응이 불가능하다. 종합점수에서는 규칙이 더 높은 항목이 있는데,",
            "규칙은 정밀도가 높은 대신 위험을 놓치기 때문이다. 두 지표를 함께 본다.",
            "",
        ]
        gain = pd.DataFrame({
            "위험유형": weather["위험유형"],
            "시계(h)": weather["시계(h)"],
            "종합점수 개선(%p)": weather.get("조치단위 개선(%p)"),
            "놓침방지 개선(%p)": weather.get("놓침방지 개선(%p)"),
        })
        lines += [table(gain), ""]

    lines += ["## 3. 핵심 실증 시나리오 시험", ""]
    if scenarios is None:
        lines += ["> 시나리오 산출물(scenario_results.csv)이 없다.", ""]
    else:
        lines += [
            table(scenarios, [c for c in
                  ["시나리오", "구분", "시험횟수", "통과", "통과율(%)", "판정기준", "판정"]
                  if c in scenarios.columns]),
            "",
            f"총 {int(scenarios['시험횟수'].sum())}회 중 {int(scenarios['통과'].sum())}회 통과, "
            f"{int((scenarios['판정'] == '적합').sum())}/{len(scenarios)} 시나리오 적합.",
            "",
            "자연 기상사례가 발생하지 않아도 검증되도록 현장 설치상태에서 안전범위 내",
            "센서 입력을 모사해 시험했다. 자연 발생 결과와 구분 표기한다.",
            "",
        ]

    lines += ["## 4. 추론시간", ""]
    if timing is None:
        lines += ["> 측정 산출물(benchmark.json)이 없다.", ""]
    else:
        rows = timing.get("결과", {})
        frame = pd.DataFrame([{"단계": k, **v} for k, v in rows.items()])
        lines += [table(frame, [c for c in
                  ["단계", "평균(ms)", "중앙값(ms)", "p95(ms)", "최대(ms)"] if c in frame.columns]), ""]
        end = rows.get("끝에서 끝까지", {})
        if end:
            seconds = end["평균(ms)"] / 1000
            lines += [
                f"판단 1회 끝에서 끝까지 평균 {end['평균(ms)']:.0f}ms "
                f"(p95 {end['p95(ms)']:.0f}ms). 목표 평균 {TARGETS['inference_seconds']:.0f}초 이내 — "
                f"**{verdict(seconds, TARGETS['inference_seconds'], higher_is_better=False)}**.",
                "",
                f"> 측정 환경은 {timing.get('환경', {}).get('machine', '?')} 서버급 CPU다. "
                "엣지 제어기(ARM)는 5~10배 느릴 수 있다. 실제 제어기 반입 후 같은 "
                "스크립트로 다시 측정해 성적서에 반영해야 한다.",
                "",
                "> 제어응답(명령 발생 → 구동 개시)은 밸브·개폐기 실측이 필요해 미측정이다.",
                "",
            ]

    lines += ["## 5. 비전 병해 진단 성능", ""]
    if vision is None:
        lines += ["> 비전 산출물(performance.json)이 없다.", ""]
    else:
        frame = pd.DataFrame([
            {"과제": "병해 종류(정상·3종)", **vision.get("disease", {})},
            {"과제": "심각도(정상·초기·중기·말기)", **vision.get("risk", {})},
        ]).rename(columns={"macro_f1": "Macro F1", "accuracy": "정확도"})
        for column in ("Macro F1", "정확도"):
            if column in frame.columns:
                frame[column] = frame[column].round(3)
        lines += [table(frame), "", f"학습 경로: {vision.get('path', '?')}", ""]

    lines += [
        "## 6. 측정하지 못한 항목",
        "",
        "| 항목 | 사유 | 측정 가능 시점 |",
        "|---|---|---|",
        "| 현장 제어응답 | 밸브·개폐기 실측 필요 | 제어함·관수라인 설치 후 |",
        "| 차광막 완전 회수 시간 | 구동부 실측 필요 | 구조물·차광 설치 후 |",
        "| 로컬 독립운전 72시간 | 실제 통신 차단 시험 필요 | 제어기 설치 후 |",
        "| 데이터 정상 수집·저장률 | 현장 수집 개시 후 집계 | 센서 설치 후 |",
        "| 강풍 시나리오 자연사례 | 3초 순간최대풍속 계측기 필요 | 기상 마스트 설치 후 |",
        "",
    ]
    return "\n".join(lines)


def model_card(args, data: dict) -> str:
    thresholds = RiskThresholds()
    importance = data["importance"]

    lines = [
        "# AI 모델 설명서",
        "",
        f"- 모델명 : 노지 대파 농지위험 판단모델 (ofdf-weather)",
        f"- 버전 : {__version__} ({git_revision()})",
        f"- 작성일 : {date.today():%Y-%m-%d}",
        f"- 수행기관 : {args.org}",
        "",
        "## 1. 용도",
        "",
        "나주 노지 대파 포장에서 **앞으로 1시간·3시간 안에 도달할 위험등급**을 판단하고,",
        "판단근거와 추천조치를 함께 내놓는다. 기상을 다시 예측하는 모델이 아니라,",
        "예보와 농지 관측을 합쳐 **그 포장에서 무엇을 해야 하는가**를 판단하는 모델이다.",
        "",
        "용도 밖 사용 금지:",
        "",
        "- 병해 **진단**(어떤 병인지)은 이 모델의 일이 아니다. 병해 유리환경(발생하기",
        "  좋은 조건)까지만 판단한다. 진단은 비전 모델이 따로 한다.",
        "- 강풍·도복은 AI 판단 대상이 아니다. 3초 순간최대풍속 기준의 현장 즉응규칙(L1)이며,",
        "  AI가 멈춰도 작동해야 한다.",
        "- 나주 외 지역, 대파 외 작물에는 그대로 쓸 수 없다. 위험 기준값과 농지 보정이",
        "  지역·작물마다 다르다.",
        "",
        "## 2. 입력",
        "",
        "현재 시각 이하의 값만 쓴다. 미래값은 라벨 계산에만 쓰고 입력에 넣지 않는다.",
        "",
        "| 묶음 | 내용 |",
        "|---|---|",
        "| 관측값 | 기온, 상대습도, 강수량, 일사량, 풍속 |",
        "| 파생 | 이슬점, 수증기압 부족(VPD), 엽면습윤, 초관부 온도, 토양수분 지수 |",
        "| 구간 통계 | 1·3·6·12·24시간 평균/최대/최소/합 |",
        "| 변화추세 | 1·3·6시간 전 대비 변화량 |",
        "| 지속시간 | 습윤·고습·강우·무강우·고온·저온 연속 시간 |",
        "| 시각·계절 | 시각·절기 주기함수, 야간 여부 |",
        "| 일자료 | 운량, 기준증발산량 (AgERA5) |",
        "",
        "현장 센서가 없는 구간에서는 초관부 온도·엽면습윤·토양수분을 **대리지표**로 만든다.",
        "실측이 붙으면 같은 이름의 컬럼으로 교체하며, 토양수분만은 척도가 달라",
        "분위수 사상이 필요하다.",
        "",
        "## 3. 출력",
        "",
        "위험유형별로 정상(0)·주의(1)·경계(2) 등급, 판단 신뢰도, 주요 판단근거 3개,",
        "추천조치를 낸다. 복합위험은 따로 학습하지 않고 구성 위험의 예측을 조합한다.",
        "",
        f"판단 신뢰도가 {CONFIDENCE_THRESHOLD:.2f} 미만이면 자동 실행하지 않고 농가 승인을 받는다.",
        "",
        "## 4. 학습",
        "",
        "| 항목 | 내용 |",
        "|---|---|",
        "| 알고리즘 | LightGBM 다중분류 (위험유형 × 예측시계마다 1개) |",
        "| 학습자료 | 나주 농업기상관측 시간자료 5지점, AgERA5 일자료 |",
        "| 대상기간 | 작기 9~11월 |",
        "| 분할 | 기상사례 블록 단위 (같은 사례가 학습·시험에 쪼개지지 않음) |",
        "| 클래스 불균형 | 등급별 가중치 적용 |",
        "| 비교 기준 | 단순 임계값 규칙, 로지스틱 회귀(현재값만) |",
        "",
        "## 5. 위험 판정 기준값",
        "",
        "사업계획서 '위험 정답값(라벨) 정의' 표의 초기 기준이다. 3자 검토",
        "(실증농가·재배/병해 전문가·AI 담당)에서 확정한 값으로 교체한다.",
        "",
        "| 위험유형 | 주의 | 경계 |",
        "|---|---|---|",
        f"| 고온·건조 | 기온 {thresholds.heat_caution_temp:.0f}℃ + 일사 {thresholds.heat_caution_solar:.0f}W/m² "
        f"또는 토양수분 하한 | 기온 {thresholds.heat_warning_temp:.0f}℃ + 일사 {thresholds.heat_warning_solar:.0f}W/m² |",
        f"| 강우·과습 | 3시간 강우 {thresholds.rain_caution_3h_mm:.0f}mm + 토양수분 상한 "
        f"{thresholds.wet_caution_hours}시간 | 6시간 강우 {thresholds.rain_warning_6h_mm:.0f}mm "
        f"또는 상한 {thresholds.wet_warning_hours}시간 |",
        f"| 병해 유리환경 | 엽면습윤 {thresholds.disease_caution_wet_hours}시간 + RH "
        f"{thresholds.disease_caution_rh:.0f}% + {thresholds.disease_caution_temp[0]:.0f}~"
        f"{thresholds.disease_caution_temp[1]:.0f}℃ | 엽면습윤 {thresholds.disease_warning_wet_hours}시간 + RH "
        f"{thresholds.disease_warning_rh:.0f}% |",
        f"| 저온·서리 | 최저기온 {thresholds.frost_caution_tmin:.0f}℃ + 풍속 "
        f"{thresholds.frost_caution_wind:.0f}m/s 이하 + 맑음, 또는 초관부 "
        f"{thresholds.frost_caution_canopy:.0f}℃ | 초관부 {thresholds.frost_warning_canopy:.0f}℃ "
        f"{thresholds.frost_warning_canopy_hours}시간 |",
        "| 복합위험 | 과습+병해 또는 과습+서리 동시 주의 | 구성 위험 중 1개 이상 경계 |",
        "",
        "## 6. 주요 영향변수",
        "",
    ]

    if importance:
        for key, rows in list(importance.items())[:4]:
            risk, _, horizon = key.rpartition("_h")
            top = ", ".join(r["feature"] for r in rows[:5])
            lines.append(f"- **{RISK_LABELS_KO.get(risk, risk)} {horizon}시간** : {top}")
        lines.append("")
    else:
        lines += ["> 중요도 산출물(feature_importance.json)이 없다.", ""]

    lines += [
        "개별 판단의 근거는 전역 중요도가 아니라 그 판단의 기여도(SHAP)로 뽑아",
        "화면과 로그에 상위 3개를 남긴다.",
        "",
        "## 7. 데이터 품질관리",
        "",
        "원본은 바꾸지 않고 이상 표시만 붙인 뒤, 범위이탈·급변·교차모순·영점 채움을",
        "결측 처리한 사본으로 학습한다.",
        "",
        "| 검사 | 내용 |",
        "|---|---|",
        "| 범위이탈 | 물리적으로 불가능한 값 |",
        "| 급변 | 1시간 변화가 물리 한계 초과 |",
        "| 고정값 | 센서가 멈춰 같은 값이 이어짐 |",
        "| 영점 채움 | 통신 두절 구간을 0으로 메운 자리 |",
        "| 교차 모순 | 강우 중 저습, 야간 일사 등 |",
        "| 시간중복·결측 | 지점별로 센다 |",
        "",
        "누적 컬럼 주의 — 농업기상 자료의 강수량·일사량·일조시간은 자정부터의",
        "**누적값**이다. 하루 단위로 차분해야 시간값이 된다.",
        "",
        "## 8. 알려진 한계",
        "",
        "- **강우·과습**은 관측만으로 1~3시간 뒤 강수를 맞히는 데 원리적 한계가 있다.",
        "  기상청 초단기예보를 입력으로 연계하면 재현율이 개선된다.",
        "- **초관부 온도·엽면습윤·토양수분**은 현장 실측 전까지 대리지표다. 특히",
        "  초관부 온도는 지면상태·지형 냉기호수를 반영하지 않은 1차 근사다.",
        "- **관측소 센서 고장**이 지점·연도마다 다르다. 기준 관측소를 고르기 전에",
        "  `quality.station_health` 로 건전성을 확인해야 한다.",
        "- **복합위험·강우·과습은 위험표본이 적다**. 시험 이벤트가 적은 유형은",
        "  성능 신뢰구간이 넓다.",
        "- **ONNX 변환본은 원본과 다른 등급을 낸다**. 네이티브 LightGBM 을 쓴다.",
        "",
        "## 9. 재현",
        "",
        "```bash",
        "python scripts/train_weather.py --aaos-dir <농업기상> --agera5 <AgERA5> \\",
        "    --cache artifacts/dataset.pkl --out artifacts/weather",
        "python scripts/run_scenarios.py --out artifacts/scenarios",
        "python scripts/benchmark_inference.py --cache artifacts/dataset.pkl \\",
        "    --models artifacts/weather --out artifacts/benchmark",
        "python scripts/build_report.py --artifacts artifacts --out reports",
        "```",
        "",
        f"- 분할 씨앗 고정(기본 42), 소프트웨어 {__version__} ({git_revision()})",
        f"- 실행 환경 : Python {platform.python_version()} / {platform.machine()}",
        f"- 물리 유효범위 등 품질 기준은 `ofdf.data.quality` 에 상수로 둔다 "
        f"({len(VALID_RANGE)}개 항목)",
        f"- 차광 제한 : {SHADE_MAX_MINUTES:.0f}분/일",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    root = Path(args.artifacts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "weather": read_csv(root / args.weather / "performance.csv"),
        "importance": read_json(root / args.weather / "feature_importance.json"),
        "scenarios": read_csv(root / "scenarios" / "scenario_results.csv"),
        "timing": read_json(root / "benchmark" / "benchmark.json"),
        "vision": read_json(root / "vision" / "performance.json"),
    }
    missing = [k for k, v in data.items() if v is None]
    if missing:
        print(f"[주의] 산출물 없음: {', '.join(missing)} — 해당 절은 비워 둔다")

    reports = {
        "성능평가서.md": performance_report(args, data),
        "AI모델설명서.md": model_card(args, data),
    }
    for name, text in reports.items():
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        print(f"[출력] {path}  ({len(text.splitlines())}줄)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
