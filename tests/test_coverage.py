"""수집 범위 점검.

이 시험이 지키는 것은 하나다 — **없는 센서를 있다고 보고하지 않는다.**
'모든 센서 데이터를 받는다'는 보고가 틀리면 아무도 확인하지 않게 되고,
빠진 변수는 비가 오는 날이나 서리가 내리는 날에야 드러난다.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ofdf.data import coverage, ecowitt, rtl433
from ofdf.data.field import FIELD_VARIABLES

CONFIG = """
devices:
  soil-treat-01:
    zone: treatment
    sensor_id: SOIL-T1
    map:
      temperature: soil_temp
    sends: [soil_moisture]
  valve-treat-01:
    zone: facility
    sensor_id: VALVE-01
    map:
      state: valve_state
      flow: irrigation_flow
rf:
  Fineoffset-WS90:
    zone: mast
    sensor_id: WS90-01
ecowitt:
  gateway:
    zone: mast
    sensor_id: WS90-01
  channels: {}
"""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "devices.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def status_of(frame: pd.DataFrame, variable: str) -> str:
    row = frame[frame["variable"] == variable]
    assert not row.empty, f"{variable} 이 표에 없다"
    return str(row.iloc[0]["status"])


def test_설계_변수_전부가_표에_나온다(config):
    frame = coverage.coverage(config)
    assert set(frame["variable"]) == set(FIELD_VARIABLES)


def test_설정에_적힌_장치가_보내는_변수는_연결됨(config):
    frame = coverage.coverage(config)
    assert status_of(frame, "soil_temp") == "연결됨"
    assert status_of(frame, "valve_state") == "연결됨"
    assert status_of(frame, "t_air") == "연결됨"       # WS90


def test_기본매핑에_이름이_있어도_장치가_없으면_연결됨이_아니다(config):
    """누수 센서가 토양수분을 보내지는 않는다.

    ``Z2M_DEFAULT_MAP`` 에 ``soil_ec`` 이름이 있다는 것은 '그 이름이 오면
    이 변수'라는 뜻일 뿐이다. 그것을 '받고 있다'로 읽으면 매핑표가 곧
    장비 목록이 되어 버린다.
    """
    frame = coverage.coverage(config)
    assert status_of(frame, "soil_ec") == "경로있음"


def test_WS90_가_재지_않는_것을_받는다고_적지_않는다(config):
    """WS90 은 엽면습윤도 토양수분도 재지 않는다.

    환산표(:data:`RTL433_MAP`)에는 두 이름이 다 있다 — WH35·WH51 을 사면
    같은 동글로 들어오기 때문이다. 그러나 WS90 하나만 달아 놓고 '엽면습윤도
    받는다'고 보고하면 없는 센서를 있다고 적는 셈이 된다.
    """
    assert "leaf_wetness" in rtl433.RTL433_MAP
    assert "moisture" in rtl433.RTL433_MAP
    assert "leaf_wetness" not in rtl433.MODEL_FIELDS["Fineoffset-WS90"]

    frame = coverage.coverage(config)
    assert status_of(frame, "leaf_wetness") != "연결됨"
    assert status_of(frame, "soil_moisture") == "연결됨"   # SOIL-T1 이 보낸다


def test_모르는_모델은_받는다고_단정하지_않는다(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text("rf:\n  Acme-Unknown:\n    sensor_id: X-01\n", encoding="utf-8")
    frame = coverage.coverage(path)
    # 무엇을 보내는지 알 수 없으므로 '연결됨'이 하나도 없어야 한다
    assert "연결됨" not in set(frame["status"])


def test_받을_장비가_없으면_무엇을_사면_되는지_적는다(config):
    frame = coverage.coverage(config)
    missing = frame[frame["status"] == "장비없음"]
    assert not missing.empty
    assert (missing["note"].str.len() > 10).all()
    assert status_of(frame, "wind_gust_3s") == "장비없음"


def test_실제로_들어온_값은_수신확인으로_올라간다(config):
    observed = pd.DataFrame({
        "ts": pd.to_datetime(["2026-10-14T05:00"] * 2),
        "variable": ["t_air", "valve_state"],
        "value": [5.0, 1.0],
        "sensor_id": ["WS90-01", "VALVE-01"],
    })
    frame = coverage.coverage(config, observed=observed)
    assert status_of(frame, "t_air") == "수신확인"
    assert status_of(frame, "valve_state") == "수신확인"
    assert status_of(frame, "soil_temp") == "연결됨"      # 아직 안 들어왔다
    row = frame[frame["variable"] == "t_air"].iloc[0]
    assert row["rows"] == 1
    assert str(row["last_seen"]).startswith("2026-10-14")


def test_결측만_있는_변수는_수신확인이_아니다(config):
    """행은 있는데 값이 비어 있으면 받은 것이 아니다."""
    observed = pd.DataFrame({
        "ts": pd.to_datetime(["2026-10-14T05:00"]),
        "variable": ["t_air"], "value": [float("nan")], "sensor_id": ["WS90-01"],
    })
    frame = coverage.coverage(config, observed=observed)
    assert status_of(frame, "t_air") == "연결됨"


def test_설정이_없어도_표는_나온다():
    frame = coverage.coverage(None)
    assert len(frame) == len(FIELD_VARIABLES)
    assert "연결됨" not in set(frame["status"])


def test_설정이_깨져도_표는_나온다(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("devices: [이건: 사전이: 아니다", encoding="utf-8")
    frame = coverage.coverage(path)
    assert len(frame) == len(FIELD_VARIABLES)


def test_에코윗_채널을_적으면_연결됨이_된다(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(
        "ecowitt:\n"
        "  gateway: {sensor_id: WS90-01}\n"
        "  channels:\n"
        "    leaf1: {zone: treatment, sensor_id: LW-T1}\n",
        encoding="utf-8")
    frame = coverage.coverage(path)
    assert status_of(frame, "leaf_wetness") == "연결됨"


def test_사람이_읽는_보고서가_모든_상태를_담는다(config):
    text = coverage.format_report(coverage.coverage(config))
    assert "[연결됨]" in text
    assert "[장비없음]" in text
    assert "수집 변수" in text


def test_모델별_필드가_환산표_안에_있다():
    """모델이 보낸다고 적은 필드는 환산표나 진단표에 있어야 한다.

    없으면 그 값은 들어와도 버려지는데, 점검표는 '받는다'고 적게 된다.
    """
    known = set(rtl433.RTL433_MAP) | set(rtl433.CUMULATIVE_MAP) \
        | set(rtl433.RTL433_DIAGNOSTICS) | set(rtl433.IGNORED_FIELDS)
    for model, fields in rtl433.MODEL_FIELDS.items():
        unknown = fields - known
        assert not unknown, f"{model}: {sorted(unknown)} 를 받을 자리가 없다"


def test_에코윗_센서별_필드가_환산표_안에_있다():
    known = set(ecowitt.ECOWITT_MAP) | set(ecowitt.CUMULATIVE_MAP) \
        | set(ecowitt.DIAGNOSTICS) | set(ecowitt.IGNORED)
    for model, fields in ecowitt.SENSOR_FIELDS.items():
        unknown = fields - known
        assert not unknown, f"{model}: {sorted(unknown)} 를 받을 자리가 없다"


def test_기본매핑으로_안_들어오는_변수는_전부_설명이_있다():
    """'장비없음'이 될 수 있는 변수는 왜 없는지, 무엇을 사면 되는지 적혀 있어야 한다.

    '받을 경로가 없다'만 적혀 있으면 읽는 사람이 다음에 무엇을 해야 하는지
    알 수 없다. 변수를 새로 더하는 사람이 그 한 줄을 쓰도록 여기서 막는다.
    """
    available = coverage._zigbee_available()
    for source in (coverage._rf_available(), coverage._ecowitt_available()):
        available.update(source)
    for variable in FIELD_VARIABLES:
        if variable in available or variable in coverage.PROXY_VARIABLES:
            continue
        assert variable in coverage.MISSING_HARDWARE, (
            f"{variable}: 받을 경로가 없는데 왜 없는지 적혀 있지 않다"
        )


def test_측정점_수를_구간별로_센다(tmp_path):
    """1점은 그 센서가 죽으면 판단이 빈다. 몇 점인지가 보여야 외줄을 안다."""
    path = tmp_path / "d.yaml"
    path.write_text(
        "kinds:\n"
        "  soil: {map: {temperature: soil_temp}, sends: [soil_moisture]}\n"
        "  valve: {map: {state: valve_state}}\n"
        "devices:\n"
        "  a: {kind: soil, zone: treatment, sensor_id: S1}\n"
        "  b: {kind: soil, zone: treatment, sensor_id: S2}\n"
        "  c: {kind: soil, zone: control, sensor_id: S3}\n"
        "  v1: {kind: valve, zone: facility, sensor_id: V1}\n"
        "  v2: {kind: valve, zone: facility, sensor_id: V2}\n",
        encoding="utf-8")
    points = coverage.points_per_zone(path)
    by = {(r["zone"], r["variable"]): r["points"] for _, r in points.iterrows()}
    assert by[("treatment", "soil_moisture")] == 2
    assert by[("treatment", "soil_temp")] == 2
    assert by[("control", "soil_moisture")] == 1          # 외줄
    assert ("facility", "valve_state") not in by          # 장치별 변수는 점이 아니다


def test_페어링됐지만_설정에_없는_장치를_찾는다(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text("devices:\n  known: {zone: treatment, sensor_id: K1}\n",
                    encoding="utf-8")
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "zigbee_devices.json").write_text(json.dumps({
        "ts": "2026-10-14T05:00:00",
        "devices": [
            {"friendly_name": "known", "model": "TS0601_soil", "vendor": "Tuya",
             "description": "Soil", "ieee_address": "0x1", "supported": True},
            {"friendly_name": "0xdeadbeef", "model": "TS0201", "vendor": "Tuya",
             "description": "T&H", "ieee_address": "0xdeadbeef", "supported": True},
        ]}), encoding="utf-8")
    missing = coverage.unconfigured_devices(path, capture)
    assert [d["friendly_name"] for d in missing] == ["0xdeadbeef"]
    assert coverage.unconfigured_devices(path, tmp_path / "nope") == []


def test_설정에_있는_장치끼리는_어긋남이_없다(tmp_path):
    """밸브 둘·차단기 둘이 같은 구간에 있어도 장치별로 남으므로 경고가 아니다."""
    path = tmp_path / "d.yaml"
    path.write_text(
        "kinds:\n  valve: {map: {state: valve_state}}\n"
        "devices:\n"
        "  v1: {kind: valve, zone: facility, sensor_id: V1}\n"
        "  v2: {kind: valve, zone: facility, sensor_id: V2}\n",
        encoding="utf-8")
    assert coverage.conflicts(path) == []
