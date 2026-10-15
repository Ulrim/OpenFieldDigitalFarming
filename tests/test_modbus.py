"""RS-485 수질 센서 수집 검증.

Shrimp365 와 같은 하드웨어다. 프레임·레지스터 해석이 그쪽과 같아야 두 장비를
오가며 설정을 바꿀 일이 없다. 직렬 포트 없이 돈다.
"""

from __future__ import annotations

import pytest

from ofdf.data import modbus
from ofdf.data.field import FIELD_VARIABLES


def regs(pairs: dict) -> list[int]:
    """{레지스터: (원시값, 소수자릿수, 단위코드)} 로 16개 레지스터를 만든다."""
    out = [0] * 16
    for idx, (raw, decimals, unit) in pairs.items():
        out[idx] = raw
        out[idx + 1] = (decimals << 8) | unit
    return out


PH_REGS = regs({0: (702, 2, 0x0A), 4: (250, 0, 0x00), 8: (213, 1, 0x0B)})
DO_REGS = regs({0: (812, 2, 0x0E), 2: (985, 1, 0x13), 8: (215, 1, 0x0B)})
EC_REGS = regs({0: (1240, 0, 0x07), 2: (620, 0, 0x11), 8: (210, 1, 0x0B)})


def test_CRC_가_MODBUS_표준과_같다():
    """01 04 00 00 00 10 의 CRC 는 F1 C6 (표준 예제)."""
    assert modbus.crc16(bytes.fromhex("010400000010")) == bytes.fromhex("f1c6")


def test_읽기_요청_프레임():
    assert modbus.build_read_request(1) == bytes.fromhex("010400000010f1c6")


def test_응답을_해석한다():
    body = bytes([1, 4, 4, 0x02, 0xBE, 0x02, 0x0A])      # 702, (2자리, pH)
    frame = body + modbus.crc16(body)
    assert modbus.parse_read_response(frame, 1) == [702, 0x020A]


@pytest.mark.parametrize("frame,msg", [
    (b"", "응답 없음"),
    (bytes([2, 4, 2, 0, 0, 0, 0]), "다른 기기"),
    (bytes([1, 0x84, 2, 0, 0]), "예외 응답"),
    (bytes([1, 4, 2, 0, 1, 0xAA, 0xBB]), "CRC"),
])
def test_실패_사유를_고칠_수_있는_말로(frame, msg):
    with pytest.raises(modbus.ModbusError, match=msg):
        modbus.parse_read_response(frame, 1)


def test_값과_단위를_해석한다():
    value, unit = modbus.decode([702, 0x020A], 0)
    assert value == pytest.approx(7.02)
    assert unit == "pH"
    assert modbus.decode([5, 0x0900], 0) is None        # 소수 9자리 = 쓰레기값
    assert modbus.decode([5], 0) is None


def _readings():
    return {"ph": PH_REGS, "do": DO_REGS, "ec": EC_REGS}


def _cfg(**over):
    base = dict(enabled=True, sensors={"ph": 1, "do": 3, "ec": 4})
    base.update(over)
    return modbus.Rs485Config(**base)


def test_세_센서를_우리_변수로_바꾼다():
    rows = modbus.registers_to_rows(_readings(), _cfg())
    values = {r["variable"]: r["value"] for r in rows}
    assert values["water_ph"] == pytest.approx(7.02)
    assert values["water_orp"] == 250.0
    assert values["water_do"] == pytest.approx(8.12)
    assert values["water_do_sat"] == pytest.approx(98.5)
    assert values["water_ec"] == pytest.approx(1.24)          # 1240 uS -> 1.24 mS/cm
    assert values["water_tds"] == 620.0
    assert values["water_salinity"] == pytest.approx(0.682)   # 1240 * 0.55 / 1000
    for r in rows:
        assert r["unit"] == FIELD_VARIABLES[r["variable"]][0]
        assert r["sensor_id"] == "WQ-01"


def test_수온은_하나만_낸다_DO_가_먼저():
    """같은 값을 셋 다 보내면 중앙값이 엉뚱한 센서를 고른다. DO 변환기가 자기
    수온으로 포화도를 보정하므로 그 수온을 남겨야 근거가 맞는다."""
    rows = [r for r in modbus.registers_to_rows(_readings(), _cfg()) if r["variable"] == "water_temp"]
    assert len(rows) == 1
    assert rows[0]["value"] == pytest.approx(21.5)            # DO 센서의 215/10


def test_DO_센서가_없으면_pH_수온을_쓴다():
    readings = {"ph": PH_REGS, "ec": EC_REGS}
    rows = [r for r in modbus.registers_to_rows(readings, _cfg()) if r["variable"] == "water_temp"]
    assert rows[0]["value"] == pytest.approx(21.3)


def test_응답_없는_센서는_건너뛴다():
    rows = modbus.registers_to_rows({"ph": None, "do": DO_REGS}, _cfg())
    assert not any(r["variable"] == "water_ph" for r in rows)
    assert any(r["variable"] == "water_do" for r in rows)


def test_전극_고장값은_버리고_알린다():
    broken = regs({0: (0x7FFF, 2, 0x0A), 8: (213, 1, 0x0B)})
    unknown = []
    rows = modbus.registers_to_rows({"ph": broken}, _cfg(), unknown=unknown)
    assert not any(r["variable"] == "water_ph" for r in rows)
    assert any("전극" in n for n in unknown)


def test_공중의_DO_전극을_잡는다():
    """값은 범위 안(20mg/L)이라 범위검사로는 못 잡는다. 21℃ 포화는 8.9 라 220%."""
    air = regs({0: (2000, 2, 0x0E), 8: (215, 1, 0x0B)})
    unknown = []
    rows = modbus.registers_to_rows({"do": air}, _cfg(), unknown=unknown)
    do = next(r for r in rows if r["variable"] == "water_do")
    assert do["quality"] == "range"
    assert any("공중" in n for n in unknown)


def test_단위_환산():
    assert modbus.normalize("water_temp", 70.0, "F") == pytest.approx(21.11)
    assert modbus.normalize("water_ec", 2.5, "mS") == 2.5
    assert modbus.normalize("water_ec", 0.0025, "S") == 2.5
    assert modbus.normalize("water_do", 8120, "ug/L") == pytest.approx(8.12)
    assert modbus.normalize("diff_pressure", 1.0, "bar") == 100.0
    assert modbus.normalize("diff_pressure", 100.0, "mbar") == 10.0


def test_선을_훑어_ID_를_찾는다():
    bus = {1: PH_REGS, 3: DO_REGS, 4: EC_REGS, 7: EC_REGS,
           9: regs({0: (5, 0, 0x14)})}                    # mbar — 모르는 변환기
    result = modbus.scan(lambda sid: bus.get(sid), 1, 10)
    assert result["assign"] == {"ph": 1, "do": 3, "ec": 4}
    assert result["conflicts"] == {"ec": [4, 7]}
    assert result["others"][0]["id"] == 9
    assert result["missing"] == []


def test_설정을_읽는다(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text("rs485:\n  enabled: true\n  port: /dev/serial0\n  sensors: {ph: 1, ec: 4, flow: 0}\n",
                    encoding="utf-8")
    cfg = modbus.load_rs485(path)
    assert cfg.enabled and cfg.port == "/dev/serial0"
    assert cfg.sensors == {"ph": 1, "ec": 4}                  # flow: 0 은 꺼짐


def test_기본은_꺼짐이다(tmp_path):
    """노지 실증에서 수질은 보조 지표다. 없는 장비를 읽으려 하면 안 된다."""
    path = tmp_path / "d.yaml"
    path.write_text("devices: {}\n", encoding="utf-8")
    cfg = modbus.load_rs485(path)
    assert cfg.enabled is False
    assert cfg.sensors == {}


def test_저장소_설정은_꺼져_있고_켜면_점검표에_연결됨이_된다(tmp_path):
    from pathlib import Path
    from ofdf.data import coverage
    repo = Path(__file__).resolve().parents[1] / "configs" / "devices.yaml"
    assert modbus.load_rs485(repo).enabled is False
    frame = coverage.coverage(repo)
    row = frame[frame["variable"] == "water_ph"].iloc[0]
    assert row["status"] == "경로있음"

    on = tmp_path / "d.yaml"
    on.write_text(repo.read_text(encoding="utf-8").replace("enabled: false", "enabled: true"),
                  encoding="utf-8")
    frame = coverage.coverage(on)
    row = frame[frame["variable"] == "water_ph"].iloc[0]
    assert row["status"] == "연결됨"
    assert any("RS485" in p for p in row["paths"])
