"""RS-485 수질 센서 -> 현장 수집 파일 브리지 (수질 센서가 있을 때만).

Shrimp365 수질 장비와 같은 USB-RS485 변환기·Nengshi 센서를 읽는다.

    python scripts/modbus_bridge.py                # 설정대로 상주 수집
    python scripts/modbus_bridge.py --scan         # 선을 훑어 ID 찾기
    python scripts/modbus_bridge.py --check        # 지금 한 번 읽어 센서별 상태
    python scripts/modbus_bridge.py --replay data/field/capture/modbus_20261101.jsonl --out /tmp/x

변환기가 없으면(포트 없음) 종료코드 0 으로 조용히 끝난다. 나머지 수집은
영향이 없어야 하고, 서비스가 재시작 루프를 돌아서도 안 된다.

읽은 레지스터는 ``capture/`` 에 그대로 남는다. 해석이 틀렸는지 센서가
이상한지 가리려면 원본이 있어야 한다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf.data import modbus  # noqa: E402

FILE_PATTERN = "modbus_%Y%m%d.csv"
CAPTURE_PATTERN = "modbus_%Y%m%d.jsonl"
COLUMNS = ["ts", "zone", "sensor_id", "variable", "value", "unit", "quality"]


class SerialTransport:
    """포트 하나를 센서 여러 대가 돌려 쓴다. 프레임 사이 침묵과 상대 교체
    여유는 Shrimp365 코드에서 현장 검증된 값을 그대로 쓴다."""

    SLAVE_SWITCH_GAP = 0.05

    def __init__(self, port: str, baudrate: int, timeout: float = 1.0) -> None:
        import serial                                 # pyserial

        self._serial = serial
        self.ser = serial.Serial(port=port, baudrate=baudrate, bytesize=8,
                                 parity="N", stopbits=1, timeout=timeout)
        self.baudrate = baudrate
        self._last_use = 0.0
        self._last_slave: int | None = None

    def _gap(self, slave_id: int) -> float:
        gap = max(0.008, 3.5 * 11.0 / self.baudrate)
        if self._last_slave is not None and slave_id != self._last_slave:
            gap = max(gap, self.SLAVE_SWITCH_GAP)
        return gap

    def read(self, slave_id: int) -> list[int] | None:
        idle = time.monotonic() - self._last_use
        gap = self._gap(slave_id)
        if idle < gap:
            time.sleep(gap - idle)
        self.ser.reset_input_buffer()
        self.ser.write(modbus.build_read_request(slave_id))
        self._last_slave = slave_id
        response = self.ser.read(5 + modbus.REGISTER_COUNT * 2)
        self._last_use = time.monotonic()
        try:
            return modbus.parse_read_response(response, slave_id)
        except modbus.ModbusError:
            self._drain()
            return None

    def _drain(self) -> None:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            pending = self.ser.in_waiting
            if pending:
                self.ser.read(pending)
                continue
            time.sleep(0.02)
            if not self.ser.in_waiting:
                break
        self.ser.reset_input_buffer()
        self._last_use = time.monotonic()

    def close(self) -> None:
        self.ser.close()


def append_rows(out_dir: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / dt.datetime.now().strftime(FILE_PATTERN)
    new = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new:
            handle.write(",".join(COLUMNS) + "\n")
        for row in rows:
            handle.write(",".join(str(row[c]) for c in COLUMNS) + "\n")
    return len(rows)


def capture(capture_dir: Path | None, readings: dict) -> None:
    if not capture_dir:
        return
    capture_dir.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": dt.datetime.now().isoformat(), "readings": readings},
                      ensure_ascii=False)
    with (capture_dir / dt.datetime.now().strftime(CAPTURE_PATTERN)).open(
            "a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def open_transport(config: modbus.Rs485Config) -> SerialTransport | None:
    if not Path(config.port).exists():
        print(f"[없음] {config.port} 가 없다 — USB-RS485 변환기가 안 꽂혔다. "
              f"수질 센서 없이 간다.")
        return None
    try:
        return SerialTransport(config.port, config.baudrate)
    except ImportError:
        print("pyserial 이 없다. pip install pyserial", file=sys.stderr)
        return None
    except Exception as exc:                          # 권한·점유
        print(f"[오류] {config.port} 를 열지 못했다: {exc}  "
              f"(sudo usermod -aG dialout $USER 후 재로그인)", file=sys.stderr)
        return None


def run_scan(transport: SerialTransport) -> int:
    print("[훑기] ID 1~32 — 센서에 쓰기는 하지 않는다")
    result = modbus.scan(transport.read)
    for key, slave in result["assign"].items():
        print(f"  ID {slave:>2}  {modbus.SENSOR_SPECS[key].label}")
    for key, ids in result["conflicts"].items():
        print(f"  ! {modbus.SENSOR_SPECS[key].label} 가 둘 이상: ID {ids} — 하나만 남기거나 ID 를 바꾼다")
    for other in result["others"]:
        print(f"  ? ID {other['id']:>2}  모르는 변환기 ({other['note']})")
    if result["missing"]:
        print(f"  없음: {', '.join(modbus.SENSOR_SPECS[k].label for k in result['missing'])}")
    if result["assign"]:
        print("\nconfigs/devices.yaml 의 rs485.sensors 에 이렇게 적는다:")
        for key, slave in result["assign"].items():
            print(f"    {key}: {slave}")
    return 0


def run_check(transport: SerialTransport, config: modbus.Rs485Config) -> int:
    ok = True
    readings = {}
    for key, slave in config.sensors.items():
        regs = transport.read(slave)
        readings[key] = regs
        label = modbus.SENSOR_SPECS[key].label
        if regs is None:
            ok = False
            print(f"[✗] {label} (ID {slave}) 응답 없음 — 배선·전원(DC 9~24V)·ID 확인")
            continue
        print(f"[✓] {label} (ID {slave})")
    unknown: list[str] = []
    rows = modbus.registers_to_rows(readings, config, unknown=unknown)
    for row in rows:
        mark = "  " if row["quality"] == "ok" else "▲ "
        print(f"    {mark}{row['variable']:<16} {row['value']:>9} {row['unit']}")
    for note in unknown:
        print(f"    ! {note}")
    print("\n점검 통과" if ok and rows else "\n막힌 곳부터 고치고 다시 --check")
    return 0 if ok and rows else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", default="configs/devices.yaml")
    p.add_argument("--out", default="data/field")
    p.add_argument("--capture", default="data/field/capture")
    p.add_argument("--interval", type=int, default=None, help="측정 주기(초). 설정보다 우선")
    p.add_argument("--scan", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--once", action="store_true", help="한 번 읽고 끝낸다")
    p.add_argument("--replay", default=None)
    p.add_argument("--force", action="store_true",
                   help="설정이 enabled: false 여도 돈다 (점검용)")
    args = p.parse_args(argv)

    config = modbus.load_rs485(args.devices) if Path(args.devices).exists() \
        else modbus.Rs485Config()
    out_dir = Path(args.out)
    capture_dir = Path(args.capture) if args.capture else None

    if args.replay:
        rows = []
        for line in Path(args.replay).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            when = dt.datetime.fromisoformat(rec["ts"]) if rec.get("ts") else None
            rows += modbus.registers_to_rows(rec.get("readings", {}), config, now=when)
        frame = modbus.rows_to_frame(rows)
        if frame.empty:
            print("기록에서 읽은 행이 없다.", file=sys.stderr)
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "modbus_replay.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        print(f"[출력] {path}  ({len(frame):,}행)")
        print(frame.groupby("variable")["value"].agg(["count", "min", "max"]).to_string())
        return 0

    if not config.enabled and not (args.force or args.scan or args.check):
        print("[꺼짐] configs/devices.yaml 의 rs485.enabled 가 false 다. 수질 센서 없이 간다.")
        return 0

    transport = open_transport(config)
    if transport is None:
        return 0 if not (args.scan or args.check) else 1

    try:
        if args.scan:
            return run_scan(transport)
        if args.check:
            return run_check(transport, config)

        interval = args.interval or 60
        print(f"[수집] {config.port} · {', '.join(f'{k}=ID{v}' for k, v in config.sensors.items())}"
              f" · {interval}초")
        seen_unknown: set[str] = set()
        total = 0
        while True:
            started = time.monotonic()
            readings = {key: transport.read(slave) for key, slave in config.sensors.items()}
            capture(capture_dir, readings)
            unknown: list[str] = []
            rows = modbus.registers_to_rows(readings, config, unknown=unknown)
            total += append_rows(out_dir, rows)
            for note in unknown:
                if note not in seen_unknown:
                    seen_unknown.add(note)
                    print(f"[확인 필요] {note}", file=sys.stderr)
            dead = [k for k, r in readings.items() if r is None]
            if dead:
                print(f"[응답 없음] {', '.join(dead)}", file=sys.stderr)
            if args.once:
                print(f"[완료] {len(rows)}행")
                return 0
            time.sleep(max(1.0, interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        print(f"\n[종료] 누적 {total:,}행")
        return 0
    finally:
        transport.close()


if __name__ == "__main__":
    raise SystemExit(main())
