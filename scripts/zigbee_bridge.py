"""Zigbee2MQTT -> 현장 수집 파일 브리지.

라즈베리파이에서 상주로 돌며 Zigbee 센서 값을 받아 수집 디렉터리에 쌓는다.
엣지 제어기(``ofdf.cli edge``)가 그 파일을 읽는다.

    python scripts/zigbee_bridge.py \
        --devices configs/devices.yaml --out data/field

받은 메시지를 **그대로** ``capture/`` 에 JSON Lines 로도 남긴다. 현장에
나가지 않고 같은 자료로 파이프라인을 다시 돌릴 수 있어야 공인시험의 재현
절차가 성립한다.

의존성
------
MQTT 구독에는 ``paho-mqtt`` 가 필요하다(``pip install paho-mqtt``). 없으면
``--replay`` 로 저장해 둔 기록을 읽어 같은 출력을 만들 수 있다 — 설치 전
형식 검증과 시험에 쓴다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf.data import zigbee  # noqa: E402
from ofdf.data.counter import CumulativeCounter  # noqa: E402

#: 수집 파일을 하루 단위로 나눈다. 한 파일이 끝없이 커지지 않게 한다.
FILE_PATTERN = "zigbee_%Y%m%d.csv"
CAPTURE_PATTERN = "capture_%Y%m%d.jsonl"
COLUMNS = ["ts", "zone", "sensor_id", "variable", "value", "unit", "quality"]


class Writer:
    """수집 행과 원본 메시지를 날짜별 파일에 덧붙인다."""

    def __init__(self, out_dir: Path, capture_dir: Path | None) -> None:
        self.out_dir = out_dir
        self.capture_dir = capture_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        if capture_dir:
            capture_dir.mkdir(parents=True, exist_ok=True)
        self.rows = 0
        self.messages = 0

    def _path(self, when: dt.datetime) -> Path:
        return self.out_dir / when.strftime(FILE_PATTERN)

    def write_rows(self, rows: list[dict]) -> None:
        if not rows:
            return
        path = self._path(dt.datetime.now())
        new = not path.exists()
        with path.open("a", encoding="utf-8") as handle:
            if new:
                handle.write(",".join(COLUMNS) + "\n")
            for row in rows:
                handle.write(",".join(str(row[c]) for c in COLUMNS) + "\n")
        self.rows += len(rows)

    def capture(self, topic: str, payload: str) -> None:
        if not self.capture_dir:
            return
        now = dt.datetime.now()
        line = json.dumps(
            {"ts": now.isoformat(), "topic": topic, "payload": payload},
            ensure_ascii=False,
        )
        with (self.capture_dir / now.strftime(CAPTURE_PATTERN)).open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(line + "\n")
        self.messages += 1


def run_replay(args, devices, writer: Writer) -> int:
    """저장해 둔 기록을 읽어 같은 출력을 만든다."""
    frame = zigbee.read_capture(args.replay, devices, base_topic=args.base_topic)
    if frame.empty:
        print("기록에서 읽은 행이 없다. 장치 이름이 설정과 맞는지 확인한다.",
              file=sys.stderr)
        return 1
    out = Path(args.out) / "zigbee_replay.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False, encoding="utf-8")
    print(f"[출력] {out}  ({len(frame):,}행)")
    print(frame.groupby(["zone", "sensor_id", "variable"]).size().to_string())
    return 0


def run_subscribe(args, devices, writer: Writer) -> int:
    try:
        import paho.mqtt.client as mqtt
    except ImportError:
        print("paho-mqtt 가 없다. pip install paho-mqtt 하거나 "
              "--replay 로 저장된 기록을 쓴다.", file=sys.stderr)
        return 2

    unknown_seen: set[str] = set()
    # 적산값(전력량·관수량)은 메시지 사이에 이어서 세야 한다. 메시지마다
    # 새로 만들면 매번 '처음 본 값'이 되어 구간 증가분이 늘 0 이 된다.
    counter = CumulativeCounter()

    def on_connect(client, userdata, flags, rc, properties=None):
        topic = f"{args.base_topic}/+"
        client.subscribe(topic)
        print(f"[연결] {args.host}:{args.port}  구독 {topic}")

    def on_message(client, userdata, message):
        payload = message.payload.decode("utf-8", errors="replace")
        writer.capture(message.topic, payload)
        unknown: list[str] = []
        try:
            rows = zigbee.message_to_rows(
                message.topic, payload, devices,
                base_topic=args.base_topic, unknown=unknown,
                cumulative=counter,
            )
        except ValueError:
            return                      # JSON 이 아닌 메시지는 넘긴다
        writer.write_rows(rows)
        # 같은 경고를 끝없이 찍지 않는다. 처음 한 번만 알린다.
        for note in unknown:
            if note not in unknown_seen:
                unknown_seen.add(note)
                print(f"[확인 필요] {note}", file=sys.stderr)

    client = mqtt.Client()
    if args.username:
        client.username_pw_set(args.username, args.password or None)
    client.on_connect = on_connect
    client.on_message = on_message

    def stop(signum, frame):
        print(f"\n[종료] 메시지 {writer.messages:,}건 / 수집 {writer.rows:,}행")
        client.disconnect()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    client.connect(args.host, args.port, keepalive=60)
    client.loop_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", default="configs/devices.yaml")
    p.add_argument("--out", default="data/field", help="수집 CSV 출력 디렉터리")
    p.add_argument("--capture", default="data/field/capture",
                   help="원본 메시지 기록 디렉터리. 빈 문자열이면 남기지 않는다")
    p.add_argument("--host", default="127.0.0.1", help="MQTT 브로커")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--username", default=None)
    p.add_argument("--password", default=None)
    p.add_argument("--base-topic", default="zigbee2mqtt")
    p.add_argument("--replay", default=None,
                   help="저장된 기록(JSON Lines)을 읽는다. MQTT 에 붙지 않는다")
    args = p.parse_args(argv)

    devices = zigbee.load_devices(args.devices)
    if not devices:
        print(f"{args.devices} 에 장치가 없다.", file=sys.stderr)
        return 1
    print(f"[설정] 장치 {len(devices)}대")

    writer = Writer(Path(args.out), Path(args.capture) if args.capture else None)
    return (run_replay if args.replay else run_subscribe)(args, devices, writer)


if __name__ == "__main__":
    raise SystemExit(main())
