"""rtl_433 -> 현장 수집 파일 브리지 (Ecowitt WS90, 433MHz).

WS90 은 433MHz 전용 RF 라 Zigbee 로 들어오지 않는다. 라즈베리파이에 꽂은
RTL-SDR 동글로 ``rtl_433`` 이 받아 JSON 을 내보내면 이 브리지가 수집
디렉터리에 쌓는다.

    rtl_433 -f 433.92M -F json | python scripts/rtl433_bridge.py --out data/field

``rtl_433`` 을 직접 띄우게 할 수도 있다::

    python scripts/rtl433_bridge.py --spawn --out data/field

저장해 둔 기록을 다시 돌리려면::

    python scripts/rtl433_bridge.py --replay capture/rtl433_20261101.jsonl --out /tmp/x

받은 줄은 ``capture/`` 에 그대로 남는다. 현장에 나가지 않고 같은 자료로
파이프라인을 다시 돌릴 수 있어야 공인시험 재현 절차가 성립한다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf.data import rtl433  # noqa: E402
from ofdf.data.field import ZONE_MAST  # noqa: E402

FILE_PATTERN = "rtl433_%Y%m%d.csv"
CAPTURE_PATTERN = "rtl433_%Y%m%d.jsonl"
COLUMNS = ["ts", "zone", "sensor_id", "variable", "value", "unit", "quality"]

#: 이 모델만 받는다. 이웃집 온도계까지 수집 파일에 섞이면 안 된다.
DEFAULT_MODELS = ("Fineoffset-WS90",)


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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/field")
    p.add_argument("--capture", default="data/field/capture",
                   help="원본 줄 기록 디렉터리. 빈 문자열이면 남기지 않는다")
    p.add_argument("--devices", default="configs/devices.yaml",
                   help="설치 정보 YAML. rf: 칸에서 구간·센서이름을 읽는다")
    p.add_argument("--zone", default=ZONE_MAST, help="설정에 없을 때의 기본 구간")
    p.add_argument("--sensor-id", default="WS90-01")
    p.add_argument("--models", default="",
                   help="받을 모델. 비우면 설정의 rf: 칸, 'all' 이면 모두")
    p.add_argument("--replay", default=None)
    p.add_argument("--spawn", action="store_true",
                   help="rtl_433 을 직접 띄운다(기본은 표준입력으로 받는다)")
    p.add_argument("--frequency", default="433.92M")
    args = p.parse_args(argv)

    out_dir = Path(args.out)
    capture_dir = Path(args.capture) if args.capture else None

    # 설정에 적힌 모델만 받는다. 433MHz 는 페어링이 없어서 **이웃집 온도계도
    # 들어온다** — 수집 파일에 섞이면 품질관리가 통째로 흔들린다.
    rf_devices = {}
    if Path(args.devices).exists():
        try:
            rf_devices = rtl433.load_rf_devices(args.devices)
        except Exception as exc:
            print(f"[경고] 설정을 읽지 못했다: {exc}", file=sys.stderr)
    if args.models == "all":
        wanted = None
    elif args.models:
        wanted = set(args.models.split(","))
    elif rf_devices:
        wanted = {d.model for d in rf_devices.values()}
    else:
        wanted = set(DEFAULT_MODELS)
    if wanted:
        print(f"[대상] {', '.join(sorted(wanted))}")

    if args.replay:
        frame = rtl433.read_capture(args.replay, zone=args.zone,
                                    sensor_id=args.sensor_id)
        if frame.empty:
            print("기록에서 읽은 행이 없다.", file=sys.stderr)
            return 1
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "rtl433_replay.csv"
        frame.to_csv(path, index=False, encoding="utf-8")
        print(f"[출력] {path}  ({len(frame):,}행)")
        print(frame.groupby("variable").size().to_string())
        return 0

    if args.spawn:
        cmd = ["rtl_433", "-f", args.frequency, "-F", "json"]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        except FileNotFoundError:
            print("rtl_433 이 없다. sudo apt install rtl-433", file=sys.stderr)
            return 2
        stream = proc.stdout
        print(f"[실행] {' '.join(cmd)}")
    else:
        stream = sys.stdin
        print("[대기] 표준입력에서 rtl_433 JSON 을 읽는다")

    rain = rtl433.RainAccumulator()
    seen_unknown: set[str] = set()
    total = messages = 0

    try:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if not isinstance(record, dict):
                continue
            model = str(record.get("model", ""))
            if wanted is not None and model not in wanted:
                continue

            messages += 1
            if capture_dir:
                capture_dir.mkdir(parents=True, exist_ok=True)
                name = dt.datetime.now().strftime(CAPTURE_PATTERN)
                with (capture_dir / name).open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")

            unknown: list[str] = []
            rows = rtl433.message_to_rows(
                record, zone=args.zone, sensor_id=args.sensor_id,
                rain=rain, unknown=unknown,
                device=rtl433.match_device(record, rf_devices),
            )
            total += append_rows(out_dir, rows)
            for note in unknown:
                if note not in seen_unknown:
                    seen_unknown.add(note)
                    print(f"[확인 필요] {note}", file=sys.stderr)
    except KeyboardInterrupt:
        pass

    print(f"\n[종료] 메시지 {messages:,}건 / 수집 {total:,}행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
