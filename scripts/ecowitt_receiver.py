"""에코윗 게이트웨이 수신 서버 (와이파이 경로).

WS90 자체에는 와이파이가 없다. 게이트웨이(GW1200·GW2000 등)가 433MHz 를
받아 와이파이로 이 서버에 HTTP POST 한다.

    python scripts/ecowitt_receiver.py --out data/field --port 8088

게이트웨이 설정(웹UI 또는 WSView 앱)
------------------------------------
    Customized 업로드   켜기
    Protocol Type       Ecowitt
    Server / IP         라즈베리파이 주소 (예: 192.168.0.50)
    Path                /data/report/
    Port                8088
    Upload Interval     60 초

**바깥 인터넷을 거치지 않는다.** 같은 공유기 안에서만 오가므로 통신이
끊겨도 돌고, 72시간 독립운전 요건과 어긋나지 않는다.

받은 본문은 ``capture/`` 에 그대로 남는다. 현장에 나가지 않고 같은 자료로
파이프라인을 다시 돌릴 수 있어야 공인시험 재현 절차가 성립한다.

표준 라이브러리만 쓴다. 추가 설치가 없다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ofdf.data import ecowitt  # noqa: E402
from ofdf.data.field import ZONE_MAST  # noqa: E402
from ofdf.data.rtl433 import RainAccumulator  # noqa: E402

FILE_PATTERN = "ecowitt_%Y%m%d.csv"
CAPTURE_PATTERN = "ecowitt_%Y%m%d.jsonl"
COLUMNS = ["ts", "zone", "sensor_id", "variable", "value", "unit", "quality"]


class State:
    def __init__(self, args) -> None:
        self.args = args
        self.out_dir = Path(args.out)
        self.capture_dir = Path(args.capture) if args.capture else None
        self.rain = RainAccumulator()
        self.gateway = None
        self.channels: dict = {}
        if args.devices and Path(args.devices).exists():
            try:
                self.gateway, self.channels = ecowitt.load_channels(args.devices)
            except Exception as exc:
                print(f"[경고] 설정을 읽지 못했다: {exc}", file=sys.stderr)
        self.seen_unknown: set[str] = set()
        self.reports = 0
        self.rows = 0

    def handle(self, body: str) -> int:
        now = dt.datetime.now()
        if self.capture_dir:
            self.capture_dir.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"ts": now.isoformat(), "body": body}, ensure_ascii=False)
            with (self.capture_dir / now.strftime(CAPTURE_PATTERN)).open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(line + "\n")

        unknown: list[str] = []
        rows = ecowitt.report_to_rows(
            body, zone=self.args.zone, sensor_id=self.args.sensor_id,
            rain=self.rain, unknown=unknown,
            gateway=self.gateway, channels=self.channels,
        )
        for note in unknown:
            if note not in self.seen_unknown:
                self.seen_unknown.add(note)
                print(f"[확인 필요] {note}", file=sys.stderr)

        if rows:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / now.strftime(FILE_PATTERN)
            new = not path.exists()
            with path.open("a", encoding="utf-8") as handle:
                if new:
                    handle.write(",".join(COLUMNS) + "\n")
                for row in rows:
                    handle.write(",".join(str(row[c]) for c in COLUMNS) + "\n")
        self.reports += 1
        self.rows += len(rows)
        return len(rows)


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _ok(self) -> None:
            # 게이트웨이는 200 만 확인한다. 본문이 없어도 된다.
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self) -> None:  # noqa: N802
            if state.args.path not in ("*", self.path.split("?")[0]):
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length).decode("utf-8", errors="replace")
            try:
                count = state.handle(body)
            except Exception as exc:              # 한 건 때문에 서버가 죽으면 안 된다
                print(f"[오류] {type(exc).__name__}: {exc}", file=sys.stderr)
                self._ok()
                return
            self._ok()
            if state.reports == 1 or state.reports % 60 == 0:
                print(f"[수신] {state.reports:,}건 / 누적 {state.rows:,}행 "
                      f"(마지막 {count}행)")

        def do_GET(self) -> None:  # noqa: N802
            # 설정이 맞는지 브라우저로 확인할 수 있게 해 둔다
            self.send_response(200)
            text = (f"ofdf 에코윗 수신기\n경로 {state.args.path}\n"
                    f"수신 {state.reports:,}건 / {state.rows:,}행\n").encode()
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(text)))
            self.end_headers()
            self.wfile.write(text)

        def log_message(self, *args) -> None:
            pass                                  # 접속 로그는 남기지 않는다

    return Handler


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/field")
    p.add_argument("--capture", default="data/field/capture")
    p.add_argument("--host", default="0.0.0.0",
                   help="포장 내부망 전용. 공인 IP 로 열지 않는다")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--path", default=ecowitt.DEFAULT_PATH,
                   help="게이트웨이 설정의 Path 와 같아야 한다. '*' 면 아무 경로나 받는다")
    p.add_argument("--devices", default="configs/devices.yaml",
                   help="설치 정보 YAML. ecowitt: 칸에서 게이트웨이·채널을 읽는다")
    p.add_argument("--zone", default=ZONE_MAST, help="설정에 없을 때의 기본 구간")
    p.add_argument("--sensor-id", default="WS90-01")
    p.add_argument("--replay", default=None, help="저장된 기록(JSON Lines)을 읽는다")
    args = p.parse_args(argv)

    state = State(args)

    if args.replay:
        bodies = []
        for line in Path(args.replay).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                bodies.append(json.loads(line)["body"])
            except (ValueError, KeyError):
                continue
        frame = ecowitt.reports_to_frame(
            bodies, zone=args.zone, sensor_id=args.sensor_id,
            gateway=state.gateway, channels=state.channels)
        if frame.empty:
            print("기록에서 읽은 행이 없다.", file=sys.stderr)
            return 1
        out = Path(args.out) / "ecowitt_replay.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out, index=False, encoding="utf-8")
        print(f"[출력] {out}  ({len(frame):,}행)")
        print(frame.groupby("variable")["value"].agg(["count", "min", "max"]).to_string())
        return 0

    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    print(f"[대기] http://{args.host}:{args.port}{args.path}")
    print("       게이트웨이 설정에서 Protocol Type 을 Ecowitt 로 둔다")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"\n[종료] 수신 {state.reports:,}건 / 수집 {state.rows:,}행")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
