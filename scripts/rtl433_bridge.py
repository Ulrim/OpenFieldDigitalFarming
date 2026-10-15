"""rtl_433 -> 현장 수집 파일 브리지 (Ecowitt WS90, 433MHz).

WS90 은 433MHz 전용 RF 라 Zigbee 로 들어오지 않는다. 라즈베리파이에 꽂은
RTL-SDR 동글로 ``rtl_433`` 이 받아 JSON 을 내보내면 이 브리지가 수집
디렉터리에 쌓는다.

    rtl_433 -f 433.92M -F json | python scripts/rtl433_bridge.py --out data/field

``rtl_433`` 을 직접 띄우게 할 수도 있다::

    python scripts/rtl433_bridge.py --spawn --out data/field

저장해 둔 기록을 다시 돌리려면::

    python scripts/rtl433_bridge.py --replay capture/rtl433_20261101.jsonl --out /tmp/x

동글을 꽂고 처음 확인할 때는 점검 모드를 쓴다. 동글·커널 모듈·디코더·
실제 수신까지 순서대로 보고 어디서 막혔는지 말해 준다::

    python scripts/rtl433_bridge.py --check

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


def rtl433_command(frequency: str, *, seconds: int | None = None) -> list[str]:
    """``rtl_433`` 실행 줄.

    ``-M level`` 로 수신 세기(RSSI·SNR)를 함께 받는다. 전파가 약해지는 것은
    서서히 진행하므로 끊기기 전에 보여야 하고, 안테나 위치를 잡을 때도 이
    숫자가 유일한 길잡이다.
    """
    cmd = ["rtl_433", "-f", frequency, "-F", "json", "-M", "level", "-M", "time:iso"]
    if seconds:
        cmd += ["-T", str(seconds)]
    return cmd


def _run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout or "") + (done.stderr or "")
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired as exc:
        return 124, (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else str(exc.stdout or "")


def run_check(args) -> int:
    """동글이 꽂힌 파이에서 '왜 안 들어오나'를 순서대로 가른다.

    각 단계가 앞 단계를 전제한다. 동글이 안 보이면 디코더를 따질 이유가
    없고, 디코더가 있어도 커널이 동글을 TV 수신기로 잡고 있으면 한 줄도
    안 온다. 막힌 곳에서 멈추고 그 고치는 법을 적는다.
    """
    ok = True

    def step(title: str, passed: bool, detail: str, fix: str = "") -> None:
        nonlocal ok
        mark = "✓" if passed else "✗"
        print(f"[{mark}] {title}")
        if detail:
            for line in detail.rstrip().splitlines():
                print(f"      {line}")
        if not passed:
            ok = False
            if fix:
                print(f"      → {fix}")

    # 1. 동글이 USB 에 보이나
    code, out = _run(["lsusb"])
    found = rtl433.find_dongle(out)
    step("USB 동글", bool(found),
         "\n".join(found) if found else "RTL2832U 동글이 lsusb 에 없다",
         "동글을 다른 USB 포트에 꽂는다. 허브를 쓰면 전원 부족일 수 있으니 파이에 직접 꽂는다")
    if not found:
        return 1

    # 2. 커널이 TV 수신기로 가로채지 않았나
    code, out = _run(["lsmod"])
    grabbed = rtl433.dvb_modules_loaded(out)
    step("커널 DVB 모듈 차단", not grabbed,
         f"올라온 모듈: {', '.join(grabbed)}" if grabbed else "가로채는 모듈 없음",
         "sudo bash scripts/setup_rtlsdr.sh 를 돌리거나, "
         "/etc/modprobe.d/blacklist-rtlsdr.conf 에 'blacklist dvb_usb_rtl28xxu' 를 적고 재부팅")
    if grabbed:
        return 1

    # 3. rtl_433 이 있고 WS90 디코더가 있나
    code, out = _run(["rtl_433", "-R", "help"])
    if code == 127:
        step("rtl_433 설치", False, "rtl_433 명령이 없다", "sudo apt install rtl-433")
        return 1
    has = rtl433.decoder_supported(out, "WS90")
    step("Fineoffset-WS90 디코더", has,
         "rtl_433 -R help 에 WS90 있음" if has else "이 rtl_433 에는 WS90 디코더가 없다 (22.11 이전 버전)",
         "sudo apt install rtl-433 으로 22.11 이상을 설치하거나 소스로 빌드")
    if not has:
        return 1

    # 4. 동글을 열 수 있나 (rtl_test 로 1초)
    code, out = _run(["rtl_test", "-t"], timeout=10)
    opened = "Found" in out and "usb_claim_interface error" not in out
    tuner = next((l.strip() for l in out.splitlines() if "tuner" in l.lower()), "")
    step("동글 열기", opened, tuner or out.strip().splitlines()[-1] if out.strip() else "",
         "사용자를 plugdev 그룹에 넣고 다시 로그인: sudo usermod -aG plugdev $USER")
    if not opened:
        return 1

    # 5. 실제로 들린다 — 설정의 모델만 센다
    wanted = set(DEFAULT_MODELS)
    if Path(args.devices).exists():
        try:
            wanted = {d.model for d in rtl433.load_rf_devices(args.devices).values()} or wanted
        except Exception:
            pass
    print(f"[…] {args.listen}초 동안 듣는다 — {', '.join(sorted(wanted))} 를 기다린다")
    code, out = _run(rtl433_command(args.frequency, seconds=args.listen),
                     timeout=args.listen + 15)
    records = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    summary = rtl433.listen_summary(records, args.listen)
    heard = {m: v for m, v in summary.items() if m in wanted}
    others = {m: v for m, v in summary.items() if m not in wanted}
    lines = []
    for model, v in heard.items():
        rssi = f"RSSI {v['rssi_mean']}dB" if v["rssi_mean"] is not None else "세기 없음"
        lines.append(f"{model} id={','.join(v['ids'])}  {v['count']}회 "
                     f"({v['per_minute']}회/분)  {rssi}")
    if others:
        lines.append("이웃 장치(설정에 없어 버림): " +
                     ", ".join(f"{m} {v['count']}회" for m, v in others.items()))
    step("WS90 수신", bool(heard), "\n".join(lines) or "아무것도 안 들렸다",
         "WS90 전원(슈퍼캡 충전)을 확인하고, 안테나를 금속 함체 밖으로 빼서 세우고, "
         "거리를 줄여 다시 시도. 이웃 장치가 들렸으면 동글은 정상이다")
    if heard:
        worst = min(v["rssi_mean"] for v in heard.values() if v["rssi_mean"] is not None) \
            if any(v["rssi_mean"] is not None for v in heard.values()) else None
        if worst is not None and worst < -20:
            print(f"      ※ 세기가 약하다({worst}dB). 설치 후 끊김이 잦으면 안테나 위치·높이를 올린다")

    print()
    print("점검 통과 — 서비스를 띄우면 된다: sudo systemctl enable --now ofdf-rtl433"
          if ok else "막힌 단계부터 고치고 다시 --check")
    return 0 if ok else 1


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
    p.add_argument("--check", action="store_true",
                   help="동글·드라이버·디코더·수신을 순서대로 점검하고 끝낸다")
    p.add_argument("--listen", type=int, default=30,
                   help="--check 에서 전파를 들어 볼 시간(초). WS90 은 약 9초마다 보낸다")
    args = p.parse_args(argv)

    if args.check:
        return run_check(args)

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
        cmd = rtl433_command(args.frequency)
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
