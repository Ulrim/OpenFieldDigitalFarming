"""수집 범위 점검 — 어느 변수가 어느 경로로 들어오는지.

'모든 센서 데이터를 받는다'는 말은 **확인할 수 있어야** 한다. 센서를 달고
코드를 고친 다음 화면에 뭔가 나오는 것을 보고 끝내면, 빠진 변수는 비가 오는
날이나 서리가 내리는 날에야 드러난다. 그때는 이미 늦다.

이 모듈은 설계상 필요한 변수 목록(:data:`ofdf.data.field.FIELD_VARIABLES`)과
세 수집 경로의 실제 매핑표를 맞대어, 변수마다 어디서 들어오는지를 **코드에서
직접 계산한다.** 손으로 적은 표를 두면 매핑을 고친 날 표가 틀어지고, 틀어진
표는 없는 것보다 나쁘다 — 받고 있다고 적혀 있으니 아무도 확인하지 않는다.

    python -m ofdf.cli sensors --devices configs/devices.yaml

상태는 넷이다.

``수신확인``
    수집 파일에 **실제로 값이 들어와 있다.** 설정이 맞다는 가장 확실한
    증거이고, 공인시험 이전에 이것부터 확인한다.
``연결됨``
    설정에 적힌 장치가 그 변수를 보낸다. 아직 받은 기록은 없다 —
    설치 전이거나, 그 조건이 아직 없었거나(비가 안 왔다), 통신이 안 된다.
``경로있음``
    매핑은 있지만 그 변수를 보내는 장치가 설정에 없다. 센서를 달고 설정에
    한 줄 적으면 코드를 고치지 않고 들어온다.
``대리지표``
    실측 장비는 없고 다른 관측값으로 계산한다. 쓸 수는 있으나 실측이 아니다.
``장비없음``
    받을 길이 없다. 무엇을 사면 되는지를 :data:`MISSING_HARDWARE` 에 적어 둔다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ofdf.data import ecowitt, rtl433, zigbee
from ofdf.data.field import FIELD_VARIABLES, PER_DEVICE_VARIABLES

#: 실측 장비가 없을 때 대리지표로 계산되는 변수.
#: :func:`ofdf.features.derived.add_derived` 가 만든다.
PROXY_VARIABLES = {
    "leaf_wetness": "기온·습도·강수·일사로 계산(실측이 있으면 실측이 이긴다)",
    "canopy_temp": "기온·습도·풍속·일사·운량으로 계산",
    "solar": "조도(lux)에서 환산 — WS90 처럼 조도계만 있을 때",
}

#: 받을 장비가 없는 변수와, 무엇을 사면 되는지.
#:
#: 이것만은 손으로 적는다. 코드가 알 수 없는 사실(무엇을 사야 하는가)이고,
#: 추천이 틀려도 수집은 틀어지지 않는다. 다만 **없는 것을 있다고 적지는
#: 않는다** — 상태는 매핑표에서 계산하고, 여기는 설명만 붙인다.
MISSING_HARDWARE: dict[str, str] = {
    "wind_gust_1s": "WS90 은 2초 간격 계측이라 1초 순간최대를 못 만든다. "
                    "초음파 풍속계(RS-485) 또는 1초 샘플링 풍속계가 필요하다",
    "wind_gust_3s": "같은 이유. WS90 은 28초 창 최댓값(wind_gust_window)만 준다. "
                    "강풍 안전규칙을 사업계획서 그대로 적용하려면 참 3초 계측기가 필요하다",
    "soil_ec": "구매한 토양센서는 수분·온도만 잰다. EC 를 보려면 "
               "3-in-1(수분·온도·EC) 센서가 필요하다",
    "leaf_wetness": "엽면습윤 센서가 없다. Ecowitt WH35 를 게이트웨이에 붙이면 "
                    "leafwetness_ch1 로 바로 들어온다(코드 수정 없음)",
    "shade_position": "차광 구동부는 별도 진행으로 확정됐다. 위치 되읽기 센서는 "
                      "구동부와 함께 들어온다",
    "motor_current": "보조회로 차단기(breaker-aux-01)가 전류를 주지만, 그것이 "
                     "구동부 전류인지는 구동부가 붙은 뒤에 정해진다",
    "spray_flow": "살수 라인 유량계가 없다. 관수 라인은 VALVE-01 이 센다. "
                  "살수를 따로 보려면 같은 모델 1대가 더 필요하다",
    "emergency_stop": "비상정지 접점이 없다. Zigbee 접점 입력 모듈(dry contact) "
                      "1대로 받을 수 있다 — contact 필드를 설정에 적으면 된다",
    "rain_detect": "에코윗 게이트웨이 경로면 srain_piezo 로 들어온다. "
                   "RTL-SDR 경로에는 없으므로 강우강도>0 으로 대신 봐야 한다",
    #: 상태 되읽기는 기본 매핑에 둘 수 없다. 밸브의 state 는 열림·닫힘이고
    #: 차단기의 state 는 통전·차단이라 뜻이 갈리기 때문이다. 그래서 설정에
    #: 적지 않으면 '장비없음'으로 나온다 — 장치는 있는데 설정이 없는 것이다.
    "valve_state": "밸브가 state 를 보내지만 설정에 적어야 받는다. "
                   "configs/devices.yaml 의 해당 장치 map 에 "
                   "'state: valve_state' 를 적는다",
    "circuit_state": "차단기가 state 를 보내지만 설정에 적어야 받는다. "
                     "configs/devices.yaml 의 해당 장치 map 에 "
                     "'state: circuit_state' 를 적는다",
    "aux_circuit_state": "보조 회로 차단기의 state 다. 관수펌프와 다른 회로라 "
                         "변수를 따로 둔다 — 같은 변수에 담으면 구간 통합이 "
                         "둘의 중앙값(0.5)을 낸다",
    "aux_power": "보조 회로 차단기의 전력이다. 설정 map 에 'power: aux_power'",
    "aux_energy": "보조 회로 차단기의 적산 전력량이다. map 에 'energy: aux_energy'",
}


def _zigbee_sources(devices: dict[str, zigbee.Device]) -> dict[str, list[str]]:
    """설정에 적힌 Zigbee 장치가 보낼 수 있는 변수."""
    out: dict[str, list[str]] = {}
    for device in devices.values():
        # 설정이 말해 준 필드만 센다. 기본 매핑에 이름이 있다는 것은
        # '그 이름이 오면 이 변수'라는 뜻일 뿐, 이 장치가 그것을 보낸다는
        # 뜻이 아니다 — 누수 센서가 토양수분을 보내지는 않는다.
        for field_name in sorted(set(device.map) | set(device.sends)):
            variable = device.variable_for(field_name)
            if variable is None:
                continue
            out.setdefault(variable, []).append(
                f"Zigbee {device.sensor_id}.{field_name}")
    return out


def _zigbee_available() -> dict[str, list[str]]:
    """Zigbee 경로가 매핑을 가진 변수(장치를 달면 들어온다)."""
    out: dict[str, list[str]] = {}
    for field_name, variable in zigbee.Z2M_DEFAULT_MAP.items():
        out.setdefault(variable, []).append(f"Zigbee {field_name}")
    return out


def _rf_available() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for field_name, variable in rtl433.RTL433_MAP.items():
        out.setdefault(variable, []).append(f"433MHz {field_name}")
    for field_name, variable in rtl433.CUMULATIVE_MAP.items():
        out.setdefault(variable, []).append(f"433MHz {field_name}(적산)")
    out.setdefault("wind_dir", []).append("433MHz wind_dir_deg")
    return out


def _ecowitt_available() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for field_name, (variable, _) in ecowitt.ECOWITT_MAP.items():
        out.setdefault(variable, []).append(f"에코윗 {field_name}")
    for field_name, (variable, _) in ecowitt.CUMULATIVE_MAP.items():
        out.setdefault(variable, []).append(f"에코윗 {field_name}(적산)")
    for pattern, family, variable, _, ambiguous in ecowitt.CHANNEL_PATTERNS:
        label = f"에코윗 {family} 채널" + ("(설정 필요)" if ambiguous else "")
        out.setdefault(variable, []).append(label)
    return out


def observed_variables(frame: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    """수집 자료에 실제로 들어온 변수와 마지막 수신 정보."""
    if frame is None or frame.empty or "variable" not in frame.columns:
        return {}
    out: dict[str, dict[str, Any]] = {}
    valid = frame.dropna(subset=["value"]) if "value" in frame.columns else frame
    for variable, part in valid.groupby("variable"):
        out[str(variable)] = {
            "rows": int(len(part)),
            "last_seen": part["ts"].max() if "ts" in part.columns else None,
            "sensors": sorted({str(s) for s in part.get("sensor_id", [])}),
        }
    return out


def coverage(
    devices_path: str | Path | None = None,
    observed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """변수마다 수집 상태를 매긴 표를 만든다.

    ``observed`` 로 수집 자료(세로 형식)를 넘기면, 실제로 값이 들어온 변수를
    ``수신확인`` 으로 올린다. 설정이 맞다는 가장 확실한 증거다.

    Returns
    -------
    pd.DataFrame
        ``variable, unit, status, paths, note, rows, last_seen`` 컬럼.
        상태 순서대로 정렬한다.
    """
    devices: dict[str, zigbee.Device] = {}
    gateway = None
    channels: dict[str, Any] = {}
    rf: dict[str, Any] = {}

    if devices_path and Path(devices_path).exists():
        try:
            devices = zigbee.load_devices(devices_path)
            gateway, channels = ecowitt.load_channels(devices_path)
            rf = rtl433.load_rf_devices(devices_path)
        except Exception:                 # 설정이 깨져도 표는 나와야 한다
            pass

    connected = _zigbee_sources(devices)
    tentative: dict[str, list[str]] = {}

    for device in rf.values():
        fields = rtl433.MODEL_FIELDS.get(device.model)
        # 모르는 모델은 무엇을 보내는지 알 수 없다. '받는다'고 단정하지 않고
        # '경로있음'으로 둔다 — 없는 센서를 있다고 적는 것이 가장 나쁘다.
        target = connected if fields is not None else tentative
        for field_name in sorted(fields or set(rtl433.RTL433_MAP)):
            variable = device.variable_for(field_name)
            if variable:
                target.setdefault(variable, []).append(
                    f"433MHz {device.sensor_id}.{field_name}")
        for field_name, variable in rtl433.CUMULATIVE_MAP.items():
            if fields is not None and field_name not in fields:
                continue
            target.setdefault(variable, []).append(
                f"433MHz {device.sensor_id}.{field_name}(적산)")
        if fields is None or "wind_dir_deg" in fields:
            target.setdefault("wind_dir", []).append(
                f"433MHz {device.sensor_id}.wind_dir_deg")

    if gateway is not None:
        sent: set[str] = set()
        for model in (gateway.sensors or ecowitt.DEFAULT_SENSORS):
            sent |= ecowitt.SENSOR_FIELDS.get(model, set())
        for field_name, (variable, _) in ecowitt.ECOWITT_MAP.items():
            if field_name in sent:
                connected.setdefault(variable, []).append(
                    f"에코윗 {gateway.sensor_id}.{field_name}")
        for field_name, (variable, _) in ecowitt.CUMULATIVE_MAP.items():
            if field_name in sent:
                connected.setdefault(variable, []).append(
                    f"에코윗 {gateway.sensor_id}.{field_name}(적산)")

    for channel in channels.values():
        for _, family, variable, _, _ in ecowitt.CHANNEL_PATTERNS:
            if family != channel.family:
                continue
            mapped = channel.variable_for(variable)
            if mapped:
                connected.setdefault(mapped, []).append(
                    f"에코윗 {channel.sensor_id}")

    available = _zigbee_available()
    for source in (_rf_available(), _ecowitt_available(), tentative):
        for variable, paths in source.items():
            available.setdefault(variable, []).extend(paths)

    seen = observed_variables(observed)

    rows = []
    for variable, (unit, _, _) in FIELD_VARIABLES.items():
        if variable in seen:
            info = seen[variable]
            status = "수신확인"
            paths = sorted(set(connected.get(variable, [])))
            note = f"{info['rows']:,}행 · " + ", ".join(info["sensors"][:3])
        elif variable in connected:
            status, paths = "연결됨", sorted(set(connected[variable]))
            note = ""
        elif variable in available:
            status, paths = "경로있음", sorted(set(available[variable]))
            note = "센서를 달고 설정에 한 줄 적으면 들어온다"
        elif variable in PROXY_VARIABLES:
            status, paths, note = "대리지표", [], PROXY_VARIABLES[variable]
        else:
            status, paths = "장비없음", []
            note = MISSING_HARDWARE.get(variable, "받을 경로가 없다")
        if status in {"연결됨", "경로있음", "장비없음"} and variable in PROXY_VARIABLES:
            note = f"{note} · 그때까지는 대리지표: {PROXY_VARIABLES[variable]}"
        info = seen.get(variable) or {}
        rows.append({
            "variable": variable, "unit": unit, "status": status,
            "paths": paths, "note": note,
            "rows": info.get("rows", 0), "last_seen": info.get("last_seen"),
        })

    order = {"수신확인": 0, "연결됨": 1, "경로있음": 2, "대리지표": 3, "장비없음": 4}
    frame = pd.DataFrame(rows)
    frame["_o"] = frame["status"].map(order)
    return frame.sort_values(["_o", "variable"]).drop(columns="_o").reset_index(drop=True)


def conflicts(devices_path: str | Path | None) -> list[str]:
    """설정이 서로 어긋나는 곳을 찾는다.

    가장 흔한 실수는 WS90 을 **두 경로로 동시에** 받는 것이다. RTL-SDR 과
    에코윗 게이트웨이는 둘 중 하나만 있으면 되는데, 둘 다 켜 놓고 같은
    센서이름으로 쓰면 같은 비가 두 번 들어온다. 강수는 구간 적산이라 두 배가
    되고, 그 값으로 관수 판단이 돈다 — 비가 왔는데 더 뿌리지 않는 쪽이면
    모르지만, 반대면 작물이 마른다.
    """
    if not devices_path or not Path(devices_path).exists():
        return []
    notes: list[str] = []
    try:
        rf = rtl433.load_rf_devices(devices_path)
        gateway, channels = ecowitt.load_channels(devices_path)
        devices = zigbee.load_devices(devices_path)
    except Exception as exc:
        return [f"설정을 읽지 못했다: {exc}"]

    rf_ids = {d.sensor_id for d in rf.values()}
    if gateway is not None and gateway.sensor_id in rf_ids:
        notes.append(
            f"{gateway.sensor_id}: rf: 와 ecowitt: 가 같은 센서이름을 쓴다. "
            f"두 경로를 동시에 켜면 같은 비가 두 번 들어와 강수가 두 배가 된다. "
            f"RTL-SDR 과 게이트웨이 중 **하나만** 쓰고 나머지 칸은 지운다")

    return notes


def unconfigured_devices(
    devices_path: str | Path | None, capture_dir: str | Path | None
) -> list[dict[str, str]]:
    """페어링은 됐는데 설정에 없는 Zigbee 장치.

    센서를 여러 대 붙이면 반드시 생기는 틈이다. 페어링한 장치의 값은
    '모르는 장치'로 버려지는데, 화면에는 아무것도 안 나오므로 **센서가
    고장난 것과 구별이 안 된다.** 브리지가 남긴 페어링 목록과 설정을 맞대야
    보인다.
    """
    if not devices_path or not capture_dir:
        return []
    path = Path(capture_dir) / "zigbee_devices.json"
    if not path.exists() or not Path(devices_path).exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        devices = zigbee.load_devices(devices_path)
    except Exception:
        return []
    paired = data.get("devices", []) if isinstance(data, dict) else data
    return zigbee.unconfigured([d for d in paired if isinstance(d, dict)], devices)


def points_per_zone(devices_path: str | Path | None) -> pd.DataFrame:
    """구간·변수마다 측정점이 몇 개인지.

    설계는 '초관부 온도 2점·토양 2점'이다. 1점이면 그 센서가 죽는 순간 그
    구간의 판단이 빈다. 몇 점인지가 보여야 어디가 외줄인지 안다.
    """
    columns = ["zone", "variable", "points", "sensors"]
    if not devices_path or not Path(devices_path).exists():
        return pd.DataFrame(columns=columns)
    try:
        devices = zigbee.load_devices(devices_path)
    except Exception:
        return pd.DataFrame(columns=columns)
    rows: dict[tuple[str, str], list[str]] = {}
    for device in devices.values():
        for field_name in sorted(set(device.map) | set(device.sends)):
            variable = device.variable_for(field_name)
            if variable is None or variable not in FIELD_VARIABLES:
                continue
            if variable in PER_DEVICE_VARIABLES:
                continue                   # 장치별 변수는 '점'이 아니다
            rows.setdefault((device.zone, variable), []).append(device.sensor_id)
    out = [{"zone": z, "variable": v, "points": len(ids), "sensors": sorted(ids)}
           for (z, v), ids in rows.items()]
    frame = pd.DataFrame(out, columns=columns)
    return frame.sort_values(["zone", "variable"]).reset_index(drop=True)


def summary(frame: pd.DataFrame) -> dict[str, int]:
    """상태별 개수."""
    return {str(k): int(v) for k, v in frame["status"].value_counts().items()}


def format_report(frame: pd.DataFrame) -> str:
    """사람이 읽을 표로 만든다."""
    counts = summary(frame)
    total = len(frame)
    lines = [
        f"수집 변수 {total}개 — " + " · ".join(
            f"{k} {v}" for k, v in counts.items()),
        "",
    ]
    for status in ("수신확인", "연결됨", "경로있음", "대리지표", "장비없음"):
        part = frame[frame["status"] == status]
        if part.empty:
            continue
        lines.append(f"[{status}] {len(part)}개")
        for _, row in part.iterrows():
            head = f"  {row['variable']:<20} {row['unit']:<7}"
            if status == "수신확인":
                lines.append(f"{head} {row['note']}")
            elif row["paths"]:
                lines.append(f"{head} {', '.join(row['paths'][:3])}")
            else:
                lines.append(f"{head} {row['note']}")
        lines.append("")
    return "\n".join(lines)
