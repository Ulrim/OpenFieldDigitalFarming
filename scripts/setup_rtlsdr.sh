#!/usr/bin/env bash
# RTL-SDR 동글(V2·V3·Nooelec, R820T2) 을 라즈베리파이에 붙인다.
#
#   sudo bash scripts/setup_rtlsdr.sh
#
# 하는 일 네 가지. 셋째가 요점이다 — 라즈베리파이 OS 는 동글을 꽂으면
# **TV 수신기로 잡아 버린다**(dvb_usb_rtl28xxu). 그러면 rtl_433 이 장치를
# 열지 못해 한 줄도 안 들어오는데, 오류도 뚜렷하지 않아 반나절을 잃는다.
#
#   1. rtl-sdr(드라이버·rtl_test) 와 rtl-433 설치
#   2. 서비스 사용자(ofdf) 를 plugdev 에 넣어 동글을 열 수 있게
#   3. 커널 DVB 모듈 차단
#   4. 지금 올라와 있는 모듈은 내린다 (재부팅 없이 바로 쓰려고)
#
# R828D(V4) 동글은 이 스크립트로 부족하다. rtl-sdr-blog 드라이버를 따로
# 빌드해야 한다 — docs/라즈베리파이-배포.md 참고.
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "sudo 로 돌려야 한다: sudo bash $0" >&2
  exit 1
fi

OFDF_USER="${OFDF_USER:-ofdf}"

echo "[1/4] 패키지 설치"
apt-get update -qq
apt-get install -y -qq rtl-sdr rtl-433 usbutils

echo "[2/4] $OFDF_USER 를 plugdev 그룹에"
if id "$OFDF_USER" >/dev/null 2>&1; then
  usermod -aG plugdev "$OFDF_USER"
else
  echo "      사용자 $OFDF_USER 가 없다. 엣지 서비스를 만들 때 같이 만든다"
fi
# 지금 로그인한 사람도 바로 쓸 수 있게
if [[ -n "${SUDO_USER:-}" ]]; then usermod -aG plugdev "$SUDO_USER"; fi

echo "[3/4] 커널 DVB 모듈 차단"
cat > /etc/modprobe.d/blacklist-rtlsdr.conf <<'EOF'
# RTL-SDR 동글을 TV 수신기로 잡지 않게 한다. rtl_433 이 직접 연다.
blacklist dvb_usb_rtl28xxu
blacklist rtl2832
blacklist rtl2830
blacklist r820t
EOF

echo "[4/4] 올라와 있는 모듈 내리기"
for m in dvb_usb_rtl28xxu rtl2832 rtl2830 r820t; do
  modprobe -r "$m" 2>/dev/null || true
done

echo
echo "끝. 동글을 뽑았다 다시 꽂고 점검한다:"
echo "    python scripts/rtl433_bridge.py --check"
echo "그룹 변경은 다시 로그인해야 적용된다."
