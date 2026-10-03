"""게시용 미리보기 페이지를 현장 화면에서 만들어 낸다.

현장 화면(``dashboard/index.html``)은 라즈베리파이가 ``http.server`` 로
그대로 내보내는 파일이라 **완전한 HTML 문서**여야 한다. charset 이 없으면
한글이 깨지고, doctype 이 없으면 쿼크스 모드로 뜨고, viewport 가 없으면
휴대폰이 축소해서 띄운다.

반면 claude.ai 게시 경로는 올린 페이지를 문서 뼈대로 **자동으로 감싼다**.
같은 파일을 그대로 올리면 ``<body>`` 안에 ``<!doctype html><html>`` 이 또
들어가 중첩 문서가 된다.

요구가 반대이므로 한쪽을 원본으로 두고 다른 쪽을 만들어 낸다. 원본은
제어기용이다. 제품이 그쪽이기 때문이다.

    python scripts/build_artifact_page.py --out artifacts/artifact-page.html
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dashboard" / "index.html"

#: 게시 경로가 스스로 넣어 주는 것들. 이 줄들만 떼어낸다.
WRAPPER_SUPPLIED = re.compile(
    r"""^\s*(?:
          <!doctype\s+html>
        | </?html\b[^>]*>
        | </?head\b[^>]*>
        | </?body\b[^>]*>
        | <meta\s+charset=[^>]*>
        | <meta\s+name="viewport"[^>]*>
        | <link\s+rel="icon"[^\n]*>   # href 의 SVG 에 '>' 가 들어 있다
    )\s*$""",
    re.IGNORECASE | re.VERBOSE,
)


def strip_document_shell(page: str) -> str:
    kept = [ln for ln in page.splitlines() if not WRAPPER_SUPPLIED.match(ln)]
    return "\n".join(kept).lstrip("\n") + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=SOURCE)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    page = args.source.read_text(encoding="utf-8")
    out = strip_document_shell(page)

    # 떼어낸 뒤에도 남아 있으면 게시 경로와 충돌한다.
    for tag in ("<!doctype", "<html", "<head>", "<body"):
        if tag in out.lower():
            print(f"문서 뼈대가 남았다: {tag}", file=sys.stderr)
            return 1
    # 떼어내면 안 되는 것이 같이 날아갔는지 본다.
    for keep in ("<title>", "<style>", "<script>", 'id="stale"', "setInterval(load"):
        if keep not in out:
            print(f"있어야 할 것이 사라졌다: {keep}", file=sys.stderr)
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(out, encoding="utf-8")
    print(f"{args.out} ({len(out.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
