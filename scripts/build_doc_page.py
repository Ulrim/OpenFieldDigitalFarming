"""docs/ 의 마크다운 해설서를 게시용 HTML 한 장으로 만든다.

    python scripts/build_doc_page.py docs/센서수집-기술해설.md --out /tmp/page.html

외부 라이브러리 없이 이 저장소 문서가 쓰는 문법만 다룬다 — 제목, 문단,
표, 코드 블록, 목록, 인용, 굵게·코드 인라인. 게시 경로가 문서 뼈대를
감싸므로 doctype·html·head·body 는 넣지 않는다 (build_artifact_page.py 와
같은 이유).
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return text


def slug(text: str, used: set[str]) -> str:
    base = re.sub(r"[^\w가-힣]+", "-", text).strip("-").lower() or "s"
    out, n = base, 2
    while out in used:
        out, n = f"{base}-{n}", n + 1
    used.add(out)
    return out


def convert(md: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = md.splitlines()
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []
    used: set[str] = set()
    i = 0
    para: list[str] = []

    def flush_para() -> None:
        if para:
            out.append(f"<p>{inline(' '.join(para))}</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_para()
            lang = stripped[3:].strip()
            block: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out.append(f'<pre data-lang="{html.escape(lang)}"><code>'
                       f"{html.escape(chr(10).join(block))}</code></pre>")
            continue

        m = re.match(r"^(#{1,3})\s+(.*)$", stripped)
        if m:
            flush_para()
            level = len(m.group(1))
            text = m.group(2).strip()
            if level == 1:
                out.append(f"<h1>{inline(text)}</h1>")
            else:
                sid = slug(text, used)
                toc.append((level, sid, text))
                out.append(f'<h{level} id="{sid}">{inline(text)}</h{level}>')
            i += 1
            continue

        if stripped in ("---", "***"):
            flush_para()
            out.append("<hr>")
            i += 1
            continue

        if stripped.startswith(">"):
            flush_para()
            quote: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(f"<blockquote><p>{inline(' '.join(quote))}</p></blockquote>")
            continue

        if stripped.startswith("|"):
            flush_para()
            rows: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip())
                i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            if len(cells) >= 2 and all(re.match(r"^:?-+:?$", c) for c in cells[1]):
                head, body = cells[0], cells[2:]
            else:
                head, body = None, cells
            parts = ['<div class="tw"><table>']
            if head:
                parts.append("<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head)
                             + "</tr></thead>")
            parts.append("<tbody>")
            for r in body:
                parts.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            parts.append("</tbody></table></div>")
            out.append("".join(parts))
            continue

        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            flush_para()
            ordered = m.group(2)[0].isdigit()
            tag = "ol" if ordered else "ul"
            items: list[str] = []
            while i < len(lines):
                m2 = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if m2:
                    items.append(m2.group(3))
                    i += 1
                elif lines[i].startswith("  ") and lines[i].strip() and items:
                    items[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            out.append(f"<{tag}>" + "".join(f"<li>{inline(t)}</li>" for t in items) + f"</{tag}>")
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        para.append(stripped)
        i += 1
    flush_para()
    return "\n".join(out), toc


STYLE = """
<style>
/* 레이아웃: 한 단 읽기 폭(70자), 왼쪽에 얇은 차례. 현장 화면과 같은 글꼴 계열. */
:root {
  --bg: #fbfaf7; --surface: #ffffff; --ink: #1f2320; --ink-2: #4b5048; --ink-3: #7c8072;
  --line: #e2e0d8; --accent: #1e6b3a; --accent-soft: #e6f1ea; --code-bg: #f1f0ea;
  --font-ui: "IBM Plex Sans KR", system-ui, "Noto Sans KR", "Malgun Gothic", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #15171a; --surface: #1c1f23; --ink: #e8e6e0; --ink-2: #b9b7b0; --ink-3: #868b84;
  --line: #2d3136; --accent: #6dc48a; --accent-soft: #1d2f24; --code-bg: #22262b; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #15171a; --surface: #1c1f23; --ink: #e8e6e0; --ink-2: #b9b7b0; --ink-3: #868b84;
  --line: #2d3136; --accent: #6dc48a; --accent-soft: #1d2f24; --code-bg: #22262b; color-scheme: dark; }

body { background: var(--bg); color: var(--ink); font-family: var(--font-ui); font-size: 15.5px;
  line-height: 1.7; padding-block: 0 48px;
  padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.wrap { max-width: 1080px; margin: 0 auto; display: grid; grid-template-columns: 220px minmax(0, 1fr); gap: 40px; }
@media (max-width: 760px) { .wrap { grid-template-columns: 1fr; gap: 0; } }

header.top { grid-column: 1 / -1; padding: 28px 0 18px; border-bottom: 1px solid var(--line); margin-bottom: 20px; }
header .eyebrow { font-size: 11.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
header h1 { font-size: clamp(22px, 3.4vw, 30px); margin: 4px 0 6px; letter-spacing: -0.01em; text-wrap: balance; }
header p { margin: 0; color: var(--ink-3); font-size: 13.5px; }

nav.toc { position: sticky; top: env(safe-area-inset-top, 0px); align-self: start; padding-top: 6px; font-size: 13px; }
nav.toc .lbl { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-3); margin-bottom: 6px; }
nav.toc a { display: block; color: var(--ink-2); text-decoration: none; padding: 3px 0 3px 10px; border-left: 2px solid var(--line); }
nav.toc a code { background: none; padding: 0; font-size: .92em; }
nav.toc a.l3 { padding-left: 22px; color: var(--ink-3); font-size: 12.5px; }
nav.toc a:hover, nav.toc a:focus-visible { color: var(--accent); border-left-color: var(--accent); outline: none; }
@media (max-width: 760px) { nav.toc { position: static; margin-bottom: 24px; } nav.toc a.l3 { display: none; } }

main { min-width: 0; max-width: 72ch; }
main h2 { font-size: 20px; margin: 44px 0 12px; padding-top: 10px; letter-spacing: -0.01em; text-wrap: balance; }
main h2:first-of-type { margin-top: 0; }
main h3 { font-size: 16px; margin: 28px 0 8px; color: var(--ink); }
main p { margin: 0 0 14px; }
main hr { border: 0; border-top: 1px solid var(--line); margin: 36px 0; }
main hr:first-child { display: none; }
main hr + h2 { margin-top: 0; }
main ul, main ol { padding-left: 22px; margin: 0 0 14px; }
main li { margin: 4px 0; }
main blockquote { margin: 0 0 18px; padding: 10px 16px; border-left: 3px solid var(--accent); background: var(--accent-soft); border-radius: 0 6px 6px 0; }
main blockquote p { margin: 0; }
code { font-family: var(--font-mono); font-size: .9em; background: var(--code-bg); padding: 1px 5px; border-radius: 4px; }
pre { background: var(--code-bg); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; overflow-x: auto;
  margin: 0 0 18px; font-size: 12.8px; line-height: 1.55; }
pre code { background: none; padding: 0; font-size: inherit; }
.tw { overflow-x: auto; margin: 0 0 18px; border: 1px solid var(--line); border-radius: 8px; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { font-size: 12px; letter-spacing: .04em; color: var(--ink-3); background: var(--surface); font-weight: 600; }
tr:last-child td { border-bottom: 0; }
td code, th code { white-space: nowrap; }
footer { grid-column: 1 / -1; margin-top: 40px; padding-top: 14px; border-top: 1px solid var(--line); color: var(--ink-3); font-size: 12.5px; }
</style>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eyebrow", default="노지작물 재해위험 AI 현장제어 · 기술 해설")
    args = ap.parse_args(argv)

    md = args.source.read_text(encoding="utf-8")
    title_m = re.search(r"^#\s+(.*)$", md, re.MULTILINE)
    title = title_m.group(1).strip() if title_m else args.source.stem
    body_md = md[title_m.end():] if title_m else md
    # 머리의 인용(문서 설명)은 헤더 부제로 쓴다
    sub = ""
    m = re.match(r"\s*((?:>.*\n?)+)", body_md)
    if m:
        sub = " ".join(l.strip()[1:].strip() for l in m.group(1).splitlines())
        body_md = body_md[m.end():]

    body, toc = convert(body_md)
    short_title = title.split("—")[0].strip()
    nav = "\n".join(
        f'<a class="l{lvl}" href="#{sid}">{inline(text)}</a>' for lvl, sid, text in toc)
    page = f"""<title>{html.escape(short_title)}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
{STYLE}
<div class="wrap">
<header class="top">
  <div class="eyebrow">{html.escape(args.eyebrow)}</div>
  <h1>{inline(title)}</h1>
  <p>{inline(sub)}</p>
</header>
<nav class="toc" aria-label="차례"><div class="lbl">차례</div>
{nav}
</nav>
<main>
{body}
</main>
<footer>원본: <code>{html.escape(str(args.source))}</code> · 저장소 문서와 같은 내용이다. 고칠 때는 마크다운을 고치고 이 페이지를 다시 만든다.</footer>
</div>
"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page, encoding="utf-8")
    print(f"[출력] {args.out}  ({len(page):,}자, 절 {len(toc)}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
