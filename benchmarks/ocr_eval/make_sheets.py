"""
Make the problem sheet you photograph: dataset/sheets.html.

    .venv/bin/python benchmarks/ocr_eval/make_sheets.py
    open dataset/sheets.html

One problem per screen / printed page, in problems.csv order, with math
typeset by KaTeX. Nothing else is on the page (no ids or page numbers), so
the OCR only sees the problem. The browser tab title shows "Problem 3 / 30".

  Screen photos:  full-screen the browser (⌃⌘F), photograph, press → or space.
  Printed photos: File → Print (one problem per page), then photograph each sheet.
  Handwritten:    copy the problems by hand in the same order.

Take the photos in this order, delete any bad shots on the phone, then run
import_photos.py.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from dataset import load_problems

ROOT = Path(__file__).resolve().parents[2]
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Problem 1 / {n}</title>
<link rel="stylesheet" href="{katex}/katex.min.css">
<script defer src="{katex}/katex.min.js"></script>
<script defer src="{katex}/contrib/auto-render.min.js"></script>
<style>
  html, body {{ margin: 0; background: #fff; color: #111; }}
  body {{ font: 36px/1.6 "Times New Roman", Times, serif; }}
  .p {{ box-sizing: border-box; height: 100vh; display: flex; align-items: center;
        padding: 0 9vw; scroll-snap-align: start; }}
  .p div {{ max-width: 1100px; }}
  html {{ scroll-snap-type: y mandatory; }}
  .katex {{ font-size: 1.05em; }}
  @media print {{
    body {{ font-size: 22pt; }}
    .p {{ height: auto; min-height: 90vh; page-break-after: always; break-after: page; padding: 0 0.8in; }}
  }}
</style></head><body>
{pages}
<script>
const pages = [...document.querySelectorAll('.p')];
let i = 0;
function go(k) {{
  i = Math.max(0, Math.min(pages.length - 1, k));
  pages[i].scrollIntoView();
  document.title = `Problem ${{i + 1}} / ${{pages.length}}`;
}}
addEventListener('keydown', e => {{
  if (['ArrowRight', 'ArrowDown', ' ', 'PageDown'].includes(e.key)) {{ e.preventDefault(); go(i + 1); }}
  if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(e.key)) {{ e.preventDefault(); go(i - 1); }}
}});
addEventListener('DOMContentLoaded', () => renderMathInElement(document.body, {{
  delimiters: [{{left: '\\\\[', right: '\\\\]', display: true}}, {{left: '$', right: '$', display: false}}],
  ignoredClasses: ['plain'], throwOnError: false }}));
</script>
</body></html>
"""


def sheet_html(probs) -> str:
    """The sheet page for these problems (also used by render_pages.py, one problem at a time)."""
    pages = []
    for p in probs:
        cls = "p plain" if p.format == "plain" else "p"       # GSM8K: $ is money, don't typeset
        pages.append(f'<section class="{cls}" id="{p.problem_id}"><div>{html.escape(p.reference)}</div></section>')
    return PAGE.format(n=len(probs), katex=KATEX, pages="\n".join(pages))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    a = ap.parse_args()
    probs = list(load_problems(a.dataset).values())
    out = a.dataset / "sheets.html"
    out.write_text(sheet_html(probs), encoding="utf-8")
    (a.dataset / "sheet_order.json").write_text(json.dumps([p.problem_id for p in probs], indent=1))
    print(f"wrote {out} ({len(probs)} problems)")


if __name__ == "__main__":
    main()
