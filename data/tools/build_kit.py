"""
Build the capture kit for a problem set (default: data/pilot/).

Reads problems.jsonl (from select_pilot.py) and writes, next to it:
  sheets.html / sheets.pdf  one problem per page, for the printed condition and
                            to copy from for the handwritten one
  screen.html               one problem at a time on a monitor, for the screen
                            condition (arrow keys / space to move)
  synthetic/<id>_synthetic.jpg + synthetic/params.csv
                            each problem rendered and degraded once (blur, low
                            contrast, tilt, noise, JPEG artifacts), seeded per
                            problem so it is reproducible
  manifest.csv              one row per problem x condition; synthetic paths are
                            filled in, photo paths are filled by import_photos.py

The problem ID never appears inside the photographed area, because OCR would
read its digits as extra numbers. Sheets put the shot number and ID below a
fold line; the screen page shows them outside the problem card.

MATH problems are typeset with KaTeX (loaded from cdnjs); GSM8K problems are
shown as plain text so their dollar signs are not read as math.
PDF and synthetic renders use headless Google Chrome.

Run:
  .venv/bin/python data/tools/build_kit.py
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9"
CONDITIONS = ("printed", "handwritten", "screen", "synthetic")

HEAD = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{title}}</title>
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"></script>
<style>
  :root {{ --ink: #111; --muted: #666; --paper: #fff; --line: #bbb; }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; background: var(--paper); color: var(--ink);
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }}
  .problem {{ font-size: 22pt; line-height: 1.55; }}
  .meta {{ color: var(--muted); font-size: 10pt; }}
  {{extra_css}}
</style></head><body>
"""

def head(title: str, extra_css: str) -> str:
    # HEAD is already an f-string (KaTeX URL filled in), so fill the rest by replacement.
    return HEAD.replace("{title}", title).replace("{extra_css}", extra_css)


RENDER_JS = """
<script>
const PROBLEMS = __DATA__;
function fill(el, p) {
  // Full-size fractions everywhere: inline \\frac shrinks nested parts until they are
  // hard to copy by hand and unfair to OCR.
  el.textContent = p.source === "MATH" ? p.text.replace(/\\\\frac/g, "\\\\dfrac") : p.text;
  if (p.source === "MATH") {
    renderMathInElement(el, {delimiters: [
      {left: "$$", right: "$$", display: true}, {left: "\\\\[", right: "\\\\]", display: true},
      {left: "\\\\(", right: "\\\\)", display: false}, {left: "$", right: "$", display: false}],
      throwOnError: false});
  }
}
</script>
"""


def sheets_html(problems: list[dict]) -> str:
    css = """
  @page { size: letter; margin: 0.9in; }
  .page { min-height: 9.2in; display: flex; flex-direction: column; page-break-after: always; }
  .page:last-child { page-break-after: auto; }
  .problem { flex: 1; padding-top: 0.4in; }
  .fold { border-top: 1px dashed var(--line); padding-top: 8pt; display: flex; justify-content: space-between; }
  .cover h1 { font-size: 20pt; } .cover li { margin: 6pt 0; font-size: 12pt; line-height: 1.4; }
"""
    out = [head("Lens pilot sheets", css)]
    out.append(f"""<section class="page cover"><h1>Lens pilot: {len(problems)} problems</h1><ol>
<li><b>Printed:</b> photograph each page in order, keeping everything below the dashed line out of the shot.</li>
<li><b>Handwritten:</b> copy each problem onto its own sheet, exactly as printed. Write the shot number and ID on the <b>back</b>, not the front.</li>
<li>Photograph in this order (1 to {len(problems)}) for every condition; the import script names files by order.</li>
<li>If you mis-copy something, leave it and note the shot number in the manifest's <code>notes</code> column.</li>
</ol><p class="meta">This cover page is not photographed.</p></section>""")
    for i, p in enumerate(problems, 1):
        out.append(f"""<section class="page"><div class="problem" data-i="{i - 1}"></div>
<div class="fold meta"><span>▲ photograph above this line only</span><span>#{i:02d} · {html.escape(p['id'])}</span></div></section>""")
    out.append(RENDER_JS.replace("__DATA__", json.dumps(problems)))
    out.append("""<script>
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".problem").forEach(el => fill(el, PROBLEMS[+el.dataset.i]));
});
</script></body></html>""")
    return "".join(out)


def screen_html(problems: list[dict]) -> str:
    css = """
  body { min-height: 100vh; display: flex; flex-direction: column; }
  .stage { flex: 1; display: flex; align-items: center; justify-content: center; padding: 40px; }
  .card { width: min(1100px, 100%); }
  .problem { font-size: clamp(22px, 3vw, 34px); }
  .bar { display: flex; justify-content: space-between; align-items: center; padding: 12px 20px;
         border-top: 1px solid var(--line); }
  button { font: inherit; padding: 6px 14px; }
  body.bare .bar { display: none; }
  body.bare .stage { padding: 60px; }
  body.bare .problem { font-size: 42px; }
"""
    out = [head("Lens pilot screen", css)]
    out.append("""<div class="stage"><div class="card"><div class="problem" id="problem"></div></div></div>
<div class="bar meta"><button id="prev">◀ Prev</button>
<span id="label"></span><span>← → or space · keep this bar out of the photo</span>
<button id="next">Next ▶</button></div>""")
    out.append(RENDER_JS.replace("__DATA__", json.dumps(problems)))
    out.append("""<script>
document.addEventListener("DOMContentLoaded", () => {
  const q = new URLSearchParams(location.search);
  if (q.get("bare")) document.body.classList.add("bare");
  let i = Math.max(0, Math.min(PROBLEMS.length - 1, (+q.get("i") || 1) - 1));
  const show = () => {
    fill(document.getElementById("problem"), PROBLEMS[i]);
    document.getElementById("label").textContent =
      `#${String(i + 1).padStart(2, "0")} of ${PROBLEMS.length} · ${PROBLEMS[i].id}`;
  };
  const go = d => { i = (i + d + PROBLEMS.length) % PROBLEMS.length; show(); };
  document.getElementById("prev").onclick = () => go(-1);
  document.getElementById("next").onclick = () => go(1);
  document.addEventListener("keydown", e => {
    if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); go(1); }
    if (e.key === "ArrowLeft") go(-1);
  });
  show();
});
</script></body></html>""")
    return "".join(out)


def chrome(args: list[str], profile: Path, output: Path, timeout: float = 60.0) -> None:
    """Run headless Chrome until `output` is written. Headless Chrome sometimes
    finishes writing but never exits, so wait for the file to stop growing and
    then close Chrome ourselves."""
    output.unlink(missing_ok=True)
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--no-first-run",
                             "--no-default-browser-check", "--hide-scrollbars",
                             f"--user-data-dir={profile}", "--virtual-time-budget=8000", *args],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline, last = time.monotonic() + timeout, -1
    try:
        while time.monotonic() < deadline:
            size = output.stat().st_size if output.exists() else -1
            if size > 0 and size == last:
                return
            if proc.poll() is not None and size > 0:
                return
            last = size
            time.sleep(0.5)
        raise RuntimeError(f"Chrome did not produce {output.name} within {timeout:.0f}s")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


def degrade(img: Image.Image, rng: random.Random) -> tuple[Image.Image, dict]:
    p = {
        "rotate_deg": round(rng.uniform(-4, 4), 2),
        "contrast": round(rng.uniform(0.55, 0.85), 2),
        "brightness": round(rng.uniform(0.85, 1.08), 2),
        "blur_radius": round(rng.uniform(0.6, 1.6), 2),
        "noise_sigma": round(rng.uniform(6, 14), 1),
        "scale": round(rng.uniform(0.55, 0.8), 2),
        "jpeg_quality": rng.randint(30, 55),
    }
    img = img.convert("RGB").rotate(p["rotate_deg"], resample=Image.BICUBIC, expand=True,
                                     fillcolor=(255, 255, 255))
    img = ImageEnhance.Contrast(img).enhance(p["contrast"])
    img = ImageEnhance.Brightness(img).enhance(p["brightness"])
    img = img.filter(ImageFilter.GaussianBlur(p["blur_radius"]))
    arr = np.asarray(img).astype(np.float32)
    noise = np.random.default_rng(rng.randrange(2**32)).normal(0, p["noise_sigma"], arr.shape)
    img = Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    img = img.resize((int(img.width * p["scale"]), int(img.height * p["scale"])), Image.LANCZOS)
    return img, p


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the capture kit.")
    ap.add_argument("--kit", type=Path, default=ROOT / "data" / "pilot")
    ap.add_argument("--skip-synthetic", action="store_true")
    args = ap.parse_args(argv)

    problems = [json.loads(l) for l in (args.kit / "problems.jsonl").read_text("utf-8").splitlines() if l]
    for p in problems:  # keep only what the pages need
        p.pop("reference", None)

    (args.kit / "sheets.html").write_text(sheets_html(problems), "utf-8")
    (args.kit / "screen.html").write_text(screen_html(problems), "utf-8")
    print("Wrote sheets.html and screen.html")

    profile = Path(tempfile.mkdtemp(prefix="lens-chrome-"))
    try:
        chrome(["--no-pdf-header-footer", f"--print-to-pdf={args.kit / 'sheets.pdf'}",
                (args.kit / "sheets.html").as_uri()], profile, args.kit / "sheets.pdf")
        print("Wrote sheets.pdf")

        synth_dir = args.kit / "synthetic"
        params_rows = []
        if not args.skip_synthetic:
            synth_dir.mkdir(exist_ok=True)
            raw = profile / "raw.png"
            for i, p in enumerate(problems, 1):
                chrome(["--window-size=1400,1000", f"--screenshot={raw}",
                        f"{(args.kit / 'screen.html').as_uri()}?bare=1&i={i}"], profile, raw)
                img, params = degrade(Image.open(raw), random.Random(f"{p['id']}-synthetic"))
                out = synth_dir / f"{p['id']}_synthetic.jpg"
                img.save(out, "JPEG", quality=params["jpeg_quality"])
                params_rows.append({"problem_id": p["id"], **params})
            with (synth_dir / "params.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(params_rows[0]))
                w.writeheader()
                w.writerows(params_rows)
            print(f"Wrote {len(params_rows)} synthetic images")
    finally:
        shutil.rmtree(profile, ignore_errors=True)

    manifest = args.kit / "manifest.csv"
    if manifest.exists():
        print("manifest.csv exists; left unchanged")
    else:
        cols = ["image_id", "problem_id", "shot", "condition", "split", "path",
                "device", "lighting", "angle", "writer", "notes"]
        with manifest.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for cond in CONDITIONS:
                for i, p in enumerate(problems, 1):
                    synth = cond == "synthetic"
                    w.writerow({"image_id": f"{p['id']}_{cond}", "problem_id": p["id"], "shot": i,
                                "condition": cond, "split": p.get("pilot_split", "dev"),
                                "path": f"synthetic/{p['id']}_synthetic.jpg" if synth else "",
                                "device": "render" if synth else "", "lighting": "render" if synth else "",
                                "angle": "render" if synth else "", "writer": "" , "notes": ""})
        print("Wrote manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
