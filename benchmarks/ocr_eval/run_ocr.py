"""
Step 1: read every photo with every OCR model and save the readings.

    .venv/bin/python benchmarks/ocr_eval/run_ocr.py                       # both models
    .venv/bin/python benchmarks/ocr_eval/run_ocr.py --models pix2text --resume

Runs the models in-process with the same settings as the app (OCR/engines.py),
so no server is needed. Each photo is turned upright from its EXIF tag first
(the "sideways photos" problem in Update #2). Every model gets one untimed
warm-up read so its first photo isn't charged for loading.

Output: one JSON line per (photo, model) in benchmarks/results/readings.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "OCR"))

from dataset import find_photos, load_problems           # noqa: E402
from metrics import latex_to_plain                         # noqa: E402


def load_image(path: Path) -> Image.Image:
    im = Image.open(path)
    return ImageOps.exif_transpose(im).convert("RGB")


def plain_transcript(lines) -> str:
    """The text we score: Pix2Text formulas converted from LaTeX, $ left as currency."""
    out = []
    for ln in lines:
        segs = ln.get("segments")
        if segs:
            out.append(" ".join(latex_to_plain(s["text"]) if s["type"] == "math" else s["text"]
                                for s in segs))
        else:
            out.append(ln["text"])
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "results" / "readings.jsonl")
    ap.add_argument("--models", nargs="+", default=["easyocr", "pix2text"])
    ap.add_argument("--conf", type=float, default=0.0,
                    help="drop lines below this confidence (default 0: score everything the model read)")
    ap.add_argument("--types", nargs="+", help="only these photo types, e.g. printed screen")
    ap.add_argument("--resume", action="store_true", help="skip (photo, model) pairs already in --out")
    args = ap.parse_args()

    from engines import get_engine                       # heavy import, after arg parsing

    problems = load_problems(args.dataset)
    photos = find_photos(args.dataset, problems)
    if args.types:
        photos = [p for p in photos if p.photo_type in args.types]
    if not photos:
        sys.exit(f"No photos found under {args.dataset / 'photos'}")

    done = set()
    if args.resume and args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["photo_id"], r["model"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"

    print(f"{len(photos)} photos x {len(args.models)} models  ->  {args.out}")
    with args.out.open(mode, encoding="utf-8") as f:
        for model in args.models:
            engine = get_engine(model)
            t0 = time.perf_counter()
            engine.load()
            engine.read(load_image(photos[0].path), conf=args.conf)    # warm-up, untimed
            print(f"[{model}] loaded + warm-up in {time.perf_counter() - t0:.1f}s")

            for i, ph in enumerate(photos, 1):
                if (ph.photo_id, model) in done:
                    continue
                im = load_image(ph.path)
                t = time.perf_counter()
                lines = engine.read(im, conf=args.conf)
                secs = time.perf_counter() - t
                rec = {
                    "photo_id": ph.photo_id, "problem_id": ph.problem_id,
                    "photo_type": ph.photo_type, "source": problems[ph.problem_id].source,
                    "model": model, "model_version": engine.version,
                    "model_settings": engine.settings, "conf": args.conf,
                    "seconds": round(secs, 4), "image_size": [im.width, im.height],
                    "raw_text": " ".join(l["text"] for l in lines),
                    "plain_text": plain_transcript(lines),
                    "lines": lines,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{model}] {i:>3}/{len(photos)} {ph.photo_id:<36} {secs:6.2f}s")


if __name__ == "__main__":
    main()
