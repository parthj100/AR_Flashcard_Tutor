"""
Bring phone photos into the dataset, matched to problems by the order you took them.

    .venv/bin/python benchmarks/ocr_eval/import_photos.py --type screen  --from ~/Downloads/screen
    .venv/bin/python benchmarks/ocr_eval/import_photos.py --type printed --from ~/Downloads/printed
    .venv/bin/python benchmarks/ocr_eval/import_photos.py --type handwritten --from ~/Downloads/hw --partial

Photos are sorted by when they were taken (EXIF time, then file name) and
matched to the problems in sheet order (dataset/sheet_order.json, made by
make_sheets.py). iPhone HEIC photos are converted to JPEG with macOS `sips`.

--partial  fewer photos than problems: match them to the first N problems
           (e.g. you hand-copied only the first 10).

Always look at the contact sheet it writes (dataset/photos/<type>_contact.jpg):
each thumbnail is labelled with the problem it was matched to.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from dataset import REAL_TYPES, load_problems

ROOT = Path(__file__).resolve().parents[2]
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def to_jpeg(src: Path, tmp: Path) -> Path:
    if src.suffix.lower() not in (".heic", ".heif"):
        return src
    out = tmp / (src.stem + ".jpg")
    subprocess.run(["sips", "-s", "format", "jpeg", "-s", "formatOptions", "92", str(src), "--out", str(out)],
                   check=True, stdout=subprocess.DEVNULL)
    return out


def taken_at(p: Path) -> str:
    try:
        exif = Image.open(p).getexif()
        return exif.get_ifd(0x8769).get(36867) or exif.get(306) or ""     # DateTimeOriginal, DateTime
    except Exception:
        return ""


def contact_sheet(items, out: Path, cols: int = 6, w: int = 260):
    thumbs = []
    for pid, path in items:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        im.thumbnail((w, w))
        tile = Image.new("RGB", (w, w + 28), "white")
        tile.paste(im, ((w - im.width) // 2, 0))
        ImageDraw.Draw(tile).text((6, w + 6), pid, fill="black")
        thumbs.append(tile)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + 8), rows * (w + 36)), "#ddd")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * (w + 8), (i // cols) * (w + 36)))
    sheet.save(out, quality=85)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--type", required=True, choices=REAL_TYPES)
    ap.add_argument("--from", dest="src", required=True, type=Path)
    ap.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--replace", action="store_true", help="overwrite photos already imported for this type")
    a = ap.parse_args()

    order_file = a.dataset / "sheet_order.json"
    order = (json.loads(order_file.read_text()) if order_file.exists()
             else list(load_problems(a.dataset)))
    files = [p for p in a.src.expanduser().iterdir() if p.suffix.lower() in EXTS]
    if not files:
        sys.exit(f"no photos in {a.src}")
    if len(files) > len(order) or (len(files) < len(order) and not a.partial):
        sys.exit(f"{len(files)} photos but {len(order)} problems. Delete extra/retaken shots, "
                 f"or pass --partial if you photographed only the first N problems.")

    dest = a.dataset / "photos" / a.type
    if dest.exists() and any(dest.iterdir()):
        if not a.replace:
            sys.exit(f"{dest} already has photos; pass --replace to overwrite them")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        jpgs = [(to_jpeg(f, Path(tmp)), f.name) for f in files]
        jpgs.sort(key=lambda x: (taken_at(x[0]), x[1]))
        matched = []
        for pid, (path, name) in zip(order, jpgs):
            out = dest / f"{pid}{path.suffix.lower().replace('.jpeg', '.jpg')}"
            shutil.copy2(path, out)
            matched.append((pid, out))
            print(f"{name:<28} -> {a.type}/{out.name}")
    sheet = a.dataset / "photos" / f"{a.type}_contact.jpg"
    contact_sheet(matched, sheet)
    print(f"\nCheck the matching: open {sheet}")


if __name__ == "__main__":
    main()
