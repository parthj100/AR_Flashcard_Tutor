"""
Import one condition's phone photos into the kit, matched to problems by
shot order.

Take the photos in sheet order (1 to N), put them in a folder (AirDrop is
fine; HEIC, JPEG and PNG all work), then:

  .venv/bin/python data/tools/import_photos.py --condition printed --from ~/Downloads/printed \\
      --device "iPhone 15" --lighting "room, daylight" --angle "overhead" --writer ""

Photos are ordered by when they were taken (EXIF time, then file time, then
name), checked against the number of problems, and for each one:
  - rotated upright (phones store rotation in metadata, which the OCR services ignore)
  - stripped of all metadata, including GPS location
  - saved as photos/<problem_id>_<condition>.jpg
  - written into manifest.csv with the setup you give

It also writes photos/contact_<condition>.jpg: thumbnails labeled with shot
number and problem ID, so you can check at a glance that nothing is out of
order. Use --dry-run to see the matching without writing anything.
For handwritten sheets written by different people, import each person's
photos with --writer and --shots (e.g. --shots 1-15), or edit the writer
column afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[2]
PHOTO_CONDITIONS = ("printed", "handwritten", "screen")
EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def taken_at(path: Path) -> tuple:
    """Sort key: EXIF capture time, else Spotlight creation date, else file time; then name."""
    t = None
    if path.suffix.lower() in (".jpg", ".jpeg"):
        try:
            exif = Image.open(path).getexif().get_ifd(0x8769)  # Exif IFD
            raw = exif.get(36867) or exif.get(36868)  # DateTimeOriginal / Digitized
            if raw:
                t = datetime.strptime(raw, "%Y:%m:%d %H:%M:%S").timestamp()
        except Exception:
            pass
    if t is None:
        try:
            out = subprocess.run(["mdls", "-raw", "-name", "kMDItemContentCreationDate", str(path)],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            t = datetime.strptime(out[:19], "%Y-%m-%d %H:%M:%S").timestamp()
        except Exception:
            t = path.stat().st_mtime
    # Natural name order as tie-break: "img-2" before "img-10"
    return (t, [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)])


def not_an_image(path: Path) -> str | None:
    """Reason this file is not a usable photo, or None. Catches failed AirDrop
    transfers, which leave small Apple property-list files with photo names."""
    head = path.read_bytes()[:16]
    if head.startswith(b"bplist"):
        return "Apple property list (failed AirDrop transfer), not an image"
    if path.stat().st_size < 10_000:
        return f"only {path.stat().st_size} bytes"
    return None


def load_upright(path: Path, tmp: Path) -> Image.Image:
    if path.suffix.lower() in (".heic", ".heif"):
        jpg = tmp / (path.stem + ".jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", str(path), "--out", str(jpg)],
                       check=True, capture_output=True, timeout=60)
        path = jpg
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # apply the phone's rotation to the pixels
    img = img.convert("RGB")              # fresh image: no EXIF, no GPS
    # Fixed preprocessing: long side 2048 px for every model (pilot: EasyOCR ~2x faster, same CER)
    img.thumbnail((2048, 2048), Image.LANCZOS)
    return img


def parse_shots(spec: str | None, n: int) -> list[int]:
    if not spec:
        return list(range(1, n + 1))
    shots: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        shots += list(range(int(a), int(b or a) + 1))
    if any(s < 1 or s > n for s in shots):
        sys.exit(f"error: --shots must be within 1-{n}")
    return shots


def contact_sheet(items: list[tuple[int, str, Path]], out: Path) -> None:
    thumb, cols = 260, 6
    rows = (len(items) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * (thumb + 28)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (shot, pid, path) in enumerate(items):
        im = Image.open(path)
        im.thumbnail((thumb - 10, thumb - 10))
        x, y = (k % cols) * thumb, (k // cols) * (thumb + 28)
        sheet.paste(im, (x + 5, y + 5))
        draw.text((x + 6, y + thumb + 6), f"#{shot:02d} {pid}", fill="black")
    sheet.save(out, "JPEG", quality=80)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Import phone photos for one condition.")
    ap.add_argument("--kit", type=Path, default=ROOT / "data" / "pilot")
    ap.add_argument("--condition", choices=PHOTO_CONDITIONS, required=True)
    ap.add_argument("--from", dest="src", type=Path, required=True, help="folder of photos")
    ap.add_argument("--shots", help="which shot numbers these photos are, e.g. 1-15 (default: all)")
    ap.add_argument("--device", default="")
    ap.add_argument("--lighting", default="")
    ap.add_argument("--angle", default="")
    ap.add_argument("--writer", default="")
    ap.add_argument("--rotate-shots", help="shots that came out sideways, e.g. 2-20,24 (phone held flat "
                    "over a table can save the wrong orientation)")
    ap.add_argument("--rotate", type=int, choices=(90, 180, 270), default=90,
                    help="counter-clockwise degrees to turn --rotate-shots upright (default 90)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    problems = [json.loads(l) for l in (args.kit / "problems.jsonl").read_text("utf-8").splitlines() if l]
    shots = parse_shots(args.shots, len(problems))
    rotate_shots = set(parse_shots(args.rotate_shots, len(problems))) if args.rotate_shots else set()
    files = sorted((p for p in args.src.expanduser().iterdir() if p.suffix.lower() in EXTS), key=taken_at)
    bad = [(f, why) for f in files if (why := not_an_image(f))]
    if bad:
        print(f"error: {len(bad)} file(s) in {args.src} are not photos:")
        for f, why in bad:
            print(f"  {f.name}: {why}")
        sys.exit("Re-send those photos (or remove them) and run again.")
    if len(files) != len(shots):
        sys.exit(f"error: found {len(files)} photo(s) in {args.src} but expected {len(shots)} "
                 f"(shots {shots[0]}-{shots[-1]}). Remove extra or retake missing shots first.")

    print(f"{'shot':>4}  {'photo':28s} -> problem")
    for shot, f in zip(shots, files):
        print(f"{shot:>4}  {f.name:28s} -> {problems[shot - 1]['id']}")
    if args.dry_run:
        print("Dry run: nothing written.")
        return 0

    photos = args.kit / "photos"
    photos.mkdir(exist_ok=True)
    written: list[tuple[int, str, Path]] = []
    tmp = Path(tempfile.mkdtemp(prefix="lens-import-"))
    try:
        for shot, f in zip(shots, files):
            pid = problems[shot - 1]["id"]
            out = photos / f"{pid}_{args.condition}.jpg"
            img = load_upright(f, tmp)
            if shot in rotate_shots:  # straightening step, recorded in the manifest notes
                img = img.rotate(args.rotate, expand=True)
            img.save(out, "JPEG", quality=92)
            written.append((shot, pid, out))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    manifest = args.kit / "manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        cols, rows = reader.fieldnames, list(reader)
    by_id = {f"{pid}_{args.condition}": (shot, out) for shot, pid, out in written}
    for r in rows:
        if r["image_id"] in by_id:
            r["path"] = by_id[r["image_id"]][1].relative_to(args.kit).as_posix()
            for k in ("device", "lighting", "angle", "writer"):
                if getattr(args, k):
                    r[k] = getattr(args, k)
            note = f"rotated {args.rotate} deg CCW at import (saved sideways)"
            if by_id[r["image_id"]][0] in rotate_shots and note not in r["notes"]:
                r["notes"] = "; ".join(x for x in (r["notes"], note) if x)
    with manifest.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    sheet = photos / f"contact_{args.condition}.jpg"
    contact_sheet(written, sheet)
    print(f"Imported {len(written)} photo(s) into {photos}; manifest updated.")
    print(f"Check the order at a glance: {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
