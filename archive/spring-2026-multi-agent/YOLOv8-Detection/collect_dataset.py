"""
School-object dataset collector for YOLOv8 benchmarking.

Scrapes images via Bing, removes exact duplicates via MD5 hashing, and groups
everything by class for Roboflow annotation. The original six classes that
produced the Update 4 benchmark are preserved; the remaining classes were
added in Update 5 to broaden the detection corpus for the comparative
benchmark described in Research Update 5.

Usage:
    python collect_dataset.py                 # collect all classes
    python collect_dataset.py --classes textbook,whiteboard
    python collect_dataset.py --per-class 150
    python collect_dataset.py --output my_dataset/images/raw
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from icrawler.builtin import BingImageCrawler


DEFAULT_OUTPUT_DIR = "dataset/images/raw"
DEFAULT_IMAGES_PER_CLASS = 100


# Classes used in the Update 4 benchmark — kept stable so the committed CSV
# results stay comparable to any re-run of this script.
ORIGINAL_CLASSES = {
    "textbook": [
        "school textbook on desk",
        "open textbook close up",
        "stack of textbooks",
    ],
    "whiteboard": [
        "whiteboard in classroom",
        "whiteboard with writing school",
        "wall mounted whiteboard",
    ],
    "desk_chair": [
        "school desk and chair",
        "student desk classroom",
        "classroom furniture desk chair",
    ],
    "chemistry_flask": [
        "chemistry flask laboratory",
        "erlenmeyer flask science lab",
        "glass beaker flask chemistry",
    ],
    "laptop_computer": [
        "laptop on school desk",
        "student using laptop computer",
        "open laptop close up",
    ],
    "ruler_pencil_pen": [
        "ruler pencil pen on desk",
        "school stationery ruler pencil",
        "pencil and ruler close up",
    ],
}

# Additional classes introduced in Update 5. These are annotated separately
# and merged into the final Roboflow dataset.
EXPANDED_CLASSES = {
    "microscope": [
        "compound microscope school lab",
        "student using microscope in biology class",
        "lab microscope on desk",
    ],
    "calculator": [
        "scientific calculator on desk",
        "graphing calculator close up",
        "calculator and math textbook",
    ],
    "backpack": [
        "school backpack on desk",
        "student wearing backpack in hallway",
        "school bag on classroom floor",
    ],
    "periodic_table_poster": [
        "periodic table poster on classroom wall",
        "periodic table of the elements chart",
        "chemistry classroom with periodic table poster",
    ],
    "globe_model": [
        "classroom globe on teacher desk",
        "school globe of the world",
        "spinning globe in geography class",
    ],
    "safety_goggles": [
        "science lab safety goggles on bench",
        "student wearing safety goggles chemistry",
        "protective lab glasses close up",
    ],
}

ALL_CLASSES = {**ORIGINAL_CLASSES, **EXPANDED_CLASSES}


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def remove_duplicates(folder: Path) -> int:
    """Remove exact duplicate files using MD5 hashing. Returns count removed."""
    seen = set()
    removed = 0
    for f in folder.rglob("*"):
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            h = hashlib.md5(f.read_bytes()).hexdigest()
        except OSError:
            continue
        if h in seen:
            f.unlink(missing_ok=True)
            removed += 1
        else:
            seen.add(h)
    return removed


def collect_images(classes_subset: dict[str, list[str]], output_dir: str, per_class: int) -> dict:
    print("=" * 60)
    print("  School Dataset Image Collector")
    print(f"  Target: {len(classes_subset)} classes x {per_class} images")
    print("=" * 60)

    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    summary = {}

    for class_name, queries in classes_subset.items():
        print(f"\nCollecting: {class_name}")
        class_dir = base / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        per_query = max(1, per_class // len(queries))

        for query in queries:
            print(f"   '{query}' -> {per_query} images")

            tmp_dir = base / "_tmp"
            tmp_dir.mkdir(exist_ok=True)

            try:
                crawler = BingImageCrawler(
                    storage={"root_dir": str(tmp_dir)},
                    feeder_threads=1,
                    parser_threads=1,
                    downloader_threads=4,
                )
                crawler.crawl(
                    keyword=query,
                    max_num=per_query,
                    min_size=(100, 100),
                    overwrite=True,
                )

                for img in tmp_dir.iterdir():
                    if img.suffix.lower() in IMAGE_EXTS:
                        safe_query = query.replace(" ", "_")[:30]
                        dest = class_dir / f"{class_name}_{safe_query}_{img.name}"
                        shutil.move(str(img), str(dest))

            except Exception as e:
                print(f"   ERROR: {e}")
            finally:
                shutil.rmtree(str(tmp_dir), ignore_errors=True)

        dupes = remove_duplicates(class_dir)
        count = len([p for p in class_dir.glob("*.*") if p.suffix.lower() in IMAGE_EXTS])
        summary[class_name] = count
        status = "OK" if count >= int(per_class * 0.8) else "LOW"
        print(f"   [{status}] {count} images saved (removed {dupes} dupes) -> {class_dir}")

    print("\n" + "=" * 60)
    print("  COLLECTION SUMMARY")
    print("=" * 60)
    total = 0
    for class_name, count in summary.items():
        status = "OK " if count >= int(per_class * 0.8) else "LOW"
        print(f"  [{status}] {class_name:<25} {count:>4} images")
        total += count
    print(f"\n  Total images collected : {total}")
    print(f"  Saved to               : {base.resolve()}")
    print("\n  Next step: Label with Roboflow -> https://roboflow.com")
    print("=" * 60)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classes", type=str, default="",
                    help="Comma-separated class subset (default: all). "
                         f"Available: {','.join(ALL_CLASSES.keys())}")
    ap.add_argument("--per-class", type=int, default=DEFAULT_IMAGES_PER_CLASS,
                    help=f"Target images per class (default: {DEFAULT_IMAGES_PER_CLASS})")
    ap.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR,
                    help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    ap.add_argument("--expanded-only", action="store_true",
                    help="Only collect the Update 5 expanded classes")
    args = ap.parse_args()

    if args.expanded_only:
        subset = EXPANDED_CLASSES
    elif args.classes:
        requested = [c.strip() for c in args.classes.split(",") if c.strip()]
        missing = [c for c in requested if c not in ALL_CLASSES]
        if missing:
            raise SystemExit(f"Unknown class(es): {', '.join(missing)}")
        subset = {c: ALL_CLASSES[c] for c in requested}
    else:
        subset = ALL_CLASSES

    collect_images(subset, args.output, args.per_class)


if __name__ == "__main__":
    main()
