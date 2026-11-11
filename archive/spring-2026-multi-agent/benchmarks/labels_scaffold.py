"""
Generate a Tier 2 labels.json scaffold for the rewards benchmark.

What this can fill in automatically:
  - gt_topic:    derived from the folder name (e.g. "backpack")
  - clip_routing_id: same as gt_topic — used to score CLIP top-1 routing
  - notes:       a TODO marker so reviewers know where human input is needed

What this CANNOT fill in (needs human or external tool):
  - gt_box:    requires Roboflow / hand annotation. Left as `null`.
  - gt_text:   for OCR images, requires manual transcription. Left as `null`.
  - gt_card_id: optional pointer into prototype/data.js authored cards
                when the topic has a hand-authored card. Pre-filled when an
                obvious match exists.
  - factuality: needs sampled human grading, not stored per-image.

Run:
    python benchmarks/labels_scaffold.py \\
        --images-dir YOLOv8-Detection/dataset/images/raw \\
        --classes microscope calculator backpack periodic_table_poster globe_model safety_goggles \\
        --per-class 5 \\
        --out benchmarks/labels.json

The resulting file is the input to a future Tier-2 version of
benchmark_rewards.py (see BENCHMARKS.md). Reviewers should:
  1. Open the JSON.
  2. Replace "TODO" markers with real values where they can.
  3. Optionally label a few boxes by hand for IoU rewards.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Known mapping from school-object class names to authored card ids in
# prototype/data.js. Extend as we authored more cards.
CARD_ID_HINTS: dict[str, str | None] = {
    "microscope":             None,
    "calculator":             None,
    "backpack":               None,
    "periodic_table_poster":  "periodic-table",
    "globe_model":            None,
    "safety_goggles":         None,
    "textbook":               None,
    "whiteboard":             None,
}


def collect_balanced(images_dir: Path, classes: list[str], per_class: int):
    out = []
    for cls in classes:
        d = images_dir / cls
        if not d.is_dir():
            print(f"  WARN: missing class dir {d}")
            continue
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for p in files[:per_class]:
            out.append((cls, p))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", default="YOLOv8-Detection/dataset/images/raw")
    p.add_argument("--classes", nargs="+",
                   default=["microscope", "calculator", "backpack",
                            "periodic_table_poster", "globe_model", "safety_goggles"])
    p.add_argument("--per-class", type=int, default=5)
    p.add_argument("--out",     default="benchmarks/labels.json")
    args = p.parse_args()

    images = collect_balanced(Path(args.images_dir), args.classes, args.per_class)
    if not images:
        print("  ERROR: no images collected")
        return 1

    labels = []
    repo_root = Path.cwd()
    for cls, path in images:
        try:
            rel = path.resolve().relative_to(repo_root)
        except ValueError:
            rel = path  # fall back to the path as given
        labels.append({
            "image":            str(rel),
            "gt_topic":         cls,                          # auto
            "clip_routing_id":  cls,                          # auto
            "gt_card_id":       CARD_ID_HINTS.get(cls),       # auto (mostly null)
            "gt_box":           None,                         # TODO: hand-label
            "gt_text":          None,                         # TODO: transcribe
            "notes":            "TODO: verify; add gt_box if doing IoU rewards",
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "version": 1,
        "tier": 2,
        "image_count": len(labels),
        "fields": {
            "image":           "repo-relative path",
            "gt_topic":        "folder-name; used for CLIP routing accuracy",
            "clip_routing_id": "expected CLIP top-1 id; usually = gt_topic",
            "gt_card_id":      "authored card id in prototype/data.js, or null",
            "gt_box":          "[x, y, w, h] in pixels, or null until labelled",
            "gt_text":         "ground-truth OCR transcript, or null",
        },
        "labels": labels,
    }, indent=2) + "\n")

    todo_box = sum(1 for r in labels if r["gt_box"] is None)
    todo_txt = sum(1 for r in labels if r["gt_text"] is None)
    print(f"  Wrote {args.out}")
    print(f"    {len(labels)} entries")
    print(f"    {todo_box} still need gt_box (manual)")
    print(f"    {todo_txt} still need gt_text (manual; only matters for OCR rewards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
