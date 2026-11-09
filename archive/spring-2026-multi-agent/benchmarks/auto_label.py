"""
Auto-populate the Tier-2 labels.json with suggestions from YOLO + OCR.

THE POINT — and the caveat:
Using the same YOLO server to *generate* a ground-truth box that we then
*score* YOLO against is circular. Same for OCR. So the populated labels
are NOT rigorous Tier-2 ground truth; they are *suggestions* that a
human reviewer should verify before being used to publish Tier-2
numbers.

Each filled field carries a `_source` sibling indicating provenance:
  "yolo"     — top detection above the conf threshold
  "easyocr"  — full extracted text concatenated
  "folder"   — derived from the dataset class folder name
  "human"    — verified or hand-corrected (set this when you review)

A `verified` boolean defaults to false. Set it to true after you've
checked the row, so a future "rigorous Tier-2 only" filter can drop
unverified rows.

Run:
    python benchmarks/auto_label.py            # populate in place
    python benchmarks/auto_label.py --input benchmarks/labels.json \\
                                    --output benchmarks/labels_autofilled.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agents import yolo_health, yolo_detect, ocr_health, ocr_extract


def autolabel(label_row: dict, *, conf: float, ocr_conf: float, repo_root: Path) -> dict:
    """Fill in gt_box / gt_text using YOLO + OCR as suggestion engines."""
    img = repo_root / label_row["image"]
    if not img.is_file():
        label_row["notes"] = f"missing file: {img}"
        return label_row

    # ── Box suggestion (YOLO top detection) ───────────────────────────
    y = yolo_detect(img, conf=conf)
    if y.get("ok") and y["n_detections"] > 0:
        top = y["detections"][0]
        box = top.get("box", {})
        label_row["gt_box"] = [box.get("x"), box.get("y"),
                               box.get("w"), box.get("h")]
        label_row["gt_box_source"]    = "yolo"
        label_row["gt_box_yolo_class"] = top.get("class_name")
        label_row["gt_box_yolo_conf"]  = round(top.get("confidence", 0.0), 4)
    else:
        label_row["gt_box"] = None
        label_row["gt_box_source"] = "none"

    # ── Text suggestion (OCR full transcript) ─────────────────────────
    o = ocr_extract(img, conf=ocr_conf)
    if o.get("ok"):
        text = (o.get("raw_text") or "").strip()
        label_row["gt_text"] = text or None
        label_row["gt_text_source"] = "easyocr" if text else "none"
        label_row["gt_text_lines"]  = o.get("n_lines", 0)
        label_row["gt_text_conf"]   = round(o.get("mean_conf", 0.0), 4)
    else:
        label_row["gt_text"] = None
        label_row["gt_text_source"] = "none"

    label_row["gt_topic_source"] = "folder"
    label_row["verified"] = False
    label_row["notes"] = "auto-suggested by YOLO+OCR; verify before scoring"
    return label_row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",   default="benchmarks/labels.json")
    p.add_argument("--output",  default=None,
                   help="defaults to overwriting --input")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--ocr-conf",  type=float, default=0.5)
    p.add_argument("--repo-root", default=".",
                   help="resolve --image paths relative to this directory")
    args = p.parse_args()

    out = args.output or args.input

    # Sanity-check sidecars
    yh = yolo_health()
    oh = ocr_health()
    if not yh.get("ok"):
        print(f"  WARN: YOLO server unreachable: {yh.get('error')[:80]}")
    if not oh.get("ok"):
        print(f"  WARN: OCR server unreachable: {oh.get('error')[:80]}")
    if not (yh.get("ok") or oh.get("ok")):
        print("  ERROR: at least one of YOLO/OCR must be up to populate labels")
        return 1

    payload = json.loads(Path(args.input).read_text())
    labels = payload.get("labels", [])
    print(f"  Populating {len(labels)} labels from {args.input}")
    print(f"  YOLO conf {args.yolo_conf}  •  OCR conf {args.ocr_conf}\n")

    repo_root = Path(args.repo_root).resolve()
    n_box, n_text = 0, 0
    for i, row in enumerate(labels, 1):
        autolabel(row, conf=args.yolo_conf, ocr_conf=args.ocr_conf, repo_root=repo_root)
        if row.get("gt_box"):  n_box  += 1
        if row.get("gt_text"): n_text += 1
        print(f"    [{i}/{len(labels)}] {row['image']}  "
              f"box={'Y' if row.get('gt_box') else '·'}  "
              f"text={'Y' if row.get('gt_text') else '·'}")

    # Update fields description to document the new keys
    payload["fields"].update({
        "gt_box_source":     '"yolo" | "human" | "none"',
        "gt_text_source":    '"easyocr" | "human" | "none"',
        "gt_topic_source":   '"folder" | "human"',
        "verified":          "false until a human has reviewed the row",
        "gt_box_yolo_class": "the COCO class YOLO assigned to the suggested box",
        "gt_box_yolo_conf":  "YOLO confidence on the suggested box",
        "gt_text_lines":     "number of text lines OCR found",
        "gt_text_conf":      "mean confidence across OCR lines",
    })
    payload["tier"]        = 2
    payload["caveat"]      = ("Auto-populated by YOLO + OCR. Circular for Tier-2 "
                               "rigor — verify each row before publishing.")
    payload["auto_filled"] = True
    payload["labels"]      = labels

    Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n  Wrote {out}")
    print(f"    {n_box}/{len(labels)} have an auto-suggested gt_box")
    print(f"    {n_text}/{len(labels)} have an auto-suggested gt_text")
    print("    verified=false on every row — flip to true after human review")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
