"""
Dataset layout for the OCR study.

    dataset/
      problems.csv                   problem_id,source,reference[,format]
      photos/
        printed/      gsm8k_001.jpg  gsm8k_001_take2.jpg ...
        screen/       math_004.png ...
        handwritten/  ...

problems.csv
    problem_id  unique id; each photo's filename must start with it
    source      "gsm8k" or "math"
    reference   the correct text of the problem
    format      "plain" ($ = currency) or "latex" ($..$ = math, \\$ = currency).
                Optional: defaults to latex for MATH problems, plain otherwise,
                so MATH problems can be pasted straight from the dataset.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PHOTO_TYPES = ["printed", "screen", "handwritten", "rendered", "rendered_blur"]
# rendered*: typeset from the text by render_pages.py (not AI-generated); reported separately
REAL_TYPES  = ["printed", "screen", "handwritten"]
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class Problem:
    problem_id: str
    source: str
    reference: str
    format: str


@dataclass
class Photo:
    path: Path
    photo_id: str          # "<photo_type>/<file stem>"
    problem_id: str
    photo_type: str


def load_problems(dataset: Path) -> Dict[str, Problem]:
    path = dataset / "problems.csv"
    probs = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["problem_id"].strip()
            src = row.get("source", "").strip().lower()
            fmt = (row.get("format") or "").strip().lower() or ("latex" if src == "math" else "plain")
            probs[pid] = Problem(pid, src, row["reference"], fmt)
    return probs


def find_photos(dataset: Path, problems: Dict[str, Problem]) -> List[Photo]:
    ids = sorted(problems, key=len, reverse=True)       # longest id wins: math_1 vs math_10
    photos = []
    for ptype in PHOTO_TYPES:
        folder = dataset / "photos" / ptype
        if not folder.is_dir():
            continue
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                if p.suffix.lower() in {".heic", ".heif"}:
                    print(f"skip {p.name}: export HEIC photos as JPEG first", file=sys.stderr)
                continue
            pid = next((i for i in ids if p.stem == i or
                        (p.stem.startswith(i) and not p.stem[len(i)].isalnum())), None)
            if pid is None:
                print(f"skip {ptype}/{p.name}: name doesn't start with a problem_id", file=sys.stderr)
                continue
            photos.append(Photo(p, f"{ptype}/{p.stem}", pid, ptype))
    return photos
