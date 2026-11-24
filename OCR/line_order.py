"""
Put OCR text boxes into reading order: top line first, left to right.

EasyOCR returns boxes roughly top to bottom, but a word that sits a little
lower or higher than its neighbours (common in photos taken at an angle and
in handwriting) can come out after the rest of its line. In the pilot this
showed up as words moved to the end of the sentence:

  "James decides to birthday candles ... on candles? buy"

The fix does not change what was read, only the order:
  1. Go through boxes from top to bottom.
  2. A box joins an existing line when its vertical middle falls inside that
     line's band, or when it overlaps the band by at least half its own
     height. Otherwise it starts a new line.
  3. Lines are ordered top to bottom, boxes inside a line left to right.

A box is a dict with x1, y1, x2, y2 (pixels), as returned by the services.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def _box(item) -> dict:
    return item["bbox"] if isinstance(item, dict) else item.bbox


def reading_order(items: Sequence[T], min_overlap: float = 0.5) -> list[T]:
    """Return items sorted into reading order. Items carry a bbox (dict key or attribute)."""
    lines: list[dict] = []  # each: {"top", "bottom", "items"}
    for it in sorted(items, key=lambda it: (_box(it)["y1"] + _box(it)["y2"]) / 2):
        b = _box(it)
        mid = (b["y1"] + b["y2"]) / 2
        height = max(b["y2"] - b["y1"], 1e-6)
        best, best_overlap = None, 0.0
        for ln in lines:
            overlap = (min(b["y2"], ln["bottom"]) - max(b["y1"], ln["top"])) / height
            inside = ln["top"] <= mid <= ln["bottom"]
            if (inside or overlap >= min_overlap) and overlap > best_overlap:
                best, best_overlap = ln, overlap
        if best is None:
            lines.append({"top": b["y1"], "bottom": b["y2"], "items": [it]})
        else:
            best["items"].append(it)
            # The band follows the average of its boxes, so one tall box
            # (a fraction, a root sign) does not swallow the next line.
            n = len(best["items"])
            best["top"] += (b["y1"] - best["top"]) / n
            best["bottom"] += (b["y2"] - best["bottom"]) / n
    lines.sort(key=lambda ln: (ln["top"] + ln["bottom"]) / 2)
    return [it for ln in lines for it in sorted(ln["items"], key=lambda it: _box(it)["x1"])]
