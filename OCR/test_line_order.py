"""
Tests for OCR/line_order.py.

Run from the repo root:
  python3 -m unittest discover -s OCR -p "test_*.py" -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from line_order import reading_order  # noqa: E402


def box(text, x1, y1, x2, y2):
    return {"text": text, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}


def text(items):
    return " ".join(i["text"] for i in reading_order(items))


class TestReadingOrder(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(reading_order([]), [])

    def test_already_ordered(self):
        items = [box("a", 0, 0, 50, 20), box("b", 60, 0, 100, 20), box("c", 0, 30, 50, 50)]
        self.assertEqual(text(items), "a b c")

    def test_word_sitting_lower_stays_in_its_line(self):
        # Pilot case: "buy" sat a few pixels low and came out last.
        items = [
            box("James decides to", 0, 0, 200, 30),
            box("birthday candles", 260, 2, 460, 32),
            box("for his 2 sons.", 0, 40, 200, 70),
            box("buy", 205, 8, 255, 38),
        ]
        self.assertEqual(text(items), "James decides to buy birthday candles for his 2 sons.")

    def test_slanted_line(self):
        # Each word a little lower than the one before, as in a tilted photo.
        items = [box(w, i * 100, i * 6, i * 100 + 90, i * 6 + 30) for i, w in enumerate("a b c d".split())]
        items += [box(w, i * 100, 60 + i * 6, i * 100 + 90, 90 + i * 6) for i, w in enumerate("e f g h".split())]
        self.assertEqual(text(list(reversed(items))), "a b c d e f g h")

    def test_tall_box_does_not_merge_two_lines(self):
        items = [
            box("x =", 0, 0, 60, 30),
            box("(tall fraction)", 70, -10, 200, 45),
            box("next line", 0, 50, 150, 80),
        ]
        self.assertEqual(text(items), "x = (tall fraction) next line")

    def test_objects_with_bbox_attribute(self):
        class Line:
            def __init__(self, t, b):
                self.text, self.bbox = t, b
        items = [Line("b", {"x1": 60, "y1": 0, "x2": 100, "y2": 20}),
                 Line("a", {"x1": 0, "y1": 1, "x2": 50, "y2": 21})]
        self.assertEqual([i.text for i in reading_order(items)], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
