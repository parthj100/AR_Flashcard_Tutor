"""Tests for how run_ocr.py picks and names the OCR tool.

  python -m unittest discover -s benchmarks -p "test_*.py"
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_ocr import pick_model  # noqa: E402

BOTH = {"ok": True, "ocr": "easyocr · pix2text", "default_model": "easyocr",
        "models": {"easyocr": {"label": "easyocr+order", "version": "1.7.2"},
                   "pix2text": {"label": "pix2text", "version": "1.1.7"}}}
ONE = {"ok": True, "ocr": "pix2text 1.1.7 · en", "model": "pix2text", "version": "1.1.7"}


class PickModel(unittest.TestCase):
    def test_default_is_the_services_default(self):
        self.assertEqual(pick_model(BOTH, None), ("easyocr", "easyocr+order", "1.7.2"))

    def test_chosen_model(self):
        self.assertEqual(pick_model(BOTH, "pix2text"), ("pix2text", "pix2text", "1.1.7"))

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            pick_model(BOTH, "tesseract")

    def test_one_model_service(self):
        self.assertEqual(pick_model(ONE, None), (None, "pix2text", "1.1.7"))
        self.assertEqual(pick_model(ONE, "pix2text"), (None, "pix2text", "1.1.7"))

    def test_one_model_service_wrong_model(self):
        with self.assertRaises(ValueError):
            pick_model(ONE, "easyocr")


if __name__ == "__main__":
    unittest.main()
