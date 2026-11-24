"""
Tests for benchmarks/label_agreement.py.

Run from the repo root:
  python3 -m unittest discover -s benchmarks -p "test_*.py" -v
"""

from __future__ import annotations

import csv
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import label_agreement as la  # noqa: E402


class TestKappa(unittest.TestCase):
    def test_perfect(self):
        self.assertEqual(la.kappa(list("yynn"), list("yynn")), (1.0, 1.0))

    def test_chance_level(self):
        agree, k = la.kappa(list("yynn"), list("ynyn"))
        self.assertAlmostEqual(agree, 0.5)
        self.assertAlmostEqual(k, 0.0)

    def test_known_value(self):
        # 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no: the textbook example, kappa = 0.4
        a = list("y" * 25 + "n" * 25)
        b = list("y" * 20 + "n" * 5 + "y" * 10 + "n" * 15)
        agree, k = la.kappa(a, b)
        self.assertAlmostEqual(agree, 0.7)
        self.assertAlmostEqual(k, 0.4)

    def test_everyone_says_the_same(self):
        self.assertEqual(la.kappa(list("nnn"), list("nnn")), (1.0, 1.0))


class TestMakeAndScore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        (self.dir / "refs.csv").write_text(
            "problem_id,reference\np1,It costs $3.00 each.\np2,She has 18 eggs.\n", encoding="utf-8")
        rows = ["image_id,problem_id,model,condition,split,hypothesis",
                "i1,p1,m,printed,dev,It costs $3.00 each.",
                "i2,p1,m,screen,dev,It costs 53.00 each.",
                "i3,p2,m,printed,dev,She has 13 eggs.",
                "i4,p2,m,screen,dev,She has 18 egs.",
                "i5,p2,m,screen,test,She has 99 eggs."]
        (self.dir / "preds.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        self.out = self.dir / "labels"

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *argv):
        with redirect_stdout(io.StringIO()) as buf:
            la.main(list(argv))
        return buf.getvalue()

    def make(self, *extra):
        return self.run_cli("make", "--refs", str(self.dir / "refs.csv"), "--preds", str(self.dir / "preds.csv"),
                            "--out-dir", str(self.out), "--n", "4", *extra)

    def rows(self, name):
        with (self.out / name).open(encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_sheets_are_blank_and_hide_the_source(self):
        self.make()
        sheet, key = self.rows("labels_A.csv"), self.rows("key.csv")
        self.assertEqual(len(sheet), 4)                       # the test-split row is left out
        self.assertTrue(all(r["critical_error"] == "" for r in sheet))
        self.assertNotIn("model", sheet[0])
        self.assertEqual(sorted(r["critical_error"] for r in key), ["n", "n", "y", "y"])
        self.assertEqual(self.rows("labels_A.csv"), self.rows("labels_B.csv"))

    def test_does_not_overwrite_filled_sheets(self):
        self.make()
        with self.assertRaises(SystemExit):
            self.make()
        self.make("--force")

    def test_score(self):
        self.make()
        key = {r["item"]: r for r in self.rows("key.csv")}
        sheet = self.rows("labels_A.csv")
        for r in sheet:  # labeler A agrees with the scorer on every item
            r["critical_error"] = key[r["item"]]["critical_error"].upper()
            for c in la.CATEGORIES:
                r[c] = "y" if key[r["item"]][c] == "y" else ""
        with (self.out / "labels_A.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=la.SHEET_COLS)
            w.writeheader()
            w.writerows(sheet)
        out = self.run_cli("score", "--dir", str(self.out))
        self.assertIn("labeler A: 4 of 4 items labeled", out)
        self.assertIn("labeler B: 0 of 4 items labeled", out)
        self.assertIn("| scorer vs labeler A | 4 | any critical error | 100% | 1.00 |", out)
        self.assertIn("disagree on \"any critical error\": none", out)

    def test_score_needs_labels(self):
        self.make()
        with self.assertRaises(SystemExit):
            self.run_cli("score", "--dir", str(self.out))


if __name__ == "__main__":
    unittest.main()
