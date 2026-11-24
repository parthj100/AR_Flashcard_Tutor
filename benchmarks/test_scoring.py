"""
Tests for benchmarks/scoring.py and benchmarks/score_ocr.py.

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

import score_ocr  # noqa: E402
from scoring import (  # noqa: E402
    DECIMAL, FRACTION_EXPONENT, NUMBER, SIGN_OPERATOR, UNIT_CURRENCY,
    cer, critical_errors, quantity_tokens, score, wer,
)


def cats(ref: str, hyp: str) -> list[str]:
    return sorted(e.category for e in critical_errors(ref, hyp))


class TestCerWer(unittest.TestCase):
    def test_identical(self):
        s = score("Tom has 3 apples.", "Tom has 3 apples.")
        self.assertEqual((s.cer, s.wer, s.critical_count), (0.0, 0.0, 0))

    def test_cer_value(self):
        self.assertAlmostEqual(cer("abc", "abd")[0], 1 / 3)
        self.assertAlmostEqual(cer("abc", "")[0], 1.0)

    def test_wer_value(self):
        self.assertAlmostEqual(wer("the cat sat", "the bat sat")[0], 1 / 3)
        self.assertAlmostEqual(wer("the cat sat", "the cat")[0], 1 / 3)

    def test_whitespace_and_line_breaks_ignored(self):
        self.assertEqual(cer("Tom has\n3 apples", "Tom  has 3 apples")[0], 0.0)

    def test_empty_reference(self):
        self.assertEqual(cer("", "")[0], 0.0)
        self.assertEqual(cer("", "x")[0], 1.0)


class TestRubricExamples(unittest.TestCase):
    def test_dollar_read_as_five(self):
        # The rubric's own example: "$3.00" read as "53.00".
        errs = critical_errors("It costs $3.00 each.", "It costs 53.00 each.")
        self.assertTrue(errs)
        self.assertEqual(sorted(e.category for e in errs), [NUMBER, UNIT_CURRENCY])

    def test_changed_number(self):
        self.assertEqual(cats("She has 18 eggs", "She has 13 eggs"), [NUMBER])

    def test_decimal_point_lost(self):
        self.assertEqual(cats("It weighs 3.50 kg", "It weighs 350 kg"), [DECIMAL])

    def test_leading_decimal_lost(self):
        self.assertEqual(cats("add .5 cups", "add 5 cups"), [DECIMAL])

    def test_sign_lost(self):
        self.assertEqual(cats("The temperature is -5 degrees", "The temperature is 5 degrees"),
                         [SIGN_OPERATOR])

    def test_operator_changed(self):
        self.assertEqual(cats("8 + 3 = 11", "8 - 3 = 11"), [SIGN_OPERATOR])

    def test_exponent_lost(self):
        self.assertEqual(cats("Find x^2 + 1", "Find x2 + 1"), [FRACTION_EXPONENT])

    def test_exponent_changed(self):
        self.assertEqual(cats("x^2", "x^3"), [FRACTION_EXPONENT])

    def test_fraction_changed(self):
        self.assertEqual(cats("She ate 1/2 of the pie", "She ate 1/3 of the pie"),
                         [FRACTION_EXPONENT])

    def test_special_characters(self):
        self.assertEqual(cats("√(16) = 4", "v(16) = 4"), [SIGN_OPERATOR])
        self.assertEqual(cats("x ≠ 3", "x = 3"), [SIGN_OPERATOR])
        self.assertEqual(cats("y = ±3", "y = +3"), [SIGN_OPERATOR])
        self.assertEqual(cats("area = πr^2", "area = nr^2"), [NUMBER])
        self.assertEqual(cats("x ≥ 2", "x > 2"), [SIGN_OPERATOR])

    def test_currency_symbol_lost(self):
        self.assertEqual(cats("It costs $40", "It costs 40"), [UNIT_CURRENCY])

    def test_unit_word_misread(self):
        self.assertEqual(cats("He ran 5 km", "He ran 5 kn"), [UNIT_CURRENCY])

    def test_percent_lost(self):
        self.assertEqual(cats("a 20% discount", "a 20 discount"), [UNIT_CURRENCY])

    def test_extra_number_inserted(self):
        errs = critical_errors("Tom has 3 apples", "Tom has 3 apples 7")
        self.assertEqual([(e.op, e.category) for e in errs], [("ins", NUMBER)])

    def test_empty_hypothesis_loses_everything(self):
        self.assertEqual(len(critical_errors("Pay $3 for 2 kg", "")), 4)


class TestNotCritical(unittest.TestCase):
    def test_spelling(self):
        s = score("Tom has 3 apples.", "Tom has 3 appies.")
        self.assertGreater(s.cer, 0)
        self.assertTrue(s.critical_free)

    def test_punctuation(self):
        self.assertEqual(cats("It costs $3.00.", "It costs $3.00"), [])
        self.assertEqual(cats("Is it 4.5?", "Is it 4.5 ."), [])

    def test_case(self):
        self.assertEqual(cats("5 KG", "5 kg"), [])

    def test_hyphens_are_not_minus(self):
        self.assertEqual(cats("a half-hour walk", "a half hour walk"), [])
        self.assertEqual(cats("a 5-year-old", "a 5 year old"), [])

    def test_thousands_separator(self):
        self.assertEqual(cats("It costs $1,000", "It costs $1000"), [])

    def test_equivalent_notation(self):
        self.assertEqual(cats("x^2", "x²"), [])
        self.assertEqual(cats("3 × 4", "3 * 4"), [])
        self.assertEqual(cats("3 x 4", "3 * 4"), [])
        self.assertEqual(cats("8 − 3", "8 - 3"), [])
        self.assertEqual(cats("3½ cups", "3 1/2 cups"), [])

    def test_unit_abbreviation_without_number_ignored(self):
        # "km" only counts as a unit right after a number.
        self.assertEqual(cats("the km marker", "the kn marker"), [])

    def test_misspelled_unit_word(self):
        # Spelling of a full unit word is not a quantity change.
        self.assertEqual(cats("she did 50 minutes", "she did 50 minuts"), [])

    def test_variable_x_is_not_times(self):
        # Found in the pilot: "x_1x_2" has "1x2" inside it, which is not multiplication.
        self.assertFalse(any(t.kind == "op" for t in quantity_tokens("Calculate x_1x_2...x_8.")))
        self.assertEqual(cats("Calculate x_1x_2...x_8.", "Calculate x_1x_2...x_8."), [])

    def test_and_or_slash_ignored(self):
        self.assertEqual(quantity_tokens("apples and/or pears"), [])


class TestScoreOcrCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.refs = self.dir / "refs.csv"
        self.refs.write_text(
            "problem_id,reference\n"
            "p1,It costs $3.00 each.\n"
            "p2,She has 18 eggs.\n",
            encoding="utf-8")
        self.preds = self.dir / "preds.csv"
        rows = [
            # image_id, problem_id, model, condition, split, energy_j, hypothesis
            ("i1", "p1", "easyocr", "printed", "dev", "2.0", "It costs $3.00 each."),
            ("i2", "p1", "easyocr", "handwritten", "dev", "2.0", "It costs 53.00 each."),
            ("i3", "p2", "easyocr", "synthetic", "dev", "2.0", "She has 18 eggs"),
            ("i4", "p2", "easyocr", "screen", "test", "2.0", "She has 13 eggs."),
        ]
        with self.preds.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "problem_id", "model", "condition", "split",
                        "energy_j", "hypothesis"])
            w.writerows(rows)

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *extra):
        out = self.dir / "out"
        with redirect_stdout(io.StringIO()) as buf:
            score_ocr.main(["--refs", str(self.refs), "--preds", str(self.preds),
                            "--out-dir", str(out), *extra])
        summary = next(out.glob("*_summary.csv"))
        with summary.open(encoding="utf-8") as f:
            return list(csv.DictReader(f)), buf.getvalue()

    def test_test_split_held_out_by_default(self):
        rows, printed = self.run_cli()
        self.assertIn("Skipped 1 held-out test row", printed)
        self.assertFalse(any(r["split"] == "test" for r in rows))

    def test_synthetic_not_pooled_with_real(self):
        rows, _ = self.run_cli()
        by_cond = {r["condition"]: r for r in rows}
        self.assertEqual(by_cond["real (all)"]["n"], "2")      # printed + handwritten
        self.assertEqual(by_cond["synthetic"]["n"], "1")
        self.assertEqual(by_cond["real (all)"]["critical_free_rate"], "0.5")

    def test_energy_per_critical_free_transcript(self):
        rows, _ = self.run_cli()
        real = next(r for r in rows if r["condition"] == "real (all)")
        self.assertEqual(real["energy_j_per_image"], "2.0")
        self.assertEqual(real["energy_j_per_critical_free"], "4.0")  # 4 J / 1 clean

    def test_include_test(self):
        rows, _ = self.run_cli("--include-test")
        self.assertTrue(any(r["split"] == "test" for r in rows))

    def test_unknown_problem_id_fails(self):
        self.preds.write_text(
            "image_id,problem_id,model,condition,hypothesis\ni9,p404,easyocr,printed,x\n",
            encoding="utf-8")
        with self.assertRaises(SystemExit):
            self.run_cli()


if __name__ == "__main__":
    unittest.main()
