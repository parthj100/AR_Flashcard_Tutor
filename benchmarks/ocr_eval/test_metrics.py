"""
Tests for the scorer. Run from the repo root:

    .venv/bin/python -m unittest discover -s benchmarks/ocr_eval -v
"""

import unittest

from metrics import (cer, critical_errors, critical_tokens, latex_to_plain,
                     normalize, score, to_plain, wer)


def crit(ref, hyp, fmt="plain"):
    return score(ref, hyp, fmt)["critical"]


class LatexToPlain(unittest.TestCase):
    def test_fraction_root_pi_power(self):
        self.assertEqual(latex_to_plain(r"\frac{3}{4}"), "3/4")
        self.assertEqual(latex_to_plain(r"\dfrac{x+1}{2}"), "(x+1)/2")
        self.assertEqual(latex_to_plain(r"\sqrt{16}"), "√16")
        self.assertEqual(latex_to_plain(r"200\pi"), "200π")
        self.assertEqual(latex_to_plain(r"x^{2}"), "x^2")
        self.assertEqual(latex_to_plain(r"2^{x+1}"), "2^(x+1)")

    def test_pix2text_spaced_digits(self):
        # Pix2Text writes 48 as "4 8" inside math; spaces don't count in LaTeX
        self.assertEqual(latex_to_plain(r"\frac{\sqrt{4 8}} {2}"), "√48/2")
        self.assertEqual(latex_to_plain(r"1 6"), "16")
        self.assertEqual(latex_to_plain(r"3 . 5"), "3.5")

    def test_cfrac(self):
        # Pix2Text sometimes writes \cfrac for a fraction
        self.assertEqual(latex_to_plain(r"z+\cfrac{\mathrm{1}} {z}"), "z+1/z")

    def test_function_names_kept(self):
        self.assertEqual(normalize(latex_to_plain(r"\tan A")), "tan A")

    def test_nested(self):
        self.assertEqual(latex_to_plain(r"\frac{\sqrt{2}}{2}"), "√2/2")

    def test_operators_and_units(self):
        self.assertEqual(normalize(latex_to_plain(r"3 \times 4 \div 2")), "3×4÷2")
        self.assertEqual(normalize(latex_to_plain(r"90^\circ")), "90°")
        self.assertEqual(normalize(latex_to_plain(r"50\%")), "50%")

    def test_latex_reference_keeps_currency(self):
        # MATH style: $..$ is math, \$ is money
        self.assertEqual(normalize(to_plain(r"It costs \$5. Find $\frac{1}{2}$ of it.", "latex")),
                         "It costs $5. Find 1/2 of it.")

    def test_plain_reference_dollar_is_currency(self):
        self.assertEqual(to_plain("She sells eggs for $2 each and $3 total.", "plain"),
                         "She sells eggs for $2 each and $3 total.")


class Normalize(unittest.TestCase):
    def test_operator_spacing_and_unicode(self):
        self.assertEqual(normalize("3  +  4 − 2 * 5"), "3+4-2×5")
        self.assertEqual(normalize("x²"), "x^2")


class CriticalTokens(unittest.TestCase):
    def test_numbers_and_thousands(self):
        self.assertEqual(critical_tokens("1,000 and 3.50"), [("number", "1000"), ("decimal", "3.50")])

    def test_hyphenated_words_are_not_minus(self):
        self.assertEqual(critical_tokens("a two-thirds share for a 15-year-old"), [("number", "15")])
        self.assertIn(("operator", "-"), critical_tokens("x-3"))
        self.assertIn(("operator", "-"), critical_tokens("-3"))


class CriticalErrors(unittest.TestCase):
    def test_update2_dollar_example_counts(self):
        # "$3.00" read as "53.00": number changed AND currency sign lost
        ce = critical_errors(normalize("$3.00"), normalize("53.00"))
        self.assertEqual(ce["by_category"]["decimal"], 1)
        self.assertEqual(ce["by_category"]["unit/currency"], 1)
        self.assertEqual(ce["total"], 2)

    def test_pi_read_as_7(self):
        # Update #2: EasyOCR read "200π" as "2007"
        self.assertEqual(crit("Find $200\\pi$.", "Find 2007.", "latex"), 2)

    def test_spelling_and_punctuation_do_not_count(self):
        self.assertEqual(crit("Janet's ducks lay 16 eggs per day.", "Janets duks lay 16 eggs per day"), 0)

    def test_letter_digit_slip_in_word_does_not_count(self):
        # "to" read as "t0" is a spelling slip, not an extra number
        self.assertEqual(crit("She needs to make 20 masks.", "She needs t0 make 20 masks."), 0)
        self.assertEqual(crit("one of them", "one 0f them"), 0)

    def test_currency_misread_as_letter_counts_once(self):
        # Pix2Text read "€1500" as "f1500": the euro sign is lost, the number is right
        self.assertEqual(crit("a budget of €1500", "a budget of f1500"), 1)

    def test_word_order_does_not_count(self):
        self.assertEqual(crit("She has 3 apples and 5 pears", "5 pears and She has 3 apples"), 0)

    def test_dropped_fraction_bar_and_root(self):
        ref = r"Simplify $\frac{\sqrt{8}}{2}$."
        self.assertEqual(crit(ref, "Simplify 8 2.", "latex"), 2)       # lost / and √
        self.assertEqual(crit(ref, "Simplify √8/2.", "latex"), 0)

    def test_dropped_minus(self):
        self.assertEqual(crit("Solve $x - 3 = 7$", "Solve x 3 = 7", "latex"), 1)

    def test_extra_number_from_show_through(self):
        self.assertEqual(crit("Add 2 and 3.", "Add 2 and 3. 48"), 1)

    def test_pix2text_style_latex_matches_plain_reference(self):
        # Pix2Text "$\frac{3}{4}$" converted by run_ocr, vs a plain "3/4"
        hyp = latex_to_plain(r"\frac{3}{4}")
        self.assertEqual(crit("What is 3/4 of 20?", f"What is {hyp} of 20?"), 0)


class Rates(unittest.TestCase):
    def test_cer_wer(self):
        self.assertEqual(cer("abcd", "abcd"), 0.0)
        self.assertAlmostEqual(cer("abcd", "abxd"), 0.25)
        self.assertAlmostEqual(wer("the cat sat", "the hat sat"), 1 / 3)

    def test_low_cer_can_hide_critical_error(self):
        # H4: one wrong digit in a long problem is a tiny CER but a critical error
        ref = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning."
        s = score(ref, ref.replace("16", "18"))
        self.assertLess(s["cer"], 0.05)
        self.assertEqual(s["critical"], 1)


if __name__ == "__main__":
    unittest.main()
