"""
Tests for OCR/latex_plain.py.

Run from the repo root:
  python3 -m unittest discover -s OCR -p "test_*.py" -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from latex_plain import latex_to_plain  # noqa: E402

CASES = [
    (r"\frac{1}{2}", "1/2"),
    (r"\dfrac { 3 } { 4 }", "3/4"),
    (r"\frac{a+b}{c}", "(a + b)/c"),
    (r"x^{2}-4=21", "x^2 - 4 = 21"),
    (r"x ^ { n + 1 }", "x^(n + 1)"),
    (r"10^{-3}", "10^-3"),
    (r"90^{\circ}", "90°"),
    (r"\sqrt{x}", "√(x)"),
    (r"\sqrt[3]{27}", "∛(27)"),
    (r"3 . 7 5", "3.75"),
    (r"1 , 0 0 0", "1,000"),
    (r"1 2 \times 4", "12 × 4"),
    (r"-5+3", "-5 + 3"),
    (r"x \leq -2", "x ≤ -2"),
    (r"\$ 3 . 0 0", "$3.00"),
    (r"5 0 \%", "50%"),
    (r"5\text{ cm}", "5 cm"),
    (r"\left( 2 x + 1 \right) ^ { 2 }", "(2x + 1)^2"),
    (r"\pi r^{2}", "πr^2"),
    (r"$$x=2$$", "x = 2"),
    (r"$x=2$", "x = 2"),
    (r"2 \div \frac { 1 } { 4 }", "2 ÷ 1/4"),
    (r"y = \pm 3", "y = ±3"),
    (r"a \neq b", "a ≠ b"),
    # Seen in Pix2Text output on OCR/test_images
    (r"\bf a", "a"),
    (r"{\bf a} > {\bf g}", "a > g"),
    (r"\mathbf{a}=0", "a = 0"),
    (r"\mu> 0", "μ > 0"),
    (r"0 . 2 5", "0.25"),
    (r"\S2 , \! 0 0 0 . 0 0", "§2,000.00"),  # pilot: "$2,000.00" misread as a section sign
    (r"f ( x )=x^{5} \sqrt{x^{2}-x+7}", "f(x) = x^5√(x^2 - x + 7)"),
    (r"\begin{matrix} a \\ b \end{matrix}", "a b"),
    (r"\begin{aligned} x &= 1 \\ y &= 2 \end{aligned}", "x = 1 y = 2"),
]


class TestLatexToPlain(unittest.TestCase):
    def test_cases(self):
        for latex, plain in CASES:
            with self.subTest(latex=latex):
                self.assertEqual(latex_to_plain(latex), plain)

    def test_empty(self):
        self.assertEqual(latex_to_plain(""), "")
        self.assertEqual(latex_to_plain(None), "")


if __name__ == "__main__":
    unittest.main()
