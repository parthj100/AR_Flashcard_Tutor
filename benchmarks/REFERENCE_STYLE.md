# Writing reference transcripts

CER and WER count every character, so the references need one consistent way of
writing math. Pix2Text's LaTeX is converted into this same style
([`OCR/latex_plain.py`](../OCR/latex_plain.py)), so neither model gains or loses
points for notation alone.

Write the problem exactly as it appears in the photo (same words, same numbers,
same line order), using these conventions for math:

| Write | Not | Notes |
|---|---|---|
| `x^2`, `x^(n + 1)`, `10^-3` | `x²`, `x**2` | Brackets only when the exponent is more than one term |
| `1/2`, `(a + b)/c`, `3 1/2` | `½`, `\frac{1}{2}` | Mixed numbers: whole number, space, fraction |
| `x^2 - 4 = 21` | `x^2-4=21` | One space around `=`, `+`, `-`, `×`, `÷`, `<`, `>`, `≤`, `≥`, `≠`, `≈` |
| `-5`, `x ≤ -2` | `- 5` | No space after a minus sign that belongs to a number |
| `3 × 4` | `3*4`, `3 x 4` | Use `×` when the problem shows a multiplication sign, `·` for a dot |
| `√(16)`, `∛(27)` | `sqrt(16)` | Always bracket what is under the root |
| `90°`, `50%`, `$3.00` | `90 degrees` | Unless the problem itself spells it out |
| `πr^2`, `θ` | `pi r^2` | Greek letters as Unicode |
| `x_1`, `x_(n - 1)` | `x1`, `x₁` | Subscripts like exponents: brackets only for more than one term |
| `...` | `…`, `\cdots` | Three periods for any ellipsis |
| `1,000` | `1000` | Keep the separator if the problem has one |

Line breaks in the photo can be a single space or a new line; both are treated as
one space.

Check every reference against the original problem (GSM8K or your own source),
not against an OCR output, so neither model's mistakes end up in the reference.
