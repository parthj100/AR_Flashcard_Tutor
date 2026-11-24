# Labeling critical errors by hand

You have a sheet (`labels_A.csv` or `labels_B.csv`). Open it in Excel, Numbers or Google Sheets.
Each row shows the correct text (`reference`) and what the OCR tool read (`ocr_output`).

**Do not open `key.csv`, and do not compare answers with the other labeler until both are done.**

For each row:

1. `critical_error`: type `y` if the OCR output changed, dropped or added anything that would change the
   math. Otherwise type `n`.
2. If `y`, put `y` under each kind of error you see (leave the others blank):

| Column | Means | Example (correct -> OCR) |
|---|---|---|
| `number` | a number is different, missing or extra | 18 -> 13 |
| `decimal` | same digits, but the decimal point moved or vanished | 3.50 -> 350 |
| `sign_operator` | a minus sign or an operator (+ - x / = < > root) changed or vanished | 8 + 3 -> 8 - 3 |
| `fraction_exponent` | a fraction or power lost its shape | x^2 -> x2, 1/2 -> 12 |
| `unit_currency` | a $, %, degree sign or unit (cm, kg...) changed or vanished | $40 -> 40 |

**Not critical** (answer `n` if this is all that is wrong): spelling mistakes in words, capital letters,
punctuation, spacing, line breaks, "1,000" vs "1000", and different ways of writing the same thing
(`x^2` and `x²`, `3 × 4` and `3 * 4`).

Use `comment` for anything you were unsure about. Save as CSV with the same file name when done.
