"""
Transcript scoring for the Fall 2026 OCR study.

Compares an OCR transcript (hypothesis) with its checked reference transcript
and reports:

  - CER: character edit distance / reference length
  - WER: word edit distance / reference word count
  - critical errors: changes to a quantity, labeled with one of the rubric's
    categories (see CATEGORIES)

Standard library only, so it runs anywhere the OCR service runs.

How critical errors are found
-----------------------------
Both texts are reduced to a sequence of "quantity tokens", the only parts of a
math problem where a misread can change the answer:

  number            3, 3.00, 1,000, .5, π
  sign / operator   + - * = < > <= >= ± ≠ ≈ √ ∛   (a "-" counts only before a number,
                    so "half-hour" and "5-year-old" are hyphens, not minus signs)
  fraction/exponent "/" next to a number, "^", and the numbers they bind
  unit / currency   $ € £ ¥ ¢ % °, and unit abbreviations (kg, km, mph, ...)
                    that directly follow a number. Full unit words such as
                    "minutes" are not tokens, so misspelling them is not critical.

The two token sequences are aligned with edit distance. Every substitution,
deletion or insertion in that alignment is one critical error. Everything
else (spelling, punctuation, letter case, spacing) affects CER/WER but is
never critical, matching the rubric: "Punctuation and spelling that do not
change a quantity do not count."

Normalization applied before tokenizing, so equivalent ways of writing the
same quantity are not errors:
  x^2 and x² are the same; 3½ is 3 1/2; 1,000 is 1000; the Unicode minus,
  en dash, times and divide signs map to - * /; "3 x 4" is 3 * 4.

These rules are an automatic first pass. The rubric locked on October 28 is
the authority: compare a sample of these labels with your hand labels and
report the agreement before relying on them.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --- Categories ---------------------------------------------------------------

NUMBER = "number"
DECIMAL = "decimal"
SIGN_OPERATOR = "sign_operator"
FRACTION_EXPONENT = "fraction_exponent"
UNIT_CURRENCY = "unit_currency"

CATEGORIES = (NUMBER, DECIMAL, SIGN_OPERATOR, FRACTION_EXPONENT, UNIT_CURRENCY)

# Unit symbols written as letters. They count only when they directly follow a
# number ("5 kg", "12km"). Full unit words (minutes, dollars) are left out on
# purpose: "minuts" for "minutes" is a spelling error, which the rubric excludes.
# "in" and "l" are left out because "put 3 in the bag" and 1/l confusions make
# them too noisy.
UNIT_WORDS = frozenset("""
    mm cm m km ft yd mi
    mg g kg lb lbs oz
    ml gal qt pt
    s sec secs min mins h hr hrs
    mph kph
""".split())

CURRENCY_UNIT_SYMBOLS = "$€£¥¢%°"

# --- Normalization -------------------------------------------------------------

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUPERSCRIPT_RUN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+")

_VULGAR_FRACTIONS = {
    "½": "1/2", "⅓": "1/3", "⅔": "2/3", "¼": "1/4", "¾": "3/4",
    "⅕": "1/5", "⅖": "2/5", "⅗": "3/5", "⅘": "4/5", "⅙": "1/6", "⅚": "5/6",
    "⅐": "1/7", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8", "⅞": "7/8", "⅑": "1/9", "⅒": "1/10",
}

_SYMBOL_MAP = str.maketrans({
    "−": "-",  # minus sign
    "–": "-",  # en dash
    "‒": "-",  # figure dash
    "×": "*",
    "·": "*",
    "∙": "*",
    "÷": "/",
    "⁄": "/",  # fraction slash
    "∕": "/",  # division slash
})

_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFKC + collapsed whitespace. Used for CER and WER."""
    return _WS.sub(" ", unicodedata.normalize("NFKC", text or "")).strip()


def normalize_math(text: str) -> str:
    """Normalization for quantity tokens: equivalent notations become identical."""
    text = text or ""
    text = _SUPERSCRIPT_RUN.sub(lambda m: "^" + m.group(0).translate(_SUPERSCRIPTS), text)
    for frac, plain in _VULGAR_FRACTIONS.items():
        text = text.replace(frac, f" {plain}")
    text = text.translate(_SYMBOL_MAP)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("≤", "<=").replace("≥", ">=")
    return text


# --- Edit distance ---------------------------------------------------------------

def levenshtein(a, b) -> int:
    """Edit distance between two sequences (strings or lists)."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Returns (rate, edits, reference length) on normalize_text()'d strings."""
    ref, hyp = normalize_text(reference), normalize_text(hypothesis)
    edits = levenshtein(ref, hyp)
    return _rate(edits, len(ref), len(hyp)), edits, len(ref)


def wer(reference: str, hypothesis: str) -> tuple[float, int, int]:
    """Returns (rate, edits, reference word count). Words are whitespace-separated."""
    ref, hyp = normalize_text(reference).split(), normalize_text(hypothesis).split()
    edits = levenshtein(ref, hyp)
    return _rate(edits, len(ref), len(hyp)), edits, len(ref)


def _rate(edits: int, ref_len: int, hyp_len: int) -> float:
    if ref_len == 0:
        return 0.0 if hyp_len == 0 else 1.0
    return edits / ref_len


# --- Quantity tokens -------------------------------------------------------------

@dataclass(frozen=True)
class Token:
    kind: str      # "num", "op", "unit"
    value: str     # normalized value, compared during alignment
    text: str      # as written, for reports
    category: str  # rubric category if this token is lost or changed


_SCAN = re.compile(r"""
    (?P<num>\d+(?:[.,]\d+)*|(?<![\d.,])[.,]\d+|π)
  | (?P<op><=|>=|[-+*/=<>^±∓√∛∜≠≈])
  | (?P<sym>[$€£¥¢%°])
  | (?P<word>[A-Za-z]+)
""", re.VERBOSE)

_THOUSANDS = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


def _number_value(s: str) -> str:
    if _THOUSANDS.match(s):
        s = s.replace(",", "")
    if s.startswith((".", ",")):
        s = "0" + s
    return s


def quantity_tokens(text: str) -> list[Token]:
    """Reduce text to the tokens whose change would change a quantity."""
    norm = normalize_math(text)
    matches = list(_SCAN.finditer(norm))
    raw = [(m.lastgroup, m.group(0)) for m in matches]

    def spaced(i):  # whitespace on both sides of token i ("3 x 4", not "x_1x_2")
        s, e = matches[i].span()
        return s > 0 and norm[s - 1].isspace() and e < len(norm) and norm[e].isspace()

    def kind_at(i):
        return raw[i][0] if 0 <= i < len(raw) else None

    tokens: list[Token] = []
    for i, (kind, s) in enumerate(raw):
        prev_num = kind_at(i - 1) == "num"
        next_num = kind_at(i + 1) == "num"

        if kind == "num":
            binds = (kind_at(i - 1) == "op" and raw[i - 1][1] in "/^") or \
                    (kind_at(i + 1) == "op" and raw[i + 1][1] == "/")
            cat = FRACTION_EXPONENT if binds else NUMBER
            tokens.append(Token("num", _number_value(s), s, cat))
        elif kind == "op":
            if s == "-" and not next_num:
                continue  # hyphen in "half-hour" or "5-year-old"
            if s == "/":
                if not (prev_num or next_num):
                    continue  # "and/or"
                tokens.append(Token("op", s, s, FRACTION_EXPONENT))
            elif s == "^":
                tokens.append(Token("op", s, s, FRACTION_EXPONENT))
            else:
                tokens.append(Token("op", s, s, SIGN_OPERATOR))
        elif kind == "sym":
            tokens.append(Token("unit", s, s, UNIT_CURRENCY))
        elif kind == "word":
            w = s.lower()
            if w == "x" and prev_num and next_num and spaced(i):
                tokens.append(Token("op", "*", s, SIGN_OPERATOR))  # "3 x 4"
            elif w in UNIT_WORDS and (prev_num or (i >= 2 and raw[i - 1] == ("op", "-") and kind_at(i - 2) == "num")):
                # "5 kg", and also "5-year-old", so a dropped hyphen is not a unit change
                tokens.append(Token("unit", w, s, UNIT_CURRENCY))
    return tokens


# --- Critical errors ----------------------------------------------------------------

@dataclass(frozen=True)
class CriticalError:
    op: str        # "sub", "del" (in reference, missing from OCR), "ins" (extra in OCR)
    category: str
    ref: str
    hyp: str

    def describe(self) -> str:
        if self.op == "sub":
            return f"{self.category}: {self.ref!r} -> {self.hyp!r}"
        if self.op == "del":
            return f"{self.category}: {self.ref!r} missing"
        return f"{self.category}: extra {self.hyp!r}"


def _digits(s: str) -> str:
    return re.sub(r"[.,]", "", s).lstrip("0") or "0"


def _sub_category(r: Token, h: Token) -> str:
    if r.kind == "num" and h.kind == "num" and r.category == NUMBER \
            and _digits(r.value) == _digits(h.value):
        return DECIMAL  # same digits, decimal point moved, lost or added: 3.50 -> 350
    return r.category


def critical_errors(reference: str, hypothesis: str) -> list[CriticalError]:
    """Align quantity tokens and return every change, in reference order."""
    ref, hyp = quantity_tokens(reference), quantity_tokens(hypothesis)
    n, m = len(ref), len(hyp)

    # Substituting across kinds (a number for a "$") costs more than a
    # delete + insert, so the alignment keeps numbers paired with numbers.
    def sub_cost(i, j):
        if ref[i].value == hyp[j].value:
            return 0
        return 1 if ref[i].kind == hyp[j].kind else 3

    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + sub_cost(i - 1, j - 1),
                          d[i - 1][j] + 1,
                          d[i][j - 1] + 1)

    errors: list[CriticalError] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + sub_cost(i - 1, j - 1):
            r, h = ref[i - 1], hyp[j - 1]
            if r.value != h.value:
                errors.append(CriticalError("sub", _sub_category(r, h), r.text, h.text))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            r = ref[i - 1]
            errors.append(CriticalError("del", r.category, r.text, ""))
            i -= 1
        else:
            h = hyp[j - 1]
            errors.append(CriticalError("ins", h.category, "", h.text))
            j -= 1
    errors.reverse()
    return errors


# --- One transcript ---------------------------------------------------------------------

@dataclass
class TranscriptScore:
    cer: float
    cer_edits: int
    ref_chars: int
    wer: float
    wer_edits: int
    ref_words: int
    errors: list[CriticalError] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return len(self.errors)

    @property
    def critical_free(self) -> bool:
        return not self.errors

    def category_counts(self) -> dict[str, int]:
        counts = dict.fromkeys(CATEGORIES, 0)
        for e in self.errors:
            counts[e.category] += 1
        return counts


def score(reference: str, hypothesis: str) -> TranscriptScore:
    c, c_edits, c_len = cer(reference, hypothesis)
    w, w_edits, w_len = wer(reference, hypothesis)
    return TranscriptScore(c, c_edits, c_len, w, w_edits, w_len,
                           critical_errors(reference, hypothesis))
