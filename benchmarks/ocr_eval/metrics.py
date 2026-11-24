"""
Scoring for the OCR study: CER, WER and critical errors.

Everything is compared in one canonical *plain* form, so a reference typed
as LaTeX (MATH problems) and a Pix2Text reading in LaTeX can be compared with
a plain reference (GSM8K) and EasyOCR's plain text:

    \\frac{3}{4}      -> 3/4         \\sqrt{16} -> √16      \\pi -> π
    x^{2}            -> x^2          \\times, \\cdot, * -> ×  − -> -
    2^{x+1}          -> 2^(x+1)      ^\\circ -> °          1,000 -> 1000 (numbers only)

Critical error (Research Update #2): a misread that changes a number or a
math symbol. Spelling and punctuation do not count. We extract the
"critical tokens" of each text and compare them as multisets (order-free,
so EasyOCR's line-order mix-ups are not double-counted here; they still
show up in CER). Per category, errors = max(missing, extra), so a
substitution like "3.00" -> "53.00" is one error, not two.

Categories:
    number         integers                       16, 1000
    decimal        numbers with a decimal point   3.00, 0.5
    operator       signs and relations            + - × ÷ = < > ≤ ≥ ≠ ±
    frac/pow/root  structure of math              / ^ √ π
    unit/currency                                 $ % ° ¢ € £
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Dict, List, Sequence, Tuple

# --- LaTeX -> plain ---------------------------------------------------------

_NAMED = {
    "pi": "π", "times": "×", "cdot": "×", "ast": "×", "div": "÷", "pm": "±",
    "leq": "≤", "le": "≤", "leqslant": "≤", "geq": "≥", "ge": "≥", "geqslant": "≥",
    "neq": "≠", "ne": "≠", "circ": "°", "degree": "°", "infty": "∞",
    "cdots": "...", "ldots": "...", "dots": "...", "lt": "<", "gt": ">",
    "minus": "-", "colon": ":",
}
# Function names stay as words: \tan A -> "tan A".
_NAMED.update({f: f" {f} " for f in ("sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "exp",
                                     "lim", "max", "min", "det", "gcd", "deg")})
_WRAPPERS = r"text|textrm|textbf|textit|mathrm|mathbf|mathit|mathsf|operatorname|boldsymbol|mbox|hbox"
_FRAC  = re.compile(r"\\[dtc]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_FRAC_SHORT = re.compile(r"\\[dtc]?frac\s*(\w)\s*(\w)")
_SQRT  = re.compile(r"\\sqrt\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}")
_SQRT_SHORT = re.compile(r"\\sqrt\s*(\w)")


def _wrap(x: str) -> str:
    """Parenthesise a multi-token group: {x+1} -> (x+1), {2} -> 2."""
    x = x.strip()
    return x if re.fullmatch(r"√?[\w.π]+", x) else f"({x})"


def latex_to_plain(s: str) -> str:
    """Convert one LaTeX math fragment (no $ delimiters) to plain linear text."""
    # Spaces are insignificant in LaTeX math, and Pix2Text emits "4 8" for 48
    # and "3 . 5" for 3.5. Join digit runs so they score as one number.
    s = re.sub(r"(?<=[\d.])\s+(?=[\d.])", "", s)
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\%", "%").replace("\\$", "$")
    s = re.sub(r"\\[,;:! ]", " ", s)                       # spacing commands
    s = re.sub(r"\^\s*\{?\s*\\circ\s*\}?", "°", s)          # 90^\circ -> 90°
    for _ in range(4):
        s = re.sub(rf"\\(?:{_WRAPPERS})\s*\{{([^{{}}]*)\}}", r"\1", s)
    s = re.sub(r"\\(?:bf|it|rm|displaystyle|textstyle)\b\s*", "", s)
    for _ in range(6):                                     # innermost first, nested ok
        before = s
        s = _FRAC.sub(lambda m: f"{_wrap(m[1])}/{_wrap(m[2])}", s)
        s = _FRAC_SHORT.sub(r"\1/\2", s)
        s = _SQRT.sub(lambda m: "√" + _wrap(m[1]), s)
        s = _SQRT_SHORT.sub(r"√\1", s)
        s = re.sub(r"\^\s*\{([^{}]*)\}", lambda m: "^" + _wrap(m[1]), s)
        s = re.sub(r"_\s*\{([^{}]*)\}", r"\1", s)
        if s == before:
            break
    s = re.sub(r"\\([A-Za-z]+)", lambda m: _NAMED.get(m[1], " "), s)
    s = s.replace("_", "").replace("{", "").replace("}", "")
    return s


_MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.S)


def to_plain(text: str, fmt: str = "plain") -> str:
    """fmt="plain": $ is a currency sign. fmt="latex": $..$ wraps math, \\$ is currency."""
    if fmt != "latex":
        return text
    text = text.replace("\\$", "\x00")
    out = _MATH_SPAN.sub(lambda m: latex_to_plain(next(g for g in m.groups() if g is not None)), text)
    out = out.replace("\\%", "%")
    return out.replace("\x00", "$")


# --- normalisation ------------------------------------------------------------

_PRE = {
    "²": "^2", "³": "^3", "¹": "^1", "⁰": "^0",
    "−": "-", "–": "-", "‐": "-", "‑": "-",
    "∗": "×", "*": "×", "·": "×", "⋅": "×", "✕": "×",
    "⁄": "/", "∕": "/",
    "“": '"', "”": '"', "‘": "'", "’": "'",
    "√ ": "√",
}
_OPS = "+-×÷=<>≤≥≠±/^"


def normalize(s: str) -> str:
    """Canonical plain form used for CER, WER and critical tokens."""
    for a, b in _PRE.items():
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Spacing around operators is layout, not content: "3 + 4" == "3+4".
    s = re.sub(rf"\s*([{re.escape(_OPS)}])\s*", r"\1", s)
    s = re.sub(r"√\s+", "√", s)
    return s


# --- critical tokens ------------------------------------------------------------

CATEGORIES = ["number", "decimal", "operator", "frac/pow/root", "unit/currency"]
_NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
_OPERATORS = set("+×÷=<>≤≥≠±-")
_STRUCT = set("/^√π")
_UNITS = set("$%°¢€£")


def _is_word_joiner(s: str, i: int) -> bool:
    """A '-' or '/' joining words ("two-thirds", "15-year-old", "and/or") is not math."""
    nxt = s[i + 1:i + 3]          # followed by a word of 2+ letters (not "x-y" or "km/h")
    return len(nxt) == 2 and nxt.isalpha()


def _is_letter_slip(s: str, m) -> bool:
    """A lone 0 or 1 glued to a letter ("t0", "0f", "a1l") is a misread o/l inside a word:
    a spelling slip, not a number. Applied to reference and reading alike."""
    if m[0] not in ("0", "1"):
        return False
    before = s[m.start() - 1] if m.start() > 0 else " "
    after = s[m.end()] if m.end() < len(s) else " "
    return before.isalpha() or after.isalpha()


def critical_tokens(s: str) -> List[Tuple[str, str]]:
    """(category, token) for every number and math symbol in a normalised string."""
    toks: List[Tuple[str, str]] = []
    covered = set()
    for m in _NUM.finditer(s):
        num = m[0].replace(",", "")
        if _is_letter_slip(s, m):
            covered.update(range(m.start(), m.end()))
            continue
        toks.append(("decimal" if "." in num else "number", num))
        covered.update(range(m.start(), m.end()))
    for i, ch in enumerate(s):
        if i in covered:
            continue
        if ch in ("-", "/") and _is_word_joiner(s, i):
            continue
        if ch in _OPERATORS:
            toks.append(("operator", ch))
        elif ch in _STRUCT:
            toks.append(("frac/pow/root", ch))
        elif ch in _UNITS:
            toks.append(("unit/currency", ch))
    return toks


def critical_errors(ref: str, hyp: str) -> Dict:
    """Compare normalised strings. Returns total, per-category counts, and the diff."""
    r, h = Counter(critical_tokens(ref)), Counter(critical_tokens(hyp))
    missing, extra = r - h, h - r
    by_cat = {}
    for cat in CATEGORIES:
        m = sum(n for (c, _), n in missing.items() if c == cat)
        e = sum(n for (c, _), n in extra.items() if c == cat)
        by_cat[cat] = max(m, e)
    return {
        "total": sum(by_cat.values()),
        "by_category": by_cat,
        "missing": sorted(t for (_, t), n in missing.items() for _ in range(n)),
        "extra":   sorted(t for (_, t), n in extra.items() for _ in range(n)),
    }


# --- edit distance ----------------------------------------------------------------

def levenshtein(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate on normalised strings (can exceed 1 when OCR adds text)."""
    return levenshtein(ref, hyp) / max(1, len(ref))


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    return levenshtein(r, h) / max(1, len(r))


def score(ref_text: str, hyp_text: str, ref_fmt: str = "plain") -> Dict:
    """Score one reading. hyp_text must already be plain (see run_ocr.plain_transcript)."""
    ref = normalize(to_plain(ref_text, ref_fmt))
    hyp = normalize(hyp_text)
    ce = critical_errors(ref, hyp)
    return {
        "ref_norm": ref, "hyp_norm": hyp,
        "cer": cer(ref, hyp), "wer": wer(ref, hyp),
        "critical": ce["total"], "error_free": ce["total"] == 0,
        "by_category": ce["by_category"],
        "missing": ce["missing"], "extra": ce["extra"],
    }
